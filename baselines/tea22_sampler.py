# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Adapted for MORCA; see licenses/NOTICE.txt.
import json
import gc, logging, math, random, sys
from contextlib import contextmanager
import torch
import torch.distributed as dist
from tqdm import tqdm
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .tea22 import TeaCacheConfig as TeaCacheConfig, TeaCacheController
from wan.modules.model import sinusoidal_embedding_1d


def generate(
    self,
    input_prompt,
    size=(1280, 720),
    frame_num=81,
    shift=5.0,
    sample_solver="unipc",
    sampling_steps=50,
    guide_scale=5.0,
    n_prompt="",
    seed=-1,
    offload_model=True,
    teacache_config=None,
):
    guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale
    F = frame_num
    target_shape = (
        self.vae.model.z_dim,
        (F - 1) // self.vae_stride[0] + 1,
        size[1] // self.vae_stride[1],
        size[0] // self.vae_stride[2],
    )
    seq_len = (
        math.ceil(
            target_shape[2]
            * target_shape[3]
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        )
        * self.sp_size
    )
    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    teacache = None
    if teacache_config is not None:
        teacache = TeaCacheController(teacache_config)
    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device("cpu"))
        context_null = self.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]
    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g,
        )
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low_noise = getattr(self.low_noise_model, "no_sync", noop_no_sync)
    no_sync_high_noise = getattr(self.high_noise_model, "no_sync", noop_no_sync)
    with torch.amp.autocast(
        "cuda", dtype=self.param_dtype
    ), torch.no_grad(), no_sync_low_noise(), no_sync_high_noise():
        boundary = self.boundary * self.num_train_timesteps
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=self.device, sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError("Unsupported solver.")
        runtime_protocol = {
            "task": "t2v-A14B",
            "size_wh": [int(size[0]), int(size[1])],
            "frame_num": int(frame_num),
            "sampling_steps": int(sampling_steps),
            "sample_solver": sample_solver,
            "shift": float(shift),
            "guide_scale_low_high": [float(guide_scale[0]), float(guide_scale[1])],
            "boundary": float(self.boundary),
            "param_dtype": str(self.param_dtype),
            "use_ret_steps": False,
        }
        if teacache is not None:
            pass
        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        previous_model_stage = None
        for step_index, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = [t]
            timestep = torch.stack(timestep)
            model_stage = "high" if t.item() >= boundary else "low"
            if (
                teacache is not None
                and previous_model_stage is not None
                and (previous_model_stage != model_stage)
            ):
                teacache.clear_stage(previous_model_stage)
                torch.cuda.empty_cache()
                logging.info(f"Cleared TeaCache state for completed {previous_model_stage} stage.")
            previous_model_stage = model_stage
            model = self._prepare_model_for_timestep(t, boundary, offload_model)
            sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]
            noise_pred_cond = model(
                latent_model_input,
                t=timestep,
                teacache=teacache,
                teacache_stage=model_stage,
                teacache_branch="cond",
                teacache_step_index=step_index,
                teacache_num_steps=len(timesteps),
                **arg_c,
            )[0]
            noise_pred_uncond = model(
                latent_model_input,
                t=timestep,
                teacache=teacache,
                teacache_stage=model_stage,
                teacache_branch="uncond",
                teacache_step_index=step_index,
                teacache_num_steps=len(timesteps),
                **arg_null,
            )[0]
            noise_pred = noise_pred_uncond + sample_guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]
        x0 = latents
        if teacache is not None:
            teacache.release()
            torch.cuda.empty_cache()
        if offload_model:
            self.low_noise_model.cpu()
            self.high_noise_model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)
            if teacache is not None:
                summary = teacache.summary()
                logging.info(
                    "TeaCache summary: %s",
                    json.dumps(
                        {key: value for (key, value) in summary.items() if key != "decisions"},
                        sort_keys=True,
                    ),
                )
                trace_path = teacache.write_trace(extra={"runtime_protocol": runtime_protocol})
                if trace_path is not None:
                    logging.info("TeaCache trace: %s", trace_path)
    del noise, latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return videos[0] if self.rank == 0 else None


def forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    teacache=None,
    teacache_stage=None,
    teacache_branch=None,
    teacache_step_index=None,
    teacache_num_steps=None,
):
    if self.model_type == "i2v":
        pass
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    if y is not None:
        x = [torch.cat([u, v], dim=0) for (u, v) in zip(x, y)]
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    x = torch.cat(
        [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
    )
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float()
        )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
    context_lens = None
    context = self.text_embedding(
        torch.stack(
            [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
        )
    )
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )
    if teacache is None:
        for block in self.blocks:
            x = block(x, **kwargs)
    else:
        compact_e = e[:, 0, :]
        should_reuse = teacache.plan_step(
            teacache_stage, teacache_step_index, teacache_num_steps, compact_e
        )
        if should_reuse:
            x = x + teacache.reuse_residual(teacache_stage, teacache_branch, teacache_step_index)
        else:
            block_input = x.detach().clone()
            for block in self.blocks:
                x = block(x, **kwargs)
            teacache.record_recompute(
                teacache_stage, teacache_branch, teacache_step_index, x - block_input
            )
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


__all__ = ["TeaCacheConfig", "generate", "forward"]
