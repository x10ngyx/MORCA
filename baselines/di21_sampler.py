# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Adapted for MORCA; see licenses/NOTICE.txt.
from __future__ import annotations
import gc
import logging
import math
import random
import sys
import types
from typing import Any
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
from .di21 import DiCacheConfig, DiCacheController


def dicache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    dicache_branch=None,
    dicache_step_index=None,
    dicache_num_steps=None,
):
    from wan.modules.model import sinusoidal_embedding_1d

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    x = [self.patch_embedding(item.unsqueeze(0)) for item in x]
    grid_sizes = torch.stack([torch.tensor(item.shape[2:], dtype=torch.long) for item in x])
    x = [item.flatten(2).transpose(1, 2) for item in x]
    seq_lens = torch.tensor([item.size(1) for item in x], dtype=torch.long)
    x = torch.cat(
        [
            torch.cat([item, item.new_zeros(1, seq_len - item.size(1), item.size(2))], dim=1)
            for item in x
        ]
    )
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
    context_lens = None
    context = self.text_embedding(
        torch.stack(
            [
                torch.cat([item, item.new_zeros(self.text_len - item.size(0), item.size(1))])
                for item in context
            ]
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
    x = self.dicache_controller.execute(
        blocks=self.blocks,
        x=x,
        kwargs=kwargs,
        branch=dicache_branch,
        step_index=dicache_step_index,
        num_steps=dicache_num_steps,
    )
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    return [item.float() for item in x]


def t2v_generate(
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
    offload_model=False,
):
    from contextlib import contextmanager as local_contextmanager
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    controller: DiCacheController = self.model.dicache_controller
    controller.reset()
    target_shape = (
        self.vae.model.z_dim,
        (frame_num - 1) // self.vae_stride[0] + 1,
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
    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device("cpu"))
        context_null = self.text_encoder([n_prompt], torch.device("cpu"))
        context = [item.to(self.device) for item in context]
        context_null = [item.to(self.device) for item in context_null]
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

    @local_contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, "no_sync", noop_no_sync)
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
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
        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        for step_index, t in enumerate(tqdm(timesteps)):
            timestep = torch.stack([t])
            self.model.to(self.device)
            noise_pred_cond = self.model(
                latents,
                t=timestep,
                dicache_branch="cond",
                dicache_step_index=step_index,
                dicache_num_steps=len(timesteps),
                **arg_c,
            )[0]
            noise_pred_uncond = self.model(
                latents,
                t=timestep,
                dicache_branch="uncond",
                dicache_step_index=step_index,
                dicache_num_steps=len(timesteps),
                **arg_null,
            )[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]
        if self.rank == 0:
            logging.info("DiCache summary: %s", controller.summary())
        controller.write_trace(
            extra={
                "task": "t2v",
                "sampling_steps": int(sampling_steps),
                "sample_solver": sample_solver,
                "shift": float(shift),
                "guide_scale": float(guide_scale),
                "frame_num": int(frame_num),
                "size_wh": [int(size[0]), int(size[1])],
            }
        )
        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        videos = self.vae.decode(x0) if self.rank == 0 else None
    del noise, latents, sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return videos[0] if self.rank == 0 else None


def apply_dicache(
    pipeline: Any,
    *,
    task: str,
    threshold: float,
    probe_depth: int,
    retention_ratio: float,
    dcta: bool,
    trace_path: str | None,
) -> None:
    config = DiCacheConfig(
        threshold=threshold,
        probe_depth=probe_depth,
        retention_ratio=retention_ratio,
        dcta=dcta,
        trace_path=trace_path,
    )
    pipeline.generate = types.MethodType(t2v_generate, pipeline)
    pipeline.model.forward = types.MethodType(dicache_forward, pipeline.model)
    pipeline.model.dicache_controller = DiCacheController(config)
    logging.info("Enabled DiCache4Wan21: %s", config)
