import importlib
import json
from copy import deepcopy
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType
import torch

ROOT = Path(__file__).resolve().parents[1]


def inference_config(model):
    return json.loads((ROOT / "config" / model / "model.json").read_text())


def load_pipeline(model, source, checkpoint):
    sys.path.insert(0, str(Path(source).resolve()))
    import wan
    from wan.configs import WAN_CONFIGS

    settings = inference_config(model)["pipeline"]
    config = deepcopy(WAN_CONFIGS["t2v-1.3B" if model == "wan21" else "t2v-A14B"])
    config.param_dtype = getattr(torch, settings["dit_dtype"])
    if model == "wan22":
        config.boundary = settings["boundary"]
    pipe = wan.WanT2V(
        config=config,
        checkpoint_dir=str(checkpoint),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        t5_cpu=settings["t5_cpu"],
        **(
            dict(use_sp=False, convert_model_dtype=True)
            if model == "wan22"
            else dict(use_usp=False)
        ),
    )
    if model == "wan21":
        pipe.model.to(device=pipe.device, dtype=config.param_dtype)
        if not settings["t5_cpu"]:
            pipe.text_encoder.model.to(pipe.device)
        pipe.vae.model.to(pipe.device)
    return pipe


def generation_options(model):
    options = inference_config(model)["generation"]
    options["size"] = tuple(options["size"])
    if isinstance(options["guide_scale"], list):
        options["guide_scale"] = tuple(options["guide_scale"])
    return options


@contextmanager
def method_context(
    pipe,
    model,
    method,
    *,
    policy=None,
    budget=25,
    threshold=0.2,
    retention=0.2,
    max_skip=4,
):
    version = model[-2:]
    models = [pipe.model] if model == "wan21" else [pipe.high_noise_model, pipe.low_noise_model]
    generate = pipe.generate
    forwards = [m.forward for m in models]
    context = nullcontext()
    factory = None
    controller = None
    options = {}
    try:
        if method in ("morca", "random", "seacache"):
            sampler = importlib.import_module(f"baselines.sea{version}_sampler")
            if method != "seacache":
                cls = importlib.import_module(f".controller{version}", __package__).Controller
                controller = cls(policy, budget)
            if model == "wan21":
                if controller is None:
                    controller = sampler.SeaCacheController(
                        sampler.SeaCacheConfig(threshold=threshold)
                    )
                pipe.model.seacache_controller = controller
                pipe.generate = MethodType(sampler.t2v_generate, pipe)
                original = sampler.seacache_forward
            else:
                if controller is None:
                    controller = sampler.SeaCacheController(
                        sampler.SeaCacheConfig(threshold=threshold)
                    )
                factory = sampler.SeaCacheController
                sampler.SeaCacheController = lambda config: controller
                options["seacache_config"] = sampler.SeaCacheConfig(threshold=threshold)
                pipe.generate = MethodType(sampler.generate, pipe)
                original = sampler.forward
            for m in models:

                def forward(self, x, *args, **kwargs):
                    if method != "seacache" and kwargs.get("seacache_branch") == "cond":
                        with torch.autocast(x[0].device.type, enabled=False):
                            controller.observe_latent(x[0], kwargs["seacache_step_index"])
                    return original(self, x, *args, **kwargs)

                m.forward = MethodType(forward, m)
        elif method == "teacache":
            if model == "wan21":
                from baselines import tea21 as tea

                pipe.generate = MethodType(tea.t2v_generate, pipe)
                m = pipe.model
                m.forward = MethodType(tea.teacache_forward, m)
                m.enable_teacache = True
                m.cnt, m.num_steps = (0, 100)
                m.teacache_thresh = threshold
                m.accumulated_rel_l1_distance_even = m.accumulated_rel_l1_distance_odd = 0
                m.previous_e0_even = m.previous_e0_odd = None
                m.previous_residual_even = m.previous_residual_odd = None
                m.use_ref_steps, m.ret_steps, m.cutoff_steps = (False, 2, 98)
                m.coefficients = list(tea.T2V_COEFFICIENTS["1.3B", False])
            else:
                from baselines import tea22_sampler as tea

                pipe.generate = MethodType(tea.generate, pipe)
                for m in models:
                    m.forward = MethodType(tea.forward, m)
                options["teacache_config"] = tea.TeaCacheConfig(
                    threshold=threshold, coefficients_path=str(ROOT / "data/teacache_wan22.json")
                )
        elif method in ("magcache", "dicache"):
            short = method[:3] if method == "magcache" else "di"
            module = importlib.import_module(
                f"baselines.{short}{version}" + ("_sampler" if model == "wan21" else ""),
            )
            if model == "wan21":
                fn = getattr(module, "apply_" + method)
                kwargs = dict(
                    task="t2v-1.3B", threshold=threshold, retention_ratio=retention, trace_path=None
                )
                kwargs.update(
                    dict(max_skip_steps=max_skip)
                    if method == "magcache"
                    else dict(probe_depth=1, dcta=True)
                )
                fn(pipe, **kwargs)
            else:
                kwargs = dict(threshold=threshold, retention_ratio=retention, sample_steps=50)
                if method == "magcache":
                    kwargs["k"] = max_skip
                args = module.method_args(**kwargs)
                extra = dict(split_steps=32) if method == "magcache" else dict(pipeline=pipe)
                context = module.official_forward(
                    models, core=module.load_official(), args=args, **extra
                )
        with context:
            yield (controller, options)
    finally:
        pipe.generate = generate
        for m, forward in zip(models, forwards):
            m.forward = forward
        if factory is not None:
            sampler.SeaCacheController = factory
        for m in models:
            for key in ("seacache_controller", "magcache_controller", "dicache_controller"):
                m.__dict__.pop(key, None)
