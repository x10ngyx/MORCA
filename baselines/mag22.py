from __future__ import annotations
from contextlib import contextmanager
from types import SimpleNamespace

STATE_NAMES = (
    "forward",
    "cnt",
    "num_steps",
    "split_step",
    "mode",
    "magcache_thresh",
    "K",
    "accumulated_err",
    "accumulated_steps",
    "accumulated_ratio",
    "retention_ratio",
    "residual_cache",
    "mag_ratios",
    "norm_ratio",
    "norm_std",
    "cos_dis",
)
MISSING = object()


def load_official(sinusoidal_embedding=None):
    from . import mag22_core

    return mag22_core


def method_args(threshold=0.06, k=2, retention_ratio=0.4, sample_steps=50):
    return SimpleNamespace(
        magcache_thresh=threshold,
        magcache_K=k,
        retention_ratio=retention_ratio,
        sample_steps=sample_steps,
    )


@contextmanager
def official_forward(models, *, core, args, split_steps):
    models = list(models)
    cls = type(models[0])
    class_state = {n: cls.__dict__.get(n, MISSING) for n in STATE_NAMES}
    instance_state = [{n: m.__dict__.get(n, MISSING) for n in STATE_NAMES} for m in models]
    cls._magcache4wan22_active = True
    try:
        core.init_magcache(models[0], list(core.ratios), args, split_steps)
        for model in models:
            for name in STATE_NAMES:
                if name in model.__dict__:
                    delattr(model, name)
        yield
    finally:
        for model, saved in zip(models, instance_state):
            for name, original in saved.items():
                if original is MISSING:
                    if name in model.__dict__:
                        delattr(model, name)
                else:
                    setattr(model, name, original)
        for name, original in class_state.items():
            if original is MISSING:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, original)
        delattr(cls, "_magcache4wan22_active")
