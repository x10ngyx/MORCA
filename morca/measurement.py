import time
from contextlib import contextmanager, nullcontext
import torch


@contextmanager
def measure(pipe, model, profile=False):
    totals = {k: dict(seconds=0.0, tflops=0.0 if profile else None) for k in ("t5", "dit", "vae")}
    saved = []

    def wrap(obj, name, component, proxy=False):
        original = getattr(obj, name)
        saved.append((obj, name, original))

        def measured(*args, **kwargs):
            from torch.utils.flop_counter import FlopCounterMode

            counter = FlopCounterMode(display=False) if profile else nullcontext()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with counter:
                value = original(*args, **kwargs)
            torch.cuda.synchronize()
            totals[component]["seconds"] += time.perf_counter() - tick
            if profile:
                totals[component]["tflops"] += counter.get_total_flops() / 1000000000000.0
            return value

        if proxy:

            class Encoder:
                def __getattr__(self, key):
                    return getattr(original, key)

                def __call__(self, *args, **kwargs):
                    return measured(*args, **kwargs)

            setattr(obj, name, Encoder())
        else:
            setattr(obj, name, measured)

    wrap(pipe, "text_encoder", "t5", proxy=True)
    wrap(pipe.vae, "decode", "vae")
    for model_object in (
        [pipe.model] if model == "wan21" else [pipe.high_noise_model, pipe.low_noise_model]
    ):
        wrap(model_object, "forward", "dit")
    try:
        torch.cuda.synchronize()
        tick = time.perf_counter()
        yield totals
        torch.cuda.synchronize()
        totals["inference_seconds"] = time.perf_counter() - tick
    finally:
        for obj, name, fn in reversed(saved):
            setattr(obj, name, fn)
