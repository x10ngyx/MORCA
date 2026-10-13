from __future__ import annotations
import math
from contextlib import contextmanager
from types import SimpleNamespace, MethodType

STATE_NAMES = (
    "cnt",
    "num_steps",
    "probe_depth",
    "rel_l1_thresh",
    "ret_ratio",
    "accumulated_rel_l1_distance",
    "residual_cache",
    "probe_residual_cache",
    "residual_window",
    "probe_residual_window",
    "previous_internal_states",
    "previous_input",
    "previous_output",
    "resume_flag",
)
MISSING = object()


def load_official():
    from .dicache_blocks import blocks

    def adapted(self, x, kwargs):
        x, skip, delta, gamma = blocks(self, x, kwargs)
        return (x, dict(skip=skip, delta_y=delta, gamma=gamma))

    return SimpleNamespace(blocks=adapted)


def method_args(threshold=0.08, retention_ratio=0.2, sample_steps=50):
    return SimpleNamespace(
        rel_l1_thresh=threshold, ret_ratio=retention_ratio, sample_steps=sample_steps, probe_depth=1
    )


def initialize(model, args):
    model.cnt, model.num_steps = (0, args.sample_steps * 2)
    model.probe_depth, model.rel_l1_thresh, model.ret_ratio = (
        1,
        args.rel_l1_thresh,
        args.ret_ratio,
    )
    clear_history(model)


def clear_history(model):
    model.accumulated_rel_l1_distance = [0.0, 0.0]
    model.residual_cache = [None, None]
    model.probe_residual_cache = [None, None]
    model.residual_window = [[], []]
    model.probe_residual_window = [[], []]
    model.previous_internal_states = [None, None]
    model.previous_input = [None, None]
    model.previous_output = [None, None]
    model.resume_flag = [False, False]


def compose_forward(native_forward, block_call):
    from .di22_forward import forward

    return lambda self, *args, **kwargs: forward(
        self, *args, _dicache_block_call=block_call, **kwargs
    )


def scalar(value):
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else str(result)


@contextmanager
def official_forward(models, *, core, args, records=None, pipeline=None):
    models = list(models)
    names = ("forward",) + STATE_NAMES
    saved = [{k: m.__dict__.get(k, MISSING) for k in names} for m in models]
    current = {"count": 0, "expert": 0, "active": None}
    local_counts = [0] * len(models)
    handles = []
    stages = ("high", "low") if len(models) == 2 else ("high",)
    saved_prepare = MISSING

    def block_call(model, x, kwargs):
        row = current["active"]
        branch = row["call_index"] % 2
        before = scalar(model.accumulated_rel_l1_distance[branch])
        x, info = core.blocks(model, x, kwargs)
        row.update(
            action="reuse" if info["skip"] else "recompute",
            accumulated_error_before=before,
            accumulated_error_after=scalar(model.accumulated_rel_l1_distance[branch]),
            probe_relative_l1=scalar(info["delta_y"]),
            gamma=scalar(info["gamma"]),
            residual_window_length=len(model.residual_window[branch]),
            block_output_dtype=str(x.dtype),
        )
        return x

    try:
        if pipeline is not None:
            saved_prepare = pipeline.__dict__.get("_prepare_model_for_timestep", MISSING)
            native_prepare = pipeline._prepare_model_for_timestep

            def prepare(t, boundary, offload_model):
                if t.item() < boundary and current["expert"] == 0:
                    clear_history(models[0])
                return native_prepare(t, boundary, offload_model)

            pipeline._prepare_model_for_timestep = prepare
        for expert, model in enumerate(models):
            model._dicache_active = True
            initialize(model, args)
            native = compose_forward(model.forward, block_call)

            def count_block(_module, _inputs):
                if current["active"] is not None:
                    current["active"]["blocks_executed"] += 1

            for block in model.blocks:
                handles.append(block.register_forward_pre_hook(count_block))

            def forward(self, *inputs, _expert=expert, _native=native, **kwargs):
                i = current["count"]
                if _expert != current["expert"]:
                    clear_history(models[current["expert"]])
                    current["expert"] = _expert
                initial = local_counts[_expert] < 2
                self.cnt = i % 2 if initial else i
                row = dict(
                    call_index=i,
                    step_index=i // 2,
                    model_stage=stages[_expert],
                    cfg_branch=("cond", "uncond")[i % 2],
                    expert_call_index=local_counts[_expert],
                    official_counter=self.cnt,
                    expert_initialization=initial,
                    probe_depth=1,
                    global_retention=i < int(args.sample_steps * 2 * args.ret_ratio),
                    blocks_executed=0,
                )
                current["active"] = row
                try:
                    result = _native(self, *inputs, **kwargs)
                finally:
                    current["active"] = None
                if records is not None:
                    records.append(row)
                local_counts[_expert] += 1
                current["count"] += 1
                if current["count"] == args.sample_steps * 2:
                    for item in models:
                        clear_history(item)
                return result

            model.forward = MethodType(forward, model)
        yield
    finally:
        if pipeline is not None:
            if saved_prepare is MISSING:
                pipeline.__dict__.pop("_prepare_model_for_timestep", None)
            else:
                pipeline._prepare_model_for_timestep = saved_prepare
        for handle in handles:
            handle.remove()
        for model, state in zip(models, saved):
            for name, original in state.items():
                if original is MISSING:
                    model.__dict__.pop(name, None)
                else:
                    setattr(model, name, original)
            model.__dict__.pop("_dicache_active", None)
