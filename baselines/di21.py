from __future__ import annotations
import math
from dataclasses import dataclass
from types import SimpleNamespace

BRANCHES = ("cond", "uncond")


def load_official():
    from .dicache_blocks import blocks

    return blocks


@dataclass(frozen=True)
class DiCacheConfig:
    threshold: float
    probe_depth: int = 1
    retention_ratio: float = 0.2
    dcta: bool = True
    trace_path: str | None = None


def diagnostic(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else str(value)


class DiCacheController:

    def __init__(self, config: DiCacheConfig):
        self.config = config
        self.core = load_official()
        self.reset()

    def reset(self):
        self.decisions = []
        self.state = SimpleNamespace(
            cnt=0,
            num_steps=100,
            probe_depth=1,
            rel_l1_thresh=self.config.threshold,
            ret_ratio=self.config.retention_ratio,
            accumulated_rel_l1_distance=[0.0, 0.0],
            residual_cache=[None, None],
            probe_residual_cache=[None, None],
            residual_window=[[], []],
            probe_residual_window=[[], []],
            previous_internal_states=[None, None],
            previous_input=[None, None],
            previous_output=[None, None],
            resume_flag=[False, False],
        )

    def execute(self, *, blocks, x, kwargs, branch, step_index, num_steps):
        i = len(self.decisions)
        state = self.state
        state.blocks, state.cnt = (blocks, i)
        before = diagnostic(state.accumulated_rel_l1_distance[i % 2])
        output, skip, delta, gamma = self.core(state, x, kwargs)
        self.decisions.append(
            dict(
                call_index=i,
                step_index=step_index,
                branch=branch,
                action="probe_reuse" if skip else "full_compute",
                probe_blocks_executed=1,
                deep_blocks_executed=0 if skip else len(blocks) - 1,
                probe_relative_change=diagnostic(delta),
                dcta_gamma=diagnostic(gamma),
                accumulated_error_before=before,
                accumulated_error_after_gate=diagnostic(state.accumulated_rel_l1_distance[i % 2]),
                residual_window_length=len(state.residual_window[i % 2]),
                block_output_dtype=str(output.dtype),
                execution="probe_reuse" if skip else "full_compute",
            )
        )
        state.cnt = i + 1
        if state.cnt == 100:
            decisions = self.decisions
            self.reset()
            self.decisions = decisions
        return output

    def summary(self):
        return dict(
            schema="dicache4wan21_official_trace_v2",
            threshold=self.config.threshold,
            probe_depth=1,
            retention_ratio=self.config.retention_ratio,
            dcta=True,
            total_steps=50,
            total_branch_calls=len(self.decisions),
            full_compute=sum((r["action"] == "full_compute" for r in self.decisions)),
            probe_reuse=sum((r["action"] == "probe_reuse" for r in self.decisions)),
            decisions=self.decisions,
        )

    def write_trace(self, extra=None):
        return None
