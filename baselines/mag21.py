from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import torch

BRANCHES = ("cond", "uncond")
WAN21_T2V13B_MAG_RATIOS = (
    1.0,
    1.0,
    1.0124,
    1.02213,
    1.00166,
    1.0041,
    0.99791,
    1.00061,
    0.99682,
    0.99762,
    0.99634,
    0.99685,
    0.99567,
    0.99586,
    0.99416,
    0.99422,
    0.99578,
    0.99575,
    0.9957,
    0.99563,
    0.99511,
    0.99506,
    0.99535,
    0.99531,
    0.99552,
    0.99549,
    0.99541,
    0.99539,
    0.9954,
    0.99536,
    0.99489,
    0.99485,
    0.99518,
    0.99514,
    0.99484,
    0.99478,
    0.99481,
    0.99479,
    0.99415,
    0.99413,
    0.99419,
    0.99416,
    0.99396,
    0.99393,
    0.99388,
    0.99386,
    0.99349,
    0.99349,
    0.99309,
    0.99304,
    0.9927,
    0.9927,
    0.99228,
    0.99226,
    0.99171,
    0.9917,
    0.99137,
    0.99135,
    0.99068,
    0.99063,
    0.99005,
    0.99003,
    0.98944,
    0.98942,
    0.98849,
    0.98849,
    0.98758,
    0.98757,
    0.98644,
    0.98643,
    0.98504,
    0.98503,
    0.9836,
    0.98359,
    0.98202,
    0.98201,
    0.97977,
    0.97978,
    0.97717,
    0.97718,
    0.9741,
    0.97411,
    0.97003,
    0.97002,
    0.96538,
    0.96541,
    0.9593,
    0.95933,
    0.95086,
    0.95089,
    0.94013,
    0.94019,
    0.92402,
    0.92414,
    0.90241,
    0.9026,
    0.86821,
    0.86868,
    0.81838,
    0.81939,
)


@dataclass(frozen=True)
class MagCacheConfig:
    threshold: float
    max_skip_steps: int
    retention_ratio: float = 0.2
    trace_path: Optional[str] = None


class MagCacheController:

    def __init__(self, config: MagCacheConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.residuals: dict[str, torch.Tensor] = {}
        self.accumulated_ratio = {branch: 1.0 for branch in BRANCHES}
        self.accumulated_error = {branch: 0.0 for branch in BRANCHES}
        self.accumulated_steps = {branch: 0 for branch in BRANCHES}
        self.decisions: list[dict[str, Any]] = []
        self._num_steps: Optional[int] = None
        self._pending: Optional[dict[str, Any]] = None
        self._expected_call_index = 0

    def plan_step(self, branch: str, step_index: int, num_steps: int) -> bool:
        call_index = self._expected_call_index
        retention_calls = int(num_steps * len(BRANCHES) * self.config.retention_ratio)
        ratio = WAN21_T2V13B_MAG_RATIOS[call_index]
        action, reason = ("recompute", "retention")
        candidate_ratio = self.accumulated_ratio[branch]
        candidate_error = self.accumulated_error[branch]
        candidate_steps = self.accumulated_steps[branch]
        if call_index >= retention_calls:
            candidate_ratio *= ratio
            candidate_steps += 1
            candidate_error += abs(1.0 - candidate_ratio)
            can_reuse = (
                branch in self.residuals
                and candidate_error < self.config.threshold
                and (candidate_steps <= self.config.max_skip_steps)
            )
            if can_reuse:
                action, reason = ("reuse", "within_budget")
                self.accumulated_ratio[branch] = candidate_ratio
                self.accumulated_error[branch] = candidate_error
                self.accumulated_steps[branch] = candidate_steps
            else:
                if candidate_error >= self.config.threshold:
                    reason = "error_budget"
                elif candidate_steps > self.config.max_skip_steps:
                    reason = "max_skip_steps"
                else:
                    reason = "missing_residual"
                self.accumulated_ratio[branch] = 1.0
                self.accumulated_error[branch] = 0.0
                self.accumulated_steps[branch] = 0
        row = {
            "call_index": call_index,
            "step_index": step_index,
            "branch": branch,
            "magnitude_ratio": ratio,
            "candidate_accumulated_ratio": candidate_ratio,
            "candidate_accumulated_error": candidate_error,
            "candidate_accumulated_steps": candidate_steps,
            "action": action,
            "reason": reason,
        }
        self.decisions.append(row)
        self._pending = row
        self._expected_call_index += 1
        return action == "reuse"

    def reuse_residual(self, branch: str, step_index: int) -> torch.Tensor:
        self._finish(branch, step_index, "reuse")
        return self.residuals[branch]

    def record_recompute(self, branch: str, step_index: int, residual: torch.Tensor) -> None:
        self._finish(branch, step_index, "recompute")
        self.residuals[branch] = residual.detach().clone()

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "magcache4wan21_trace_v1",
            "gate_mode": "official_magnitude_curve_independent_cfg_branches",
            "threshold": self.config.threshold,
            "max_skip_steps": self.config.max_skip_steps,
            "retention_ratio": self.config.retention_ratio,
            "magnitude_curve": "official_Wan2.1_T2V_1.3B_50step_interleaved",
            "total_steps": self._num_steps or 0,
            "total_branch_calls": len(self.decisions),
            "reuse": sum((row["action"] == "reuse" for row in self.decisions)),
            "recompute": sum((row["action"] == "recompute" for row in self.decisions)),
            "decisions": self.decisions,
        }

    def write_trace(
        self, path: Optional[str] = None, extra: Optional[Mapping[str, Any]] = None
    ) -> Optional[Path]:
        return None

    def _finish(self, branch: str, step_index: int, action: str) -> None:
        self._pending["execution"] = action
        self._pending = None


__all__ = ["MagCacheConfig", "MagCacheController", "WAN21_T2V13B_MAG_RATIOS"]
