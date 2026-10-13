from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple
import torch

BRANCHES = ("cond", "uncond")


@dataclass(frozen=True)
class SeaCacheConfig:
    threshold: float
    trace_path: Optional[str] = None
    use_ret_steps: bool = False
    power_exp: float = 3.0
    power_const: float = 1.0
    eps: float = 1e-16
    norm_mode: str = "mean"


class SeaCacheController:

    def __init__(self, config: SeaCacheConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.previous_features: dict[str, torch.Tensor] = {}
        self.accumulators = {branch: 0.0 for branch in BRANCHES}
        self.scheduler_sigmas: Optional[torch.Tensor] = None
        self.residuals: dict[str, torch.Tensor] = {}
        self.decisions: list[dict[str, Any]] = []
        self._expected_call_index = 0
        self._num_steps: Optional[int] = None
        self._pending_decision: Optional[dict[str, Any]] = None

    def set_scheduler_sigmas(self, sigmas: Optional[torch.Tensor]) -> None:
        self.scheduler_sigmas = sigmas

    def plan_step(
        self,
        *,
        branch: str,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
    ) -> bool:
        filtered = self._filter_feature(feature, grid_size, step_index, num_steps)
        ret_steps, cutoff_steps = self._ret_cutoff_steps(num_steps)
        forced = step_index < ret_steps or step_index >= cutoff_steps
        relative_l1: Optional[float] = None
        accumulator_before = self.accumulators[branch]
        if forced:
            stored_feature = filtered
            action = "recompute"
            reason = "forced_boundary"
            self.accumulators[branch] = 0.0
            feature_state = "sea_filtered"
        else:
            previous = self.previous_features.get(branch)
            relative_l1 = self._relative_l1(filtered, previous)
            self.accumulators[branch] += relative_l1
            stored_feature = filtered
            feature_state = "sea_filtered"
            if self.accumulators[branch] < self.config.threshold:
                action = "reuse"
                reason = "accumulator_below_threshold"
            else:
                action = "recompute"
                reason = "threshold_reached"
                self.accumulators[branch] = 0.0
        self.previous_features[branch] = stored_feature.detach().clone()
        decision = {
            "call_index": self._expected_call_index,
            "step_index": step_index,
            "branch": branch,
            "action": action,
            "reason": reason,
            "relative_l1": relative_l1,
            "accumulator_before": accumulator_before,
            "accumulator_after": self.accumulators[branch],
            "stored_feature": feature_state,
            "execution": None,
        }
        self.decisions.append(decision)
        self._pending_decision = decision
        self._expected_call_index += 1
        return action == "reuse"

    def reuse_residual(self, branch: str, step_index: int) -> torch.Tensor:
        residual = self.residuals.get(branch)
        self._pending_decision["execution"] = "reuse"
        self._pending_decision = None
        return residual

    def record_recompute(self, branch: str, step_index: int, residual: torch.Tensor) -> None:
        self.residuals[branch] = residual.detach().clone()
        self._pending_decision["execution"] = "recompute"
        self._pending_decision = None

    def summary(self) -> dict[str, Any]:
        per_branch = {
            branch: {
                "reuse": sum(
                    (row["branch"] == branch and row["action"] == "reuse" for row in self.decisions)
                ),
                "recompute": sum(
                    (
                        row["branch"] == branch and row["action"] == "recompute"
                        for row in self.decisions
                    )
                ),
                "reuse_path": [
                    row["step_index"]
                    for row in self.decisions
                    if row["branch"] == branch and row["action"] == "reuse"
                ],
                "recompute_path": [
                    row["step_index"]
                    for row in self.decisions
                    if row["branch"] == branch and row["action"] == "recompute"
                ],
            }
            for branch in BRANCHES
        }
        return {
            "schema": "seacache4wan21_trace_v3",
            "gate_mode": "corrected_independent_cfg_branches_filtered_boundary",
            "threshold": self.config.threshold,
            "use_ret_steps": self.config.use_ret_steps,
            "power_exp": self.config.power_exp,
            "power_const": self.config.power_const,
            "eps": self.config.eps,
            "norm_mode": self.config.norm_mode,
            "total_steps": self._num_steps or 0,
            "total_branch_calls": len(self.decisions),
            "reuse": sum((row["action"] == "reuse" for row in self.decisions)),
            "recompute": sum((row["action"] == "recompute" for row in self.decisions)),
            "per_branch": per_branch,
            "decisions": self.decisions,
        }

    def write_trace(
        self, path: Optional[str] = None, extra: Optional[Mapping[str, Any]] = None
    ) -> Optional[Path]:
        return None

    def _ret_cutoff_steps(self, num_steps: int) -> Tuple[int, int]:
        if self.config.use_ret_steps:
            return (min(5, num_steps), num_steps)
        return (min(1, num_steps), max(0, num_steps - 1))

    def _filter_feature(
        self, feature: torch.Tensor, grid_size: torch.Tensor, step_index: int, num_steps: int
    ) -> torch.Tensor:
        f, h, w = [int(value) for value in grid_size.detach().cpu().tolist()]
        feature_5d = feature.reshape(feature.shape[0], f, h, w, feature.shape[-1])
        a, b = self._ab_from_flow_scheduler(step_index, num_steps)
        filtered = self._apply_sea_from_ab(feature_5d, a, b, dims=(-2, -3, -4))
        return filtered.reshape(filtered.shape[0], -1, filtered.shape[-1]).detach()

    def _ab_from_flow_scheduler(self, step_index: int, num_steps: int) -> Tuple[float, float]:
        if self.scheduler_sigmas is not None:
            sigma = float(self.scheduler_sigmas[step_index].detach().cpu().item())
        else:
            sigma = 1.0 - (step_index + 1) / float(num_steps)
        sigma = max(1e-06, min(1.0 - 1e-06, sigma))
        return (1.0 - sigma, sigma)

    def _apply_sea_from_ab(
        self, x: torch.Tensor, a: float, b: float, dims: Tuple[int, ...]
    ) -> torch.Tensor:
        original_dtype = x.dtype
        x32 = x.contiguous().to(torch.float32)
        spectrum = torch.fft.fftn(x32, dim=dims)
        gain: Optional[torch.Tensor] = None
        for axis in dims:
            frequency = torch.fft.fftfreq(
                x32.shape[axis], device=x32.device, dtype=torch.float32
            ).abs()
            signal_power = self.config.power_const / (
                frequency.pow(self.config.power_exp) + self.config.eps
            )
            axis_gain = a * signal_power / (a * a * signal_power + b * b + self.config.eps)
            shape = [1] * x32.ndim
            shape[axis] = axis_gain.shape[0]
            axis_gain = axis_gain.reshape(shape)
            gain = axis_gain if gain is None else gain * axis_gain
        normalizer = torch.amax(gain) if self.config.norm_mode == "peak" else torch.mean(gain)
        if torch.isfinite(normalizer) and normalizer > 0:
            gain = gain / normalizer
        return torch.fft.ifftn(spectrum * gain, dim=dims).real.to(original_dtype)

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        numerator = (current - previous).abs().mean()
        denominator = previous.abs().mean() + self.config.eps
        return float((numerator / denominator).detach().cpu())


__all__ = ["SeaCacheConfig", "SeaCacheController"]
