from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import torch

STAGES = ("high", "low")
BRANCHES = ("cond", "uncond")


@dataclass(frozen=True)
class TeaCacheConfig:
    threshold: float
    coefficients_path: str
    trace_path: Optional[str] = None
    eps: float = 1e-12
    use_ret_steps: bool = False


@dataclass
class _GateState:
    previous_feature: Optional[torch.Tensor] = None
    accumulator: float = 0.0


class TeaCacheController:

    def __init__(self, config: TeaCacheConfig):
        self.config = config
        coefficient_path = Path(config.coefficients_path).expanduser().resolve()
        raw = coefficient_path.read_bytes()
        self.coefficient_path = coefficient_path
        self.coefficient_payload = json.loads(raw.decode("utf-8"))
        self.coefficients, self.protocol = self._parse_coefficients(self.coefficient_payload)
        self._gate_states: Dict[str, _GateState] = {}
        self._residuals: Dict[Tuple[str, str], torch.Tensor] = {}
        self._decisions: list[dict[str, Any]] = []
        self._current_plan_step: Optional[int] = None
        self._current_plan_stage: Optional[str] = None
        self._current_plan_feature: Optional[torch.Tensor] = None

    @staticmethod
    def _parse_coefficients(
        payload: Mapping[str, Any],
    ) -> tuple[dict[str, tuple[float, ...]], dict[str, Any]]:
        stage_payload = payload.get("stages", {})
        protocol = dict(payload.get("protocol", {}))
        coefficients: dict[str, tuple[float, ...]] = {}
        for stage in STAGES:
            values = tuple(
                (float(value) for value in stage_payload[stage]["coefficients_descending"])
            )
            coefficients[stage] = values
        return (coefficients, protocol)

    @staticmethod
    def _protocol_values_equal(expected: Any, observed: Any) -> bool:
        if isinstance(expected, (list, tuple)) or isinstance(observed, (list, tuple)):
            try:
                expected_values = list(expected)
                observed_values = list(observed)
            except TypeError:
                return False
            return len(expected_values) == len(observed_values) and all(
                (
                    TeaCacheController._protocol_values_equal(left, right)
                    for (left, right) in zip(expected_values, observed_values)
                )
            )
        if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
            return math.isclose(float(expected), float(observed), rel_tol=0.0, abs_tol=1e-12)
        return expected == observed

    def plan_step(
        self, stage: str, step_index: int, num_steps: int, compact_time_embedding: torch.Tensor
    ) -> bool:
        feature = compact_time_embedding.detach().float().reshape(1, -1).clone()
        if self._current_plan_step == step_index:
            return self._decisions[-1]["action"] == "reuse"
        state = self._gate_states.setdefault(stage, _GateState())
        accumulator_before = state.accumulator
        relative_l1: Optional[float] = None
        rescaled_distance: Optional[float] = None
        accumulator_candidate: Optional[float] = None
        forced_reason: Optional[str] = None
        if step_index == 0:
            forced_reason = "global_first"
        elif state.previous_feature is None:
            forced_reason = "stage_first"
        elif step_index == num_steps - 1:
            forced_reason = "global_final"
        elif any(((stage, branch) not in self._residuals for branch in BRANCHES)):
            forced_reason = "missing_branch_residual"
        if forced_reason is not None:
            action = "recompute"
            state.accumulator = 0.0
        else:
            relative_l1 = self._relative_l1(feature, state.previous_feature)
            rescaled_distance = self._polyval(self.coefficients[stage], relative_l1)
            accumulator_candidate = accumulator_before + rescaled_distance
            if accumulator_candidate < self.config.threshold:
                action = "reuse"
                state.accumulator = accumulator_candidate
            else:
                action = "recompute"
                state.accumulator = 0.0
        state.previous_feature = feature
        decision = {
            "step_index": step_index,
            "stage": stage,
            "action": action,
            "forced_reason": forced_reason,
            "relative_l1": relative_l1,
            "rescaled_distance": rescaled_distance,
            "accumulator_before": accumulator_before,
            "accumulator_candidate": accumulator_candidate,
            "accumulator_after": state.accumulator,
            "threshold": self.config.threshold,
            "branches": {},
        }
        self._decisions.append(decision)
        self._current_plan_step = step_index
        self._current_plan_stage = stage
        self._current_plan_feature = feature
        return action == "reuse"

    def reuse_residual(self, stage: str, branch: str, step_index: int) -> torch.Tensor:
        key = (stage, branch)
        residual = self._residuals.get(key)
        self._decisions[-1]["branches"][branch] = "reuse"
        return residual

    def record_recompute(
        self, stage: str, branch: str, step_index: int, residual: torch.Tensor
    ) -> None:
        self._residuals[stage, branch] = residual.detach()
        self._decisions[-1]["branches"][branch] = "recompute"

    def clear_stage(self, stage: str) -> None:
        self._gate_states.pop(stage, None)
        for branch in BRANCHES:
            self._residuals.pop((stage, branch), None)
        if self._current_plan_stage == stage:
            self._current_plan_step = None
            self._current_plan_stage = None
            self._current_plan_feature = None

    def summary(self) -> dict[str, Any]:
        per_stage = {}
        for stage in STAGES:
            rows = [row for row in self._decisions if row["stage"] == stage]
            per_stage[stage] = {
                "steps": len(rows),
                "reuse": sum((row["action"] == "reuse" for row in rows)),
                "recompute": sum((row["action"] == "recompute" for row in rows)),
                "reuse_path": [row["step_index"] for row in rows if row["action"] == "reuse"],
                "recompute_path": [
                    row["step_index"] for row in rows if row["action"] == "recompute"
                ],
            }
        return {
            "schema": "teacache4wan22_trace_v1",
            "threshold": self.config.threshold,
            "use_ret_steps": False,
            "coefficients_path": str(self.coefficient_path),
            "coefficient_protocol": self.protocol,
            "total_steps": len(self._decisions),
            "reuse": sum((row["action"] == "reuse" for row in self._decisions)),
            "recompute": sum((row["action"] == "recompute" for row in self._decisions)),
            "per_stage": per_stage,
            "decisions": self._decisions,
        }

    def write_trace(
        self, path: Optional[str] = None, extra: Optional[Mapping[str, Any]] = None
    ) -> Optional[Path]:
        return None

    def release(self) -> None:
        self._gate_states.clear()
        self._residuals.clear()
        self._current_plan_step = None
        self._current_plan_stage = None
        self._current_plan_feature = None

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        numerator = (current - previous).abs().mean()
        denominator = previous.abs().mean().clamp_min(self.config.eps)
        value = numerator / denominator
        return float(value.detach().item())

    @staticmethod
    def _polyval(coefficients: Sequence[float], value: float) -> float:
        result = 0.0
        for coefficient in coefficients:
            result = result * value + coefficient
        return result


__all__ = ["TeaCacheConfig", "TeaCacheController"]
