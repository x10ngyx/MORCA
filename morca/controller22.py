import torch
from baselines import sea22 as sea
from .features import History, scalar_state, required_hard_budget_action

FORCED = (0, 49) if 22 == 21 else (0, 32, 49)
STEPS = 50
observation = lambda mode, **kw: scalar_state(
    kw["step"],
    kw["budget"],
    kw["used"],
    kw["consecutive"],
    kw["cached_valid"],
    kw["adjacent"],
    kw["accumulated"],
)


class Controller(sea.SeaCacheController):

    def __init__(self, policy, k):
        super().__init__(sea.SeaCacheConfig(threshold=1.0))
        self.policy, self.k = (policy, k)
        self.used = self.consecutive = 0
        self.history = History((0, 32))
        self.saved_states, self.actions, self.actor_masks = ([], [], [])

    def observe_latent(self, latent, step):
        self.latent_feature = self.history.observe(latent, step).cpu()

    def summary(self):
        return dict(reuse=self.used, recompute=50 - self.used)

    def write_trace(self, *args, **kwargs):
        return None

    def plan_step(self, *, stage, step_index, num_steps, feature, grid_size):
        if self._current_step == step_index:
            return self.decisions[-1]["action"] == "reuse"
        state = self.states.setdefault(stage, sea._StageState())
        filtered = self._filter_feature(feature, grid_size, step_index, num_steps)
        cached = set(state.residuals) == {"cond", "uncond"}
        d = total = 0.0
        before = state.accumulator
        if step_index not in FORCED:
            d = self._relative_l1(filtered, state.previous_feature)
            total = before + d
        scalar = scalar_state(step_index, self.k, self.used, self.consecutive, cached, d, total)
        required, reason = required_hard_budget_action(
            step_index=step_index,
            used_skips=self.used,
            skip_budget=self.k,
            num_steps=50,
            forced_steps=FORCED,
            current_forced_recompute=not cached,
        )
        self.saved_states.append(torch.cat((self.latent_feature, scalar)))
        self.actor_masks.append(float(required is None))
        if required is None:
            action, probability = self.policy.choose(torch.cat((self.latent_feature, scalar)))
            reason = getattr(self.policy, "action_mode", "policy_argmax")
        else:
            action, probability = (required, None)
        self.history.commit(action)
        self.actions.append(action)
        state.accumulator = total if action else 0.0
        state.previous_feature = filtered.detach().clone()
        self._current_step, self._current_stage = (step_index, stage)
        behavior = None if probability is None else probability
        self.decisions.append(
            dict(
                step_index=step_index,
                stage=stage,
                action="reuse" if action else "recompute",
                reason=reason,
                branches={},
                state=scalar.tolist(),
                state_mode="cnn_G1",
                skip_budget=self.k,
                used_skips_before=self.used,
                consecutive_skips_before=self.consecutive,
                p_skip=probability,
                behavior_p_skip=behavior,
                behavior_action_probability=(
                    None if behavior is None else behavior if action else 1 - behavior
                ),
                policy_queried=required is None,
                actor_mask=float(required is None),
                native_forced_recompute=step_index in FORCED,
                relative_l1=d,
                accumulator_before=before,
                accumulated_distance_with_current=total,
                accumulator_after=state.accumulator,
            )
        )
        self.used += action
        self.consecutive = self.consecutive + 1 if action else 0
        return bool(action)
