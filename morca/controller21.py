import torch
from baselines import sea21 as sea
from .features import History, scalar_state, required_hard_budget_action

FORCED = (0, 49) if 21 == 21 else (0, 32, 49)
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

    def __init__(self, policy, budget):
        self.policy, self.budget = (policy, budget)
        super().__init__(sea.SeaCacheConfig(threshold=1.0))

    def reset(self):
        super().reset()
        self.feature_history = History((0,))
        self.current_latent_feature = None
        self.used = dict(cond=0, uncond=0)
        self.consecutive = dict(cond=0, uncond=0)
        self.saved_states, self.actions, self.actor_masks = ([], [], [])

    def observe_latent(self, latent, step):
        self.policy.current_feature = self.feature_history.observe(latent, step).cpu()

    def summary(self):
        return dict(reuse=self.used["cond"], recompute=50 - self.used["cond"])

    def write_trace(self, *args, **kwargs):
        return None

    def plan_step(self, *, branch, step_index, num_steps, feature, grid_size):
        filtered = self._filter_feature(feature, grid_size, step_index, num_steps)
        before = self.accumulators[branch]
        d = total = 0.0
        if step_index not in FORCED:
            d = self._relative_l1(filtered, self.previous_features[branch])
            total = before + d
        state = observation(
            self.policy.mode,
            step=step_index,
            budget=self.budget,
            used=self.used[branch],
            consecutive=self.consecutive[branch],
            cached_valid=branch in self.residuals,
            adjacent=d,
            accumulated=total,
            latent=self.current_latent_feature,
        )
        required, reason = required_hard_budget_action(
            step_index=step_index,
            used_skips=self.used[branch],
            skip_budget=self.budget,
            num_steps=STEPS,
            forced_steps=FORCED,
            current_forced_recompute=branch not in self.residuals,
        )
        if branch == "cond":
            self.saved_states.append(torch.cat((self.policy.current_feature, state)))
            self.actor_masks.append(float(required is None))
            if required is None:
                action, probability = self.policy.choose(state)
                reason = getattr(self.policy, "action_mode", "policy_argmax")
            else:
                action, probability = (required, None)
        else:
            cond = self.decisions[-1]
            action = int(cond["action"] == "reuse")
            probability, reason = (cond["p_skip"], cond["reason"])
        if branch == "cond" and self.feature_history is not None:
            self.feature_history.commit(action)
            self.actions.append(action)
        self.accumulators[branch] = total if action else 0.0
        self.previous_features[branch] = filtered.detach().clone()
        decision = dict(
            call_index=self._expected_call_index,
            branch=branch,
            step_index=step_index,
            action="reuse" if action else "recompute",
            reason=reason,
            state=state.tolist(),
            state_mode=self.policy.mode,
            skip_budget=self.budget,
            used_skips_before=self.used[branch],
            consecutive_skips_before=self.consecutive[branch],
            p_skip=probability,
            policy_queried=branch == "cond" and required is None,
            actor_mask=float(required is None),
            native_forced_recompute=step_index in FORCED,
            behavior_action_probability=(
                (probability if action else 1 - probability) if probability is not None else 1.0
            ),
            filtered_relative_l1=d,
            accumulated_distance_before=before,
            accumulated_distance_with_current=total,
            accumulated_distance_after=self.accumulators[branch],
            stored_feature="sea_filtered",
            execution=None,
        )
        self.used[branch] += action
        self.consecutive[branch] = self.consecutive[branch] + 1 if action else 0
        self.decisions.append(decision)
        self._pending_decision = decision
        self._expected_call_index += 1
        return bool(action)
