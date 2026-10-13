import torch
import torch.nn.functional as F


def pool(x):
    direct = F.adaptive_avg_pool3d(x, (4, 8, 8)).flatten()
    stats = torch.cat(
        (
            F.adaptive_avg_pool2d(x.mean(2), (8, 8)),
            F.adaptive_avg_pool2d(x.var(2, unbiased=False), (8, 8)),
        ),
        1,
    ).flatten()
    return (direct, stats)


class History:

    def __init__(self, starts):
        self.starts = starts
        self.previous = self.cache = None

    @torch.no_grad()
    def observe(self, latent, step):
        self.current = latent.detach().half().float()
        if self.current.ndim == 4:
            self.current = self.current.unsqueeze(0)
        if step in self.starts:
            self.previous = self.cache = None
        if self.cache is None:
            return self.current.new_zeros(18432)
        parts = [pool(x) for x in (self.current, self.previous, self.cache)]
        return torch.cat((torch.cat([x[0] for x in parts]), torch.cat([x[1] for x in parts])))

    def commit(self, action):
        if action == 0:
            self.cache = self.current
        self.previous = self.current


def scalar_state(step, k, used, consecutive, cached_valid, adjacent=0.0, accumulated=0.0):
    return torch.tensor(
        [
            adjacent,
            accumulated,
            float(cached_valid),
            step / 49.0,
            k / 50.0,
            used / 50.0,
            consecutive / 50.0,
        ],
        dtype=torch.float32,
    )


def required_hard_budget_action(
    *,
    step_index,
    used_skips,
    skip_budget,
    num_steps=50,
    forced_steps=(0, 49),
    current_forced_recompute=False
):
    if step_index in forced_steps or current_forced_recompute:
        return (0, "recompute")
    remaining = skip_budget - used_skips
    available = sum((t not in forced_steps for t in range(step_index, num_steps)))
    if remaining <= 0:
        return (0, "budget")
    if remaining >= available:
        return (1, "budget")
    return (None, "policy")
