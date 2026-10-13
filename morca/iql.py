from __future__ import annotations
import argparse, json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def build_mlp(
    input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def gather_action_values(q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return q_values.gather(1, actions.long().view(-1, 1)).squeeze(1)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(
        diff < 0, torch.full_like(diff, 1.0 - expectile), torch.full_like(diff, expectile)
    )
    return (weight * diff.pow(2)).mean()


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, rho: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(rho).add_(source_param.data, alpha=1.0 - rho)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for (key, value) in batch.items()}


def _actor_terms(
    *,
    state: torch.Tensor,
    action: torch.Tensor,
    actor_mask: torch.Tensor,
    value_net: nn.Module,
    q1_net: nn.Module,
    q2_net: nn.Module,
    policy_net: nn.Module,
    beta: float,
    weight_max: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    valid = actor_mask > 0.5
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return (
            None,
            {
                "actor_examples": 0.0,
                "pi_loss": 0.0,
                "advantage_mean": 0.0,
                "advantage_std": 0.0,
                "actor_weight_mean": 0.0,
                "actor_weight_max": 0.0,
            },
        )
    with torch.no_grad():
        logged_q = torch.minimum(
            gather_action_values(q1_net(state), action), gather_action_values(q2_net(state), action)
        )
        advantage = logged_q - value_net(state)
        valid_advantage = advantage[valid]
        advantage_mean = valid_advantage.mean()
        advantage_std = valid_advantage.std(unbiased=False)
        normalized = (advantage - advantage_mean) / advantage_std.clamp_min(1e-06)
        weights = torch.exp(beta * normalized).clamp(max=weight_max)
        valid_weights = weights[valid]
    logits = policy_net(state)
    log_prob = F.log_softmax(logits, dim=-1).gather(1, action.view(-1, 1)).squeeze(1)
    loss = -(weights * log_prob * actor_mask).sum() / actor_mask.sum().clamp_min(1.0)
    return (
        loss,
        {
            "actor_examples": float(valid_count),
            "pi_loss": float(loss.item()),
            "advantage_mean": float(advantage_mean.item()),
            "advantage_std": float(advantage_std.item()),
            "actor_weight_mean": float(valid_weights.mean().item()),
            "actor_weight_max": float(valid_weights.max().item()),
        },
    )


def _run_epoch(
    *,
    loader: DataLoader,
    value_net: nn.Module,
    q1_net: nn.Module,
    q2_net: nn.Module,
    target_q1: nn.Module,
    target_q2: nn.Module,
    policy_net: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer] | None,
    train_actor=True,
) -> dict[str, float]:
    training = optimizers is not None
    for network in (value_net, q1_net, q2_net, policy_net):
        network.train(training)
    totals = {
        "v_loss": 0.0,
        "q_loss": 0.0,
        "reward": 0.0,
        "skip_rate_data": 0.0,
        "skip_rate_policy": 0.0,
        "pi_loss": 0.0,
        "advantage_mean": 0.0,
        "advantage_std": 0.0,
        "actor_weight_mean": 0.0,
        "actor_weight_max": 0.0,
    }
    examples = 0
    actor_examples = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, raw_batch in enumerate(loader):
            batch = _move_batch(raw_batch, device)
            state = batch["state"].float()
            next_state = batch["next_state"].float()
            action = batch["action"].long()
            reward = batch["reward"].float()
            done = batch["done"].float()
            actor_mask = batch["actor_mask"].float()
            batch_size = int(state.shape[0])
            with torch.no_grad():
                value_q1 = target_q1 if training else q1_net
                value_q2 = target_q2 if training else q2_net
                target_logged_q = torch.minimum(
                    gather_action_values(value_q1(state), action),
                    gather_action_values(value_q2(state), action),
                )
            value_pred = value_net(state)
            value_loss = expectile_loss(target_logged_q - value_pred, args.tau)
            if training:
                value_opt, q_opt, policy_opt = optimizers
                value_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), args.grad_clip_norm)
                value_opt.step()
            with torch.no_grad():
                q_target = reward + args.gamma * (1.0 - done) * value_net(next_state)
            q1_pred = gather_action_values(q1_net(state), action)
            q2_pred = gather_action_values(q2_net(state), action)
            q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
            if training:
                q_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(q1_net.parameters()) + list(q2_net.parameters()), args.grad_clip_norm
                )
                q_opt.step()
                soft_update(target_q1, q1_net, args.target_rho)
                soft_update(target_q2, q2_net, args.target_rho)
            policy_loss, actor_metrics = _actor_terms(
                state=state,
                action=action,
                actor_mask=actor_mask,
                value_net=value_net,
                q1_net=q1_net,
                q2_net=q2_net,
                policy_net=policy_net,
                beta=args.beta,
                weight_max=args.weight_max,
            )
            if training and train_actor and (policy_loss is not None):
                policy_opt.zero_grad(set_to_none=True)
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip_norm)
                policy_opt.step()
            with torch.no_grad():
                policy_action = policy_net(state).argmax(dim=-1)
            totals["v_loss"] += float(value_loss.item()) * batch_size
            totals["q_loss"] += float(q_loss.item()) * batch_size
            totals["reward"] += float(reward.mean().item()) * batch_size
            totals["skip_rate_data"] += float(action.float().mean().item()) * batch_size
            totals["skip_rate_policy"] += float(policy_action.float().mean().item()) * batch_size
            current_actor_examples = actor_metrics["actor_examples"]
            for key in (
                "pi_loss",
                "advantage_mean",
                "advantage_std",
                "actor_weight_mean",
                "actor_weight_max",
            ):
                totals[key] += actor_metrics[key] * current_actor_examples
            actor_examples += current_actor_examples
            examples += batch_size
            if args.log_every and (batch_index + 1) % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "batch": batch_index + 1,
                            "v_loss": float(value_loss.item()),
                            "q_loss": float(q_loss.item()),
                            "pi_loss": actor_metrics["pi_loss"],
                            "actor_examples": current_actor_examples,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    result = {
        key: value / max(examples, 1)
        for (key, value) in totals.items()
        if key
        not in {
            "pi_loss",
            "advantage_mean",
            "advantage_std",
            "actor_weight_mean",
            "actor_weight_max",
        }
    }
    for key in (
        "pi_loss",
        "advantage_mean",
        "advantage_std",
        "actor_weight_mean",
        "actor_weight_max",
    ):
        result[key] = totals[key] / max(actor_examples, 1.0)
    result["actor_examples"] = actor_examples
    result["constraint_forced_examples"] = float(examples) - actor_examples
    return result
