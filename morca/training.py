import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .model import CNN, encode_normalized
from .iql import _run_epoch
from .selection import select_offline, census


class Transitions(Dataset):

    def __init__(self, states, actions, masks, rewards, indices):
        self.states, self.actions, self.masks, self.rewards = (states, actions, masks, rewards)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        terminal = i % 50 == 49
        return dict(
            state=torch.from_numpy(self.states[i].copy()),
            next_state=torch.from_numpy(self.states[i if terminal else i + 1].copy()),
            action=self.actions[i],
            actor_mask=self.masks[i],
            done=torch.tensor(float(terminal)),
            reward=self.rewards[i],
        )


def fixed_support(states, masks, indices, budgets, limit=4096):
    free = indices[masks[indices].numpy() > 0.5]
    groups = free % 50 // 10 * 7 + budgets[free] // 8
    generator = np.random.default_rng(42)
    selected, weights = ([], [])
    unique = np.unique(groups)
    for group in unique:
        positions = free[groups == group]
        count = min(len(positions), limit // len(unique))
        selected.extend(generator.choice(positions, count, replace=False))
        weights.extend([len(positions) / count] * count)
    return (
        torch.from_numpy(states[selected].copy()).float(),
        torch.tensor(weights, dtype=torch.float64),
    )


@torch.no_grad()
def actor_outputs(net, states, device):
    probabilities, actions = ([], [])
    for batch in states.split(64):
        logits = net(batch.to(device))
        probabilities.append(logits.softmax(-1)[:, 1].cpu())
        actions.append(logits.argmax(-1).cpu())
    return (torch.cat(probabilities), torch.cat(actions))


def choose_epoch(probabilities, actions, weights):
    p, a = (torch.stack(probabilities).double(), torch.stack(actions))
    w = weights / weights.sum()
    agreement = ((a[1:] == a[:-1]).double() * w).sum(1)
    drift = ((p[1:] - p[:-1]).abs() * w).sum(1)
    candidates = [
        (
            min(float(agreement[e - 3]), float(agreement[e - 2])),
            -max(float(drift[e - 3]), float(drift[e - 2])),
            e,
        )
        for e in range(11, len(p) + 1)
    ]
    return max(candidates)[2]


def run(args):
    torch.manual_seed(42)
    files = [p for folder in args.data for p in sorted(Path(folder).glob("trajectories/*.pt"))]
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    )
    if checkpoint:
        normalizer = checkpoint["normalizer"]
    else:
        total, square, count = (
            torch.zeros(18439, dtype=torch.float64),
            torch.zeros(18439, dtype=torch.float64),
            0,
        )
        for path in files:
            item = torch.load(path, map_location="cpu", weights_only=False)
            if item["split"] == "train":
                x = item["states"].double()
                total += x.sum(0)
                square += x.square().sum(0)
                count += len(x)
        mean = total / count
        normalizer = dict(
            mean=mean.float(),
            std=(square / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-06).float(),
        )
    nets = {
        name: CNN("G1", 1 if name == "value_net" else 2).to(args.device)
        for name in ("value_net", "q1_net", "q2_net", "policy_net")
    }
    for name in ("q1", "q2"):
        nets["target_" + name] = deepcopy(nets[name + "_net"]).eval().requires_grad_(False)
    if checkpoint:
        for name, net in nets.items():
            net.load_state_dict(checkpoint[name])
    opts = tuple(
        (
            torch.optim.AdamW(p, lr=lr, weight_decay=0.01)
            for (p, lr) in (
                (nets["value_net"].parameters(), 0.0001),
                (list(nets["q1_net"].parameters()) + list(nets["q2_net"].parameters()), 0.0001),
                (nets["policy_net"].parameters(), 4e-05 if args.online else 0.0001),
            )
        )
    )
    if checkpoint and args.restore_optimizer:
        for opt, state in zip(opts, checkpoint["optimizers"]):
            opt.load_state_dict(state)
    cfg = SimpleNamespace(
        tau=0.7,
        beta=1.0 if args.online else 1.5,
        weight_max=20.0 if args.online else 30.0,
        gamma=1.0,
        target_rho=0.999,
        grad_clip_norm=1.0,
        log_every=0,
    )
    weights, output = (Path(args.weights), Path(args.output))
    weights.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    n = len(files) * 50
    states = np.lib.format.open_memmap(
        output / "states.npy", mode="w+", dtype=np.float16, shape=(n, 18439)
    )
    actions, masks, rewards = (torch.zeros(n, dtype=torch.long), torch.zeros(n), torch.zeros(n))
    indices, budgets = (dict(train=[], val=[]), np.zeros(n, dtype=np.int64))
    for i, path in enumerate(files):
        item = torch.load(path, map_location="cpu", weights_only=False)
        sl = slice(i * 50, (i + 1) * 50)
        states[sl] = encode_normalized(item["states"], normalizer).numpy()
        actions[sl], masks[sl] = (item["actions"], item["actor_mask"])
        rewards[(i + 1) * 50 - 1] = item["terminal_psnr"]
        budgets[sl] = int(item["actions"].sum())
        if item["split"] in indices:
            indices[item["split"]].extend(range(i * 50, (i + 1) * 50))
    states.flush()
    indices = {k: np.asarray(v, dtype=np.int64) for (k, v) in indices.items()}
    loaders = {}
    for split in ("train", "val"):
        dataset = Transitions(states, actions, masks, rewards, indices[split])
        sampler = (
            torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=len(dataset))
            if split == "train" and args.online
            else None
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train" and sampler is None,
            sampler=sampler,
            num_workers=0,
        )
    probabilities, decisions = ([], [])
    census_epochs, census_q, census_actions = ([], [], [])
    selection_start = max(1, args.epochs - (30 if args.model == "wan21" else 20))
    val_free = indices["val"][masks[indices["val"]].numpy() > 0.5]
    if args.online:
        support, support_weights = fixed_support(
            states,
            masks,
            indices["train"][
                indices["train"] < 50 * len(list(Path(args.data[0]).glob("trajectories/*.pt")))
            ],
            budgets,
        )
    for epoch in range(1, args.epochs + 1):
        metrics = dict(epoch=epoch)
        for split, loader in loaders.items():
            metrics[split] = _run_epoch(
                loader=loader,
                **nets,
                args=cfg,
                device=torch.device(args.device),
                optimizers=opts if split == "train" else None,
                train_actor=not args.online or epoch > args.critic_epochs,
            )
        if args.online and epoch > args.critic_epochs:
            probability, action = actor_outputs(nets["policy_net"], support, args.device)
            probabilities.append(probability)
            decisions.append(action)
        if not args.online and epoch >= selection_start and len(val_free):
            q, a = census(nets, states, val_free, args.device)
            census_epochs.append(epoch)
            census_q.append(q)
            census_actions.append(a)
        print(json.dumps(metrics), flush=True)
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        payload = dict(
            model=args.model,
            epoch=epoch,
            normalizer=normalizer,
            **{name: net.state_dict() for (name, net) in nets.items()},
            optimizers=[o.state_dict() for o in opts],
        )
        torch.save(payload, weights / f"epoch_{epoch:03d}.pt")
    selected = args.epochs
    if args.online and len(probabilities) >= 11:
        selected = args.critic_epochs + choose_epoch(probabilities, decisions, support_weights)
    if not args.online and len(census_epochs) >= 3:
        selected = select_offline(
            census_epochs, np.stack(census_q), np.stack(census_actions), args.model
        )
    import shutil

    shutil.copyfile(weights / f"epoch_{selected:03d}.pt", weights / "selected.pt")
    (output / "selection.json").write_text(json.dumps(dict(epoch=selected)) + "\n")
    return weights / "selected.pt"
