import numpy as np
import torch


def midpoint_iqr(values):
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    centers = (np.arange(len(ordered)) + 0.5) / len(ordered)
    return float(np.interp(0.75, centers, ordered) - np.interp(0.25, centers, ordered))


def select_offline(epochs, q, actions, model):
    scales = [midpoint_iqr(z) for z in q]
    adjacent = []
    for i in range(1, len(epochs)):
        dq = float(np.abs(q[i].astype(float) - q[i - 1]).mean())
        adjacent.append(
            dict(
                left=epochs[i - 1],
                right=epochs[i],
                agreement=float((actions[i] == actions[i - 1]).mean()),
                dq=dq,
                normalized=dq / max((scales[i] + scales[i - 1]) / 2, 1e-12),
            )
        )
    candidates = epochs[1:-1] if model == "wan21" else epochs
    ranked = []
    for epoch in candidates:
        rows = [r for r in adjacent if epoch in (r["left"], r["right"])]
        ranked.append(
            dict(
                epoch=epoch,
                agreement=min((r["agreement"] for r in rows)),
                dq=float(np.mean([r["dq"] for r in rows])),
                normalized=float(np.mean([r["normalized"] for r in rows])),
            )
        )
    gate = (
        0.96
        if any((r["agreement"] >= 0.96 for r in ranked))
        else (
            0.95
            if any((r["agreement"] >= 0.95 for r in ranked))
            else max((r["agreement"] for r in ranked))
        )
    )
    eligible = [r for r in ranked if r["agreement"] >= gate]
    return min(eligible, key=lambda r: (r["normalized"], r["dq"], -r["agreement"], -r["epoch"]))[
        "epoch"
    ]


@torch.no_grad()
def census(nets, states, indices, device):
    q, actions = ([], [])
    for start in range(0, len(indices), 256):
        x = torch.from_numpy(states[indices[start : start + 256]].copy()).to(device).float()
        q.append(torch.minimum(nets["q1_net"](x), nets["q2_net"](x)).cpu().numpy())
        actions.append(nets["policy_net"](x).argmax(-1).cpu().numpy())
    return (np.concatenate(q), np.concatenate(actions))
