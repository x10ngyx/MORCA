import json
import random
from pathlib import Path
import numpy as np
import torch
from .generation import generate_one
from .io import read_prompts, write_json
from .pipeline import load_pipeline, generation_options
from .policy import Policy
from .metrics.core import psnr_per_frame
from .metrics.video import decode_video_rgb


def warmup(pipe, args, prompt):
    pipe.generate(prompt, **generation_options(args.model))


def calibrate(args):
    rows = read_prompts(args.prompts)
    pipe = load_pipeline(args.model, args.source, args.checkpoint)
    warmup(pipe, args, rows[0]["prompt"])
    records = []
    for row in rows:
        timing, _ = generate_one(pipe, args, row, Path(args.output) / "original", method="original")
        records.append(
            dict(sample_id=row["sample_id"], k=None, seconds=timing["inference_seconds"])
        )
        for k in args.budgets:
            timing, _ = generate_one(
                pipe,
                args,
                row,
                Path(args.output) / f"k{k}",
                method="random",
                policy=Policy(seed=42),
                budget=k,
            )
            records.append(
                dict(sample_id=row["sample_id"], k=k, seconds=timing["inference_seconds"])
            )
    fitted = [r for r in records if r["k"] is not None]
    slope, intercept = np.polyfit([r["k"] for r in fitted], [r["seconds"] for r in fitted], 1)
    write_json(
        Path(args.output) / "calibration.json",
        dict(
            model=args.model,
            baseline_seconds=float(np.mean([r["seconds"] for r in records if r["k"] is None])),
            latency_intercept=float(intercept),
            latency_per_skip=float(slope),
            measurements=records,
        ),
    )


def collect(args):
    rows = read_prompts(args.prompts)
    rng = random.Random(args.seed)
    ids = sorted({r["sample_id"] for r in rows})
    rng.shuffle(ids)
    nval = max(1, round(0.2 * len(ids)))
    val = set(ids[:nval]) if len(ids) > 1 else set()
    splits = {
        r["sample_id"]: r.get("split", "val" if r["sample_id"] in val else "train") for r in rows
    }
    calibration = json.loads(Path(args.calibration).read_text())
    pipe = load_pipeline(args.model, args.source, args.checkpoint)
    warmup(pipe, args, rows[0]["prompt"])
    sample_rows = (
        rng.choices(rows, k=args.samples)
        if args.samples
        else [r for r in rows for _ in range(args.trajectories)]
    )
    output = Path(args.output)
    (output / "trajectories").mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(sample_rows):
        reference_video = output / "original" / f"{row['sample_id']}.mp4"
        if not reference_video.exists():
            generate_one(pipe, args, row, output / "original", method="original")
        speed = rng.uniform(*args.speed_range)
        k = round(
            (calibration["baseline_seconds"] / speed - calibration["latency_intercept"])
            / calibration["latency_per_skip"]
        )
        k = max(0, min(48 if args.model == "wan21" else 47, k))
        policy = Policy(
            args.policy, sampling=True, seed=args.seed + i, exploration=0.1 if args.policy else 0.0
        )
        candidate = dict(row, sample_id=f"{row['sample_id']}_{i:06d}")
        _, trace = generate_one(
            pipe, args, candidate, output / "videos", method="random", policy=policy, budget=k
        )
        reward = float(
            psnr_per_frame(
                decode_video_rgb(reference_video),
                decode_video_rgb(output / "videos" / f"{candidate['sample_id']}.mp4"),
            ).mean()
        )
        trace.update(
            prompt_id=row["sample_id"],
            prompt=row["prompt"],
            split=splits[row["sample_id"]],
            terminal_psnr=reward,
            model=args.model,
            budget=k,
        )
        torch.save(trace, output / "trajectories" / f"{candidate['sample_id']}.pt")
