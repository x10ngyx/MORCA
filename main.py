#!/usr/bin/env python3
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MORCA: adaptive cache reuse for video diffusion.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("infer", "calibrate", "collect", "train", "evaluate"):
        p = commands.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
        if command != "evaluate":
            p.add_argument("--model", choices=("wan21", "wan22"), default="wan21")
        if command in ("infer", "calibrate", "collect"):
            p.add_argument(
                "--source",
                type=Path,
                help="Wan source directory (default: third_party/Wan2.1 or Wan2.2)",
            )
            p.add_argument("--checkpoint", type=Path, help="base model directory under models/")
            p.add_argument("--threshold", type=float)
            p.add_argument("--retention", type=float)
            p.add_argument("--max-skip", type=int)
        if command == "infer":
            p.add_argument(
                "--method",
                choices=(
                    "morca",
                    "original",
                    "seacache",
                    "teacache",
                    "magcache",
                    "dicache",
                ),
                default="morca",
            )
            prompts = p.add_mutually_exclusive_group(required=True)
            prompts.add_argument("--prompt")
            prompts.add_argument("--prompts", type=Path)
            p.add_argument("--policy", type=Path)
            budget = p.add_mutually_exclusive_group()
            budget.add_argument("--skip-budget", type=int)
            budget.add_argument("--speedup", type=float)
            p.add_argument("--calibration", type=Path)
            p.add_argument(
                "--profile",
                action="store_true",
                help="separate FLOP-counted pass after timed generation",
            )
        elif command == "calibrate":
            p.add_argument("--prompts", type=Path, required=True)
            p.add_argument("--budgets", type=int, nargs="+", default=[0, 16, 24, 32, 40])
        elif command == "collect":
            p.add_argument("--prompts", type=Path, required=True)
            p.add_argument("--calibration", type=Path, required=True)
            p.add_argument("--trajectories", type=int, default=3)
            p.add_argument("--samples", type=int, help="sample this many prompts with replacement")
            p.add_argument("--speed-range", nargs=2, type=float, default=[1.5, 3.5])
            p.add_argument(
                "--policy",
                type=Path,
                help="online collection: 90%% actor + 10%% uniform exploration",
            )
            p.add_argument("--seed", type=int, default=42)
        elif command == "train":
            p.add_argument("--data", nargs="+", type=Path, required=True)
            p.add_argument(
                "--weights", type=Path, required=True, help="checkpoint directory under models/"
            )
            p.add_argument("--epochs", type=int)
            p.add_argument("--batch-size", type=int, default=256)
            p.add_argument("--device", default="cuda")
            p.add_argument("--resume", type=Path)
            p.add_argument("--online", action="store_true")
            p.add_argument("--critic-epochs", type=int, default=5)
            p.add_argument("--restore-optimizer", action="store_true")
        else:
            p.add_argument("--reference", type=Path, required=True)
            p.add_argument("--candidate", type=Path, required=True)
            p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if hasattr(args, "model"):
        version = "Wan2.1" if args.model == "wan21" else "Wan2.2"
        if hasattr(args, "source"):
            args.source = args.source or Path("third_party") / version
            args.checkpoint = args.checkpoint or Path(os.environ.get("MODEL_ROOT", "models")) / (
                "Wan2.1-T2V-1.3B" if args.model == "wan21" else "Wan2.2-T2V-A14B"
            )
    if args.command == "infer":
        config_path = Path(__file__).resolve().parent / "config" / args.model / f"{args.method}.json"
        settings = json.loads(config_path.read_text())
        explicit_budget = args.skip_budget is not None or args.speedup is not None
        for key, value in settings.items():
            if key in ("skip_budget", "speedup") and explicit_budget:
                continue
            if getattr(args, key) is None:
                setattr(args, key, Path(value) if key in ("policy", "calibration") and value else value)
        if args.speedup is not None:
            args.skip_budget = None
    if args.command == "infer" and args.method == "morca":
        if args.policy is None:
            parser.error("MORCA inference requires --policy")
        if args.skip_budget is None and (args.speedup is None or args.calibration is None):
            parser.error("provide --skip-budget or --speedup with --calibration")
    model_root = Path(os.environ.get("MODEL_ROOT", "models")).resolve()
    os.environ.setdefault("TORCH_HOME", str(model_root / "torch"))
    os.environ.setdefault("HF_HOME", str(model_root / "huggingface"))
    if args.command == "infer":
        from morca.generation import run
    elif args.command == "calibrate":
        from morca.data import calibrate as run
    elif args.command == "collect":
        from morca.data import collect as run
    elif args.command == "train":
        args.epochs = args.epochs or (25 if args.online else 200)
        from morca.training import run
    else:
        from morca.evaluation import run
    run(args)


if __name__ == "__main__":
    main()
