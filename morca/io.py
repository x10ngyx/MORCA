import json
from pathlib import Path


def read_prompts(path=None, prompt=None):
    if prompt is not None:
        return [dict(sample_id="sample", prompt=prompt)]
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [
            dict(
                sample_id=r.get("sample_id", str(i)),
                prompt=r.get("prompt", r.get("prompt_en")),
                **{"split": r["split"]} if "split" in r else {},
            )
            for (i, r) in enumerate(rows)
        ]
    return [
        dict(sample_id=f"{i:04d}", prompt=p)
        for (i, p) in enumerate(path.read_text().splitlines())
        if p.strip()
    ]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def resolve_budget(args):
    if args.skip_budget is not None:
        return args.skip_budget
    curve = json.loads(Path(args.calibration).read_text())
    intercept, slope = (curve["latency_intercept"], curve["latency_per_skip"])
    k = round((curve["baseline_seconds"] / args.speedup - intercept) / slope)
    return max(0, min(48 if args.model == "wan21" else 47, k))
