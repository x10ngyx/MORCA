from pathlib import Path
import torch
from .io import read_prompts, write_json, resolve_budget
from .measurement import measure
from .pipeline import load_pipeline, generation_options, inference_config, method_context
from .policy import Policy


def generate_one(pipe, args, row, output, *, method=None, policy=None, budget=None, profile=False):
    from wan.utils.utils import cache_video

    method = method or args.method
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with method_context(
        pipe,
        args.model,
        method,
        policy=policy,
        budget=budget or 0,
        **{
            key: getattr(args, key)
            for key in ("threshold", "retention", "max_skip")
            if getattr(args, key) is not None
        },
    ) as (controller, options):
        with measure(pipe, args.model, profile=profile) as timing:
            video = pipe.generate(row["prompt"], **generation_options(args.model), **options)
        trace = None
        if controller is not None and hasattr(controller, "saved_states"):
            trace = dict(
                states=torch.stack(controller.saved_states),
                actions=torch.tensor(controller.actions),
                actor_mask=torch.tensor(controller.actor_masks),
            )
    cache_video(
        tensor=video[None],
        save_file=str(output / (row["sample_id"] + ".mp4")),
        fps=inference_config(args.model)["fps"],
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    write_json(
        output / (row["sample_id"] + ".json"), dict(**row, method=method, budget=budget, **timing)
    )
    return (timing, trace)


def run(args):
    rows = read_prompts(args.prompts, args.prompt)
    pipe = load_pipeline(args.model, args.source, args.checkpoint)
    policy = Policy(args.policy) if args.method == "morca" else None
    budget = resolve_budget(args) if args.method == "morca" else 0
    with method_context(
        pipe,
        args.model,
        args.method,
        policy=policy,
        budget=budget,
        **{
            key: getattr(args, key)
            for key in ("threshold", "retention", "max_skip")
            if getattr(args, key) is not None
        },
    ) as (_, options):
        pipe.generate(rows[0]["prompt"], **generation_options(args.model), **options)
    for row in rows:
        generate_one(pipe, args, row, args.output, policy=policy, budget=budget)
        if args.profile:
            generate_one(
                pipe,
                args,
                row,
                Path(args.output) / "profile",
                policy=policy,
                budget=budget,
                profile=True,
            )
