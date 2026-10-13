import numpy as np
from .io import write_json
from .metrics.core import psnr_per_frame, ssim_per_frame, LPIPSComputer
from .metrics.video import decode_video_rgb


def run(args):
    lpips = LPIPSComputer(device=args.device)
    scores = []
    for candidate in sorted(args.candidate.glob("*.mp4")):
        reference = args.reference / candidate.name
        a, b = (decode_video_rgb(reference), decode_video_rgb(candidate))
        scores.append(
            dict(
                video=candidate.name,
                psnr=float(psnr_per_frame(a, b).mean()),
                ssim=float(ssim_per_frame(a, b).mean()),
                lpips=float(lpips.per_frame(a, b).mean()),
            )
        )
    result = {
        key: float(np.mean([row[key] for row in scores])) for key in ("psnr", "ssim", "lpips")
    }
    result["videos"] = scores
    write_json(args.output / "quality.json", result)
