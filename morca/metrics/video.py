from __future__ import annotations
from pathlib import Path
import imageio.v2 as imageio
import numpy as np


def decode_video_rgb(path: str | Path) -> np.ndarray:
    video_path = Path(path).resolve(strict=True)
    reader = imageio.get_reader(str(video_path), "ffmpeg", output_params=["-threads", "1"])
    frames: list[np.ndarray] = []
    try:
        for index, frame in enumerate(reader):
            array = np.asarray(frame)
            frames.append(array)
    finally:
        reader.close()
    video_thwc = np.stack(frames)
    return np.ascontiguousarray(
        video_thwc.transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255.0)
    )
