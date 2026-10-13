from __future__ import annotations
import math
import cv2
import numpy as np

cv2.setNumThreads(1)
PROTOCOL_ID = "rgb_full_reference_v1"
PSNR_METRIC_NAME = "psnr_rgb_db"
SSIM_METRIC_NAME = "ssim_rgb"
LPIPS_METRIC_NAME = "lpips_alex_v0_1_spatial"
PSNR_MSE_CAP_THRESHOLD = 1e-10
PSNR_CAP_DB = 100.0


def psnr_per_frame(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference, candidate = (
        np.asarray(reference, dtype=np.float32),
        np.asarray(candidate, dtype=np.float32),
    )
    values: list[float] = []
    for ref_frame, candidate_frame in zip(reference, candidate, strict=True):
        mse = float(np.mean((ref_frame / 1.0 - candidate_frame / 1.0) ** 2))
        if mse < PSNR_MSE_CAP_THRESHOLD:
            values.append(PSNR_CAP_DB)
        else:
            values.append(20.0 * math.log10(1.0 / math.sqrt(mse)))
    return np.asarray(values, dtype=np.float64)


def _ssim_plane(reference: np.ndarray, candidate: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu_reference = cv2.filter2D(reference, -1, window)[5:-5, 5:-5]
    mu_candidate = cv2.filter2D(candidate, -1, window)[5:-5, 5:-5]
    mu_reference_sq = mu_reference**2
    mu_candidate_sq = mu_candidate**2
    mu_cross = mu_reference * mu_candidate
    sigma_reference_sq = cv2.filter2D(reference**2, -1, window)[5:-5, 5:-5] - mu_reference_sq
    sigma_candidate_sq = cv2.filter2D(candidate**2, -1, window)[5:-5, 5:-5] - mu_candidate_sq
    sigma_cross = cv2.filter2D(reference * candidate, -1, window)[5:-5, 5:-5] - mu_cross
    ssim_map = (
        (2.0 * mu_cross + c1)
        * (2.0 * sigma_cross + c2)
        / (
            (mu_reference_sq + mu_candidate_sq + c1)
            * (sigma_reference_sq + sigma_candidate_sq + c2)
        )
    )
    return float(ssim_map.mean())


def ssim_per_frame(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference, candidate = (
        np.asarray(reference, dtype=np.float32),
        np.asarray(candidate, dtype=np.float32),
    )
    values: list[float] = []
    for ref_frame, candidate_frame in zip(reference, candidate, strict=True):
        channel_values = [
            _ssim_plane(ref_frame[channel], candidate_frame[channel]) for channel in range(3)
        ]
        values.append(float(np.asarray(channel_values).mean()))
    return np.asarray(values, dtype=np.float64)


class LPIPSComputer:

    def __init__(self, device: str = "auto", batch_size: int = 8) -> None:
        import lpips
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = (
            lpips.LPIPS(net="alex", spatial=True, version="0.1", verbose=False)
            .eval()
            .to(self.device)
        )

    def per_frame(self, reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        import torch

        reference, candidate = (
            np.asarray(reference, dtype=np.float32),
            np.asarray(candidate, dtype=np.float32),
        )
        reference_tensor = torch.from_numpy(reference).mul(2.0).sub(1.0)
        candidate_tensor = torch.from_numpy(candidate).mul(2.0).sub(1.0)
        values: list[float] = []
        with torch.inference_mode():
            for start in range(0, reference_tensor.shape[0], self.batch_size):
                stop = min(start + self.batch_size, reference_tensor.shape[0])
                ref_batch = reference_tensor[start:stop].to(self.device)
                candidate_batch = candidate_tensor[start:stop].to(self.device)
                spatial_maps = self.model.forward(ref_batch, candidate_batch)
                batch_values = spatial_maps.mean(dim=(1, 2, 3)).detach().cpu()
                values.extend((float(value) for value in batch_values))
        return np.asarray(values, dtype=np.float64)
