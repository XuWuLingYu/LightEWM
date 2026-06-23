from __future__ import annotations

import math

import numpy as np

from lightewm.eval.utils import sample_frames


def compute_ssim_and_psnr(real_frames: np.ndarray, generated_frames: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    ssim_values = []
    psnr_values = []
    for real_frame, generated_frame in zip(real_frames, generated_frames):
        min_side = min(real_frame.shape[0], real_frame.shape[1])
        kwargs = {"channel_axis": 2, "data_range": 255}
        if min_side < 7:
            win_size = min_side if min_side % 2 == 1 else min_side - 1
            if win_size >= 3:
                kwargs["win_size"] = win_size
        ssim_values.append(float(structural_similarity(real_frame, generated_frame, **kwargs)))
        psnr_values.append(float(peak_signal_noise_ratio(real_frame, generated_frame, data_range=255)))
    return float(np.mean(ssim_values)), float(np.mean(psnr_values))


class LPIPSEvaluator:
    def __init__(self, *, device: str, net: str = "alex", batch_size: int = 8):
        import torch

        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS metric requires the optional 'lpips' package. Install it with `pip install lpips`."
            ) from exc

        self.torch = torch
        self.device = device
        self.batch_size = int(batch_size)
        self.model = lpips.LPIPS(net=net).to(device).eval()

    def _to_tensor(self, frames: np.ndarray):
        tensor = self.torch.from_numpy(frames).float() / 127.5 - 1.0
        return tensor.permute(0, 3, 1, 2).contiguous()

    def __call__(self, real_frames: np.ndarray, generated_frames: np.ndarray) -> float:
        values = []
        real_tensor = self._to_tensor(real_frames)
        generated_tensor = self._to_tensor(generated_frames)
        with self.torch.no_grad():
            for start in range(0, len(real_tensor), self.batch_size):
                real_batch = real_tensor[start : start + self.batch_size].to(self.device)
                generated_batch = generated_tensor[start : start + self.batch_size].to(self.device)
                distance = self.model(real_batch, generated_batch)
                values.extend(distance.detach().flatten().cpu().numpy().tolist())
        return float(np.mean(values))


class TorchvisionR3D18FVD:
    def __init__(
        self,
        *,
        device: str,
        pretrained: bool = True,
        batch_size: int = 4,
        num_frames: int = 16,
        image_size: int = 112,
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.torch = torch
        self.F = F
        self.device = device
        self.batch_size = int(batch_size)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.backend_name = "torchvision_r3d18_kinetics400" if pretrained else "torchvision_r3d18_untrained"

        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.model = r3d_18(weights=weights)
        self.model.fc = nn.Identity()
        self.model.to(device).eval()
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)

    def _prepare_batch(self, videos: list[np.ndarray]):
        processed = [sample_frames(video, self.num_frames) for video in videos]
        array = np.stack(processed, axis=0)
        tensor = self.torch.from_numpy(array).float() / 255.0
        tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()
        tensor = tensor.to(self.device)
        tensor = self.F.interpolate(
            tensor,
            size=(self.num_frames, self.image_size, self.image_size),
            mode="trilinear",
            align_corners=False,
        )
        return (tensor - self.mean) / self.std

    def extract(self, videos: list[np.ndarray]) -> np.ndarray:
        features = []
        with self.torch.no_grad():
            for start in range(0, len(videos), self.batch_size):
                batch = videos[start : start + self.batch_size]
                tensor = self._prepare_batch(batch)
                batch_features = self.model(tensor)
                features.append(batch_features.detach().cpu().numpy())
        return np.concatenate(features, axis=0)


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray, eps: float = 1e-6) -> float:
    from scipy import linalg

    features_a = np.asarray(features_a, dtype=np.float64)
    features_b = np.asarray(features_b, dtype=np.float64)
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("FVD features must be rank-2 arrays")
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError(f"Feature dimensions differ: {features_a.shape[1]} vs {features_b.shape[1]}")

    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)
    if len(features_a) > 1:
        sigma_a = np.cov(features_a, rowvar=False)
    else:
        sigma_a = np.zeros((features_a.shape[1], features_a.shape[1]), dtype=np.float64)
    if len(features_b) > 1:
        sigma_b = np.cov(features_b, rowvar=False)
    else:
        sigma_b = np.zeros((features_b.shape[1], features_b.shape[1]), dtype=np.float64)

    diff = mu_a - mu_b
    cov_product = sigma_a.dot(sigma_b)
    if np.count_nonzero(cov_product) == 0:
        covmean = np.zeros_like(sigma_a)
    else:
        covmean, _ = linalg.sqrtm(cov_product, disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_a.shape[0]) * eps
            covmean = linalg.sqrtm((sigma_a + offset).dot(sigma_b + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    distance = diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean)
    return float(max(distance, 0.0))
