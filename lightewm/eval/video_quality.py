from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np
import pandas
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm"}


@dataclass
class VideoPair:
    row_id: int
    demo_id: str
    camera_key: str
    real_path: str
    generated_path: str


def load_metadata_records(metadata_path: str) -> list[dict]:
    if metadata_path.endswith(".jsonl"):
        records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if metadata_path.endswith(".json"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Expected a list in JSON metadata: {metadata_path}")
        return records
    metadata = pandas.read_csv(metadata_path)
    return [metadata.iloc[i].to_dict() for i in range(len(metadata))]


def _metadata_video_value(value):
    if isinstance(value, dict):
        return value.get("path") or value.get("video") or value.get("file")
    return value


def resolve_dataset_path(base_path: str, value) -> str:
    value = _metadata_video_value(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("metadata video field is empty")
    path = str(value)
    if os.path.isabs(path):
        return path
    return os.path.join(base_path, path)


def generated_video_name(row_id: int, demo_id: str, camera_key: str) -> str:
    return f"{int(row_id):06d}__{demo_id}__{camera_key}.mp4"


def find_generated_video(generated_dir: str, row_id: int, demo_id: str, camera_key: str) -> str | None:
    generated_root = Path(generated_dir)
    exact_path = generated_root / generated_video_name(row_id, demo_id, camera_key)
    if exact_path.exists():
        return str(exact_path)

    candidates = sorted(generated_root.glob(f"{int(row_id):06d}__*.mp4"))
    if len(candidates) == 1:
        return str(candidates[0])
    return None


def collect_video_pairs(
    metadata_path: str,
    dataset_base_path: str,
    generated_dir: str,
    *,
    max_samples: int | None = None,
    video_key: str = "video",
    demo_id_key: str = "demo_id",
    camera_key_col: str = "camera_key",
) -> tuple[list[VideoPair], list[dict]]:
    records = load_metadata_records(metadata_path)
    if max_samples is not None:
        records = records[: int(max_samples)]

    pairs: list[VideoPair] = []
    missing: list[dict] = []
    for row_id, record in enumerate(records):
        try:
            real_path = resolve_dataset_path(dataset_base_path, record.get(video_key))
        except Exception as exc:
            missing.append({"row_id": row_id, "reason": f"bad_real_path: {exc}"})
            continue

        demo_id = str(record.get(demo_id_key, row_id))
        camera_key = str(record.get(camera_key_col, "unknown"))
        generated_path = find_generated_video(generated_dir, row_id, demo_id, camera_key)
        if generated_path is None:
            missing.append(
                {
                    "row_id": row_id,
                    "demo_id": demo_id,
                    "camera_key": camera_key,
                    "reason": "generated_video_missing",
                }
            )
            continue
        if not os.path.exists(real_path):
            missing.append(
                {
                    "row_id": row_id,
                    "real_path": real_path,
                    "reason": "real_video_missing",
                }
            )
            continue
        pairs.append(
            VideoPair(
                row_id=row_id,
                demo_id=demo_id,
                camera_key=camera_key,
                real_path=real_path,
                generated_path=generated_path,
            )
        )
    return pairs, missing


def read_video_frames(path: str, *, max_decode_frames: int | None = None) -> np.ndarray:
    frames = []
    for frame_id, frame in enumerate(iio.imiter(path)):
        if max_decode_frames is not None and frame_id >= max_decode_frames:
            break
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(frame.astype(np.uint8, copy=False))
    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0)


def sample_frames(frames: np.ndarray, num_frames: int | None) -> np.ndarray:
    if num_frames is None:
        return frames
    num_frames = int(num_frames)
    if num_frames <= 0:
        return frames
    if len(frames) == num_frames:
        return frames
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(np.int64)
        return frames[indices]
    pad_count = num_frames - len(frames)
    padding = np.repeat(frames[-1:], pad_count, axis=0)
    return np.concatenate([frames, padding], axis=0)


def resize_frames(frames: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    if size is None:
        return frames
    height, width = int(size[0]), int(size[1])
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    resized = []
    for frame in frames:
        image = Image.fromarray(frame)
        image = image.resize((width, height), Image.BICUBIC)
        resized.append(np.asarray(image, dtype=np.uint8))
    return np.stack(resized, axis=0)


def align_pair_frames(
    real_frames: np.ndarray,
    generated_frames: np.ndarray,
    *,
    num_frames: int | None,
    pair_resize: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_frames is None:
        num_frames = min(len(real_frames), len(generated_frames))
    real_frames = sample_frames(real_frames, num_frames)
    generated_frames = sample_frames(generated_frames, num_frames)

    if pair_resize is None:
        pair_resize = (int(generated_frames.shape[1]), int(generated_frames.shape[2]))
    real_frames = resize_frames(real_frames, pair_resize)
    generated_frames = resize_frames(generated_frames, pair_resize)
    return real_frames, generated_frames


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


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(np.mean(clean))


def _write_pair_csv(path: Path, pair_rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "demo_id",
        "camera_key",
        "real_path",
        "generated_path",
        "ssim",
        "psnr",
        "lpips",
        "num_frames",
        "height",
        "width",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def evaluate_video_quality(
    *,
    metadata_path: str,
    dataset_base_path: str,
    generated_dir: str,
    weight_name: str,
    output_dir: str | None = None,
    max_samples: int | None = None,
    metrics: Iterable[str] = ("fvd", "ssim", "psnr", "lpips"),
    pair_num_frames: int | None = None,
    pair_resize: tuple[int, int] | None = None,
    device: str | None = None,
    lpips_net: str = "alex",
    metric_batch_size: int = 8,
    fvd_pretrained: bool = True,
    fvd_num_frames: int = 16,
    fvd_image_size: int = 112,
    strict: bool = True,
) -> dict:
    import torch

    metrics = {metric.lower() for metric in metrics}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs, missing = collect_video_pairs(
        metadata_path=metadata_path,
        dataset_base_path=dataset_base_path,
        generated_dir=generated_dir,
        max_samples=max_samples,
    )
    if strict and missing:
        preview = missing[:5]
        raise FileNotFoundError(f"Missing {len(missing)} video pairs. First missing entries: {preview}")
    if not pairs:
        raise ValueError("No generated/real video pairs were found for evaluation.")

    lpips_evaluator = None
    if "lpips" in metrics:
        lpips_evaluator = LPIPSEvaluator(device=device, net=lpips_net, batch_size=metric_batch_size)

    fvd_extractor = None
    if "fvd" in metrics:
        fvd_extractor = TorchvisionR3D18FVD(
            device=device,
            pretrained=fvd_pretrained,
            batch_size=metric_batch_size,
            num_frames=fvd_num_frames,
            image_size=fvd_image_size,
        )

    pair_rows: list[dict] = []
    real_videos_for_fvd: list[np.ndarray] = []
    generated_videos_for_fvd: list[np.ndarray] = []
    for pair in pairs:
        real_frames = read_video_frames(pair.real_path)
        generated_frames = read_video_frames(pair.generated_path)
        if fvd_extractor is not None:
            real_videos_for_fvd.append(real_frames)
            generated_videos_for_fvd.append(generated_frames)

        real_aligned, generated_aligned = align_pair_frames(
            real_frames,
            generated_frames,
            num_frames=pair_num_frames,
            pair_resize=pair_resize,
        )
        row = asdict(pair)
        row.update(
            {
                "num_frames": int(len(real_aligned)),
                "height": int(real_aligned.shape[1]),
                "width": int(real_aligned.shape[2]),
                "ssim": None,
                "psnr": None,
                "lpips": None,
            }
        )
        if "ssim" in metrics or "psnr" in metrics:
            ssim_value, psnr_value = compute_ssim_and_psnr(real_aligned, generated_aligned)
            row["ssim"] = ssim_value
            row["psnr"] = psnr_value
        if lpips_evaluator is not None:
            row["lpips"] = lpips_evaluator(real_aligned, generated_aligned)
        pair_rows.append(row)

    summary = {
        "weight_name": weight_name,
        "generated_dir": generated_dir,
        "metadata_path": metadata_path,
        "dataset_base_path": dataset_base_path,
        "num_pairs": len(pairs),
        "num_missing": len(missing),
        "missing": missing,
        "metrics": {
            "ssim": _mean_or_none(row["ssim"] for row in pair_rows),
            "psnr": _mean_or_none(row["psnr"] for row in pair_rows),
            "lpips": _mean_or_none(row["lpips"] for row in pair_rows),
            "fvd": None,
        },
        "metric_config": {
            "device": device,
            "pair_num_frames": pair_num_frames,
            "pair_resize": pair_resize,
            "lpips_net": lpips_net if "lpips" in metrics else None,
            "fvd_backend": None,
            "fvd_pretrained": fvd_pretrained if "fvd" in metrics else None,
            "fvd_num_frames": fvd_num_frames if "fvd" in metrics else None,
            "fvd_image_size": fvd_image_size if "fvd" in metrics else None,
        },
    }

    if fvd_extractor is not None:
        real_features = fvd_extractor.extract(real_videos_for_fvd)
        generated_features = fvd_extractor.extract(generated_videos_for_fvd)
        summary["metrics"]["fvd"] = frechet_distance(real_features, generated_features)
        summary["metric_config"]["fvd_backend"] = fvd_extractor.backend_name
        summary["metric_config"]["fvd_feature_dim"] = int(real_features.shape[1])

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_pair_csv(output_path / "pairs.csv", pair_rows)
        with open(output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        if missing:
            with open(output_path / "missing.json", "w", encoding="utf-8") as f:
                json.dump(missing, f, indent=2, ensure_ascii=False)

    return summary
