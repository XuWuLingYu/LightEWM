from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from lightewm.eval.metrics import (
    LPIPSEvaluator,
    TorchvisionR3D18FVD,
    compute_ssim_and_psnr,
    frechet_distance,
)
from lightewm.eval.utils import (
    VIDEO_EXTENSIONS,
    VideoPair,
    align_pair_frames,
    collect_video_pairs,
    find_generated_video,
    generated_video_name,
    load_metadata_records,
    read_video_frames,
    resize_frames,
    resolve_dataset_path,
    sample_frames,
)


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
