from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def write_yaml(data: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def deep_merge(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def set_dot_key(cfg: dict, key: str, value: Any) -> None:
    cur = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        child = cur.get(part)
        if not isinstance(child, dict):
            child = {}
            cur[part] = child
        cur = child
    cur[parts[-1]] = value


def latent_frame_count(num_rgb_frames: int) -> int:
    if num_rgb_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_rgb_frames}")
    padded = num_rgb_frames
    remainder = (padded - 1) % 4
    if remainder:
        padded += 4 - remainder
    return ((padded - 1) // 4) + 1


def ti2v_5b_latent_shape(batch_size: int, num_frames: int, height: int, width: int) -> list[int]:
    return [batch_size, latent_frame_count(num_frames), 48, height // 16, width // 16]


def adapt_official_config(
    *,
    official_config_path: str,
    output_config_path: str,
    data_path: str,
    model_root: str,
    output_overrides: dict | None = None,
    dot_overrides: dict | None = None,
) -> dict:
    cfg = load_yaml(official_config_path)
    for key in ("wandb_host", "wandb_key", "wandb_entity"):
        cfg.pop(key, None)
    cfg.pop("generator_ckpt", None)

    cfg["data_path"] = data_path
    cfg["data_backend"] = "jsonl_video"
    cfg["load_raw_video"] = True
    cfg.setdefault("model_kwargs", {})
    cfg["model_kwargs"]["model_name"] = cfg["model_kwargs"].get("model_name", "Wan2.2-TI2V-5B")
    cfg["model_kwargs"]["model_root"] = model_root

    if output_overrides:
        deep_merge(cfg, dict(output_overrides))

    height = int(cfg.get("height", 224))
    width = int(cfg.get("width", 224))
    num_frames = int(cfg.get("num_frames", 81))
    cfg["image_or_video_shape"] = ti2v_5b_latent_shape(
        int(cfg.get("batch_size", 1)),
        num_frames,
        height,
        width,
    )

    if dot_overrides:
        for key, value in dot_overrides.items():
            set_dot_key(cfg, key, value)

    write_yaml(cfg, output_config_path)
    return cfg
