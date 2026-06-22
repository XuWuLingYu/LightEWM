#!/usr/bin/env python3
"""Pre-encode LIBERO HDR video-action-joint samples for Causal-Forcing training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "lightewm" / "vendor" / "causal_forcing"
sys.path.insert(0, str(BACKEND_ROOT))

from utils.dataset import TextVideoDataset  # noqa: E402
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper  # noqa: E402


def _resolve(path: str | Path, base: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping YAML: {path}")
    return data


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _cache_path(output_dir: Path, index: int) -> Path:
    shard = output_dir / f"{index // 1000:05d}"
    shard.mkdir(parents=True, exist_ok=True)
    return shard / f"sample_{index:07d}.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Adapted Causal-Forcing yaml or example train yaml.")
    parser.add_argument("--output-dir", default="data/libero_i2v_train/preencoded_hdr_video_action_joint_fastwam_local")
    parser.add_argument("--output-jsonl", default="data/libero_i2v_train/metadata_preencoded_hdr_video_action_joint_fastwam_local.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg_path = _resolve(args.config, REPO_ROOT)
    cfg = _load_yaml(cfg_path)
    if "runner" in cfg:
        overrides = cfg["runner"]["params"]["causal_config_overrides"]
        dataset_cfg = cfg["dataset"]["params"]
        metadata_path = _resolve(dataset_cfg["metadata_path"], REPO_ROOT)
        base_path = _resolve(dataset_cfg["base_path"], REPO_ROOT)
        rows = []
        import csv

        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get(dataset_cfg.get("filter_key", ""), "")) == str(dataset_cfg.get("filter_value", "")):
                    rows.append(row)
        prompt_key = dataset_cfg.get("prompt_key", "dense_prompt")
        jsonl_rows = []
        for row in rows:
            source_file = row["source_file"]
            if not Path(source_file).is_absolute():
                source_file = str((base_path / source_file).resolve())
            item = {
                "prompt": row.get(prompt_key) or row.get("prompt") or row.get("sparse_prompt"),
                "video_path": row.get("video") or row.get("video_path") or "",
                "source_file": source_file,
                "demo_id": row["demo_id"],
                "camera_key": row.get("camera_key", ""),
                "num_frames": row.get("num_frames", ""),
                "dense_prompt": row.get("dense_prompt", ""),
                "sparse_prompt": row.get("sparse_prompt", ""),
                "action_stats_path": str((base_path / overrides["action_stats_path"]).resolve())
                if not Path(overrides["action_stats_path"]).is_absolute()
                else overrides["action_stats_path"],
                "proprio_stats_path": str((base_path / overrides["joint_proprio_stats_path"]).resolve())
                if not Path(overrides["joint_proprio_stats_path"]).is_absolute()
                else overrides["joint_proprio_stats_path"],
            }
            jsonl_rows.append(item)
        tmp_jsonl = Path(args.output_jsonl).with_suffix(".source.jsonl")
        tmp_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with tmp_jsonl.open("w", encoding="utf-8") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        data_path = tmp_jsonl
        config_base = REPO_ROOT
    else:
        overrides = cfg
        data_path = _resolve(cfg["data_path"], cfg_path.parent)
        config_base = cfg_path.parent

    model_kwargs = overrides.get("model_kwargs", {})
    model_name = model_kwargs.get("model_name", "Wan2.2-TI2V-5B")
    model_root = _resolve(model_kwargs.get("model_root", "checkpoints"), config_base)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    dataset = TextVideoDataset(
        metadata_path=str(data_path),
        height=int(overrides["height"]),
        width=int(overrides["width"]),
        num_frames=int(overrides["num_frames"]),
        variable_num_frames=bool(overrides.get("variable_num_frames_train", False)),
        max_num_frames=overrides.get("max_training_video_frames", 253),
        video_action_joint=bool(overrides.get("video_action_joint_training", False)),
        joint_window_frames=int(overrides.get("joint_window_frames", 13)),
        joint_source_fps=float(overrides.get("joint_source_fps", 16.0)),
        joint_target_fps=float(overrides.get("joint_target_fps", 10.0)),
        joint_video_frame_stride=int(overrides.get("joint_video_frame_stride", 1)),
        joint_camera_key=overrides.get("joint_camera_key", "agentview_rgb"),
        joint_include_terminal_video_frame=bool(overrides.get("joint_include_terminal_video_frame", False)),
        joint_norm_clip=float(overrides.get("joint_norm_clip", 1.0)),
        joint_drop_tree_tokens=bool(overrides.get("joint_drop_tree_tokens", False)),
        joint_tree_from_hdf5=bool(overrides.get("joint_tree_from_hdf5", False)),
        joint_tree_camera_key=overrides.get("joint_tree_camera_key", None),
        joint_proprio_stats_path=overrides.get("joint_proprio_stats_path", None),
    )

    output_dir = _resolve(args.output_dir, REPO_ROOT)
    output_jsonl = _resolve(args.output_jsonl, REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    source_rows = _load_jsonl(Path(data_path))
    limit = int(args.limit) if args.limit else len(dataset)
    limit = min(limit, len(dataset))
    if args.num_shards <= 0:
        raise ValueError(f"--num-shards must be positive, got {args.num_shards}")
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        raise ValueError(f"--shard-rank must be in [0, {args.num_shards}), got {args.shard_rank}")
    indices = list(range(int(args.shard_rank), limit, int(args.num_shards)))

    vae = WanVAEWrapper(model_name=model_name, model_root=str(model_root)).eval().requires_grad_(False).to(device=device, dtype=dtype)
    text_encoder = WanTextEncoder(model_name=model_name, model_root=str(model_root)).eval().requires_grad_(False).to(device=device)

    written = 0
    with torch.no_grad(), output_jsonl.open("w", encoding="utf-8") as out_f:
        for local_i, idx in enumerate(indices, start=1):
            out_path = _cache_path(output_dir, idx)
            if args.overwrite or not out_path.exists():
                sample = dataset[idx]
                frames = sample["frames"].unsqueeze(0).to(device=device, dtype=dtype)
                joint_local_frames = sample["joint_local_frames"].unsqueeze(0).to(device=device, dtype=dtype)
                clean_latent = vae.encode_to_latent(frames)[0].cpu().to(torch.bfloat16)
                joint_local_latents = vae.encode_to_latent(joint_local_frames)[0].cpu().to(torch.bfloat16)
                prompt_embeds = text_encoder([sample["prompts"]])["prompt_embeds"][0].cpu().to(torch.bfloat16)
                payload = {
                    "clean_latent": clean_latent,
                    "joint_local_start_latent": joint_local_latents[:1],
                    "joint_local_video_latents": joint_local_latents[1:],
                    "prompt_embeds": prompt_embeds,
                    "joint_actions": sample["joint_actions"].cpu().to(torch.float32),
                    "joint_proprio": sample["joint_proprio"].cpu().to(torch.float32),
                    "joint_window_start": sample.get("joint_window_start"),
                    "joint_window_indices": sample.get("joint_window_indices"),
                    "joint_video_indices": sample.get("joint_video_indices"),
                }
                torch.save(payload, out_path)
            row = dict(source_rows[idx])
            row["preencoded_cache_path"] = os.path.relpath(out_path, start=output_jsonl.parent)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written == 1 or written % 25 == 0:
                print(
                    f"[Cache] shard={args.shard_rank}/{args.num_shards} "
                    f"{written}/{len(indices)} global_idx={idx} {out_path}",
                    flush=True,
                )
    print(
        f"[Cache] shard={args.shard_rank}/{args.num_shards} wrote {written} rows to {output_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
