#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = ROOT / "lightewm" / "vendor" / "fastwam"
if str(FASTWAM_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTWAM_ROOT))


def _ensure_fastwam_local_links():
    expected = ROOT / "checkpoints" / "Wan-AI" / "Wan2.2-TI2V-5B"
    source = ROOT / "checkpoints" / "Wan2.2-TI2V-5B"
    expected.parent.mkdir(parents=True, exist_ok=True)
    if not expected.exists():
        expected.symlink_to(source.resolve(), target_is_directory=True)


def _load_rows(path: Path, limit: int | None):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("camera_key") == "agentview_rgb"]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _resolve_source_file(value: str) -> Path:
    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            (ROOT / path),
            (ROOT / "data" / path),
            (ROOT / "data" / "LIBERO-datasets" / path.name),
        ])
        parts = path.parts
        if "LIBERO-datasets" in parts:
            idx = parts.index("LIBERO-datasets")
            candidates.append(ROOT / "data" / Path(*parts[idx:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve source_file={value}; tried={candidates}")


def _sample_ids(total: int, count: int):
    if total <= 1:
        return np.zeros([count], dtype=np.int64)
    return np.linspace(0, total - 1, count).round().astype(np.int64)


def _frames_to_tensor(frames: np.ndarray, size: int):
    out = []
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB").resize((size, size), resample=Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32)
        out.append(torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(out, dim=1)


def _normalize_actions(actions: np.ndarray, stats: dict):
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)
    eps = float(stats.get("eps", 1e-6))
    actions = 2.0 * (actions.astype(np.float32) - min_v) / (max_v - min_v + eps) - 1.0
    return np.clip(actions, -1.0, 1.0)


def _load_context(row: dict, context_dir: Path, context_len: int):
    stem = Path(row["video"]).stem
    path = context_dir / f"{stem}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing context embedding: {path}")
    context = torch.load(path, map_location="cpu")
    if isinstance(context, dict):
        context = context.get("context", context.get("prompt_embeds"))
    if not torch.is_tensor(context):
        raise TypeError(f"Unsupported context payload in {path}: {type(context)}")
    context = context.to(dtype=torch.bfloat16)
    if context.ndim != 2:
        raise ValueError(f"Expected context [L,D], got {tuple(context.shape)} in {path}")
    context = context[:context_len]
    if context.shape[0] < context_len:
        pad = torch.zeros(context_len - context.shape[0], context.shape[1], dtype=context.dtype)
        context = torch.cat([context, pad], dim=0)
    mask = torch.ones(context_len, dtype=torch.bool)
    return context, mask


class LightEWMLiberoFastWAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path: Path,
        base_path: Path,
        context_dir: Path,
        stats_path: Path,
        *,
        num_video_frames: int,
        action_horizon: int,
        image_size: int,
        context_len: int,
        limit: int | None,
    ):
        self.rows = _load_rows(metadata_path, limit)
        self.base_path = base_path
        self.context_dir = context_dir
        self.num_video_frames = int(num_video_frames)
        self.action_horizon = int(action_horizon)
        self.image_size = int(image_size)
        self.context_len = int(context_len)
        with stats_path.open("r", encoding="utf-8") as f:
            self.stats = json.load(f)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        source_file = _resolve_source_file(row["source_file"])
        demo_id = row["demo_id"]
        with h5py.File(source_file, "r") as f:
            demo = f["data"][demo_id]
            obs = demo["obs"]
            raw_frames = np.asarray(obs["agentview_rgb"])
            raw_actions = np.asarray(demo["actions"], dtype=np.float32)
        n = min(raw_frames.shape[0], raw_actions.shape[0])
        raw_frames = raw_frames[:n]
        raw_actions = raw_actions[:n]

        frame_ids = _sample_ids(n, self.num_video_frames)
        action_ids = _sample_ids(n, self.action_horizon)
        frames = raw_frames[frame_ids][:, ::-1, ::-1, :]
        actions = _normalize_actions(raw_actions[action_ids], self.stats)
        context, context_mask = _load_context(row, self.context_dir, self.context_len)

        return {
            "video": _frames_to_tensor(frames, self.image_size),
            "action": torch.from_numpy(actions).float(),
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "context": context,
            "context_mask": context_mask,
            "prompt": row.get("dense_prompt") or row.get("prompt") or "",
        }


def _grad_norms(model):
    buckets = {
        "total": 0.0,
        "action_expert": 0.0,
        "action_encoder": 0.0,
        "action_head": 0.0,
        "action_blocks": 0.0,
        "video_expert": 0.0,
    }
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        value = float(param.grad.detach().float().norm().item())
        sq = value * value
        buckets["total"] += sq
        if name.startswith("action_expert.") or ".mixtures.action." in name:
            buckets["action_expert"] += sq
            if "action_encoder" in name:
                buckets["action_encoder"] += sq
            elif "head" in name:
                buckets["action_head"] += sq
            elif "blocks" in name:
                buckets["action_blocks"] += sq
        elif name.startswith("video_expert.") or ".mixtures.video." in name:
            buckets["video_expert"] += sq
    return {key: math.sqrt(value) for key, value in buckets.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/libero_i2v_train/metadata_dense_prompt_hdr_video_action_joint.csv")
    parser.add_argument("--base-path", default="data/libero_i2v_train")
    parser.add_argument("--context-dir", default="data/libero_i2v_train/t5_prompt_embeddings_wan2p2_bf16")
    parser.add_argument("--stats-path", default="data/libero_i2v_train/hdr_video_action_joint_action_stats.json")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--num-video-frames", type=int, default=33)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-action-init", action="store_true")
    parser.add_argument("--contiguous-window", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    _ensure_fastwam_local_links()

    from fastwam.models.wan22.fastwam import FastWAM

    video_dit_config = {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1.0e-6,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": True,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": 7,
        "action_group_causal_mask_mode": "group_diagonal",
    }
    action_dit_config = {
        "action_dim": 7,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-6,
        "use_gradient_checkpointing": True,
    }
    model = FastWAM.from_wan22_pretrained(
        device=args.device,
        torch_dtype=torch.bfloat16,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        load_text_encoder=False,
        redirect_common_files=False,
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=(
            None
            if args.random_action_init
            else str((ROOT / "checkpoints" / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt").resolve())
        ),
        skip_dit_load_from_pretrain=False,
        mot_checkpoint_mixed_attn=True,
        video_train_shift=5.0,
        video_infer_shift=5.0,
        video_num_train_timesteps=1000,
        action_train_shift=5.0,
        action_infer_shift=5.0,
        action_num_train_timesteps=1000,
        loss_lambda_video=1.0,
        loss_lambda_action=1.0,
    )
    model.train()
    model.requires_grad_(False)
    model.mot.train().requires_grad_(True)

    dataset = LightEWMLiberoFastWAMDataset(
        metadata_path=ROOT / args.metadata,
        base_path=ROOT / args.base_path,
        context_dir=ROOT / args.context_dir,
        stats_path=ROOT / args.stats_path,
        num_video_frames=args.num_video_frames,
        action_horizon=args.action_horizon,
        image_size=args.image_size,
        context_len=args.context_len,
        limit=args.limit,
    )
    if args.contiguous_window:
        old_getitem = dataset.__class__.__getitem__

        def contiguous_getitem(self, idx):
            row = self.rows[idx]
            source_file = _resolve_source_file(row["source_file"])
            demo_id = row["demo_id"]
            with h5py.File(source_file, "r") as f:
                demo = f["data"][demo_id]
                obs = demo["obs"]
                raw_frames = np.asarray(obs["agentview_rgb"])
                raw_actions = np.asarray(demo["actions"], dtype=np.float32)
            n = min(raw_frames.shape[0], raw_actions.shape[0])
            raw_frames = raw_frames[:n]
            raw_actions = raw_actions[:n]
            max_start = max(0, n - self.action_horizon)
            start = 0 if max_start == 0 else int(np.random.randint(0, max_start + 1))
            action_ids = np.arange(start, min(start + self.action_horizon, n), dtype=np.int64)
            if action_ids.shape[0] < self.action_horizon:
                action_ids = np.concatenate([
                    action_ids,
                    np.full(self.action_horizon - action_ids.shape[0], action_ids[-1], dtype=np.int64),
                ])
            frame_ids = action_ids[::max(1, self.action_horizon // max(1, self.num_video_frames - 1))]
            frame_ids = frame_ids[:self.num_video_frames]
            if frame_ids.shape[0] < self.num_video_frames:
                frame_ids = np.concatenate([
                    frame_ids,
                    np.full(self.num_video_frames - frame_ids.shape[0], frame_ids[-1], dtype=np.int64),
                ])
            frames = raw_frames[frame_ids][:, ::-1, ::-1, :]
            actions = _normalize_actions(raw_actions[action_ids], self.stats)
            context, context_mask = _load_context(row, self.context_dir, self.context_len)
            return {
                "video": _frames_to_tensor(frames, self.image_size),
                "action": torch.from_numpy(actions).float(),
                "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
                "context": context,
                "context_mask": context_mask,
                "prompt": row.get("dense_prompt") or row.get("prompt") or "",
            }

        dataset.__class__.__getitem__ = contiguous_getitem
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.mot.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)

    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            sample = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            sample = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = model.training_loss(sample)
        loss.backward()
        norms = _grad_norms(model)
        unclipped = torch.nn.utils.clip_grad_norm_(model.mot.parameters(), 1.0)
        optimizer.step()
        print(
            f"[FastWAMGrad] step={step} loss={float(loss.detach().float().cpu()):.6f} "
            f"clip_return={float(unclipped):.6f} "
            + " ".join(f"{k}={v:.6f}" for k, v in sorted(norms.items()))
            + " "
            + " ".join(f"{k}={float(v):.6f}" for k, v in sorted(loss_dict.items())),
            flush=True,
        )


if __name__ == "__main__":
    main()
