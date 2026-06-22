#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FASTWAM_SRC = ROOT / "lightewm" / "vendor" / "fastwam"
HIDIT_ROOT = ROOT.parent / "HiDiT" / "Causal-Forcing"
for path in (str(FASTWAM_SRC), str(HIDIT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _resolve_source_file(value: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [
        ROOT / path,
        ROOT / "data" / path,
        ROOT / "data" / "LIBERO-datasets" / path.name,
    ]
    if not path.is_absolute() and "LIBERO-datasets" in path.parts:
        idx = path.parts.index("LIBERO-datasets")
        candidates.append(ROOT / "data" / Path(*path.parts[idx:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def _normalize(actions: np.ndarray, stats: dict) -> np.ndarray:
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)
    eps = float(stats.get("eps", 1e-6))
    return np.clip(2.0 * (actions.astype(np.float32) - min_v) / (max_v - min_v + eps) - 1.0, -1.0, 1.0)


class LiberoActionWindowDataset(torch.utils.data.Dataset):
    def __init__(self, metadata: Path, stats: Path, horizon: int, limit: int):
        with metadata.open("r", encoding="utf-8", newline="") as f:
            self.rows = [row for row in csv.DictReader(f) if row.get("camera_key") == "agentview_rgb"][:limit]
        with stats.open("r", encoding="utf-8") as f:
            self.stats = json.load(f)
        self.horizon = int(horizon)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with h5py.File(_resolve_source_file(row["source_file"]), "r") as f:
            actions = np.asarray(f["data"][row["demo_id"]]["actions"], dtype=np.float32)
        max_start = max(0, len(actions) - self.horizon)
        start = 0 if max_start == 0 else int(np.random.randint(0, max_start + 1))
        ids = np.arange(start, min(start + self.horizon, len(actions)), dtype=np.int64)
        if len(ids) < self.horizon:
            ids = np.concatenate([ids, np.full(self.horizon - len(ids), ids[-1], dtype=np.int64)])
        action = _normalize(actions[ids], self.stats)
        return torch.from_numpy(action).float()


def _grad_norms(module):
    buckets = {"total": 0.0, "encoder": 0.0, "head": 0.0, "blocks": 0.0, "time": 0.0, "text": 0.0}
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        norm = float(param.grad.detach().float().norm().item())
        sq = norm * norm
        buckets["total"] += sq
        if "action_encoder" in name:
            buckets["encoder"] += sq
        elif name.startswith("head."):
            buckets["head"] += sq
        elif name.startswith("blocks."):
            buckets["blocks"] += sq
        elif name.startswith("time_"):
            buckets["time"] += sq
        elif name.startswith("text_embedding."):
            buckets["text"] += sq
    return {key: math.sqrt(value) for key, value in buckets.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=52)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--random-action-init", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device(args.device)

    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
    from model.action_mot import HDRActionMoT

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
        "use_gradient_checkpointing": False,
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
        "use_gradient_checkpointing": False,
    }
    fastwam = FastWAM.from_wan22_pretrained(
        device=str(device),
        torch_dtype=dtype,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        load_text_encoder=False,
        redirect_common_files=False,
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=None,
        skip_dit_load_from_pretrain=True,
    )
    video_model = fastwam.video_expert
    if not hasattr(video_model, "dim"):
        video_model.dim = 3072
    if not hasattr(video_model, "text_dim"):
        video_model.text_dim = 4096

    action_dit = HDRActionMoT(
        video_model=video_model,
        action_dim=7,
        hidden_dim=1024,
        ffn_dim=4096,
        freq_dim=256,
        eps=1.0e-6,
        actions_per_leaf=args.horizon,
        action_attend_video="all",
        use_gradient_checkpointing=False,
    ).to(device=device, dtype=dtype)
    if not args.random_action_init:
        backbone, _ = action_dit.build_interpolated_video_backbone_state_dict(apply_alpha_scaling=True)
        action_dit.load_state_dict(backbone, strict=False)
    action_dit.train().requires_grad_(True)
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    dataset = LiberoActionWindowDataset(
        ROOT / "data/libero_i2v_train/metadata_dense_prompt_hdr_video_action_joint.csv",
        ROOT / "data/libero_i2v_train/hdr_video_action_joint_action_stats.json",
        args.horizon,
        args.limit,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    context = torch.zeros(args.batch_size, 512, 4096, device=device, dtype=dtype)
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            actions = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            actions = next(iterator)
        actions = actions.to(device=device, dtype=dtype)
        noise = torch.randn_like(actions)
        timestep = scheduler.sample_training_t(actions.shape[0], device=device, dtype=dtype)
        noisy = scheduler.add_noise(
            actions.flatten(0, 1).unsqueeze(-1).unsqueeze(-1),
            noise.flatten(0, 1).unsqueeze(-1).unsqueeze(-1),
            timestep[:, None].expand(-1, actions.shape[1]).flatten(0, 1),
        ).squeeze(-1).squeeze(-1).unflatten(0, actions.shape[:2])
        target = scheduler.training_target(actions, noise, timestep[:, None].expand(-1, actions.shape[1]))
        action_dit.zero_grad(set_to_none=True)
        state = action_dit._prepare_action_state(noisy, timestep, {"prompt_embeds": context[: actions.shape[0]]})
        x = state["tokens"]
        empty_k = torch.empty(actions.shape[0], 0, action_dit.num_heads, action_dit.attn_head_dim, device=device, dtype=dtype)
        empty_v = torch.empty_like(empty_k)
        mask = action_dit._action_self_only_mask(actions.shape[1], 0, device)
        action_dit.disable_action_text_cross_attn = True
        for block in action_dit.blocks:
            x = action_dit._action_layer(x, block, empty_k, empty_v, state, mask)
        pred = action_dit.head(x)
        loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=2).mean(dim=1)
        weight = scheduler.training_weight(timestep).to(device=loss.device, dtype=loss.dtype)
        loss = (loss * weight).mean()
        loss.backward()
        norms = _grad_norms(action_dit)
        print(
            f"[HDRActionOnlyLibero] step={step} loss={float(loss.detach().cpu()):.6f} "
            f"t={float(timestep.float().mean().cpu()):.3f} "
            + " ".join(f"{k}={v:.6f}" for k, v in sorted(norms.items())),
            flush=True,
        )


if __name__ == "__main__":
    main()
