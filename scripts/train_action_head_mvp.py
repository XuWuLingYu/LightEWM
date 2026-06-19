#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lightewm.action_head import ActionHeadMVP, LiberoActionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal LIBERO Action Head MVP.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    dataset = LiberoActionDataset(args.manifest, sample_stride=args.sample_stride)
    first = dataset[0]
    model = ActionHeadMVP(
        num_tasks=dataset.num_tasks,
        proprio_dim=int(first["proprio"].numel()),
        action_dim=int(first["action"].numel()),
    ).to(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in loader:
            image = batch["image"].to(args.device, non_blocking=True)
            proprio = batch["proprio"].to(args.device, non_blocking=True)
            task_id = batch["task_id"].to(args.device, non_blocking=True)
            action = batch["action"].to(args.device, non_blocking=True)
            pred = model(image, proprio, task_id)
            loss = loss_fn(pred, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * image.size(0)
            total += image.size(0)
        mean_loss = total_loss / max(1, total)
        history.append({"epoch": epoch, "loss": mean_loss, "frames": total})
        print(f"[Train] epoch={epoch} frames={total} loss={mean_loss:.6f}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "task_to_id": dataset.task_to_id,
        "proprio_dim": model.proprio_dim,
        "action_dim": model.action_dim,
        "manifest": str(Path(args.manifest).resolve()),
        "history": history,
    }
    ckpt_path = output_dir / "action_head_mvp.pt"
    torch.save(ckpt, ckpt_path)
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(ckpt_path), "history": history, "num_frames": len(dataset)}, f, indent=2)
    print(f"[Train] saved {ckpt_path}")


if __name__ == "__main__":
    main()
