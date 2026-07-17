#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = ROOT / "lightewm" / "vendor" / "fastwam"
for path in (FASTWAM_ROOT, ROOT / "data" / "python-packages" / "fastwam_pydeps", ROOT / "third_parties" / "LIBERO"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils import misc

register_default_resolvers()


def _repo_abs(path: str | Path) -> str:
    return str((ROOT / path).resolve())


def _build_cfg(args: argparse.Namespace):
    config_dir = str((FASTWAM_ROOT / "configs").resolve())
    overrides = [
        "task=libero_joint_2cam224_1e-4",
        "model=fastwam_joint",
        "data=libero_2cam",
        "model.redirect_common_files=false",
        "model.mot_checkpoint_mixed_attn=false",
        "model.action_attend_video=local_clean_first",
        "model.video_dit_pretrained_path=" + str((ROOT / "checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors").resolve()),
        "model.action_dit_pretrained_path=" + str((ROOT / "checkpoints/ActionDiT_linear_interp_Wan22Robot_alphascale_1024hdim.pt").resolve()),
        "model.model_id=Wan-AI/Wan2.2-TI2V-5B",
        "model.tokenizer_model_id=Wan-AI/Wan2.1-T2V-1.3B",
        "data.train.dataset_dirs=["
        + ",".join(
            [
                _repo_abs("data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"),
                _repo_abs("data/libero_mujoco3.3.2/libero_object_no_noops_lerobot"),
                _repo_abs("data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot"),
                _repo_abs("data/libero_mujoco3.3.2/libero_10_no_noops_lerobot"),
            ]
        )
        + "]",
        "data.train.text_embedding_cache_dir=" + _repo_abs("data/text_embeds_cache/libero"),
        "+data.train.pretrained_norm_stats=" + _repo_abs(args.dataset_stats),
        "data.train.is_training_set=false",
        "+data.train.hdr_enabled=true",
        "+data.train.hdr_local_rgb_frames=9",
        "+data.train.hdr_tree_rgb_frames=4",
        "+data.train.hdr_total_rgb_frames=13",
        "+data.train.hdr_tree_sampling=uniform_local_start_to_end",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name="train", overrides=overrides)


def _denormalize_action(action: torch.Tensor, processor) -> torch.Tensor:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Expected a single action key for LIBERO FastWAM.")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    return normalizer.backward(action.detach().cpu().float())[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="logs/eval/LIBERO-FASTWAMHDR/open_loop_action")
    args = parser.parse_args()

    cfg = _build_cfg(args)
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    dtype = torch.bfloat16
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(out_dir))

    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(str((ROOT / args.checkpoint).resolve()))
    model.eval()

    dataset = instantiate(cfg.data.train)
    dataset.lerobot_dataset.processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(str((ROOT / args.dataset_stats).resolve()))
    )
    sample = dataset[int(args.index)]

    video = sample["video"]
    prompt = sample["prompt"]
    input_image = video[:, 0].unsqueeze(0)
    proprio = sample["proprio"][0]
    context = sample.get("context")
    context_mask = sample.get("context_mask")
    action = sample["action"]

    infer_kwargs = {
        "prompt": None,
        "context": context,
        "context_mask": context_mask,
        "input_image": input_image,
        "action_horizon": int(action.shape[0]),
        "num_video_frames": int(video.shape[1]),
        "proprio": proprio,
        "num_inference_steps": int(args.num_inference_steps),
        "seed": int(args.seed),
        "rand_device": "cpu",
        "tiled": False,
    }
    with torch.no_grad():
        pred = model.infer_action(**infer_kwargs)["action"]

    processor = dataset.lerobot_dataset.processor
    pred_denorm = _denormalize_action(pred, processor)
    gt_denorm = _denormalize_action(action, processor)
    diff = pred_denorm - gt_denorm

    base = f"idx_{int(args.index):06d}_step_{Path(args.checkpoint).stem}"
    png_path = out_dir / f"{base}.png"
    json_path = out_dir / f"{base}.json"
    pt_path = out_dir / f"{base}.pt"

    dim_names = ["dx", "dy", "dz", "dax", "day", "daz", "gripper"]
    timesteps = list(range(pred_denorm.shape[0]))
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes_flat = axes.reshape(-1)
    for dim in range(pred_denorm.shape[1]):
        ax = axes_flat[dim]
        name = dim_names[dim] if dim < len(dim_names) else f"dim{dim}"
        ax.plot(timesteps, gt_denorm[:, dim].numpy(), label="gt", linewidth=1.8)
        ax.plot(timesteps, pred_denorm[:, dim].numpy(), label="pred", linewidth=1.4)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    axes_flat[-1].axis("off")
    axes_flat[0].legend(loc="best")
    fig.suptitle(f"Open-loop action prediction: index={args.index}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    metrics = {
        "checkpoint": str(args.checkpoint),
        "dataset_stats": str(args.dataset_stats),
        "index": int(args.index),
        "prompt": prompt,
        "num_video_frames": int(video.shape[1]),
        "num_inference_steps": int(args.num_inference_steps),
        "action_l1": float(diff.abs().mean().item()),
        "action_l2": float(diff.pow(2).mean().item()),
        "plot": str(png_path),
        "tensor_dump": str(pt_path),
    }
    torch.save({"pred": pred_denorm, "gt": gt_denorm, "diff": diff, "metrics": metrics}, pt_path)
    json_path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
