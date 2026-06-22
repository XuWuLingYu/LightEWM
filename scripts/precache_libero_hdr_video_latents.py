#!/usr/bin/env python3
"""Pre-cache fixed-5 HDR leaf video latents for LIBERO action training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


class CastingTextEncoder(torch.nn.Module):
    def __init__(self, text_encoder: torch.nn.Module, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = dtype

    def forward(self, text_prompts: list[str]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(device=self.device, dtype=self.dtype)
            for key, value in self.text_encoder(text_prompts=text_prompts).items()
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", default="../HiDiT/Causal-Forcing")
    parser.add_argument("--config-path", default="examples/LIBERO-HDR/train_action.yaml")
    parser.add_argument("--metadata-path", default="data/libero_i2v_train/metadata_dense_prompt_hdr_actions_leaf8.csv")
    parser.add_argument("--output-dir", default="data/libero_i2v_train/hdr_video_leaf_kv_fixed5_agentview")
    parser.add_argument(
        "--output-metadata-path",
        default="data/libero_i2v_train/metadata_dense_prompt_hdr_actions_leaf8_agentview_video_leaf_kv_cache.csv",
    )
    parser.add_argument("--camera-key", default="agentview_rgb")
    parser.add_argument("--generator-ckpt", default=None)
    parser.add_argument("--model-root", default="checkpoints")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--save-kv-cache",
        action="store_true",
        help="Also save full per-layer leaf K/V. This is very large and intended only for small debugging runs.",
    )
    return parser.parse_args()


def resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_backend_path(value: str | Path, backend_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((backend_root / path).resolve())


def load_generator_state(path: str | Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(path, map_location="cpu")
    if "generator" in state_dict and state_dict["generator"] is not None:
        state_dict = state_dict["generator"]
    elif "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    fixed = {}
    for key, value in state_dict.items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[key] = value
    return fixed


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_cache_name(row: dict[str, str]) -> Path:
    suite = Path(row.get("source_file", "unknown")).parent.name or "unknown"
    source = Path(row.get("source_file", "sample")).stem
    demo_id = str(row.get("demo_id", "demo"))
    return Path(suite) / f"{source}__{demo_id}.npz"


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_light_config(path: Path, repo_root: Path):
    cfg = OmegaConf.load(path)
    overrides = cfg.runner.params.causal_config_overrides
    official = OmegaConf.load(resolve_repo_path(cfg.runner.params.official_config_path, repo_root))
    merged = OmegaConf.merge(official, overrides)
    merged.model_kwargs.model_root = str(resolve_repo_path(cfg.runner.params.model_root, repo_root))
    return merged


def _make_full_block_mask(total_length: int, device: torch.device):
    from torch.nn.attention.flex_attention import create_block_mask

    padded_length = ((total_length + 127) // 128) * 128 - total_length
    valid = torch.zeros(total_length + padded_length, dtype=torch.bool, device=device)
    valid[:total_length] = True

    def attention_mask(b, h, q_idx, kv_idx):
        return valid[q_idx] & valid[kv_idx]

    return create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )


@torch.no_grad()
def compute_leaf_kv_cache(model, leaf_latents: torch.Tensor, conditional_dict: dict, dtype: torch.dtype):
    action_dit = model.action_dit
    video = model.generator.model
    device = leaf_latents.device
    batch_size, num_leaf, _, height, width = leaf_latents.shape
    timestep = torch.zeros([batch_size, num_leaf], device=device, dtype=dtype)

    x = leaf_latents.permute(0, 2, 1, 3, 4)
    x = [video.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    max_seq_len = max(u.size(1) for u in x)
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, max_seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    e = video.time_embedding(sinusoidal_embedding_1d(video.freq_dim, timestep.flatten()).type_as(x))
    e = video.time_projection(e).unflatten(1, (6, video.dim)).unflatten(dim=0, sizes=timestep.shape)
    context = conditional_dict["prompt_embeds"]
    context = video.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(video.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )
    if video.freqs.device != device:
        video.freqs = video.freqs.to(device)
    block_mask = _make_full_block_mask(int(x.shape[1]), device)

    leaf_k = []
    leaf_v = []
    for block in video.blocks:
        q, k, v, _ = action_dit._video_qkv(block, x, e, grid_sizes, video.freqs, temporal_positions=None)
        leaf_k.append(k.detach().float().cpu())
        leaf_v.append(v.detach().float().cpu())
        mixed = action_dit._video_self_attention_from_qkv(q, k, v, block_mask)
        x = action_dit._video_post(block, x, mixed, e, context, context_lens=None)
    return torch.stack(leaf_k, dim=1), torch.stack(leaf_v, dim=1)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    backend_root = resolve_repo_path(args.backend_root, repo_root)
    sys.path.insert(0, str(backend_root))

    from model.diffusion import CausalDiffusion
    from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
    from utils.dataset import TextVideoDataset
    from wan.modules.model import sinusoidal_embedding_1d

    globals()["sinusoidal_embedding_1d"] = sinusoidal_embedding_1d

    rank = int(os.environ.get("RANK", args.rank if args.rank is not None else 0))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size if args.world_size is not None else 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(int(args.seed) + rank)
    np.random.seed(int(args.seed) + rank)

    rows = [row for row in load_rows(resolve_repo_path(args.metadata_path, repo_root)) if row.get("camera_key") == args.camera_key]
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]
    output_dir = resolve_repo_path(args.output_dir, repo_root)
    output_metadata_path = resolve_repo_path(args.output_metadata_path, repo_root)

    cfg = load_light_config(resolve_repo_path(args.config_path, repo_root), repo_root)
    cfg.batch_size = 1
    cfg.image_or_video_shape[0] = 1
    cfg.height = int(args.height)
    cfg.width = int(args.width)
    cfg.num_frames = int(args.num_frames)
    cfg.disable_wandb = True
    cfg.action_training = bool(args.save_kv_cache)
    if args.save_kv_cache:
        cfg.action_dit_config = getattr(cfg, "action_dit_config", {}) or {
            "action_dim": 7,
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "freq_dim": 256,
            "eps": 1.0e-6,
            "actions_per_leaf": 8,
            "action_attend_video": "all",
            "use_gradient_checkpointing": False,
        }
    cfg.independent_first_frame = bool(getattr(cfg, "independent_first_frame", False))
    cfg.sampling_steps = int(getattr(cfg, "sampling_steps", max(cfg.vertical_step_budgets)))
    cfg.vertical_infer_fixed_denoise_steps = 5
    cfg.vertical_infer_preserve_budget_ratio = True
    cfg.vertical_infer_reference_total_steps = int(cfg.sampling_steps)
    cfg.model_kwargs.model_root = str(resolve_repo_path(args.model_root, repo_root))
    generator_ckpt = args.generator_ckpt or str(getattr(cfg, "generator_ckpt"))
    cfg.generator_ckpt = str(resolve_repo_path(generator_ckpt, repo_root))

    dtype = torch.bfloat16 if bool(cfg.mixed_precision) else torch.float32
    model = CausalDiffusion(cfg, device=device)
    model.generator = model.generator.to(device=device, dtype=dtype).eval().requires_grad_(False)
    model.text_encoder = model.text_encoder.to(device=device).eval().requires_grad_(False)
    model.vae = model.vae.to(device=device, dtype=torch.bfloat16 if bool(cfg.mixed_precision) else torch.float32)
    model.vae.eval().requires_grad_(False)
    if args.save_kv_cache:
        model.action_dit = model.action_dit.to(device=device, dtype=dtype).eval().requires_grad_(False)
    model.generator.load_state_dict(load_generator_state(cfg.generator_ckpt), strict=True)
    pipe = CausalDiffusionInferencePipeline(
        cfg,
        device=device,
        generator=model.generator,
        text_encoder=CastingTextEncoder(model.text_encoder, device=device, dtype=dtype),
        vae=model.vae,
    ).to(device=device).eval()

    generated_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(rows):
        cache_path = output_dir / safe_cache_name(row)
        row_with_cache = dict(row)
        row_with_cache["video_latent_cache_path"] = cache_path.as_posix()
        generated_rows.append(row_with_cache)
        if row_index % world_size != rank:
            continue
        if cache_path.exists() and not args.overwrite:
            continue

        sample_jsonl = output_dir / "_tmp_jsonl" / f"rank{rank}_{row_index}.jsonl"
        sample_jsonl.parent.mkdir(parents=True, exist_ok=True)
        video_path = resolve_repo_path(row["video"], resolve_repo_path(args.metadata_path, repo_root).parent)
        item = {
            "video_path": str(video_path),
            "prompt": row.get("dense_prompt") or row.get("prompt") or row.get("sparse_prompt"),
        }
        sample_jsonl.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")

        dataset = TextVideoDataset(
            metadata_path=str(sample_jsonl),
            height=int(cfg.height),
            width=int(cfg.width),
            num_frames=int(cfg.num_frames),
            variable_num_frames=False,
            max_num_frames=None,
        )
        sample = dataset[0]
        frames = sample["frames"].unsqueeze(0).to(device=device, dtype=dtype)
        prompt = [sample["prompts"]]
        with torch.no_grad():
            clean_latent = model.vae.encode_to_latent(frames[:, :, :1]).to(device=device, dtype=dtype)
            noise = torch.randn(
                [
                    1,
                    int(pipe.vertical_info["num_tokens"]),
                    int(cfg.image_or_video_shape[2]),
                    int(cfg.image_or_video_shape[3]),
                    int(cfg.image_or_video_shape[4]),
                ],
                device=device,
                dtype=dtype,
            )
            leaf_latents, _ = pipe.inference(
                noise=noise,
                text_prompts=prompt,
                initial_latent=clean_latent,
                return_latents=True,
                return_video=False,
            )
            if args.save_kv_cache:
                conditional_dict = {
                    key: value.to(device=device, dtype=dtype)
                    for key, value in model.text_encoder(text_prompts=prompt).items()
                }
                leaf_k, leaf_v = compute_leaf_kv_cache(model, leaf_latents.to(device=device, dtype=dtype), conditional_dict, dtype)
            else:
                leaf_k = None
                leaf_v = None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "leaf_latents": leaf_latents.detach().float().cpu().numpy()[0],
            "first_frame_latent": clean_latent.detach().float().cpu().numpy()[0],
            "video_timestep": np.zeros([int(pipe.vertical_info["num_leaf_frames"])], dtype=np.float32),
            "source_row": np.asarray([row_index], dtype=np.int64),
        }
        if leaf_k is not None and leaf_v is not None:
            payload["leaf_k"] = leaf_k.numpy()[0].astype(np.float16)
            payload["leaf_v"] = leaf_v.numpy()[0].astype(np.float16)
        np.savez_compressed(cache_path, **payload)
        if (row_index + 1) % 50 == 0 or rank == 0:
            print(f"[HDRVideoCache][rank {rank}] cached row={row_index} path={cache_path}", flush=True)

    if rank == 0:
        write_metadata(output_metadata_path, generated_rows)
        print(f"[HDRVideoCache] wrote metadata {output_metadata_path} rows={len(generated_rows)}", flush=True)


if __name__ == "__main__":
    main()
