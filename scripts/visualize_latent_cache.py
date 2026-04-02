#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightewm.model.wan.model_loader import ModelPool
from lightewm.utils.data import save_video


def split_file_name(file_name: str):
    result = []
    number = -1
    for char in file_name:
        if "0" <= char <= "9":
            if number == -1:
                number = 0
            number = number * 10 + ord(char) - ord("0")
        else:
            if number != -1:
                result.append(number)
                number = -1
            result.append(char)
    if number != -1:
        result.append(number)
    return tuple(result)


def list_cache_files(cache_dir: str):
    files = []
    for root, _, names in os.walk(cache_dir):
        for name in names:
            if name.endswith(".pth"):
                files.append(os.path.join(root, name))
    files.sort(key=split_file_name)
    return files


def infer_default_vae_path(latent_channels: int):
    if latent_channels == 16:
        return "checkpoints/Wan2.1-I2V-1.3B/Wan2.1_VAE.pth"
    if latent_channels == 48:
        return "checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    raise ValueError(
        f"Cannot infer VAE path from latent channels={latent_channels}. "
        "Please pass --vae-path explicitly."
    )


def load_vae(vae_path: str, device: str):
    model_pool = ModelPool()
    model_pool.auto_load_model(vae_path)
    vae = model_pool.fetch_model("wan_video_vae")
    if vae is None:
        raise RuntimeError(f"Failed to load VAE from: {vae_path}")
    vae = vae.to(device=device)
    return vae


def decode_to_pil_frames(video_tensor: torch.Tensor):
    if video_tensor.ndim != 5:
        raise ValueError(f"Expected decoded video tensor ndim=5, got {video_tensor.ndim}")
    frames = []
    for frame in video_tensor[0].permute(1, 2, 3, 0):
        frame = ((frame + 1.0) * 127.5).clamp(0, 255).to(dtype=torch.uint8, device="cpu").numpy()
        frames.append(Image.fromarray(frame))
    return frames


def main():
    parser = argparse.ArgumentParser(description="Decode cached Wan latents into preview videos.")
    parser.add_argument("--cache-dir", required=True, help="Latent cache root directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to save decoded videos.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of cache files to decode.")
    parser.add_argument("--latent-key", default="input_latents", help="Tensor key in cached sample to decode.")
    parser.add_argument("--vae-path", default=None, help="Optional explicit VAE checkpoint path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--tiled", action="store_true", help="Use tiled VAE decoding.")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cache_files = list_cache_files(args.cache_dir)
    if len(cache_files) == 0:
        raise FileNotFoundError(f"No .pth cache files found under: {args.cache_dir}")
    cache_files = cache_files[: max(0, args.num_samples)]
    os.makedirs(args.output_dir, exist_ok=True)

    first_sample = torch.load(cache_files[0], map_location="cpu", weights_only=False)
    latent = first_sample[0][args.latent_key]
    if not isinstance(latent, torch.Tensor):
        raise ValueError(
            f"Cached sample key '{args.latent_key}' is not a tensor: {type(latent).__name__}"
        )
    vae_path = args.vae_path or infer_default_vae_path(int(latent.shape[1]))
    print(f"[LatentPreview] cache_dir={args.cache_dir}")
    print(f"[LatentPreview] output_dir={args.output_dir}")
    print(f"[LatentPreview] latent_key={args.latent_key}")
    print(f"[LatentPreview] samples={len(cache_files)}")
    print(f"[LatentPreview] inferred_vae_path={vae_path}")
    print(f"[LatentPreview] device={args.device}")

    vae = load_vae(vae_path, args.device)

    for cache_path in tqdm(cache_files, desc="Decode cache"):
        sample = torch.load(cache_path, map_location="cpu", weights_only=False)
        shared = sample[0]
        latent_tensor = shared[args.latent_key]
        latent_tensor = latent_tensor.to(dtype=torch.bfloat16, device="cpu")

        rel_name = os.path.relpath(cache_path, args.cache_dir)
        stem = Path(rel_name).with_suffix("")
        save_path = os.path.join(args.output_dir, str(stem) + ".mp4")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if args.skip_existing and os.path.exists(save_path):
            continue

        with torch.no_grad():
            decoded = vae.decode(
                latent_tensor,
                device=args.device,
                tiled=args.tiled,
                tile_size=tuple(args.tile_size),
                tile_stride=tuple(args.tile_stride),
            )
        frames = decode_to_pil_frames(decoded)
        save_video(frames, save_path, fps=args.fps, quality=args.quality)
        del sample, shared, latent_tensor, decoded, frames
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
