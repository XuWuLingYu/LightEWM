import argparse
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = ROOT / "lightewm" / "vendor" / "fastwam"
for path in (
    FASTWAM_ROOT,
    ROOT / "data" / "python-packages" / "fastwam_pydeps",
    ROOT / "third_parties" / "LIBERO",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


def _rank_world():
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def _device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def _cache_path(root: Path, idx: int) -> Path:
    shard = int(idx) // 10000
    return root / f"shard_{shard:05d}" / f"{int(idx):08d}.pt"


def _load_vae(cfg, device: torch.device, dtype: torch.dtype):
    _, _, vae_config, _ = _resolve_configs(
        model_id=str(cfg.model.model_id),
        tokenizer_model_id=str(cfg.model.tokenizer_model_id),
        redirect_common_files=bool(cfg.model.redirect_common_files),
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=dtype,
        device=str(device),
    )
    vae.eval().requires_grad_(False)
    return vae


def _encode_videos(vae, videos: list[torch.Tensor], device: torch.device, dtype: torch.dtype):
    if not videos:
        raise ValueError("Expected at least one video to encode.")
    for video in videos:
        if video.ndim != 4:
            raise ValueError(f"Expected video [C,T,H,W], got {tuple(video.shape)}")
    video_batch = torch.stack(videos, dim=0).to(device=device, dtype=dtype)
    try:
        with torch.inference_mode():
            if hasattr(vae, "model") and hasattr(vae, "scale"):
                latents = vae.model.encode(video_batch, vae.scale)
            else:
                latents = vae.encode(video_batch, device=str(device), tiled=False)
    except RuntimeError as exc:
        if len(videos) <= 1 or "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = len(videos) // 2
        left = _encode_videos(vae, videos[:mid], device=device, dtype=dtype)
        right = _encode_videos(vae, videos[mid:], device=device, dtype=dtype)
        return torch.cat([left, right], dim=0)
    return latents.detach().cpu().to(torch.bfloat16)


def _payload_from_sample(sample: dict, latents: torch.Tensor):
    first_frame_latents = latents[:, :, 0:1].contiguous()
    payload = {
        "input_latents": latents.squeeze(0).contiguous(),
        "first_frame_latents": first_frame_latents.squeeze(0).contiguous(),
        "action": sample["action"].contiguous(),
        "proprio": sample["proprio"].contiguous(),
        "prompt": sample["prompt"],
        "image_is_pad": sample["image_is_pad"].contiguous(),
        "action_is_pad": sample["action_is_pad"].contiguous(),
        "proprio_is_pad": sample["proprio_is_pad"].contiguous(),
        "num_video_frames": int(sample["video"].shape[1]),
    }
    for key in (
        "local_video_frames",
        "action_video_transition_count",
        "hdr_tree_frame_indices",
        "hdr_local_frame_indices",
    ):
        if key in sample:
            value = sample[key]
            payload[key] = value.contiguous() if isinstance(value, torch.Tensor) else value
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", default=str(FASTWAM_ROOT))
    parser.add_argument("--task", default="libero_joint_2cam224_1e-4")
    parser.add_argument("--model", default="fastwam_joint")
    parser.add_argument("--data", default="libero_2cam")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--sample-workers", type=int, default=1)
    parser.add_argument("--load-retries", type=int, default=3)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world_size = _rank_world()
    device = _device()
    barrier_device_ids = [device.index] if device.type == "cuda" and device.index is not None else None
    dtype = torch.bfloat16
    out_root = Path(args.output_dir).resolve()
    if rank == 0:
        out_root.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=barrier_device_ids)
    misc.register_work_dir(str(out_root))

    fastwam_root = Path(args.fastwam_root).resolve()
    with initialize_config_dir(config_dir=str(fastwam_root / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"task={args.task}",
                f"model={args.model}",
                f"data={args.data}",
                *args.overrides,
            ],
        )
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    dataset = instantiate(cfg.data.train)
    vae = _load_vae(cfg, device=device, dtype=dtype)

    if rank == 0:
        metadata = {
            "num_samples": len(dataset),
            "world_size": world_size,
            "task": args.task,
            "model": args.model,
            "data": args.data,
            "overrides": list(args.overrides),
        }
        torch.save(metadata, out_root / "metadata.pt")
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=barrier_device_ids)

    num_samples = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))
    indices = [
        idx for idx in range(rank, num_samples, world_size)
        if args.overwrite or not _cache_path(out_root, idx).exists()
    ]
    iterator = tqdm(indices, desc=f"rank {rank}", disable=rank != 0)

    def load_sample(idx: int):
        path = _cache_path(out_root, idx)
        retries = max(int(args.load_retries), 1)
        for attempt in range(retries):
            try:
                sample = dataset._get(idx)
                return idx, path, sample
            except Exception:
                gc.collect()
                if attempt + 1 >= retries:
                    raise
                time.sleep(1.0 + attempt)

    encode_batch_size = max(int(args.encode_batch_size), 1)
    sample_workers = max(int(args.sample_workers), 1)

    def flush(batch):
        if not batch:
            return
        videos = [sample["video"] for _, _, sample in batch]
        latents = _encode_videos(vae, videos, device=device, dtype=dtype)
        for latent, (_, path, sample) in zip(latents, batch):
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = _payload_from_sample(sample, latent.unsqueeze(0))
            tmp_path = path.with_suffix(f".tmp.{os.getpid()}")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)

    batch = []
    if sample_workers == 1:
        for item in map(load_sample, iterator):
            batch.append(item)
            if len(batch) >= encode_batch_size:
                flush(batch)
                batch.clear()
    else:
        with ThreadPoolExecutor(max_workers=sample_workers) as executor:
            for item in executor.map(load_sample, iterator):
                batch.append(item)
                if len(batch) >= encode_batch_size:
                    flush(batch)
                    batch.clear()
    flush(batch)

    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=barrier_device_ids)
    if rank == 0:
        print(f"[precache] done output_dir={out_root} samples={num_samples}")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
