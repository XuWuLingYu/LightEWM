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


def _episode_ranges(dataset, num_samples: int) -> list[tuple[int, int]]:
    base_dataset = getattr(dataset, "lerobot_dataset", None)
    episode_data_index = getattr(base_dataset, "episode_data_index", None)
    if not episode_data_index:
        raise ValueError("--shard-mode episode requires dataset.lerobot_dataset.episode_data_index")
    starts = episode_data_index["from"].detach().cpu().tolist()
    ends = episode_data_index["to"].detach().cpu().tolist()
    ranges = []
    for start, end in zip(starts, ends, strict=True):
        start = max(int(start), 0)
        end = min(int(end), int(num_samples))
        if start < end:
            ranges.append((start, end))
    if not ranges:
        raise ValueError("No episode ranges overlap the requested sample range")
    return ranges


def _episode_shard_indices(dataset, num_samples: int, rank: int, world_size: int) -> list[int]:
    ranges = _episode_ranges(dataset, num_samples)
    total = sum(end - start for start, end in ranges)
    target_start = (rank * total) // world_size
    target_end = ((rank + 1) * total) // world_size
    cursor = 0
    selected = []
    for start, end in ranges:
        length = end - start
        next_cursor = cursor + length
        if next_cursor > target_start and cursor < target_end:
            selected.extend(range(start, end))
        cursor = next_cursor
    return selected


def _payload_from_sample(sample: dict, latents: torch.Tensor, episode_latents: torch.Tensor | None = None):
    if episode_latents is not None:
        payload = {
            "hdr_mode": sample.get("hdr_mode", "episode_first_hdr"),
            "local_latents": latents.squeeze(0).contiguous(),
            "episode_latents": episode_latents.squeeze(0).contiguous(),
        }
    else:
        first_frame_latents = latents[:, :, 0:1].contiguous()
        payload = {
            "input_latents": latents.squeeze(0).contiguous(),
            "first_frame_latents": first_frame_latents.squeeze(0).contiguous(),
        }
    payload.update({
        "action": sample["action"].contiguous(),
        "proprio": sample["proprio"].contiguous(),
        "prompt": sample["prompt"],
        "image_is_pad": sample["image_is_pad"].contiguous(),
        "action_is_pad": sample["action_is_pad"].contiguous(),
        "action_dim_is_pad": sample.get("action_dim_is_pad", torch.zeros(sample["action"].shape[-1], dtype=torch.bool)).contiguous(),
        "proprio_is_pad": sample["proprio_is_pad"].contiguous(),
        "num_video_frames": int(sample["video"].shape[1]) if episode_latents is None else int(sample["video"].shape[1] + sample.get("episode_video", sample["video"]).shape[1]),
    })
    for key in (
        "local_video_frames",
        "action_video_transition_count",
        "hdr_tree_frame_indices",
        "hdr_local_frame_indices",
        "hdr_episode_frame_indices",
        "hdr_episode_latent_indices",
        "hdr_episode_latent_source_frame_indices",
        "hdr_episode_latent_padded_frames",
        "hdr_mode",
        "action_adapter",
        "source_name",
        "source_episode_index",
    ):
        if key in sample:
            value = sample[key]
            payload[key] = value.contiguous() if isinstance(value, torch.Tensor) else value
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", default=str(FASTWAM_ROOT))
    parser.add_argument("--task", required=True, help="FastWAM task config name.")
    parser.add_argument("--model", default="fastwam_joint")
    parser.add_argument("--data", required=True, help="FastWAM dataset config name.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--sample-workers", type=int, default=1)
    parser.add_argument("--load-retries", type=int, default=3)
    parser.add_argument("--shard-rank", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-mode", choices=("strided", "contiguous", "episode"), default="strided")
    parser.add_argument("--timing-report", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    manual_sharding = args.shard_rank is not None or args.num_shards is not None
    if manual_sharding:
        if args.shard_rank is None or args.num_shards is None:
            raise ValueError("--shard-rank and --num-shards must be set together")
        if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
            raise ValueError("Expected 0 <= --shard-rank < --num-shards")
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world_size = (int(args.shard_rank), int(args.num_shards)) if manual_sharding else _rank_world()
    device = _device()
    barrier_device_ids = [device.index] if device.type == "cuda" and device.index is not None else None
    dtype = torch.bfloat16
    run_start = time.perf_counter()
    timing = {"load": 0.0, "encode": 0.0, "write": 0.0, "samples": 0.0}
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
            "manual_sharding": manual_sharding,
            "task": args.task,
            "model": args.model,
            "data": args.data,
            "overrides": list(args.overrides),
        }
        torch.save(metadata, out_root / "metadata.pt")
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=barrier_device_ids)

    num_samples = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))
    if args.shard_mode == "episode":
        index_iterable = _episode_shard_indices(dataset, num_samples, rank, world_size)
    elif args.shard_mode == "contiguous":
        start_idx = (rank * num_samples) // world_size
        end_idx = ((rank + 1) * num_samples) // world_size
        index_iterable = range(start_idx, end_idx)
    else:
        index_iterable = range(rank, num_samples, world_size)
    indices = [
        idx for idx in index_iterable
        if args.overwrite or not _cache_path(out_root, idx).exists()
    ]
    iterator = tqdm(indices, desc=f"rank {rank}", disable=rank != 0)

    def load_sample(idx: int):
        path = _cache_path(out_root, idx)
        retries = max(int(args.load_retries), 1)
        for attempt in range(retries):
            try:
                load_start = time.perf_counter()
                sample = dataset._get(idx)
                return idx, path, sample, time.perf_counter() - load_start
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
        timing["load"] += sum(float(item[3]) for item in batch)
        videos = [sample["video"] for _, _, sample, _ in batch]
        encode_start = time.perf_counter()
        latents = _encode_videos(vae, videos, device=device, dtype=dtype)
        episode_latents = None
        if any("episode_video" in sample for _, _, sample, _ in batch):
            episode_videos = [sample["episode_video"] for _, _, sample, _ in batch]
            episode_latents = _encode_videos(vae, episode_videos, device=device, dtype=dtype)
        timing["encode"] += time.perf_counter() - encode_start
        write_start = time.perf_counter()
        for n, (latent, (_, path, sample, _)) in enumerate(zip(latents, batch)):
            path.parent.mkdir(parents=True, exist_ok=True)
            ep_latent = None if episode_latents is None else episode_latents[n].unsqueeze(0)
            payload = _payload_from_sample(sample, latent.unsqueeze(0), episode_latents=ep_latent)
            tmp_path = path.with_suffix(f".tmp.{os.getpid()}")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        timing["write"] += time.perf_counter() - write_start
        timing["samples"] += len(batch)

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

    local_wall = time.perf_counter() - run_start
    if args.timing_report:
        timing_tensor = torch.tensor(
            [timing["samples"], timing["load"], timing["encode"], timing["write"], local_wall],
            dtype=torch.float64,
            device=device,
        )
        sum_tensor = timing_tensor.clone()
        max_tensor = timing_tensor.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        if rank == 0:
            samples, load_s, encode_s, write_s, wall_sum = [float(x) for x in sum_tensor.detach().cpu()]
            _, load_max, encode_max, write_max, wall_max = [float(x) for x in max_tensor.detach().cpu()]
            denom = max(samples, 1.0)
            print(
                "[timing] "
                f"samples={int(samples)} "
                f"wall_max={wall_max:.3f}s "
                f"throughput={samples / max(wall_max, 1e-9):.3f} samples/s "
                f"load_sum={load_s:.3f}s load_per_sample={load_s / denom:.6f}s load_max_rank={load_max:.3f}s "
                f"encode_sum={encode_s:.3f}s encode_per_sample={encode_s / denom:.6f}s encode_max_rank={encode_max:.3f}s "
                f"write_sum={write_s:.3f}s write_per_sample={write_s / denom:.6f}s write_max_rank={write_max:.3f}s"
            )
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=barrier_device_ids)
    if rank == 0:
        if manual_sharding:
            print(f"[precache] shard_done output_dir={out_root} rank={rank}/{world_size} samples={num_samples}")
        else:
            print(f"[precache] done output_dir={out_root} samples={num_samples}")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
