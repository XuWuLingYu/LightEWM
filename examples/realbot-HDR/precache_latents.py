#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp


def load_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def sample_frame_ids(start: int, stop_inclusive: int, target_frames: int) -> list[int]:
    if stop_inclusive < start:
        raise ValueError(f'Invalid clip range start={start} stop={stop_inclusive}')
    return np.linspace(start, stop_inclusive, target_frames).round().astype(np.int64).tolist()

def read_rgb_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video {path}')
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f'Video contains no decodable frames: {path}')
    return np.stack(frames, axis=0)

def make_groups(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        key = row.get('full_video') or row.get('video')
        if not key:
            raise KeyError(f'Row has neither full_video nor video: {row}')
        groups.setdefault(key, []).append(row)
    return list(groups.items())

def worker(
    rank: int,
    world_size: int,
    rows: list[dict],
    base_path: str,
    cache_dir: str,
    model_root: str,
    force: bool,
    shard_rank: int | None = None,
    shard_world_size: int | None = None,
    within_shard_rank: int | None = None,
    within_shard_world_size: int | None = None,
    encode_batch_size: int = 1,
):
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from utils.dataset import _load_video_as_tensor, _rgb_frames_to_tensor
    from utils.wan_wrapper import WanVAEWrapper

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
    vae = WanVAEWrapper(model_name='Wan2.2-TI2V-5B', model_root=model_root).to(device).eval()
    cache_root = Path(cache_dir)
    base = Path(base_path)
    done = 0
    encode_batch_size = max(1, int(encode_batch_size))

    def flush_batch(batch_items: list[tuple[torch.Tensor, dict, Path, Path, int, int, int]]):
        nonlocal done
        if not batch_items:
            return
        frames_batch = torch.stack([item[0] for item in batch_items], dim=0)
        with torch.no_grad():
            latents = vae.encode_to_latent(frames_batch.to(device, dtype=torch.float32)).detach().cpu().to(torch.bfloat16)
        if latents.ndim != 5 or tuple(latents.shape[1:]) != (16, 48, 24, 20):
            raise RuntimeError(f'Expected latent batch shape (B,16,48,24,20), got {tuple(latents.shape)}')
        for latent, (_, row, out, video_path, group_idx0, local_idx0, group_len) in zip(latents, batch_items):
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + f'.tmp.{os.getpid()}')
            torch.save({
                'clean_latent': latent,
                'video': row['video'],
                'full_video': row.get('full_video', ''),
                'clip_start': row.get('clip_start', ''),
                'prompt': row['prompt'],
            }, tmp)
            tmp.replace(out)
            done += 1
            print(
                f'[rank {rank}] cached group {group_idx0+1}/{len(groups)} '
                f'row {local_idx0+1}/{group_len} {video_path} start={row.get("clip_start", "")} -> {out}',
                flush=True,
            )

    grouped = any(row.get('full_video') for row in rows)
    groups = make_groups(rows) if grouped else [(row.get('video', ''), [row]) for row in rows]
    for group_idx, (group_video, group_rows) in enumerate(groups):
        if shard_world_size is not None and shard_rank is not None:
            if group_idx % shard_world_size != shard_rank:
                continue
            if within_shard_world_size is not None and within_shard_rank is not None:
                shard_idx = group_idx // shard_world_size
                if shard_idx % within_shard_world_size != within_shard_rank:
                    continue
        elif group_idx % world_size != rank:
            continue
        pending = [row for row in group_rows if force or not (cache_root / row['preencoded_cache_path']).exists()]
        if not pending:
            done += len(group_rows)
            continue
        if grouped:
            video = base / group_video
            decoded = read_rgb_video(video)
        batch_items: list[tuple[torch.Tensor, dict, Path, Path, int, int, int]] = []
        for local_idx, row in enumerate(group_rows):
            out = cache_root / row['preencoded_cache_path']
            if out.exists() and not force:
                done += 1
                continue
            if grouped:
                start = int(row.get('clip_start', 0) or 0)
                stop = int(row.get('clip_end') or (len(decoded) - 1))
                stop = min(stop, len(decoded) - 1)
                frame_ids = sample_frame_ids(start, stop, 61)
                clip_frames = decoded[frame_ids]
                frames = _rgb_frames_to_tensor(clip_frames, height=384, width=320)
            else:
                video = base / row['video']
                frames = _load_video_as_tensor(video_path=video, num_frames=61, height=384, width=320)
            if frames.shape != (3, 61, 384, 320):
                raise RuntimeError(f'Unexpected frames shape for {video}: {tuple(frames.shape)}')
            batch_items.append((frames, row, out, video, group_idx, local_idx, len(group_rows)))
            if len(batch_items) >= encode_batch_size:
                flush_batch(batch_items)
                batch_items = []
        flush_batch(batch_items)
    print(f'[rank {rank}] done count={done}', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True, help='Dataset directory containing the input metadata and videos.')
    ap.add_argument('--input-metadata', default='metadata_train.csv')
    ap.add_argument('--output-metadata', default='metadata_train_preencoded.csv')
    ap.add_argument('--cache-subdir', default='latent_cache')
    ap.add_argument('--model-root', required=True, help='Directory containing the Wan model artifacts.')
    ap.add_argument('--num-workers', type=int, default=None)
    ap.add_argument('--shard-rank', type=int, default=None)
    ap.add_argument('--shard-world-size', type=int, default=None)
    ap.add_argument('--within-shard-rank', type=int, default=None)
    ap.add_argument('--within-shard-world-size', type=int, default=None)
    ap.add_argument('--encode-batch-size', type=int, default=int(os.environ.get('ENCODE_BATCH_SIZE', '1')))
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    rows = load_rows(dataset_root / args.input_metadata)
    enriched = []
    for row in rows:
        source = row.get('source', 'unknown')
        split = row.get('split', 'train')
        ep = row.get('episode_id') or Path(row['video']).stem
        rel = Path(args.cache_subdir) / source / split / f'{ep}.pt'
        row = dict(row)
        row['preencoded_cache_path'] = str(rel)
        row['latent_frames'] = '16'
        row['latent_shape'] = '16x48x24x20'
        enriched.append(row)
    write_rows(dataset_root / args.output_metadata, enriched)

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if args.num_workers is not None:
        world_size = args.num_workers
    elif torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
    world_size = max(1, min(world_size, len(enriched))) if enriched else 1
    cache_dir = str(dataset_root)
    if (args.shard_rank is None) != (args.shard_world_size is None):
        raise ValueError('--shard-rank and --shard-world-size must be provided together')
    if args.shard_rank is not None:
        if not (0 <= args.shard_rank < args.shard_world_size):
            raise ValueError(f'Invalid shard {args.shard_rank}/{args.shard_world_size}')
        world_size = 1
    if (args.within_shard_rank is None) != (args.within_shard_world_size is None):
        raise ValueError('--within-shard-rank and --within-shard-world-size must be provided together')
    if args.within_shard_rank is not None:
        if args.shard_rank is None:
            raise ValueError('--within-shard-* requires --shard-*')
        if not (0 <= args.within_shard_rank < args.within_shard_world_size):
            raise ValueError(f'Invalid within shard {args.within_shard_rank}/{args.within_shard_world_size}')

    print(json.dumps({
        'rows': len(enriched),
        'groups': len(make_groups(enriched)) if enriched else 0,
        'world_size': world_size,
        'visible': visible,
        'dataset_root': str(dataset_root),
        'shard_rank': args.shard_rank,
        'shard_world_size': args.shard_world_size,
        'within_shard_rank': args.within_shard_rank,
        'within_shard_world_size': args.within_shard_world_size,
        'encode_batch_size': args.encode_batch_size,
    }, indent=2), flush=True)
    t0 = time.time()
    if world_size == 1:
        worker(
            0, 1, enriched, str(dataset_root), cache_dir, args.model_root,
            args.force, args.shard_rank, args.shard_world_size,
            args.within_shard_rank, args.within_shard_world_size,
            args.encode_batch_size,
        )
    else:
        mp.spawn(worker, args=(world_size, enriched, str(dataset_root), cache_dir, args.model_root, args.force, None, None, None, None, args.encode_batch_size), nprocs=world_size, join=True)
    print(f'[precache] done elapsed_sec={time.time()-t0:.1f}', flush=True)

if __name__ == '__main__':
    main()
