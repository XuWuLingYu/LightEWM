#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = ROOT / "lightewm" / "vendor" / "fastwam"
for p in (ROOT, FASTWAM_ROOT, ROOT / "data" / "python-packages" / "fastwam_pydeps", ROOT / "third_parties" / "LIBERO"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers

_PRECACHE = ROOT / "scripts" / "precache_fastwamhdr_latents.py"
import importlib.util
_spec = importlib.util.spec_from_file_location("precache_fastwamhdr_latents", _PRECACHE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_cache_path = _mod._cache_path
_encode_videos = _mod._encode_videos
_load_vae = _mod._load_vae
_payload_from_sample = _mod._payload_from_sample

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

register_default_resolvers()

CAMERAS_LEROBOT = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
CAMERAS_DROID = ["observations.images.cam_anchor_exterior_1", "observations.images.cam_wrist_left", "observations.images.cam_anchor_exterior_2"]


def rank_world():
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def quat_to_rpy_xyzw(q):
    x, y, z, w = [float(v) for v in q]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def pose16_to_ee14(v):
    v = [float(x) for x in v]
    left = v[0:3] + quat_to_rpy_xyzw(v[3:7]) + [v[7]]
    right = v[8:11] + quat_to_rpy_xyzw(v[11:15]) + [v[15]]
    return left + right


def to_tensor_rows(values, dim):
    arr = np.asarray([np.asarray(x, dtype=np.float32) for x in values], dtype=np.float32)
    if arr.shape[-1] != dim:
        raise ValueError(f"Expected dim {dim}, got {arr.shape}")
    return torch.from_numpy(arr)


def zscore(x: torch.Tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=torch.float32)
    std = torch.as_tensor(std, dtype=torch.float32).clamp(min=1.0e-6)
    return (x.float() - mean) / std


def mean_std_from_quantiles(stats: dict[str, Any], dim: int):
    mean = stats["mean"][:dim]
    if "std" in stats:
        std = stats["std"][:dim]
    elif "q01" in stats and "q99" in stats:
        std = [(float(hi) - float(lo)) / 4.652 for lo, hi in zip(stats["q01"][:dim], stats["q99"][:dim])]
    else:
        std = [1.0] * dim
    return mean, [max(float(x), 1.0e-6) for x in std]


def stats_from_fastwam(path: Path, key: str, dim: int):
    stats = json.load(open(path, "r"))
    item = stats[key]["default"]
    mean = item["global_mean"][:dim]
    std = item["global_std"][:dim]
    return mean, std


def stats_from_robotwin(path: Path):
    stats = json.load(open(path, "r"))
    absolute = stats["absolute"]
    return mean_std_from_quantiles(absolute, 14)


def stats_from_droid(path: Path, feature: str, dim: int):
    stats = json.load(open(path, "r"))[feature]
    return mean_std_from_quantiles(stats, dim)


def concat_robotwin_layout(camera_frames: list[torch.Tensor]) -> torch.Tensor:
    # camera_frames: each [T,C,H,W] float [0,1]
    top = TF.resize(camera_frames[0], [256, 320], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    left = TF.resize(camera_frames[1], [128, 160], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    right = TF.resize(camera_frames[2], [128, 160], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    video = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
    return (video * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous()


def decode_camera_frames(video_path: Path, timestamps: list[float], tolerance_s: float) -> torch.Tensor:
    frames = decode_video_frames(video_path, timestamps, tolerance_s, backend=None)
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    # decode_video_frames returns [T,C,H,W] float [0,1]
    return frames.to(torch.float32).contiguous()


def episode_tree_indices(local_end: int, episode_len: int, n: int = 4):
    first_candidate = local_end + 1
    if episode_len <= first_candidate:
        return [int(local_end)] * n
    tail = episode_len - 1 - local_end
    return [int(np.clip(math.ceil(local_end + tail * (i + 1) / n), first_candidate, episode_len - 1)) for i in range(n)]


def episode_first_indices(episode_len: int, n: int = 9):
    return np.rint(np.linspace(0, episode_len - 1, n)).astype(np.int64).clip(0, episode_len - 1).tolist()


class Source:
    def __init__(self, name: str, kind: str, root: Path, adapter: str, action_dim: int, stats: dict[str, Any], max_episodes: int | None):
        self.name = name
        self.kind = kind
        self.root = root
        self.adapter = adapter
        self.action_dim = action_dim
        self.stats = stats
        self.max_episodes = max_episodes
        if kind in {"robodojo", "robotwin"}:
            self.episodes = [
                ep for ep in load_jsonl(root / "meta" / "episodes.jsonl")
                if self._lerobot_episode_files_exist(ep)
            ]
            if max_episodes is not None and max_episodes > 0:
                self.episodes = self.episodes[:max_episodes]
            self.tasks = {r["task_index"]: r["task"] for r in load_jsonl(root / "meta" / "tasks.jsonl")}
            info = json.load(open(root / "meta" / "info.json"))
            self.fps = float(info["fps"])
            self.tolerance_s = 1.0 / self.fps / 2.0
        elif kind == "droid":
            df = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file_000.parquet")
            if max_episodes is not None and max_episodes > 0:
                df = df.iloc[:max_episodes]
            self.episodes = df.to_dict("records")
            self.fps = 15.0
            self.tolerance_s = 1.0 / self.fps / 2.0
        else:
            raise ValueError(f"unknown source kind {kind}")

    def _lerobot_episode_files_exist(self, ep) -> bool:
        ep_idx = int(ep["episode_index"])
        data_path = self.root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not data_path.exists():
            return False
        return all(
            (self.root / "videos" / "chunk-000" / camera / f"episode_{ep_idx:06d}.mp4").exists()
            for camera in CAMERAS_LEROBOT
        )

    def episode_len(self, ep):
        return int(ep["length"])

    def prompt(self, ep, rows=None):
        if self.kind == "droid":
            task = str(ep.get("task") or (ep.get("tasks") or [""])[0])
        else:
            task = str((ep.get("tasks") or [""])[0])
        return DEFAULT_PROMPT.format(task=task)

    def video_path(self, camera_key: str, ep, droid_timestamp_key: str | None = None) -> tuple[Path, float]:
        if self.kind in {"robodojo", "robotwin"}:
            ep_idx = int(ep["episode_index"])
            return self.root / "videos" / "chunk-000" / camera_key / f"episode_{ep_idx:06d}.mp4", 0.0
        chunk = int(ep[f"videos/{camera_key}/chunk_index"])
        file_idx = int(ep[f"videos/{camera_key}/file_index"])
        start_ts = float(ep[f"videos/{camera_key}/from_timestamp"])
        return self.root / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file_{file_idx:03d}.mp4", start_ts

    def load_episode_rows(self, ep):
        if self.kind in {"robodojo", "robotwin"}:
            ep_idx = int(ep["episode_index"])
            return pd.read_parquet(self.root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet").to_dict("records")
        data_chunk = int(ep["data/chunk_index"])
        data_file = int(ep["data/file_index"])
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        # DROID converted shards are chunk-local and contiguous. For chunk-000, global index equals file-local offset within file ranges.
        df = pd.read_parquet(self.root / "data" / f"chunk-{data_chunk:03d}" / f"file_{data_file:03d}.parquet")
        local = df[df["episode_index"] == int(ep["episode_index"])]
        if len(local) != end - start:
            local = df.iloc[start:end]
        return local.to_dict("records")

    def action_state(self, rows, start: int, action_horizon: int):
        n = len(rows)
        act_idx = [min(n - 1, start + t) for t in range(action_horizon)]
        pad = torch.tensor([start + t >= n for t in range(action_horizon)], dtype=torch.bool)
        if self.kind == "robotwin":
            actions = torch.stack([torch.tensor(pose16_to_ee14(rows[i]["action"]), dtype=torch.float32) for i in act_idx])
            states = torch.stack([torch.tensor(pose16_to_ee14(rows[i]["observation.state"]), dtype=torch.float32) for i in act_idx])
        elif self.kind == "robodojo":
            actions = to_tensor_rows([rows[i]["action"] for i in act_idx], 14)
            states = to_tensor_rows([rows[i]["observation.state"] for i in act_idx], 14)
        else:
            actions = to_tensor_rows([rows[i]["actions.raw_action"] for i in act_idx], 8)
            states = to_tensor_rows([rows[i]["observations.eef_state_14"] for i in act_idx], 14)
        actions = zscore(actions, self.stats["action_mean"], self.stats["action_std"])
        states = zscore(states, self.stats["state_mean"], self.stats["state_std"])
        if self.action_dim < 14:
            # Mixed batches need a rectangular action tensor. The adapter still
            # consumes only the first `action_dim` entries; padded dims are
            # masked from loss and metrics.
            padded = torch.zeros((actions.shape[0], 14), dtype=actions.dtype)
            padded[:, : self.action_dim] = actions
            actions = padded
            dim_pad = torch.zeros(14, dtype=torch.bool)
            dim_pad[self.action_dim :] = True
        else:
            dim_pad = torch.zeros(14, dtype=torch.bool)
        return actions.contiguous(), states.contiguous(), pad, dim_pad

    def decode_stitched(self, ep, frame_indices: list[int], droid_offsets: list[float] | None = None):
        if self.kind == "droid":
            cameras = CAMERAS_DROID
            timestamps = [float(i) / self.fps for i in frame_indices]
        else:
            cameras = CAMERAS_LEROBOT
            timestamps = [float(i) / self.fps for i in frame_indices]
        frames = []
        for cam in cameras:
            path, offset = self.video_path(cam, ep)
            ts = [offset + t for t in timestamps] if self.kind == "droid" else timestamps
            frames.append(decode_camera_frames(path, ts, self.tolerance_s))
        return concat_robotwin_layout(frames)


def build_sources(args):
    robotwin_mean, robotwin_std = stats_from_robotwin(Path(args.robotwin_stats))
    rd_action_mean, rd_action_std = stats_from_fastwam(Path(args.robodojo_stats), "action", 14)
    rd_state_mean, rd_state_std = stats_from_fastwam(Path(args.robodojo_stats), "state", 14)
    d_action_mean, d_action_std = stats_from_droid(Path(args.droid_stats), "actions.raw_action", 8)
    d_state_mean, d_state_std = stats_from_droid(Path(args.droid_stats), "observations.eef_state_14", 14)
    robotwin_dirs = sorted([p for p in Path(args.robotwin_root).iterdir() if (p / "meta" / "info.json").exists()])
    if args.robotwin_task_limit:
        robotwin_dirs = robotwin_dirs[: args.robotwin_task_limit]
    sources = [
        Source("robodojo", "robodojo", Path(args.robodojo_root), "ee14", 14, {"action_mean": rd_action_mean, "action_std": rd_action_std, "state_mean": rd_state_mean, "state_std": rd_state_std}, args.robodojo_max_episodes),
        Source("droid", "droid", Path(args.droid_root), "joint8_delta_gripper_abs", 8, {"action_mean": d_action_mean, "action_std": d_action_std, "state_mean": d_state_mean, "state_std": d_state_std}, args.droid_max_episodes),
    ]
    for i, root in enumerate(robotwin_dirs):
        sources.append(Source(f"robotwin_{i:03d}", "robotwin", root, "ee14", 14, {"action_mean": robotwin_mean, "action_std": robotwin_std, "state_mean": robotwin_mean, "state_std": robotwin_std}, args.robotwin_episodes_per_task))
    return sources


def build_manifest(sources, args):
    rng = random.Random(args.seed)
    episodes = []
    for source_id, source in enumerate(sources):
        eps = list(source.episodes)
        rng.shuffle(eps)
        heldout_n = min(args.heldout_per_source, max(1, len(eps) // 10), len(eps))
        for split, split_eps in (("val", eps[:heldout_n]), ("train", eps[heldout_n:])):
            for ep in split_eps:
                length = source.episode_len(ep)
                starts = list(range(0, max(length - args.action_horizon, 1), args.sample_stride))
                if split == "val":
                    starts = starts[: args.val_starts_per_episode]
                elif args.train_starts_per_episode > 0:
                    starts = starts[: args.train_starts_per_episode]
                episodes.append({"source_id": source_id, "split": split, "episode": ep, "starts": starts})
    # assign sample ids deterministically
    sample_id = 0
    train_indices, val_indices = [], []
    for item in episodes:
        item["sample_ids"] = []
        for _ in item["starts"]:
            item["sample_ids"].append(sample_id)
            (val_indices if item["split"] == "val" else train_indices).append(sample_id)
            sample_id += 1
    return episodes, train_indices, val_indices, sample_id


def write_prompts(sources, episodes, out_root: Path):
    prompt_dir = out_root / "prompt_dataset" / "meta"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    with (prompt_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        idx = 0
        for item in episodes:
            source = sources[item["source_id"]]
            prompt = source.prompt(item["episode"])
            task = prompt.replace("A video recorded from a robot's point of view executing the following instruction: ", "")
            if task in seen:
                continue
            seen.add(task)
            f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")
            idx += 1
    return prompt_dir.parent


def process_episode(source: Source, item, vae, dev, dtype, out_root: Path, store_video: bool):
    ep = item["episode"]
    rows = source.load_episode_rows(ep)
    n = len(rows)
    sample_pairs = list(zip(item["sample_ids"], item["starts"]))
    if not sample_pairs:
        return 0
    # Build all unique frame requests for this episode once.
    frame_set = set()
    sample_meta = {}
    for sample_id, start in sample_pairs:
        local = [min(n - 1, start + i * args_global.action_video_freq_ratio) for i in range(args_global.local_rgb_frames)]
        local_end = local[-1]
        tree = episode_tree_indices(local_end, n, args_global.tree_rgb_frames)
        ep_first = episode_first_indices(n, args_global.local_rgb_frames)
        for idx in local + tree + ep_first:
            frame_set.add(int(idx))
        sample_meta[sample_id] = (start, local, tree, ep_first)
    frame_list = sorted(frame_set)
    # Decode separately for local+tree and episode-first by slicing from the same decoded frame map.
    decoded = source.decode_stitched(ep, frame_list)
    frame_to_pos = {f: i for i, f in enumerate(frame_list)}
    wrote = 0
    for sample_id, start in sample_pairs:
        _, local, tree, ep_first = sample_meta[sample_id]
        video = decoded[:, [frame_to_pos[i] for i in (local + tree)]].contiguous()
        episode_video = decoded[:, [frame_to_pos[i] for i in ep_first]].contiguous()
        action, proprio, action_pad, dim_pad = source.action_state(rows, start, args_global.action_horizon)
        image_pad = torch.zeros(video.shape[1], dtype=torch.bool)
        sample = {
            "video": video,
            "episode_video": episode_video,
            "hdr_mode": "episode_back_mixed",
            "action": action,
            "action_is_pad": action_pad,
            "action_dim_is_pad": dim_pad,
            "proprio": proprio,
            "proprio_is_pad": action_pad.clone(),
            "image_is_pad": image_pad,
            "prompt": source.prompt(ep, rows),
            "local_video_frames": torch.tensor(args_global.local_rgb_frames, dtype=torch.long),
            "action_video_transition_count": torch.tensor(args_global.local_rgb_frames - 1, dtype=torch.long),
            "hdr_local_frame_indices": torch.tensor(local, dtype=torch.long),
            "hdr_tree_frame_indices": torch.tensor(tree, dtype=torch.long),
            "hdr_episode_frame_indices": torch.tensor(ep_first, dtype=torch.long),
            "action_adapter": source.adapter,
            "source_name": source.name,
            "source_episode_index": int(ep.get("episode_index", ep.get("source_episode_index", -1))),
        }
        local_latents = _encode_videos(vae, [video], device=dev, dtype=dtype)[0]
        episode_latents = _encode_videos(vae, [episode_video], device=dev, dtype=dtype)[0]
        payload = _payload_from_sample(sample, local_latents.unsqueeze(0), episode_latents=episode_latents.unsqueeze(0))
        payload["action_dim_is_pad"] = dim_pad
        payload["action_adapter"] = source.adapter
        payload["source_name"] = source.name
        payload["source_episode_index"] = sample["source_episode_index"]
        if store_video:
            payload["video"] = video.to(torch.float16).cpu()
            payload["episode_video"] = episode_video.to(torch.float16).cpu()
        path = _cache_path(out_root, sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        wrote += 1
    return wrote


def record_bad_episode(out_root: Path, rank: int, source: Source, item, exc: BaseException):
    bad_dir = out_root / "bad_episodes"
    bad_dir.mkdir(parents=True, exist_ok=True)
    ep = item["episode"]
    payload = {
        "rank": rank,
        "source_name": source.name,
        "source_kind": source.kind,
        "source_root": str(source.root),
        "episode_index": int(ep.get("episode_index", ep.get("source_episode_index", -1))),
        "split": item["split"],
        "sample_ids": [int(x) for x in item["sample_ids"]],
        "starts": [int(x) for x in item["starts"]],
        "error_type": type(exc).__name__,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    with (bad_dir / f"rank_{rank:02d}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def finalize_successful_indices(out_root: Path, train_indices: list[int], val_indices: list[int], num_samples: int):
    present = sorted(
        int(p.stem)
        for p in out_root.glob("shard_*/*.pt")
        if p.is_file() and p.stat().st_size > 0
    )
    present_set = set(present)
    train_present = [int(i) for i in train_indices if int(i) in present_set]
    val_present = [int(i) for i in val_indices if int(i) in present_set]
    missing = sorted(set(range(int(num_samples))) - present_set)
    torch.save({"indices": train_present, "train_indices": train_present}, out_root / "train_indices.pt")
    torch.save({"indices": val_present, "val_indices": val_present}, out_root / "val_indices.pt")
    meta_path = out_root / "metadata.pt"
    meta = torch.load(meta_path, map_location="cpu", weights_only=False) if meta_path.exists() else {}
    meta["original_num_samples"] = int(num_samples)
    meta["num_samples"] = len(present)
    meta["train_samples"] = len(train_present)
    meta["val_samples"] = len(val_present)
    meta["missing_samples"] = len(missing)
    meta["missing_sample_ids_preview"] = missing[:1000]
    bad_dir = out_root / "bad_episodes"
    meta["bad_episode_logs"] = sorted(str(p) for p in bad_dir.glob("rank_*.jsonl")) if bad_dir.exists() else []
    torch.save(meta, meta_path)
    return len(present), len(train_present), len(val_present), len(missing)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--robodojo-root", required=True, help="Path to the local RoboDojo LeRobot dataset root.")
    ap.add_argument("--robodojo-stats", required=True, help="Path to the local RoboDojo action stats JSON.")
    ap.add_argument("--robotwin-root", required=True, help="Path to the local RoboTwin dataset root (kept outside git).")
    ap.add_argument("--robotwin-stats", required=True, help="Path to the local RoboTwin action stats JSON (kept outside git).")
    ap.add_argument("--droid-root", required=True, help="Path to the local processed DROID dataset root (kept outside git).")
    ap.add_argument("--droid-stats", required=True, help="Path to the local processed DROID stats JSON (kept outside git).")
    ap.add_argument("--robotwin-task-limit", type=int, default=8)
    ap.add_argument("--robotwin-episodes-per-task", type=int, default=8)
    ap.add_argument("--robodojo-max-episodes", type=int, default=80)
    ap.add_argument("--droid-max-episodes", type=int, default=80)
    ap.add_argument("--heldout-per-source", type=int, default=6)
    ap.add_argument("--sample-stride", type=int, default=20)
    ap.add_argument("--train-starts-per-episode", type=int, default=8)
    ap.add_argument("--val-starts-per-episode", type=int, default=1)
    ap.add_argument("--local-rgb-frames", type=int, default=9)
    ap.add_argument("--tree-rgb-frames", type=int, default=4)
    ap.add_argument("--action-video-freq-ratio", type=int, default=4)
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume-existing", action="store_true")
    ap.add_argument("--store-val-video", action="store_true")
    ap.add_argument("--task", required=True, help="FastWAM task config name.")
    ap.add_argument("--model", default="fastwam_joint")
    ap.add_argument("--data", required=True, help="FastWAM dataset config name.")
    return ap.parse_args()


args_global = None


def main():
    global args_global
    args = parse_args()
    args_global = args
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", timeout=_dt.timedelta(hours=12))
    rank, world = rank_world()
    dev = device()
    barrier_device_ids = [dev.index] if dev.type == "cuda" and dev.index is not None else None
    dtype = torch.bfloat16
    out_root = Path(args.output_dir).resolve()
    if rank == 0:
        out_root.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=barrier_device_ids)
    misc.register_work_dir(str(out_root))

    sources = build_sources(args)
    episodes, train_indices, val_indices, num_samples = build_manifest(sources, args)
    if rank == 0:
        prompt_dataset = write_prompts(sources, episodes, out_root)
        torch.save({"indices": train_indices, "train_indices": train_indices}, out_root / "train_indices.pt")
        torch.save({"indices": val_indices, "val_indices": val_indices}, out_root / "val_indices.pt")
        torch.save({
            "num_samples": num_samples,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "sources": [{"name": s.name, "kind": s.kind, "root": str(s.root), "adapter": s.adapter, "action_dim": s.action_dim, "episodes": len(s.episodes)} for s in sources],
            "prompt_dataset": str(prompt_dataset),
            "args": vars(args),
        }, out_root / "metadata.pt")
        print(json.dumps({"event": "manifest", "num_samples": num_samples, "train": len(train_indices), "val": len(val_indices), "prompt_dataset": str(prompt_dataset)}, ensure_ascii=False), flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=barrier_device_ids)

    fastwam_root = FASTWAM_ROOT.resolve()
    with initialize_config_dir(config_dir=str(fastwam_root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}", f"model={args.model}", f"data={args.data}"])
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    vae = _load_vae(cfg, device=dev, dtype=dtype)

    selected = episodes[rank::world]
    start_time = time.perf_counter()
    wrote = 0
    for item in tqdm(selected, desc=f"rank {rank}/{world} episodes", disable=rank != 0):
        source = sources[item["source_id"]]
        store_video = args.store_val_video and item["split"] == "val"
        if args.resume_existing and all(_cache_path(out_root, sid).exists() for sid in item["sample_ids"]):
            continue
        if not args.overwrite and all(_cache_path(out_root, sid).exists() for sid in item["sample_ids"]):
            continue
        try:
            wrote += process_episode(source, item, vae, dev, dtype, out_root, store_video=store_video)
        except Exception as exc:
            record_bad_episode(out_root, rank, source, item, exc)
            print(
                json.dumps(
                    {
                        "event": "bad_episode_skipped",
                        "rank": rank,
                        "source": source.name,
                        "kind": source.kind,
                        "episode_index": int(item["episode"].get("episode_index", item["episode"].get("source_episode_index", -1))),
                        "sample_count": len(item["sample_ids"]),
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    elapsed = time.perf_counter() - start_time
    local = torch.tensor([wrote, elapsed], device=dev if dev.type == "cuda" else torch.device("cpu"), dtype=torch.float64)
    total = local.clone()
    maxv = local.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxv, op=dist.ReduceOp.MAX)
        dist.barrier(device_ids=barrier_device_ids)
    if rank == 0:
        samples = int(total[0].item())
        wall = float(maxv[1].item())
        available, train_available, val_available, missing = finalize_successful_indices(out_root, train_indices, val_indices, num_samples)
        print(json.dumps({"event": "precache_done", "wrote": samples, "available": available, "train": train_available, "val": val_available, "missing": missing, "wall_seconds": wall, "samples_per_second": samples / max(wall, 1e-9), "output_dir": str(out_root)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
