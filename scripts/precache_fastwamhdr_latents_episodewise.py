import argparse
import datetime
import importlib.util
import gc
import os
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as transforms_F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = ROOT / "lightewm" / "vendor" / "fastwam"
for path in (
    ROOT,
    FASTWAM_ROOT,
    ROOT / "data" / "python-packages" / "fastwam_pydeps",
    ROOT / "third_parties" / "LIBERO",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
_PRECACHE_SPEC = importlib.util.spec_from_file_location(
    "precache_fastwamhdr_latents", ROOT / "scripts" / "precache_fastwamhdr_latents.py"
)
_PRECACHE = importlib.util.module_from_spec(_PRECACHE_SPEC)
_PRECACHE_SPEC.loader.exec_module(_PRECACHE)
_cache_path = _PRECACHE._cache_path
_device = _PRECACHE._device
_encode_videos = _PRECACHE._encode_videos
_load_vae = _PRECACHE._load_vae
_payload_from_sample = _PRECACHE._payload_from_sample

register_default_resolvers()


def _rank_world():
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def _decode_full_video_uint8(video_path: Path) -> torch.Tensor:
    frames = []
    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_rgb().to_ndarray())
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).contiguous()


def _apply_image_transforms(frames: torch.Tensor, transforms):
    out = frames
    for trans in transforms:
        out = trans(out)
    return out


def _episode_specs(robot_dataset):
    base = robot_dataset.lerobot_dataset
    specs = []
    global_offset = 0
    for dataset_index, dataset in enumerate(base.multi_dataset._datasets):
        starts = dataset.episode_data_index["from"].detach().cpu().tolist()
        ends = dataset.episode_data_index["to"].detach().cpu().tolist()
        for local_ep_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
            episode_index = dataset.episodes[local_ep_idx] if dataset.episodes is not None else local_ep_idx
            start = int(start)
            end = int(end)
            specs.append(
                {
                    "dataset_index": dataset_index,
                    "dataset": dataset,
                    "local_ep_idx": local_ep_idx,
                    "episode_index": int(episode_index),
                    "global_start": global_offset + start,
                    "global_end": global_offset + end,
                    "length": end - start,
                }
            )
        global_offset += int(dataset.num_frames)
    return specs


def _shard_specs(specs, rank: int, world_size: int, max_samples: int | None):
    dataset_total = sum(int(s["length"]) for s in specs)
    total = dataset_total if max_samples is None else min(dataset_total, int(max_samples))
    target_start = (rank * total) // world_size
    target_end = ((rank + 1) * total) // world_size
    selected = []
    for spec in specs:
        episode_start = int(spec["global_start"])
        episode_end = int(spec["global_end"])
        capped_end = episode_end if max_samples is None else min(episode_end, int(max_samples))
        process_start = max(episode_start, target_start)
        process_end = min(capped_end, target_end)
        if process_start < process_end:
            item = dict(spec)
            item["process_start"] = process_start
            item["process_end"] = process_end
            item["process_length"] = process_end - process_start
            selected.append(item)
    return selected, total

def _get_task(dataset, task_index):
    task_idx = int(torch.as_tensor(task_index).item())
    return dataset.meta.tasks[task_idx]


def _build_action_state_sample(robot_dataset, episode_raw, spec, global_idx: int):
    base = robot_dataset.lerobot_dataset
    processor = base.processor
    rel = int(global_idx - spec["global_start"])
    episode_len = int(spec["global_end"] - spec["global_start"])
    num_frames = int(robot_dataset.num_frames)
    action_size = int(base.action_size)
    stride = int(base.global_sample_stride)

    obs_rel = [max(0, min(episode_len - 1, rel + t * stride)) for t in range(num_frames)]
    act_rel = [max(0, min(episode_len - 1, rel + t * stride)) for t in range(action_size)]
    obs_pad = torch.BoolTensor([(rel + t * stride < 0) or (rel + t * stride >= episode_len) for t in range(num_frames)])
    act_pad = torch.BoolTensor([(rel + t * stride < 0) or (rel + t * stride >= episode_len) for t in range(action_size)])

    action = {}
    for meta in base.action_meta:
        action[meta["key"]] = episode_raw[meta["lerobot_key"]][act_rel].float()
    state = {}
    for meta in base.state_meta:
        state[meta["key"]] = episode_raw[meta["lerobot_key"]][obs_rel].float()

    task = _get_task(spec["dataset"], episode_raw["task_index"][rel])
    data = {
        "idx": global_idx,
        "task": task,
        "action": action,
        "state": state,
        "action_is_pad": act_pad,
        "state_is_pad": obs_pad,
        "image_is_pad": obs_pad,
        "dataset_index": torch.tensor(spec["dataset_index"]),
        "episode_index": torch.tensor(spec["episode_index"]),
        "frame_index": torch.tensor(rel),
        "timestamp": episode_raw["timestamp"][rel],
    }

    sample = {
        "instruction": processor.augment_instruction(data),
        "image_is_pad": data["image_is_pad"],
        "dataset_index": data["dataset_index"],
        "episode_index": data["episode_index"],
        "frame_index": data["frame_index"],
        "timestamp": data["timestamp"],
        "idx": data["idx"],
    }

    if "action" in data and getattr(processor, "delta_action_dim_mask", None) is not None:
        action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
        if bool(action_is_pad.any().item()):
            for key, dim_mask in processor.delta_action_dim_mask.items():
                cur_action = data["action"][key]
                cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                cur_dim_mask = dim_mask.to(device=cur_action.device)
                pad_delta_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                cur_action[pad_delta_mask] = 0.0

    data = processor.action_state_transform(data)
    data = processor.normalizer.forward(data)
    data = processor.action_state_merger.forward(data)

    sample["action"] = data["action"]
    sample["action_is_pad"] = data["action_is_pad"]
    sample["action_dim_is_pad"] = data["action_dim_is_pad"]
    sample["proprio"] = data["state"]
    sample["proprio_is_pad"] = data["state_is_pad"]
    return sample

def _concat_resize_normalize(robot_dataset, video_tensor: torch.Tensor) -> torch.Tensor:
    if video_tensor.ndim == 4:
        video_tensor = video_tensor.unsqueeze(0)
    num_cameras, t, c, h, w = video_tensor.shape
    if robot_dataset.concat_multi_camera == "robotwin":
        if num_cameras != 3:
            raise ValueError(f"robotwin concat requires 3 cameras, got {num_cameras}")
        cam_top = transforms_F.resize(
            video_tensor[0], size=[256, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        cam_left = transforms_F.resize(
            video_tensor[1], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        cam_right = transforms_F.resize(
            video_tensor[2], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        bottom = torch.cat([cam_left, cam_right], dim=-1)
        video_tensor = torch.cat([cam_top, bottom], dim=-2)
    elif robot_dataset.concat_multi_camera == "realbot_top_wrist_320x384":
        if num_cameras != 2:
            raise ValueError(f"realbot_top_wrist_320x384 concat requires 2 cameras, got {num_cameras}")
        cam_right = transforms_F.resize(
            video_tensor[0], size=[224, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        cam_wrist = transforms_F.resize(
            video_tensor[1], size=[160, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        video_tensor = torch.cat([cam_right, cam_wrist], dim=-2)
    elif robot_dataset.concat_multi_camera == "realbot_top_wrist_front_320x384":
        if num_cameras != 3:
            raise ValueError(
                "realbot_top_wrist_front_320x384 concat requires "
                f"3 cameras, got {num_cameras}"
            )
        cam_right = transforms_F.resize(
            video_tensor[0], size=[224, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        cam_wrist = transforms_F.resize(
            video_tensor[1], size=[160, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        cam_front = transforms_F.resize(
            video_tensor[2], size=[160, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        bottom = torch.cat([cam_wrist, cam_front], dim=-1)
        video_tensor = torch.cat([cam_right, bottom], dim=-2)
    elif robot_dataset.concat_multi_camera == "realbot_top_wrist_320x384_right_center_crop_half":
        if num_cameras != 2:
            raise ValueError(
                "realbot_top_wrist_320x384_right_center_crop_half concat requires "
                f"2 cameras, got {num_cameras}"
            )
        crop_h = max(1, h // 2)
        crop_w = max(1, w // 2)
        cam_right_center = transforms_F.center_crop(video_tensor[0], output_size=[crop_h, crop_w])
        cam_right = transforms_F.resize(
            cam_right_center, size=[224, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        cam_wrist = transforms_F.resize(
            video_tensor[1], size=[160, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        video_tensor = torch.cat([cam_right, cam_wrist], dim=-2)
    elif robot_dataset.concat_multi_camera in {
        "realbot_top_wrist_320x384_right_crop_half_shift_0p1",
        "realbot_top_wrist_320x384_right_crop_half_shift_0p2",
        "realbot_top_wrist_320x384_right_bottom_crop_0p6",
    }:
        if num_cameras != 2:
            raise ValueError(
                f"{robot_dataset.concat_multi_camera} concat requires "
                f"2 cameras, got {num_cameras}"
            )
        if h != 480 or w != 640:
            raise ValueError(
                f"{robot_dataset.concat_multi_camera} expects original right view resolution 480x640 "
                f"before crop/resize, got {h}x{w}"
            )
        if robot_dataset.concat_multi_camera == "realbot_top_wrist_320x384_right_bottom_crop_0p6":
            crop_h = int(round(h * 0.6))
            crop_w = int(round(w * 0.6))
            top = h - crop_h
            left = w - crop_w
        else:
            shift = 0.2 if robot_dataset.concat_multi_camera.endswith("_0p2") else 0.1
            crop_h = h // 2
            crop_w = w // 2
            top = int(round((h - crop_h) / 2 + shift * h))
            left = int(round((w - crop_w) / 2 + shift * w))
            top = min(max(top, 0), h - crop_h)
            left = min(max(left, 0), w - crop_w)
        cam_right_crop = transforms_F.crop(video_tensor[0], top=top, left=left, height=crop_h, width=crop_w)
        cam_right = transforms_F.resize(
            cam_right_crop, size=[224, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        cam_wrist = transforms_F.resize(
            video_tensor[1], size=[160, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        video_tensor = torch.cat([cam_right, cam_wrist], dim=-2)
    elif num_cameras > 1:
        if robot_dataset.concat_multi_camera == "horizontal":
            video_tensor = torch.cat([video_tensor[i] for i in range(num_cameras)], dim=-1)
        elif robot_dataset.concat_multi_camera == "vertical":
            video_tensor = torch.cat([video_tensor[i] for i in range(num_cameras)], dim=-2)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {robot_dataset.concat_multi_camera}")
    else:
        video_tensor = video_tensor.squeeze(0)
    video_tensor = robot_dataset.resize_transform(video_tensor)
    video_tensor = robot_dataset.crop_transform(video_tensor)
    video_tensor = robot_dataset.normalize_transform(video_tensor)
    return video_tensor.permute(1, 0, 2, 3).contiguous()


def _pad_episode_video_for_wan_vae(video: torch.Tensor) -> tuple[torch.Tensor, int]:
    if video.ndim != 4:
        raise ValueError(f"Expected episode video [C,T,H,W], got {tuple(video.shape)}")
    t = int(video.shape[1])
    pad = (1 - t) % 4
    if pad <= 0:
        return video, 0
    tail = video[:, -1:, :, :].expand(-1, pad, -1, -1)
    return torch.cat([video, tail], dim=1).contiguous(), pad


def _select_episode_percent_latents(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if latents.ndim != 4:
        raise ValueError(f"Expected episode latents [C,T,H,W], got {tuple(latents.shape)}")
    t = int(latents.shape[1])
    if t <= 0:
        raise ValueError("Episode latents must have positive temporal length")
    percentages = torch.tensor([0.0, 0.33, 0.66], dtype=torch.float32)
    indices = torch.round(percentages * max(t - 1, 0)).to(dtype=torch.long).clamp_(0, t - 1)
    selected = latents.index_select(1, indices).contiguous()
    return selected, indices


def _build_final_sample(robot_dataset, processed_sample, episode_video, spec):
    local_indices = list(robot_dataset.video_sample_indices)
    local_start = int(torch.as_tensor(processed_sample["frame_index"]).item())
    episode_len = int(spec["global_end"] - spec["global_start"])
    stride = int(robot_dataset.lerobot_dataset.global_sample_stride)
    local_episode_indices = [max(0, min(episode_len - 1, local_start + offset * stride)) for offset in local_indices]
    hdr_enabled = bool(getattr(robot_dataset, "hdr_enabled", False))
    hdr_mode = str(getattr(robot_dataset, "hdr_mode", "back_hdr"))
    episode_first_hdr_modes = {"episode_first_hdr", "episode_back_mixed", "episode_first_special_latent"}
    back_tree_hdr_modes = {"back_hdr", "episode_back_mixed"}
    if not hdr_enabled:
        tree_indices = []
        episode_indices = []
        video = episode_video[:, local_episode_indices, :, :]
    else:
        episode_indices = (
            robot_dataset._get_episode_first_hdr_indices(episode_len)
            if hdr_mode in episode_first_hdr_modes
            else []
        )
        if hdr_mode in back_tree_hdr_modes:
            local_end = min(local_start + (robot_dataset.num_frames - 1) * stride, episode_len - 1)
            tree_indices = robot_dataset._get_hdr_tree_indices(local_end=local_end, episode_end_exclusive=episode_len)
            video = episode_video[:, local_episode_indices + tree_indices, :, :]
        else:
            tree_indices = []
            video = episode_video[:, local_episode_indices, :, :]

    image_is_pad_full = processed_sample["image_is_pad"]
    local_image_is_pad = image_is_pad_full[local_indices]
    if not hdr_enabled:
        image_is_pad = local_image_is_pad
    elif hdr_mode not in {"back_hdr", "episode_back_mixed"}:
        image_is_pad = local_image_is_pad
    else:
        hdr_image_is_pad = torch.zeros(robot_dataset.hdr_tree_rgb_frames, dtype=local_image_is_pad.dtype)
        image_is_pad = torch.cat([local_image_is_pad, hdr_image_is_pad], dim=0)

    action = processed_sample["action"]
    proprio = processed_sample["proprio"][:-1, :]
    action_video_transition_count = len(local_indices) - 1
    if action.shape[0] % action_video_transition_count != 0:
        raise ValueError(
            f"action horizon must be divisible by local video transitions, got {action.shape[0]} and {action_video_transition_count}"
        )
    instruction = robot_dataset.DEFAULT_PROMPT.format(task=processed_sample["instruction"]) if hasattr(robot_dataset, "DEFAULT_PROMPT") else None
    if instruction is None:
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        instruction = DEFAULT_PROMPT.format(task=processed_sample["instruction"])
    sample = {
        "video": video,
        "action": action,
        "proprio": proprio,
        "prompt": instruction,
        "image_is_pad": image_is_pad,
        "action_is_pad": processed_sample["action_is_pad"],
        "proprio_is_pad": processed_sample["proprio_is_pad"],
        "local_video_frames": torch.tensor(len(local_indices), dtype=torch.long),
        "action_video_transition_count": torch.tensor(action_video_transition_count, dtype=torch.long),
        "hdr_local_frame_indices": torch.tensor(local_indices, dtype=torch.long),
    }
    if hdr_enabled:
        sample["hdr_mode"] = hdr_mode
    if hdr_enabled and hdr_mode in {"episode_first_hdr", "episode_back_mixed", "episode_first_special_latent"}:
        sample["episode_video"] = episode_video[:, episode_indices, :, :]
        sample["hdr_episode_frame_indices"] = torch.tensor(episode_indices, dtype=torch.long)
    if hdr_enabled and hdr_mode in {"back_hdr", "episode_back_mixed"}:
        sample["hdr_tree_frame_indices"] = torch.tensor(tree_indices, dtype=torch.long)
    return sample


def _load_episode_video(robot_dataset, spec):
    dataset = spec["dataset"]
    if robot_dataset.concat_multi_camera in {
        "realbot_top_wrist_320x384_right_crop_half_shift_0p1",
        "realbot_top_wrist_320x384_right_crop_half_shift_0p2",
        "realbot_top_wrist_320x384_right_bottom_crop_0p6",
    }:
        if len(robot_dataset.lerobot_dataset.image_meta) != 2:
            raise ValueError(
                f"{robot_dataset.concat_multi_camera} expects 2 cameras, got "
                f"{len(robot_dataset.lerobot_dataset.image_meta)}"
            )
        raw_frames = []
        for meta in robot_dataset.lerobot_dataset.image_meta:
            camera_key = meta["lerobot_key"]
            video_path = Path(dataset.root) / dataset.meta.get_video_file_path(spec["episode_index"], camera_key)
            raw_frames.append(_decode_full_video_uint8(video_path))
        right, wrist = raw_frames
        _, _, h, w = right.shape
        if h != 480 or w != 640:
            raise ValueError(
                f"{robot_dataset.concat_multi_camera} expects original right view resolution 480x640 "
                f"before crop/resize, got {h}x{w}"
            )
        if robot_dataset.concat_multi_camera == "realbot_top_wrist_320x384_right_bottom_crop_0p6":
            crop_h = int(round(h * 0.6))
            crop_w = int(round(w * 0.6))
            top = h - crop_h
            left = w - crop_w
        else:
            shift = 0.2 if robot_dataset.concat_multi_camera.endswith("_0p2") else 0.1
            crop_h = h // 2
            crop_w = w // 2
            top = int(round((h - crop_h) / 2 + shift * h))
            left = int(round((w - crop_w) / 2 + shift * w))
            top = min(max(top, 0), h - crop_h)
            left = min(max(left, 0), w - crop_w)
        right = transforms_F.crop(right, top=top, left=left, height=crop_h, width=crop_w)
        right = transforms_F.resize(
            right, size=[224, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        wrist = transforms_F.resize(
            wrist, size=[160, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        video = torch.cat([right, wrist], dim=-2).float().div_(255.0)
        video = robot_dataset.normalize_transform(video)
        return video.permute(1, 0, 2, 3).contiguous()

    transforms = robot_dataset.lerobot_dataset.processor.train_transforms if robot_dataset.lerobot_dataset.processor.is_train else robot_dataset.lerobot_dataset.processor.val_transforms
    camera_frames = []
    for meta in robot_dataset.lerobot_dataset.image_meta:
        key = meta["key"]
        camera_key = meta["lerobot_key"]
        video_path = Path(dataset.root) / dataset.meta.get_video_file_path(spec["episode_index"], camera_key)
        frames = _decode_full_video_uint8(video_path)
        current_transforms = transforms[key] if isinstance(transforms, dict) else transforms
        camera_frames.append(_apply_image_transforms(frames, current_transforms))
    return _concat_resize_normalize(robot_dataset, torch.stack(camera_frames, dim=0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", default=str(FASTWAM_ROOT))
    parser.add_argument("--task", default="robotwin_uncond_3cam_384_1e-4")
    parser.add_argument("--model", default="fastwam_joint")
    parser.add_argument("--data", default="robotwin")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--timing-report", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        timeout_minutes = int(os.environ.get("FASTWAM_PRECACHE_DIST_TIMEOUT_MINUTES", "360"))
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )
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
            overrides=[f"task={args.task}", f"model={args.model}", f"data={args.data}", *args.overrides],
        )
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    robot_dataset = instantiate(cfg.data.train)
    vae = _load_vae(cfg, device=device, dtype=dtype)

    specs = _episode_specs(robot_dataset)
    selected_specs, total_samples = _shard_specs(specs, rank, world_size, args.max_samples)
    num_samples = len(robot_dataset) if args.max_samples is None else min(len(robot_dataset), int(args.max_samples))
    if rank == 0:
        torch.save(
            {
                "num_samples": num_samples,
                "world_size": world_size,
                "task": args.task,
                "model": args.model,
                "data": args.data,
                "episodewise": True,
                "overrides": list(args.overrides),
            },
            out_root / "metadata.pt",
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=barrier_device_ids)

    timing = {"decode": 0.0, "build": 0.0, "encode": 0.0, "write": 0.0, "samples": 0.0}
    run_start = time.perf_counter()
    encode_batch_size = max(int(args.encode_batch_size), 1)
    pending = []

    def flush(batch):
        if not batch:
            return
        videos = [sample["video"] for _, sample in batch]
        t0 = time.perf_counter()
        latents = _encode_videos(vae, videos, device=device, dtype=dtype)
        timing["encode"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for latent, (idx, sample) in zip(latents, batch):
            path = _cache_path(out_root, idx)
            path.parent.mkdir(parents=True, exist_ok=True)
            episode_latent = sample.get("episode_latents")
            payload = _payload_from_sample(
                sample,
                latent.unsqueeze(0),
                episode_latents=None if episode_latent is None else episode_latent.unsqueeze(0),
            )
            tmp_path = path.with_suffix(f".tmp.{os.getpid()}")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        timing["write"] += time.perf_counter() - t0
        timing["samples"] += len(batch)

    iterator = tqdm(selected_specs, desc=f"rank {rank} episodes", disable=rank != 0)
    for spec in iterator:
        start = spec.get("process_start", spec["global_start"])
        end = spec.get("process_end", spec["global_end"])
        if not args.overwrite and all(_cache_path(out_root, idx).exists() for idx in range(start, end)):
            continue

        t0 = time.perf_counter()
        episode_raw = spec["dataset"].get_episode_data(spec["local_ep_idx"])
        episode_video = _load_episode_video(robot_dataset, spec)
        timing["decode"] += time.perf_counter() - t0
        episode_latents = None
        episode_latent_indices = None
        episode_latent_source_frame_indices = None
        episode_latent_padded_frames = None
        hdr_mode = str(getattr(robot_dataset, "hdr_mode", "back_hdr"))
        if bool(getattr(robot_dataset, "hdr_enabled", False)) and hdr_mode in {"episode_first_hdr", "episode_back_mixed", "episode_first_special_latent"}:
            enc_t0 = time.perf_counter()
            if hdr_mode == "episode_first_special_latent":
                padded_episode_video, pad_frames = _pad_episode_video_for_wan_vae(episode_video)
                full_episode_latents = _encode_videos(vae, [padded_episode_video], device=device, dtype=dtype)[0]
                episode_latents, episode_latent_indices = _select_episode_percent_latents(full_episode_latents)
                episode_latent_padded_frames = torch.tensor(int(pad_frames), dtype=torch.long)
                # The selected latent step k is centered roughly at RGB frame 4*k after Wan temporal compression.
                episode_latent_source_frame_indices = (episode_latent_indices * int(getattr(vae, "temporal_downsample_factor", 4))).clamp(
                    max=max(0, int(spec["length"]) - 1)
                )
                del full_episode_latents, padded_episode_video
            else:
                episode_indices = robot_dataset._get_episode_first_hdr_indices(int(spec["length"]))
                episode_sample_video = episode_video[:, episode_indices, :, :]
                episode_latents = _encode_videos(vae, [episode_sample_video], device=device, dtype=dtype)[0]
            timing["encode"] += time.perf_counter() - enc_t0
        for idx in range(start, end):
            if not args.overwrite and _cache_path(out_root, idx).exists():
                continue
            t0 = time.perf_counter()
            processed = _build_action_state_sample(robot_dataset, episode_raw, spec, idx)
            sample = _build_final_sample(robot_dataset, processed, episode_video, spec)
            if episode_latents is not None:
                sample["episode_latents"] = episode_latents
                if episode_latent_indices is not None:
                    sample["hdr_episode_latent_indices"] = episode_latent_indices
                if episode_latent_source_frame_indices is not None:
                    sample["hdr_episode_latent_source_frame_indices"] = episode_latent_source_frame_indices
                if episode_latent_padded_frames is not None:
                    sample["hdr_episode_latent_padded_frames"] = episode_latent_padded_frames
            timing["build"] += time.perf_counter() - t0
            pending.append((idx, sample))
            if len(pending) >= encode_batch_size:
                flush(pending)
                pending.clear()
        flush(pending)
        pending.clear()
        del episode_raw, episode_video
        gc.collect()
    flush(pending)

    wall = time.perf_counter() - run_start
    if args.timing_report:
        timing_tensor = torch.tensor(
            [timing["samples"], timing["decode"], timing["build"], timing["encode"], timing["write"], wall],
            dtype=torch.float64,
            device=device,
        )
        sum_tensor = timing_tensor.clone()
        max_tensor = timing_tensor.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        if rank == 0:
            samples, decode_s, build_s, encode_s, write_s, _ = [float(x) for x in sum_tensor.detach().cpu()]
            _, decode_max, build_max, encode_max, write_max, wall_max = [float(x) for x in max_tensor.detach().cpu()]
            denom = max(samples, 1.0)
            print(
                "[timing] "
                f"samples={int(samples)} wall_max={wall_max:.3f}s throughput={samples / max(wall_max, 1e-9):.3f} samples/s "
                f"decode_sum={decode_s:.3f}s decode_per_sample={decode_s / denom:.6f}s decode_max_rank={decode_max:.3f}s "
                f"build_sum={build_s:.3f}s build_per_sample={build_s / denom:.6f}s build_max_rank={build_max:.3f}s "
                f"encode_sum={encode_s:.3f}s encode_per_sample={encode_s / denom:.6f}s encode_max_rank={encode_max:.3f}s "
                f"write_sum={write_s:.3f}s write_per_sample={write_s / denom:.6f}s write_max_rank={write_max:.3f}s"
            )
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=barrier_device_ids)
    if rank == 0:
        print(f"[precache] done output_dir={out_root} samples={num_samples}")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
