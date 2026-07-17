import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .lerobot.datasets.video_utils import decode_video_frames
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        action_horizon: Optional[int] = None,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", "realbot_top_wrist_320x384", "realbot_top_wrist_320x384_right_center_crop_half", "realbot_top_wrist_320x384_right_crop_half_shift_0p1", "realbot_top_wrist_320x384_right_crop_half_shift_0p2", "realbot_top_wrist_320x384_right_bottom_crop_0p6", or None
        return_tree_video: bool = False,
        hdr_enabled: bool = False,
        hdr_local_rgb_frames: int = 9,
        hdr_tree_rgb_frames: int = 4,
        hdr_total_rgb_frames: Optional[int] = None,
        hdr_tree_sampling: str = "uniform_local_start_to_end",
        hdr_mode: str = "back_hdr",
        latent_cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        video_backend: Optional[str] = None,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=(num_frames - 1 if action_horizon is None else int(action_horizon)),
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            video_backend=video_backend,
        )
    
        self.num_frames = num_frames
        self.action_horizon = num_frames - 1 if action_horizon is None else int(action_horizon)
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.return_tree_video = bool(return_tree_video)
        self.hdr_enabled = bool(hdr_enabled)
        self.hdr_local_rgb_frames = int(hdr_local_rgb_frames)
        self.hdr_tree_rgb_frames = int(hdr_tree_rgb_frames)
        self.hdr_total_rgb_frames = (
            self.hdr_local_rgb_frames + self.hdr_tree_rgb_frames
            if hdr_total_rgb_frames is None
            else int(hdr_total_rgb_frames)
        )
        self.hdr_tree_sampling = str(hdr_tree_sampling)
        self.hdr_mode = str(hdr_mode)
        self._episode_first_hdr_modes = {"episode_first_hdr", "episode_back_mixed", "episode_first_special_latent"}
        self._back_tree_hdr_modes = {"back_hdr", "episode_back_mixed"}
        if self.hdr_mode not in {"back_hdr", "episode_first_hdr", "episode_back_mixed", "episode_first_special_latent"}:
            raise ValueError(
                f"Unsupported hdr_mode: {self.hdr_mode}. "
                "Expected 'back_hdr', 'episode_first_hdr', 'episode_back_mixed', or 'episode_first_special_latent'."
            )
        self.latent_cache_dir = None if latent_cache_dir is None else str(latent_cache_dir)
        self.max_samples = None if max_samples is None else max(int(max_samples), 0)
        if self.hdr_enabled:
            if self.hdr_local_rgb_frames != len(self.video_sample_indices):
                raise ValueError(
                    "FastWAM HDR currently expects the local RGB count to match "
                    f"`video_sample_indices`, got hdr_local_rgb_frames={self.hdr_local_rgb_frames} "
                    f"and local_samples={len(self.video_sample_indices)}."
                )
            if self.hdr_tree_rgb_frames <= 0:
                raise ValueError(f"hdr_tree_rgb_frames must be > 0, got {self.hdr_tree_rgb_frames}.")
            if self.hdr_total_rgb_frames != self.hdr_local_rgb_frames + self.hdr_tree_rgb_frames:
                raise ValueError(
                    "hdr_total_rgb_frames must equal hdr_local_rgb_frames + hdr_tree_rgb_frames, "
                    f"got {self.hdr_total_rgb_frames} vs "
                    f"{self.hdr_local_rgb_frames}+{self.hdr_tree_rgb_frames}."
                )
            if self.hdr_total_rgb_frames % 4 != 1:
                raise ValueError(
                    f"hdr_total_rgb_frames must satisfy T % 4 == 1 for Wan VAE tokenization, got {self.hdr_total_rgb_frames}."
                )
            if self.hdr_mode in self._back_tree_hdr_modes and self.hdr_tree_sampling != "uniform_local_start_to_end":
                raise ValueError(f"Unsupported hdr_tree_sampling: {self.hdr_tree_sampling}.")
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        length = len(self.lerobot_dataset)
        if self.max_samples is not None:
            length = min(length, self.max_samples)
        return length

    def _requires_raw_camera_frames(self) -> bool:
        return self.concat_multi_camera in {
            "realbot_top_wrist_320x384_right_crop_half_shift_0p1",
            "realbot_top_wrist_320x384_right_crop_half_shift_0p2",
            "realbot_top_wrist_320x384_right_bottom_crop_0p6",
        }

    def _load_local_raw_video(self, sample, offsets: list[int]) -> torch.Tensor:
        episode_len, dataset, episode_index = self._episode_metadata_from_sample(sample)
        local_start = int(torch.as_tensor(sample["frame_index"]).item())
        stride = int(self.lerobot_dataset.global_sample_stride)
        indices = [max(0, min(episode_len - 1, local_start + int(offset) * stride)) for offset in offsets]
        timestamps = [index / float(dataset.fps) for index in indices]
        return self._load_episode_frames(dataset, episode_index, timestamps, raw_for_camera_crop=True)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad_full = sample["image_is_pad"]
        image_is_pad = image_is_pad_full
        tree_image_is_pad = image_is_pad_full if self.return_tree_video else None

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        tree_video = video if self.return_tree_video else None
        num_cameras = 1
        tree_indices = []
        local_indices = self.video_sample_indices
        use_raw_camera_frames = self.latent_cache_dir is None and self._requires_raw_camera_frames()
        if use_raw_camera_frames:
            # The processor resizes each camera to 224x224 before RobotVideoDataset sees it.
            # Crop layouts need the original 480x640 right camera, so online eval reloads raw frames.
            video = self._load_local_raw_video(sample, list(range(self.num_frames)))
            tree_video = video if self.return_tree_video else None
        if video.ndim == 5:
            if self.hdr_enabled:
                if self.hdr_mode in self._episode_first_hdr_modes:
                    episode_video, episode_indices = self._load_episode_first_hdr_video(sample)
                if self.hdr_mode in self._back_tree_hdr_modes:
                    hdr_tree_video, tree_indices = self._load_episode_hdr_tree_video(sample)
                    video = torch.cat([video[:, local_indices, :, :, :], hdr_tree_video], dim=1)
                    video_indices = local_indices + tree_indices
                else:
                    video = video[:, local_indices, :, :, :]
                    video_indices = local_indices
            else:
                video_indices = local_indices
                video = video[:, video_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            if self.hdr_enabled:
                if self.hdr_mode in self._episode_first_hdr_modes:
                    episode_video, episode_indices = self._load_episode_first_hdr_video(sample)
                if self.hdr_mode in self._back_tree_hdr_modes:
                    hdr_tree_video, tree_indices = self._load_episode_hdr_tree_video(sample)
                    video = torch.cat([video[local_indices, :, :, :], hdr_tree_video], dim=0)
                    video_indices = local_indices + tree_indices
                else:
                    video = video[local_indices, :, :, :]
                    video_indices = local_indices
            else:
                video_indices = local_indices
                video = video[video_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        local_image_is_pad = image_is_pad_full[local_indices]
        if self.hdr_enabled:
            if self.hdr_mode not in self._back_tree_hdr_modes:
                image_is_pad = local_image_is_pad
            else:
                hdr_image_is_pad = torch.zeros(
                    self.hdr_tree_rgb_frames,
                    dtype=local_image_is_pad.dtype,
                    device=local_image_is_pad.device,
                )
                image_is_pad = torch.cat([local_image_is_pad, hdr_image_is_pad], dim=0)
        else:
            image_is_pad = local_image_is_pad

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]

        def _concat_resize_normalize(video_tensor):
            if video_tensor.ndim == 4:
                video_tensor = video_tensor.unsqueeze(0)
            local_num_cameras, local_t, local_c, local_h, local_w = video_tensor.shape

            if self.concat_multi_camera == "robotwin":
                if local_num_cameras != 3:
                    raise ValueError(
                        f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {local_num_cameras}"
                    )
                cam_top = transforms_F.resize(
                    video_tensor[0],
                    size=[256, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 256, 320]
                cam_left = transforms_F.resize(
                    video_tensor[1],
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 128, 160]
                cam_right = transforms_F.resize(
                    video_tensor[2],
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 128, 160]
                bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
                video_tensor = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
            elif self.concat_multi_camera == "realbot_top_wrist_320x384":
                if local_num_cameras != 2:
                    raise ValueError(
                        f"`concat_multi_camera='realbot_top_wrist_320x384'` requires exactly 2 cameras, got {local_num_cameras}"
                    )
                cam_right = transforms_F.resize(
                    video_tensor[0],
                    size=[224, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 224, 320]
                cam_wrist = transforms_F.resize(
                    video_tensor[1],
                    size=[160, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 160, 320]
                video_tensor = torch.cat([cam_right, cam_wrist], dim=-2)  # [T_video, C, 384, 320]
            elif self.concat_multi_camera == "realbot_top_wrist_front_320x384":
                if local_num_cameras != 3:
                    raise ValueError(
                        "`concat_multi_camera='realbot_top_wrist_front_320x384'` "
                        f"requires exactly 3 cameras, got {local_num_cameras}"
                    )
                cam_right = transforms_F.resize(
                    video_tensor[0],
                    size=[224, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                cam_wrist = transforms_F.resize(
                    video_tensor[1],
                    size=[160, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                cam_front = transforms_F.resize(
                    video_tensor[2],
                    size=[160, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                bottom = torch.cat([cam_wrist, cam_front], dim=-1)
                video_tensor = torch.cat([cam_right, bottom], dim=-2)
            elif self.concat_multi_camera == "realbot_top_wrist_320x384_right_center_crop_half":
                if local_num_cameras != 2:
                    raise ValueError(
                        "`concat_multi_camera='realbot_top_wrist_320x384_right_center_crop_half'` "
                        f"requires exactly 2 cameras, got {local_num_cameras}"
                    )
                crop_h = max(1, local_h // 2)
                crop_w = max(1, local_w // 2)
                cam_right_center = transforms_F.center_crop(video_tensor[0], output_size=[crop_h, crop_w])
                cam_right = transforms_F.resize(
                    cam_right_center,
                    size=[224, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 224, 320]
                cam_wrist = transforms_F.resize(
                    video_tensor[1],
                    size=[160, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 160, 320]
                video_tensor = torch.cat([cam_right, cam_wrist], dim=-2)  # [T_video, C, 384, 320]
            elif self.concat_multi_camera in {
                "realbot_top_wrist_320x384_right_crop_half_shift_0p1",
                "realbot_top_wrist_320x384_right_crop_half_shift_0p2",
                "realbot_top_wrist_320x384_right_bottom_crop_0p6",
            }:
                if local_num_cameras != 2:
                    raise ValueError(
                        f"`concat_multi_camera='{self.concat_multi_camera}'` "
                        f"requires exactly 2 cameras, got {local_num_cameras}"
                    )
                if local_h == 480 and local_w == 640:
                    if self.concat_multi_camera == "realbot_top_wrist_320x384_right_bottom_crop_0p6":
                        crop_h = int(round(local_h * 0.6))
                        crop_w = int(round(local_w * 0.6))
                        top = local_h - crop_h
                        left = local_w - crop_w
                    else:
                        shift = 0.2 if self.concat_multi_camera.endswith("_0p2") else 0.1
                        crop_h = local_h // 2
                        crop_w = local_w // 2
                        top = int(round((local_h - crop_h) / 2 + shift * local_h))
                        left = int(round((local_w - crop_w) / 2 + shift * local_w))
                        top = min(max(top, 0), local_h - crop_h)
                        left = min(max(left, 0), local_w - crop_w)
                    cam_right_source = transforms_F.crop(video_tensor[0], top=top, left=left, height=crop_h, width=crop_w)
                else:
                    raise ValueError(
                        f"{self.concat_multi_camera} expects original right view resolution 480x640 "
                        f"before crop/resize, got {local_h}x{local_w}. "
                        "Online eval must reload raw camera frames before applying this layout."
                    )
                cam_right = transforms_F.resize(
                    cam_right_source,
                    size=[224, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 224, 320]
                cam_wrist = transforms_F.resize(
                    video_tensor[1],
                    size=[160, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 160, 320]
                video_tensor = torch.cat([cam_right, cam_wrist], dim=-2)  # [T_video, C, 384, 320]
            elif local_num_cameras > 1:
                if self.concat_multi_camera == "horizontal":
                    video_tensor = torch.cat(
                        [video_tensor[i] for i in range(local_num_cameras)], dim=-1
                    )  # [T_video, C, H, num_cameras*W]
                elif self.concat_multi_camera == "vertical":
                    video_tensor = torch.cat(
                        [video_tensor[i] for i in range(local_num_cameras)], dim=-2
                    )  # [T_video, C, num_cameras*H, W]
                else:
                    raise ValueError(
                        f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                        "Expected one of: horizontal, vertical, robotwin, realbot_top_wrist_320x384, "
                        "realbot_top_wrist_320x384_right_center_crop_half, "
                        "realbot_top_wrist_320x384_right_crop_half_shift_0p1, "
                        "realbot_top_wrist_320x384_right_crop_half_shift_0p2, "
                        "realbot_top_wrist_320x384_right_bottom_crop_0p6."
                    )
            else:
                video_tensor = video_tensor.squeeze(0)  # [T_video, C, H, W]

            video_tensor = self.resize_transform(video_tensor)
            video_tensor = self.crop_transform(video_tensor)
            video_tensor = self.normalize_transform(video_tensor)  # [T_video, C, H, W]
            return video_tensor.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        video = _concat_resize_normalize(video)

        if tree_video is not None:
            if tree_video.ndim not in (4, 5):
                raise ValueError(
                    f"`tree_video` source must be [T,C,H,W] or [num_cameras,T,C,H,W], got {tuple(tree_video.shape)}"
                )
            tree_video = _concat_resize_normalize(tree_video)

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        action_video_transition_count = len(local_indices) - 1 if self.hdr_enabled else video.shape[1] - 1
        if action.shape[0] % action_video_transition_count != 0:
            raise ValueError(
                f"`action` horizon must be divisible by local video transitions, got {action.shape[0]} and {action_video_transition_count}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.hdr_enabled:
            data["hdr_mode"] = self.hdr_mode
            data["local_video_frames"] = torch.tensor(len(local_indices), dtype=torch.long)
            data["action_video_transition_count"] = torch.tensor(action_video_transition_count, dtype=torch.long)
            data["hdr_local_frame_indices"] = torch.tensor(local_indices, dtype=torch.long)
            if self.hdr_mode in self._episode_first_hdr_modes:
                data["episode_video"] = _concat_resize_normalize(episode_video)
                data["hdr_episode_frame_indices"] = torch.tensor(episode_indices, dtype=torch.long)
            if self.hdr_mode in self._back_tree_hdr_modes:
                data["hdr_tree_frame_indices"] = torch.tensor(tree_indices, dtype=torch.long)
        if tree_video is not None:
            data["tree_video"] = tree_video
            data["tree_image_is_pad"] = tree_image_is_pad
        return data

    def _latent_cache_path(self, idx: int) -> Path:
        if self.latent_cache_dir is None:
            raise ValueError("latent_cache_dir is not set.")
        shard = int(idx) // 10000
        return Path(self.latent_cache_dir) / f"shard_{shard:05d}" / f"{int(idx):08d}.pt"

    def _get_cached(self, idx: int):
        cache_path = self._latent_cache_path(idx)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing FastWAM latent cache file: {cache_path}")
        payload = None
        for attempt in range(5):
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                break
            except OSError as exc:
                if attempt == 4:
                    raise OSError(
                        f"Failed to load FastWAM latent cache file after 5 attempts: {cache_path}"
                    ) from exc
                wait_s = min(2.0 * (attempt + 1), 8.0)
                logger.warning(
                    "Failed to load FastWAM latent cache file %s on attempt %d/5: %s; retrying in %.1fs",
                    cache_path,
                    attempt + 1,
                    exc,
                    wait_s,
                )
                time.sleep(wait_s)
        assert payload is not None
        prompt = payload["prompt"]
        if "context" in payload and "context_mask" in payload:
            context = payload["context"]
            context_mask = payload["context_mask"]
        else:
            context, context_mask = self._get_cached_text_context(prompt)
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
        hdr_mode = str(payload.get("hdr_mode", "back_hdr"))
        if hdr_mode in {"episode_first_hdr", "episode_back_mixed", "episode_first_special_latent"}:
            episode_latents = payload["episode_latents"]
            local_latents = payload["local_latents"]
            input_latents = torch.cat([episode_latents, local_latents], dim=1).contiguous()
            first_frame_latents = None
            clean_latent_indices = torch.tensor([0, int(episode_latents.shape[1])], dtype=torch.long)
        else:
            input_latents = payload["input_latents"]
            first_frame_latents = payload.get("first_frame_latents")
            clean_latent_indices = None
        data = {
            "input_latents": input_latents,
            "action": payload["action"],
            "proprio": payload["proprio"],
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": payload["image_is_pad"],
            "action_is_pad": payload["action_is_pad"],
            "action_dim_is_pad": payload.get("action_dim_is_pad", torch.zeros(payload["action"].shape[-1], dtype=torch.bool)),
            "proprio_is_pad": payload["proprio_is_pad"],
            "num_video_frames": int(payload["num_video_frames"]),
            "hdr_mode": hdr_mode,
        }
        if first_frame_latents is not None:
            data["first_frame_latents"] = first_frame_latents
        if clean_latent_indices is not None:
            data["clean_latent_indices"] = clean_latent_indices
            data["episode_video_latent_indices"] = torch.arange(1, int(payload["episode_latents"].shape[1]), dtype=torch.long)
            local_start = int(payload["episode_latents"].shape[1])
            data["local_video_latent_indices"] = torch.arange(local_start + 1, local_start + int(payload["local_latents"].shape[1]), dtype=torch.long)
        optional_keys = (
            "local_video_frames",
            "action_video_transition_count",
            "hdr_tree_frame_indices",
            "hdr_local_frame_indices",
            "hdr_episode_frame_indices",
            "hdr_episode_latent_indices",
            "hdr_episode_latent_source_frame_indices",
            "hdr_episode_latent_padded_frames",
            "action_adapter",
            "source_name",
            "source_episode_index",
        )
        for key in optional_keys:
            if key in payload:
                data[key] = payload[key]
        return data

    def _get_episode_first_hdr_indices(self, episode_len: int) -> list[int]:
        if episode_len <= 0:
            raise ValueError(f"episode_len must be positive, got {episode_len}")
        raw = np.linspace(0, episode_len - 1, self.hdr_local_rgb_frames)
        indices = np.rint(raw).astype(np.int64).clip(0, episode_len - 1).tolist()
        return [int(i) for i in indices]

    def _get_hdr_tree_indices(self, local_end: int, episode_end_exclusive: int) -> list[int]:
        first_candidate = local_end + 1
        if episode_end_exclusive <= first_candidate:
            return [int(local_end)] * self.hdr_tree_rgb_frames
        if self.hdr_tree_rgb_frames == 1:
            return [episode_end_exclusive - 1]
        tail = episode_end_exclusive - 1 - local_end
        raw = [
            np.ceil(local_end + tail * (i + 1) / self.hdr_tree_rgb_frames)
            for i in range(self.hdr_tree_rgb_frames)
        ]
        indices = np.asarray(raw, dtype=np.int64).clip(first_candidate, episode_end_exclusive - 1).tolist()
        return [int(i) for i in indices]

    def _load_episode_first_hdr_video(self, sample) -> tuple[torch.Tensor, list[int]]:
        episode_len, dataset, episode_index = self._episode_metadata_from_sample(sample)
        indices = self._get_episode_first_hdr_indices(episode_len)
        timestamps = [index / float(dataset.fps) for index in indices]
        return self._load_episode_frames(
            dataset,
            episode_index,
            timestamps,
            raw_for_camera_crop=(self.latent_cache_dir is None and self._requires_raw_camera_frames()),
        ), indices

    def _episode_metadata_from_sample(self, sample) -> tuple[int, object, int]:
        required_keys = ("dataset_index", "episode_index", "frame_index")
        missing = [key for key in required_keys if key not in sample]
        if missing:
            raise KeyError(f"FastWAM HDR requires sample metadata keys {missing} to load full-episode frames.")

        dataset_index = int(torch.as_tensor(sample["dataset_index"]).item())
        episode_index = int(torch.as_tensor(sample["episode_index"]).item())
        multi_dataset = self.lerobot_dataset.multi_dataset
        dataset = multi_dataset._datasets[dataset_index]
        current_ep_idx = dataset.episodes.index(episode_index) if dataset.episodes is not None else episode_index
        ep_start = int(dataset.episode_data_index["from"][current_ep_idx].item())
        ep_end = int(dataset.episode_data_index["to"][current_ep_idx].item())
        return ep_end - ep_start, dataset, episode_index

    def _load_episode_frames(
        self,
        dataset,
        episode_index: int,
        timestamps: list[float],
        raw_for_camera_crop: bool = False,
    ) -> torch.Tensor:
        decoded = []
        processor = self.lerobot_dataset.processor
        if processor is None:
            raise ValueError("FastWAM HDR requires a processor so decoded episode frames can use the same image transforms.")
        transforms = processor.train_transforms if processor.is_train else processor.val_transforms
        for meta in self.lerobot_dataset.image_meta:
            camera_key = meta["lerobot_key"]
            video_path = Path(dataset.root) / dataset.meta.get_video_file_path(episode_index, camera_key)
            frames = decode_video_frames(video_path, timestamps, dataset.tolerance_s, dataset.video_backend)
            frames = (frames.squeeze(0) * 255).to(torch.uint8)
            current_transforms = transforms[meta["key"]] if isinstance(transforms, dict) else transforms
            if raw_for_camera_crop:
                # Keep only ToTensor-like transforms. Spatial resize must happen after
                # the right-camera crop in _concat_resize_normalize.
                for trans in current_transforms:
                    if trans.__class__.__name__ == "ToTensor":
                        frames = trans(frames)
                if frames.dtype == torch.uint8:
                    frames = frames.to(torch.float32) / 255.0
            else:
                for trans in current_transforms:
                    frames = trans(frames)
            decoded.append(frames)
        if len(decoded) == 1:
            return decoded[0]
        return torch.stack(decoded, dim=0)

    def _load_episode_hdr_tree_video(self, sample) -> tuple[torch.Tensor, list[int]]:
        episode_len, dataset, episode_index = self._episode_metadata_from_sample(sample)
        local_start = int(torch.as_tensor(sample["frame_index"]).item())
        local_end = min(local_start + (self.num_frames - 1) * self.lerobot_dataset.global_sample_stride, episode_len - 1)
        tree_indices = self._get_hdr_tree_indices(local_end=local_end, episode_end_exclusive=episode_len)
        timestamps = [index / float(dataset.fps) for index in tree_indices]

        return self._load_episode_frames(
            dataset,
            episode_index,
            timestamps,
            raw_for_camera_crop=(self.latent_cache_dir is None and self._requires_raw_camera_frames()),
        ), tree_indices

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        if self.latent_cache_dir is not None:
            return self._get_cached(idx)
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
