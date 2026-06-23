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
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        return_tree_video: bool = False,
        hdr_enabled: bool = False,
        hdr_local_rgb_frames: int = 9,
        hdr_tree_rgb_frames: int = 4,
        hdr_total_rgb_frames: Optional[int] = None,
        hdr_tree_sampling: str = "uniform_local_start_to_end",
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
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
            if self.hdr_tree_sampling != "uniform_local_start_to_end":
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
        return len(self.lerobot_dataset)

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
        if video.ndim == 5:
            if self.hdr_enabled:
                hdr_tree_video, tree_indices = self._load_episode_hdr_tree_video(sample)
                video = torch.cat([video[:, local_indices, :, :, :], hdr_tree_video], dim=1)
                video_indices = local_indices + tree_indices
            else:
                video_indices = local_indices
                video = video[:, video_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            if self.hdr_enabled:
                hdr_tree_video, tree_indices = self._load_episode_hdr_tree_video(sample)
                video = torch.cat([video[local_indices, :, :, :], hdr_tree_video], dim=0)
                video_indices = local_indices + tree_indices
            else:
                video_indices = local_indices
                video = video[video_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        local_image_is_pad = image_is_pad_full[local_indices]
        if self.hdr_enabled:
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
                        "Expected one of: horizontal, vertical, robotwin."
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
            data["local_video_frames"] = torch.tensor(len(local_indices), dtype=torch.long)
            data["action_video_transition_count"] = torch.tensor(action_video_transition_count, dtype=torch.long)
            data["hdr_tree_frame_indices"] = torch.tensor(tree_indices, dtype=torch.long)
            data["hdr_local_frame_indices"] = torch.tensor(local_indices, dtype=torch.long)
        if tree_video is not None:
            data["tree_video"] = tree_video
            data["tree_image_is_pad"] = tree_image_is_pad
        return data

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

    def _load_episode_hdr_tree_video(self, sample) -> tuple[torch.Tensor, list[int]]:
        required_keys = ("dataset_index", "episode_index", "frame_index")
        missing = [key for key in required_keys if key not in sample]
        if missing:
            raise KeyError(f"FastWAM HDR requires sample metadata keys {missing} to load full-episode frames.")

        dataset_index = int(torch.as_tensor(sample["dataset_index"]).item())
        episode_index = int(torch.as_tensor(sample["episode_index"]).item())
        local_start = int(torch.as_tensor(sample["frame_index"]).item())

        multi_dataset = self.lerobot_dataset.multi_dataset
        dataset = multi_dataset._datasets[dataset_index]
        current_ep_idx = dataset.episodes.index(episode_index) if dataset.episodes is not None else episode_index
        ep_start = int(dataset.episode_data_index["from"][current_ep_idx].item())
        ep_end = int(dataset.episode_data_index["to"][current_ep_idx].item())
        episode_len = ep_end - ep_start
        local_end = min(local_start + (self.num_frames - 1) * self.lerobot_dataset.global_sample_stride, episode_len - 1)
        tree_indices = self._get_hdr_tree_indices(local_end=local_end, episode_end_exclusive=episode_len)
        timestamps = [index / float(dataset.fps) for index in tree_indices]

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
            for trans in current_transforms:
                frames = trans(frames)
            decoded.append(frames)

        if len(decoded) == 1:
            return decoded[0], tree_indices
        return torch.stack(decoded, dim=0), tree_indices

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
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
