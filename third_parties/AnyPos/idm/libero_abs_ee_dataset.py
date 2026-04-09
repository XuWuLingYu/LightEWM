import hashlib
import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from .preprocessor import segment_robot_arms


DEFAULT_SUITES = (
    "libero_10",
    "libero_90",
    "libero_goal",
    "libero_object",
    "libero_spatial",
)


@dataclass(frozen=True)
class EpisodeRecord:
    file_path: str
    suite: str
    demo_key: str
    task_name: str
    length: int


def _deterministic_split(identifier: str, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.md5(identifier.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(16 ** 8)
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def _normalized_box_to_pixels(box, width: int, height: int):
    x0, y0, x1, y1 = box
    left = max(0, min(width - 1, int(round(x0 * width))))
    top = max(0, min(height - 1, int(round(y0 * height))))
    right = max(left + 1, min(width, int(round(x1 * width))))
    bottom = max(top + 1, min(height, int(round(y1 * height))))
    return left, top, right, bottom


def flip_box_180(box):
    x0, y0, x1, y1 = box
    return (1.0 - x1, 1.0 - y1, 1.0 - x0, 1.0 - y0)


def flip_image_180(image):
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    return ImageOps.flip(ImageOps.mirror(pil_image))


def extract_single_arm_regions(image, workspace_box, arm_box):
    pil_image = flip_image_180(image)
    width, height = pil_image.size
    workspace_crop = pil_image.crop(_normalized_box_to_pixels(workspace_box, width, height))
    arm_crop = pil_image.crop(_normalized_box_to_pixels(arm_box, width, height))
    return workspace_crop, arm_crop


def extract_official_split_regions(image):
    pil_image = flip_image_180(image)
    np_image = np.array(pil_image)
    h, w = np_image.shape[:2]
    _, arm_boxes = segment_robot_arms(np_image)

    left_split = int(arm_boxes["left_split"])
    right_split = int(arm_boxes["right_split"])
    arm_split = int(arm_boxes["arm_gripper_split"])
    grip_split = int(arm_boxes["gripper_split"])

    def crop_region(index):
        if index == 0:
            box = (0, 0, left_split, arm_split)
        elif index == 1:
            box = (0, arm_split, grip_split, h)
        elif index == 2:
            box = (right_split, 0, w, arm_split)
        elif index == 3:
            box = (grip_split, arm_split, w, h)
        else:
            raise ValueError(index)
        x0, y0, x1, y1 = box
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        return pil_image.crop((x0, y0, x1, y1))

    return pil_image, arm_boxes, [crop_region(i) for i in range(4)]


class LiberoAbsoluteEEDataset(Dataset):
    def __init__(
        self,
        libero_root,
        split,
        suites=None,
        camera_key="agentview_rgb",
        target_key="ee_states",
        frame_stride=1,
        min_episode_len=1,
        val_ratio=0.05,
        test_ratio=0.05,
        disable_pbar=False,
    ):
        del disable_pbar
        self.libero_root = libero_root
        self.split = split
        self.suites = tuple(suites or DEFAULT_SUITES)
        self.camera_key = camera_key
        self.target_key = target_key
        self.frame_stride = max(1, int(frame_stride))
        self.min_episode_len = max(1, int(min_episode_len))
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self._file_cache = {}

        self.episodes = self._scan_episodes()
        if not self.episodes:
            raise RuntimeError(
                f"No episodes found for split={split}, suites={self.suites}, camera_key={camera_key}, target_key={target_key}"
            )
        self.samples = self._build_samples()
        self.target_dim = self._infer_target_dim()

    def _scan_episodes(self):
        episodes = []
        for suite in self.suites:
            suite_dir = os.path.join(self.libero_root, suite)
            if not os.path.isdir(suite_dir):
                continue
            for filename in sorted(os.listdir(suite_dir)):
                if not filename.endswith(".hdf5"):
                    continue
                file_path = os.path.join(suite_dir, filename)
                task_name = os.path.splitext(filename)[0]
                with h5py.File(file_path, "r") as h5_file:
                    for demo_key in sorted(h5_file["data"].keys()):
                        obs_group = h5_file["data"][demo_key]["obs"]
                        if self.camera_key not in obs_group or self.target_key not in obs_group:
                            continue
                        length = min(len(obs_group[self.camera_key]), len(obs_group[self.target_key]))
                        if length < self.min_episode_len:
                            continue
                        identifier = f"{suite}/{task_name}/{demo_key}"
                        if _deterministic_split(identifier, self.val_ratio, self.test_ratio) != self.split:
                            continue
                        episodes.append(
                            EpisodeRecord(
                                file_path=file_path,
                                suite=suite,
                                demo_key=demo_key,
                                task_name=task_name,
                                length=length,
                            )
                        )
        return episodes

    def _build_samples(self):
        samples = []
        for episode_index, episode in enumerate(self.episodes):
            for frame_index in range(0, episode.length, self.frame_stride):
                samples.append((episode_index, frame_index))
        return samples

    def _get_file_handle(self, file_path):
        handle = self._file_cache.get(file_path)
        if handle is None:
            handle = h5py.File(file_path, "r")
            self._file_cache[file_path] = handle
        return handle

    def _get_obs_group(self, episode: EpisodeRecord):
        file_handle = self._get_file_handle(episode.file_path)
        return file_handle["data"][episode.demo_key]["obs"]

    def _infer_target_dim(self):
        episode = self.episodes[0]
        obs = self._get_obs_group(episode)
        return int(np.asarray(obs[self.target_key][0]).shape[0])

    def compute_target_stats(self):
        total_count = 0
        sum_vector = None
        sumsq_vector = None
        for episode in self.episodes:
            with h5py.File(episode.file_path, "r") as h5_file:
                targets = np.asarray(
                    h5_file["data"][episode.demo_key]["obs"][self.target_key][:: self.frame_stride],
                    dtype=np.float64,
                )
            if len(targets) == 0:
                continue
            if sum_vector is None:
                sum_vector = targets.sum(axis=0)
                sumsq_vector = np.square(targets).sum(axis=0)
            else:
                sum_vector += targets.sum(axis=0)
                sumsq_vector += np.square(targets).sum(axis=0)
            total_count += targets.shape[0]
        if total_count == 0:
            raise RuntimeError("No targets available to compute mean/std.")
        mean = sum_vector / total_count
        var = np.maximum(sumsq_vector / total_count - np.square(mean), 1e-12)
        std = np.sqrt(var)
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode_index, frame_index = self.samples[index]
        episode = self.episodes[episode_index]
        obs = self._get_obs_group(episode)
        image = np.asarray(obs[self.camera_key][frame_index]).copy()
        target = np.asarray(obs[self.target_key][frame_index], dtype=np.float32).copy()
        return torch.from_numpy(target), image

    def get_metadata(self, index):
        episode_index, frame_index = self.samples[index]
        episode = self.episodes[episode_index]
        return {
            "suite": episode.suite,
            "task_name": episode.task_name,
            "demo_key": episode.demo_key,
            "frame_index": frame_index,
        }
