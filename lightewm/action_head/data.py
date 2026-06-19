from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DemoSpec:
    suite: str
    task: str
    hdf5_path: str
    demo_id: str


def load_action_manifest(path: str | Path) -> list[DemoSpec]:
    specs: list[DemoSpec] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            specs.append(
                DemoSpec(
                    suite=str(row["suite"]),
                    task=str(row["task"]),
                    hdf5_path=str(row["hdf5_path"]),
                    demo_id=str(row["demo_id"]),
                )
            )
    if not specs:
        raise ValueError(f"No demos found in manifest: {path}")
    return specs


def build_task_vocab(specs: Iterable[DemoSpec]) -> dict[str, int]:
    return {task: idx for idx, task in enumerate(sorted({spec.task for spec in specs}))}


class LiberoActionDataset(Dataset):
    """Frame-level supervised action dataset for LIBERO hdf5 demonstrations."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_key: str = "agentview_rgb",
        proprio_key: str = "robot_states",
        sample_stride: int = 1,
        task_to_id: dict[str, int] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.specs = load_action_manifest(self.manifest_path)
        self.image_key = image_key
        self.proprio_key = proprio_key
        self.sample_stride = max(1, int(sample_stride))
        self.task_to_id = task_to_id or build_task_vocab(self.specs)
        self.index: list[tuple[int, int]] = []
        for spec_idx, spec in enumerate(self.specs):
            with h5py.File(spec.hdf5_path, "r") as f:
                demo = f[f"data/{spec.demo_id}"]
                length = int(demo["actions"].shape[0])
            self.index.extend((spec_idx, t) for t in range(0, length, self.sample_stride))
        if not self.index:
            raise ValueError(f"No action frames indexed from {manifest_path}")

    def __len__(self) -> int:
        return len(self.index)

    @property
    def num_tasks(self) -> int:
        return len(self.task_to_id)

    def _read_frame(self, spec: DemoSpec, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(spec.hdf5_path, "r") as f:
            demo = f[f"data/{spec.demo_id}"]
            image = np.asarray(demo["obs"][self.image_key][step], dtype=np.uint8)
            if self.proprio_key in demo:
                proprio = np.asarray(demo[self.proprio_key][step], dtype=np.float32)
            else:
                proprio = np.asarray(demo["obs"][self.proprio_key][step], dtype=np.float32)
            action = np.asarray(demo["actions"][step], dtype=np.float32)
        return image, proprio, action

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        spec_idx, step = self.index[idx]
        spec = self.specs[spec_idx]
        image, proprio, action = self._read_frame(spec, step)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        return {
            "image": image_t,
            "proprio": torch.from_numpy(proprio).float(),
            "task_id": torch.tensor(self.task_to_id[spec.task], dtype=torch.long),
            "action": torch.from_numpy(action).float(),
        }
