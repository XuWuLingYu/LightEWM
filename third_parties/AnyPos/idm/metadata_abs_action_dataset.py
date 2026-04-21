import hashlib
import json
import os
from collections import OrderedDict

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


cv2.setNumThreads(1)


def _deterministic_split(identifier: str, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.md5(identifier.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(16 ** 8)
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def _load_metadata_rows(metadata_path: str):
    if metadata_path.endswith(".jsonl"):
        rows = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if metadata_path.endswith(".json"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {metadata_path}")
        return data
    frame = pandas.read_csv(metadata_path)
    return [frame.iloc[i].to_dict() for i in range(len(frame))]


def _parse_action_value(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("action string is empty")
        try:
            parsed = json.loads(stripped)
        except Exception:
            try:
                parsed = yaml.safe_load(stripped)
            except Exception:
                parsed = [float(x.strip()) for x in stripped.split(",") if x.strip()]
        return np.asarray(parsed, dtype=np.float32)
    raise TypeError(f"Unsupported action value type: {type(value).__name__}")


class MetadataAbsoluteActionDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        split,
        image_base_path=None,
        image_key="image",
        video_key="video",
        action_key="abs_action",
        split_key="split",
        id_key=None,
        val_ratio=0.05,
        test_ratio=0.05,
        max_open_videos=2,
    ):
        self.metadata_path = metadata_path
        self.split = split
        self.image_base_path = image_base_path
        self.image_key = image_key
        self.video_key = video_key
        self.action_key = action_key
        self.split_key = split_key
        self.id_key = id_key
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.max_open_videos = max(1, int(max_open_videos))
        self.metadata_dir = os.path.dirname(os.path.abspath(metadata_path))
        self._video_cache = OrderedDict()

        rows = _load_metadata_rows(metadata_path)
        if len(rows) == 0:
            raise RuntimeError(f"No metadata rows found in {metadata_path}")

        self.samples = self._build_samples(rows)
        if len(self.samples) == 0:
            raise RuntimeError(f"No rows matched split={split} in {metadata_path}")
        self.target_dim = self._infer_target_dim()

    def _resolve_media_path(self, value: str):
        if os.path.isabs(value):
            return os.path.abspath(value)
        if self.image_base_path:
            candidate = os.path.join(self.image_base_path, value)
            return os.path.abspath(candidate)
        return os.path.abspath(os.path.join(self.metadata_dir, value))

    def _row_identifier(self, row: dict, row_index: int):
        if self.id_key and self.id_key in row:
            return str(row[self.id_key])
        if self.image_key in row:
            return str(row[self.image_key])
        if self.video_key in row:
            return str(row[self.video_key])
        return f"row_{row_index}"

    def _extract_action(self, row: dict):
        if self.action_key in row:
            return _parse_action_value(row[self.action_key])

        prefix = f"{self.action_key}_"
        indexed = []
        for key, value in row.items():
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if suffix.isdigit():
                indexed.append((int(suffix), float(value)))
        if indexed:
            indexed.sort(key=lambda x: x[0])
            return np.asarray([value for _, value in indexed], dtype=np.float32)

        if self.action_key == "abs_action":
            if "action" in row:
                return _parse_action_value(row["action"])
            indexed = []
            for key, value in row.items():
                if not isinstance(key, str) or not key.startswith("action_"):
                    continue
                suffix = key[len("action_"):]
                if suffix.isdigit():
                    indexed.append((int(suffix), float(value)))
            if indexed:
                indexed.sort(key=lambda x: x[0])
                return np.asarray([value for _, value in indexed], dtype=np.float32)

        raise KeyError(
            f"Row is missing action field '{self.action_key}' and indexed columns '{self.action_key}_0...'."
        )

    def _extract_action_sequence(self, row: dict):
        actions = self._extract_action(row)
        if actions.ndim == 1:
            return actions[None, :]
        if actions.ndim != 2:
            raise ValueError(f"Expected 1D or 2D action tensor, got shape {tuple(actions.shape)}")
        return actions

    def _extract_frame_indices(self, row: dict, num_frames: int):
        frame_indices = row.get("frame_indices")
        if frame_indices is None:
            return list(range(num_frames))
        parsed = _parse_action_value(frame_indices).astype(np.int64)
        if parsed.ndim != 1:
            raise ValueError(f"frame_indices must be 1D, got shape {tuple(parsed.shape)}")
        if parsed.shape[0] != num_frames:
            raise ValueError(
                f"frame_indices length mismatch: expected {num_frames}, got {parsed.shape[0]}"
            )
        return [int(x) for x in parsed.tolist()]

    def _build_samples(self, rows):
        samples = []
        for row_index, row in enumerate(rows):
            row_split = row.get(self.split_key)
            if row_split is None or str(row_split).strip() == "":
                row_split = _deterministic_split(
                    self._row_identifier(row, row_index),
                    self.val_ratio,
                    self.test_ratio,
                )
            if str(row_split) != self.split:
                continue

            if self.image_key in row and str(row[self.image_key]).strip():
                image_path = self._resolve_media_path(str(row[self.image_key]))
                action = self._extract_action(row)
                if action.ndim != 1:
                    raise ValueError(
                        f"Image row expects a single action vector, got shape {tuple(action.shape)}"
                    )
                samples.append(
                    {
                        **row,
                        "__row_index__": row_index,
                        "__mode__": "image",
                        "__media_path__": image_path,
                        "__action__": action.astype(np.float32),
                        "__split__": str(row_split),
                    }
                )
                continue

            if self.video_key in row and str(row[self.video_key]).strip():
                video_path = self._resolve_media_path(str(row[self.video_key]))
                action_sequence = self._extract_action_sequence(row).astype(np.float32)
                frame_indices = self._extract_frame_indices(row, action_sequence.shape[0])
                for local_index, frame_index in enumerate(frame_indices):
                    samples.append(
                        {
                            **row,
                            "__row_index__": row_index,
                            "__mode__": "video",
                            "__media_path__": video_path,
                            "__frame_index__": int(frame_index),
                            "__sample_index_in_video__": int(local_index),
                            "__action__": action_sequence[local_index],
                            "__split__": str(row_split),
                        }
                    )
                continue

            raise KeyError(
                f"Row {row_index} must contain either '{self.image_key}' or '{self.video_key}'."
            )
        return samples

    def _infer_target_dim(self):
        return int(self.samples[0]["__action__"].shape[0])

    def compute_target_stats(self):
        actions = np.stack([sample["__action__"] for sample in self.samples], axis=0).astype(np.float64)
        mean = actions.mean(axis=0)
        std = np.sqrt(np.maximum(actions.var(axis=0), 1e-12))
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def _get_video_capture(self, video_path: str):
        capture = self._video_cache.get(video_path)
        if capture is None:
            capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            self._video_cache[video_path] = capture
            while len(self._video_cache) > self.max_open_videos:
                _, old_capture = self._video_cache.popitem(last=False)
                try:
                    old_capture.release()
                except Exception:
                    pass
        else:
            self._video_cache.move_to_end(video_path)
        return capture

    def _read_video_frame_imageio(self, video_path: str, frame_index: int):
        reader = imageio.get_reader(video_path)
        try:
            frame = reader.get_data(int(frame_index))
        finally:
            reader.close()
        return np.asarray(frame)

    def _read_video_frame(self, video_path: str, frame_index: int):
        try:
            capture = self._get_video_capture(video_path)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            capture.release()
            self._video_cache.pop(video_path, None)
            capture = self._get_video_capture(video_path)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        try:
            return self._read_video_frame_imageio(video_path, frame_index)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode frame {frame_index} from video: {video_path}") from exc

    def __getitem__(self, index):
        sample = self.samples[index]
        if sample["__mode__"] == "image":
            image = Image.open(sample["__media_path__"]).convert("RGB")
            image = np.asarray(image)
        else:
            image = self._read_video_frame(sample["__media_path__"], sample["__frame_index__"])
        target = torch.from_numpy(sample["__action__"].copy())
        return target, image

    def get_metadata(self, index):
        sample = self.samples[index]
        metadata = {
            "row_index": int(sample["__row_index__"]),
            "mode": sample["__mode__"],
            "media_path": sample["__media_path__"],
            "split": sample["__split__"],
        }
        if sample["__mode__"] == "video":
            metadata["frame_index"] = int(sample["__frame_index__"])
            metadata["sample_index_in_video"] = int(sample["__sample_index_in_video__"])
        return metadata

    def __del__(self):
        for capture in self._video_cache.values():
            try:
                capture.release()
            except Exception:
                pass
