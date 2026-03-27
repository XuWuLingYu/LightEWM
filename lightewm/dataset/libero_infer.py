import os
import pandas
import imageio
from PIL import Image


class LiberoInferenceDataset:
    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        video_key: str = "video",
        prompt_key: str = "prompt",
        demo_id_key: str = "demo_id",
        camera_key_col: str = "camera_key",
        max_samples: int | None = None,
    ):
        self.base_path = base_path
        self.video_key = video_key
        self.prompt_key = prompt_key
        self.demo_id_key = demo_id_key
        self.camera_key_col = camera_key_col
        metadata = pandas.read_csv(metadata_path)
        if max_samples is not None:
            metadata = metadata.iloc[: int(max_samples)]
        self.rows = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __len__(self):
        return len(self.rows)

    def _resolve_video_path(self, value: str) -> str:
        if os.path.isabs(value):
            return value
        return os.path.join(self.base_path, value)

    def _load_first_frame(self, video_path: str) -> Image.Image:
        reader = imageio.get_reader(video_path)
        try:
            frame = reader.get_data(0)
            return Image.fromarray(frame).convert("RGB")
        finally:
            reader.close()

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        prompt = str(row.get(self.prompt_key, ""))
        rel_video = str(row.get(self.video_key, ""))
        video_path = self._resolve_video_path(rel_video)
        first_frame = self._load_first_frame(video_path)
        demo_id = str(row.get(self.demo_id_key, idx))
        camera_key = str(row.get(self.camera_key_col, "unknown"))
        return {
            "row_id": idx,
            "prompt": prompt,
            "video_path": video_path,
            "input_image": first_frame,
            "demo_id": demo_id,
            "camera_key": camera_key,
        }
