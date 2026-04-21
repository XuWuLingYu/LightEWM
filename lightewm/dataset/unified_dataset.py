from .operators import *
import json
import os
import traceback

import pandas
import torch
from tqdm import tqdm


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_from_cache_local_shard = False
        self._effective_cached_data_len = None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        frame_rate=24, fix_frame_rate=False,
        frame_processor=None,
        video_sampling_mode="prefix",
    ):
        if frame_processor is None:
            frame_processor = ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> frame_processor >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=frame_processor,
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=frame_processor,
                    frame_rate=frame_rate, fix_frame_rate=fix_frame_rate,
                    video_sampling_mode=video_sampling_mode,
                )),
            ])),
            (dict, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> frame_processor >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=frame_processor,
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=frame_processor,
                    frame_rate=frame_rate, fix_frame_rate=fix_frame_rate,
                    video_sampling_mode=video_sampling_mode,
                )),
            ])),
        ])
        
    def _distributed_env_rank_and_world_size(self):
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world_size

    def _maybe_local_cache_shard_path(self):
        if self.base_path is None:
            return None
        rank, world_size = self._distributed_env_rank_and_world_size()
        if world_size <= 1:
            return None
        shard_path = os.path.join(self.base_path, str(rank))
        if os.path.isdir(shard_path):
            return shard_path
        return None

    def search_for_cached_data_files(self, path, pbar=None):
        for file_name in os.listdir(path):
            if file_name in {".context_window", "_bad_quarantine"}:
                continue
            if file_name.startswith("."):
                continue
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath, pbar=pbar)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
                if pbar is not None:
                    pbar.update(1)

    def _finalize_cached_data_stats(self, search_root):
        local_count = len(self.cached_data)
        self._effective_cached_data_len = local_count
        rank, world_size = self._distributed_env_rank_and_world_size()

        if self.load_from_cache_local_shard and torch.distributed.is_available() and torch.distributed.is_initialized():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            local_tensor = torch.tensor([local_count], device=device, dtype=torch.long)
            gathered = [torch.zeros_like(local_tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered, local_tensor)
            counts = [int(x.item()) for x in gathered]
            min_count = min(counts) if counts else local_count
            total_count = sum(counts)
            self._effective_cached_data_len = min_count
            print(
                f"[CacheSearch][rank{rank}] shard={search_root} local_files={local_count} "
                f"effective_files={self._effective_cached_data_len}"
            )
            if rank == 0:
                print(
                    f"[CacheSearch] local-shard mode enabled. "
                    f"world_size={world_size}, total_cached_files={total_count}, "
                    f"min_per_rank={min_count}, max_per_rank={max(counts)}"
                )
            return

        print(f"[CacheSearch][rank{rank}] search_root={search_root} cached_files={local_count}")

    def load_metadata(self, metadata_path):
        if metadata_path is None:
            local_shard_path = self._maybe_local_cache_shard_path()
            search_root = local_shard_path if local_shard_path is not None else self.base_path
            self.load_from_cache_local_shard = local_shard_path is not None
            rank, world_size = self._distributed_env_rank_and_world_size()
            if self.load_from_cache_local_shard:
                print(
                    f"No metadata_path. Searching cached data files from local shard only. "
                    f"rank={rank}, world_size={world_size}, shard={search_root}"
                )
            else:
                print("No metadata_path. Searching for cached data files.")
            with tqdm(desc=f"[CacheSearch][rank{rank}]", unit="file", dynamic_ncols=True) as pbar:
                self.search_for_cached_data_files(search_root, pbar=pbar)
            self._finalize_cached_data_stats(search_root)
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def _build_error_item(self, data_id, source_item, exc: Exception):
        return {
            "__load_error__": str(exc),
            "__load_error_type__": type(exc).__name__,
            "__load_error_traceback__": traceback.format_exc(limit=5),
            "__data_id__": int(data_id),
            "__load_from_cache__": bool(self.load_from_cache),
            "__source_item__": source_item,
        }

    def __getitem__(self, data_id):
        try:
            if self.load_from_cache:
                source_item = self.cached_data[data_id % len(self.cached_data)]
                data = self.cached_data_operator(source_item)
            else:
                source_item = self.data[data_id % len(self.data)].copy()
                data = source_item.copy()
                for key in self.data_file_keys:
                    if key in data:
                        if key in self.special_operator_map:
                            data[key] = self.special_operator_map[key](data[key])
                        elif key in self.data_file_keys:
                            data[key] = self.main_data_operator(data[key])
            return data
        except Exception as exc:
            return self._build_error_item(data_id, source_item, exc)

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            cached_len = self._effective_cached_data_len
            if cached_len is None:
                cached_len = len(self.cached_data)
            return cached_len * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
