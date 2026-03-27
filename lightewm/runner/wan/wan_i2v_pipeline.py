import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import accelerate
import imageio
import pandas
import torch
from pathlib import Path
from tqdm import tqdm

from lightewm.dataset import UnifiedDataset
from lightewm.dataset.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from lightewm.model.wan.training_module import WanTrainingModule
from lightewm.runner.pipeline_factory import (
    flatten_config_params,
    instantiate_component_from_section,
)
from lightewm.runner.loops import launch_training_task, launch_data_process_task
from lightewm.utils.data import save_video
from lightewm.utils.logger import ModelLogger
from lightewm.utils.parsers import build_wan_i2v_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"

WAN_I2V_DEFAULTS = {
    "dataset_base_path": "",
    "dataset_metadata_path": None,
    "dataset_repeat": 1,
    "dataset_num_workers": 0,
    "data_file_keys": "image,video",
    "model_paths": None,
    "model_id_with_origin_paths": None,
    "extra_inputs": None,
    "fp8_models": None,
    "offload_models": None,
    "learning_rate": 1e-4,
    "num_epochs": 1,
    "batch_size": 1,
    "trainable_models": None,
    "find_unused_parameters": False,
    "weight_decay": 0.01,
    "task": "sft",
    "output_path": "./logs",
    "remove_prefix_in_ckpt": "pipe.dit.",
    "save_steps": None,
    "lora_base_model": None,
    "lora_target_modules": "q,k,v,o,ffn.0,ffn.2",
    "lora_rank": 32,
    "lora_checkpoint": None,
    "preset_lora_path": None,
    "preset_lora_model": None,
    "use_gradient_checkpointing": False,
    "use_gradient_checkpointing_offload": False,
    "gradient_accumulation_steps": 1,
    "height": None,
    "width": None,
    "max_pixels": 1024 * 1024,
    "num_frames": 81,
    "fps": None,
    "resize_mode": "stretch",
    "context_window_short_video_mode": "drop",
    "context_window_stride": 81,
    "context_window_tail_align": False,
    "context_window_wait_timeout": 7200,
    "tokenizer_path": None,
    "audio_processor_path": None,
    "max_timestep_boundary": 1.0,
    "min_timestep_boundary": 0.0,
    "initialize_model_on_cpu": False,
    "framewise_decoding": False,
    "wandb_enabled": False,
    "wandb_project": "LightEWM",
    "wandb_run_name": None,
    "wandb_mode": "online",
    "wandb_log_every": 10,
}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_runtime_types(runtime: dict):
    int_keys = {
        "dataset_repeat", "dataset_num_workers", "num_epochs", "batch_size", "save_steps", "lora_rank",
        "gradient_accumulation_steps", "height", "width", "max_pixels", "num_frames",
        "context_window_stride", "context_window_wait_timeout", "wandb_log_every",
    }
    float_keys = {"learning_rate", "weight_decay", "fps", "max_timestep_boundary", "min_timestep_boundary"}
    bool_keys = {
        "find_unused_parameters", "use_gradient_checkpointing", "use_gradient_checkpointing_offload",
        "context_window_tail_align", "initialize_model_on_cpu", "framewise_decoding", "wandb_enabled",
    }

    for key in int_keys:
        if runtime.get(key) is not None:
            runtime[key] = int(runtime[key])
    for key in float_keys:
        if runtime.get(key) is not None:
            runtime[key] = float(runtime[key])
    for key in bool_keys:
        if runtime.get(key) is not None:
            runtime[key] = _as_bool(runtime[key])
    return runtime


def build_wan_i2v_runtime_args(cfg: dict, force_task: str | None = None):
    params = flatten_config_params(cfg)
    runtime = dict(WAN_I2V_DEFAULTS)
    runtime.update(params)
    runtime = _coerce_runtime_types(runtime)
    if isinstance(runtime.get("model_paths"), list):
        runtime["model_paths"] = json.dumps(runtime["model_paths"])
    if force_task is not None:
        runtime["task"] = force_task
    return SimpleNamespace(**runtime)


def wan_parser():
    return build_wan_i2v_parser()


def load_metadata_as_records(metadata_path):
    if metadata_path.endswith(".json"):
        with open(metadata_path, "r") as f:
            return json.load(f)
    if metadata_path.endswith(".jsonl"):
        records = []
        with open(metadata_path, "r") as f:
            for line in f:
                records.append(json.loads(line.strip()))
        return records
    metadata = pandas.read_csv(metadata_path)
    return [metadata.iloc[i].to_dict() for i in range(len(metadata))]


def count_available_video_frames(video_path, target_fps=None):
    reader = imageio.get_reader(video_path)
    try:
        try:
            total_original_frames = int(reader.count_frames())
        except Exception:
            try:
                length = reader.get_length()
                if length is None or length == float("inf") or length < 0:
                    raise ValueError("invalid length")
                total_original_frames = int(length)
            except Exception:
                total_original_frames = sum(1 for _ in imageio.v3.imiter(video_path))
        if target_fps is None:
            return max(total_original_frames, 0)
        meta_data = reader.get_meta_data()
        if "duration" in meta_data and meta_data["duration"] is not None:
            duration = float(meta_data["duration"])
        else:
            raw_fps = meta_data.get("fps", 0)
            duration = total_original_frames / raw_fps if raw_fps else 0
        total_available_frames = int(math.floor(duration * target_fps))
        return max(total_available_frames, 0)
    finally:
        reader.close()


def build_context_window_metadata(args):
    if args.dataset_metadata_path is None:
        return None
    if args.context_window_stride <= 0:
        raise ValueError("--context_window_stride must be a positive integer.")

    context_dir = os.path.join(args.output_path, ".context_window")
    os.makedirs(context_dir, exist_ok=True)
    hash_key = json.dumps({
        "metadata": args.dataset_metadata_path,
        "base": args.dataset_base_path,
        "num_frames": args.num_frames,
        "stride": args.context_window_stride,
        "tail_align": args.context_window_tail_align,
        "short_mode": args.context_window_short_video_mode,
        "fps": args.fps,
    }, sort_keys=True)
    file_name = f"context_metadata_{hashlib.md5(hash_key.encode('utf-8')).hexdigest()[:12]}.json"
    metadata_out = os.path.join(context_dir, file_name)

    if os.path.exists(metadata_out):
        print(f"[ContextWindow] Reusing expanded metadata: {metadata_out}")
        return metadata_out

    local_rank = os.environ.get("LOCAL_RANK", "0")
    if local_rank not in ("0", "-1"):
        timeout = int(getattr(args, "context_window_wait_timeout", 7200))
        start_time = time.time()
        while True:
            if os.path.exists(metadata_out):
                print(f"[ContextWindow] Reusing expanded metadata: {metadata_out}")
                return metadata_out
            if timeout > 0 and (time.time() - start_time) > timeout:
                raise RuntimeError(
                    f"[ContextWindow] Timed out waiting for rank-0 metadata generation after {timeout}s: {metadata_out}"
                )
            time.sleep(1)

    records = load_metadata_as_records(args.dataset_metadata_path)
    print(f"[ContextWindow] Expanding metadata from {args.dataset_metadata_path}, records={len(records)}")
    window_size = int(args.num_frames)
    stride = int(args.context_window_stride)
    expanded_records = []
    dropped_short = 0
    dropped_invalid = 0
    used_meta_num_frames = 0
    progress_start_time = time.time()

    frame_counts = {}
    count_jobs = []
    for record_id, record in enumerate(records):
        video_path = record.get("video", None)
        if not isinstance(video_path, str):
            continue

        meta_num_frames = record.get("num_frames", None)
        if meta_num_frames is not None and str(meta_num_frames).strip() != "":
            try:
                frame_counts[record_id] = max(int(float(meta_num_frames)), 0)
                used_meta_num_frames += 1
                continue
            except Exception:
                pass

        absolute_video_path = os.path.join(args.dataset_base_path, video_path)
        count_jobs.append((record_id, absolute_video_path))

    if len(count_jobs) > 0:
        num_workers = max(int(os.cpu_count() or 1), 1)
        print(f"[ContextWindow] num_frames missing for {len(count_jobs)} records, counting frames with {num_workers} threads")

        def _count_frames(job):
            record_id_, absolute_video_path_ = job
            try:
                n = count_available_video_frames(absolute_video_path_, target_fps=args.fps)
                return record_id_, n, None, absolute_video_path_
            except Exception as e:
                return record_id_, -1, str(e), absolute_video_path_

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_count_frames, job) for job in count_jobs]
            for done_id, future in enumerate(as_completed(futures), start=1):
                record_id_, total_frames_, err_msg, absolute_video_path_ = future.result()
                frame_counts[record_id_] = total_frames_
                if err_msg is not None:
                    print(f"[ContextWindow] Failed to read video '{absolute_video_path_}', drop this item. Error: {err_msg}")
                if done_id % 500 == 0 or done_id == len(count_jobs):
                    print(f"[ContextWindow] FrameCount progress {done_id}/{len(count_jobs)}")

    for record_id, record in enumerate(records):
        video_path = record.get("video", None)
        if not isinstance(video_path, str):
            expanded_records.append(record)
            continue

        total_frames = frame_counts.get(record_id, -1)
        if total_frames < 0:
            dropped_invalid += 1
            continue

        if total_frames < window_size:
            if args.context_window_short_video_mode == "drop":
                dropped_short += 1
                continue
            starts = [0]
            pad_last = True
        else:
            starts = list(range(0, total_frames - window_size + 1, stride))
            if args.context_window_tail_align:
                tail_start = total_frames - window_size
                if len(starts) == 0 or starts[-1] != tail_start:
                    starts.append(tail_start)
            pad_last = False

        for context_start in starts:
            expanded = record.copy()
            expanded["video"] = {
                "path": video_path,
                "context_start": int(context_start),
                "context_window_size": window_size,
                "pad_last": bool(pad_last),
            }
            expanded_records.append(expanded)

        if (record_id + 1) % 500 == 0 or (record_id + 1) == len(records):
            elapsed = max(time.time() - progress_start_time, 1e-6)
            speed = (record_id + 1) / elapsed
            print(f"[ContextWindow] Progress {record_id + 1}/{len(records)} (~{speed:.1f} records/s)")

    temp_out = metadata_out + f".tmp.{os.getpid()}"
    with open(temp_out, "w") as f:
        json.dump(expanded_records, f)
    os.replace(temp_out, metadata_out)
    print(
        f"[ContextWindow] Input samples: {len(records)}, output windows: {len(expanded_records)}, "
        f"dropped_short: {dropped_short}, dropped_invalid: {dropped_invalid}, "
        f"used_meta_num_frames: {used_meta_num_frames}. "
        f"Saved expanded metadata to: {metadata_out}"
    )
    return metadata_out


def execute_wan_task(args):
    use_data_process_controls = args.task.endswith(":data_process")
    if use_data_process_controls:
        args.dataset_metadata_path = build_context_window_metadata(args)
        target_fps = args.fps if args.fps is not None else 24
        fix_frame_rate = args.fps is not None
        frame_processor = ImageCropAndResize(
            args.height, args.width, args.max_pixels, 16, 16, resize_mode=args.resize_mode
        )
    else:
        target_fps = 24
        fix_frame_rate = False
        frame_processor = None

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
            frame_rate=target_fps,
            fix_frame_rate=fix_frame_rate,
            frame_processor=frame_processor,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(
                args.num_frames, 4, 1,
                frame_processor=ImageCropAndResize(
                    512, 512, None, 16, 16,
                    resize_mode=args.resize_mode if use_data_process_controls else "crop",
                ),
                frame_rate=target_fps, fix_frame_rate=fix_frame_rate,
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


class WanCacheRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        args = build_wan_i2v_runtime_args(self.config.full_config, force_task="sft:data_process")
        execute_wan_task(args)


class WanTrainRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        args = build_wan_i2v_runtime_args(self.config.full_config)
        if not args.task:
            args.task = "sft"
        execute_wan_task(args)


class WanInferRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        full_config = self.config.full_config
        model, _ = instantiate_component_from_section(full_config.model, full_config, section_name="model")
        dataset, _ = instantiate_component_from_section(full_config.dataset, full_config, section_name="dataset")

        output_dir = getattr(self.config, "output_dir", "./outputs/libero_infer")
        os.makedirs(output_dir, exist_ok=True)

        fps = int(getattr(self.config, "fps", 16))
        quality = int(getattr(self.config, "quality", 5))
        seed_base = int(getattr(self.config, "seed", 0))
        infer_kwargs = dict(getattr(self.config, "infer_kwargs", {}))
        input_image_resize_mode = getattr(self.config, "input_image_resize_mode", "stretch")
        target_height = infer_kwargs.get("height", None)
        target_width = infer_kwargs.get("width", None)
        input_image_resizer = None
        if target_height is not None and target_width is not None:
            input_image_resizer = ImageCropAndResize(
                height=int(target_height),
                width=int(target_width),
                max_pixels=None,
                height_division_factor=16,
                width_division_factor=16,
                resize_mode=input_image_resize_mode,
            )

        for item in tqdm(dataset, total=len(dataset), desc="Infer"):
            input_image = item["input_image"]
            if input_image_resizer is not None:
                input_image = input_image_resizer(input_image)
            video = model(
                prompt=item["prompt"],
                input_image=input_image,
                seed=seed_base + int(item["row_id"]),
                **infer_kwargs,
            )
            name = f"{item['row_id']:06d}__{item['demo_id']}__{item['camera_key']}.mp4"
            save_path = str(Path(output_dir) / name)
            save_video(video, save_path, fps=fps, quality=quality)


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    execute_wan_task(args)
