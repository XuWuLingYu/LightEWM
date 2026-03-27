import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import accelerate
import imageio
import pandas

from lightewm.runner.loops import launch_data_process_task
from lightewm.runner.runner_util.wan_runtime import (
    build_wan_i2v_pipeline_from_params,
    build_wan_i2v_runtime_args,
    build_wan_training_dataset,
)
from lightewm.utils.logger import ModelLogger

from .wan_training import WanTrainingModule


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
    hash_key = json.dumps(
        {
            "metadata": args.dataset_metadata_path,
            "base": args.dataset_base_path,
            "num_frames": args.num_frames,
            "stride": args.context_window_stride,
            "tail_align": args.context_window_tail_align,
            "short_mode": args.context_window_short_video_mode,
            "fps": args.fps,
        },
        sort_keys=True,
    )
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


class WanCacheRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        args = build_wan_i2v_runtime_args(self.config.full_config, force_task="sft:data_process")
        args.dataset_metadata_path = build_context_window_metadata(args)

        accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=[
                accelerate.DistributedDataParallelKwargs(
                    find_unused_parameters=args.find_unused_parameters
                )
            ],
        )
        dataset = build_wan_training_dataset(args, use_data_process_controls=True)
        pipe = build_wan_i2v_pipeline_from_params(
            {
                "model_paths": args.model_paths,
                "model_id_with_origin_paths": args.model_id_with_origin_paths,
                "tokenizer_path": args.tokenizer_path,
                "audio_processor_path": args.audio_processor_path,
                "fp8_models": args.fp8_models,
                "offload_models": args.offload_models,
                "device": "cpu" if args.initialize_model_on_cpu else accelerator.device,
                "torch_dtype": "bfloat16",
            },
            device_override="cpu" if args.initialize_model_on_cpu else accelerator.device,
        )
        model = WanTrainingModule(
            pipe=pipe,
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
            task=args.task,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
        )
        model_logger = ModelLogger(
            args.output_path,
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        )
        launch_data_process_task(accelerator, dataset, model, model_logger, args=args)
