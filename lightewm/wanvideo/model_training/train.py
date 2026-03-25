import torch, os, argparse, accelerate, warnings, json, math, hashlib, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas
import imageio
from lightewm.diffsynth.core import UnifiedDataset
from lightewm.diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from lightewm.diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from lightewm.diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS for loading videos in data-process stage. Leave empty to keep source FPS.",
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="stretch",
        choices=["stretch", "letterbox"],
        help="Frame resize mode in data-process stage. `stretch`: direct resize. `letterbox`: keep aspect ratio with black padding.",
    )
    parser.add_argument(
        "--context_window_short_video_mode",
        type=str,
        default="drop",
        choices=["drop", "repeat_last_frame"],
        help="Data-process stage only. When video length < num_frames: `drop` or `repeat_last_frame`.",
    )
    parser.add_argument(
        "--context_window_stride",
        type=int,
        default=81,
        help="Data-process stage only. Sliding window stride in frames. Default: 81.",
    )
    parser.add_argument(
        "--context_window_tail_align",
        default=False,
        action="store_true",
        help="Data-process stage only. Append one extra tail-aligned window if the tail is not covered by regular windows.",
    )
    parser.add_argument(
        "--context_window_wait_timeout",
        type=int,
        default=7200,
        help="Non-zero ranks wait this many seconds for rank-0 context metadata generation. Use <=0 to wait indefinitely.",
    )
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    return parser


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


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
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
