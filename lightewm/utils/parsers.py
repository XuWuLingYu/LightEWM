import argparse


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    return parser

def add_image_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    return parser

def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    return parser

def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--fp8_models", default=None, help="Models with FP8 precision, comma-separated.")
    parser.add_argument("--offload_models", default=None, help="Models with offload, comma-separated. Only used in splited training.")
    return parser

def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-process training batch size.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    return parser

def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    return parser

def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--preset_lora_path", type=str, default=None, help="Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    parser.add_argument("--preset_lora_model", type=str, default=None, help="Which model the preset LoRA is fused to.")
    return parser

def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser

def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    return parser


def build_wan_i2v_parser():
    parser = argparse.ArgumentParser(description="WAN I2V training/data-process parser.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS for loading videos in data-process stage. Leave empty to keep source FPS.",
    )
    parser.add_argument(
        "--video_sampling_mode",
        type=str,
        default="prefix",
        choices=["prefix", "uniform_full_video"],
        help="Video sampling mode in data-process stage. `prefix` keeps the current prefix/context-window behavior. "
             "`uniform_full_video` samples each whole video uniformly to exactly num_frames and skips sliding windows.",
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
    return parser
