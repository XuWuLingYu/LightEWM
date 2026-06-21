import argparse
import datetime
import torch
import os
import time
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, SequentialSampler
import imageio.v3 as iio

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset, TextVideoDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


def write_video(path, video, fps=16):
    """Write uint8 THWC video without relying on torchvision.io.write_video."""
    array = video.detach().cpu().numpy() if torch.is_tensor(video) else video
    iio.imwrite(path, array, fps=fps)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--detail-log", action="store_true", help="Save per-layer per-step x0_pred videos for vertical inference")
parser.add_argument(
    "--sampling_steps",
    type=int,
    default=0,
    help="Override total diffusion sampling steps for inference (<=0 keeps config/default).",
)
parser.add_argument(
    "--vertical_infer_fixed_denoise_steps",
    type=int,
    default=-1,
    help="Override fixed per-token denoise steps in vertical inference (<0 keeps config/default).",
)
parser.add_argument(
    "--vertical_infer_preserve_budget_ratio",
    action="store_true",
    help="When overriding fixed vertical denoise steps, preserve per-level budget ratio.",
)
parser.add_argument(
    "--vertical_infer_reference_total_steps",
    type=int,
    default=0,
    help="Reference total diffusion steps for ratio-preserving vertical inference (<=0 keeps config/default).",
)
args = parser.parse_args()


class ShardedSequentialSampler(Sampler[int]):
    def __init__(self, dataset, num_replicas: int, rank: int):
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}.")
        self.indices = list(range(rank, len(dataset), num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _sanitize_filename(text: str, max_len: int = 100) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in (" ", "-", "_") else "_"
        for ch in text
    ).strip()
    safe = "_".join(safe.split())
    return safe[:max_len] if safe else "sample"


def _build_output_path(output_folder: str, prompt: str, idx: int) -> str:
    filename = f"{idx:05d}_{_sanitize_filename(prompt)}.mp4"
    return os.path.join(output_folder, filename)


def _build_vertical_layer_output_dir(output_path: str) -> str:
    root, _ = os.path.splitext(output_path)
    return f"{root}_layers"


def _make_noise(latent_shape, num_frames, device, dtype):
    if num_frames <= 0:
        raise ValueError(f"Noise must contain at least one latent frame, got {num_frames}.")
    return torch.randn(
        [latent_shape[0], num_frames, latent_shape[2], latent_shape[3], latent_shape[4]],
        device=device,
        dtype=dtype,
    )


def _rgb_frames_to_latent_frames(num_rgb_frames: int) -> int:
    if num_rgb_frames <= 0:
        raise ValueError(f"num_rgb_frames must be positive, got {num_rgb_frames}.")
    padded_rgb_frames = num_rgb_frames
    remainder = (padded_rgb_frames - 1) % 4
    if remainder != 0:
        padded_rgb_frames += 4 - remainder
    return ((padded_rgb_frames - 1) // 4) + 1

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(
        backend='nccl',
        timeout=datetime.timedelta(minutes=30),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

else:
    device = torch.device("cuda")
    local_rank = 0
    global_rank = 0
    world_size = 1

set_seed(args.seed)

print(f'Rank {global_rank} free VRAM on {device}: {get_cuda_free_memory_gb(device)} GB')

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
if args.sampling_steps > 0:
    config.sampling_steps = int(args.sampling_steps)
if args.vertical_infer_fixed_denoise_steps >= 0:
    config.vertical_infer_fixed_denoise_steps = int(args.vertical_infer_fixed_denoise_steps)
if args.vertical_infer_preserve_budget_ratio:
    config.vertical_infer_preserve_budget_ratio = True
if args.vertical_infer_reference_total_steps > 0:
    config.vertical_infer_reference_total_steps = int(args.vertical_infer_reference_total_steps)
data_path = args.data_path or config.data_path
low_memory = get_cuda_free_memory_gb(device) < getattr(config, "low_memory_threshold_gb", 40)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = 'generator_ema' if args.use_ema else 'generator'
    gen_sd = state_dict[key]

    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)


# Create dataset
data_backend = getattr(config, "data_backend", "lmdb_latent")
use_video_metadata = data_backend == "jsonl_video"
if use_video_metadata:
    dataset = TextVideoDataset(
        metadata_path=data_path,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        variable_num_frames=getattr(config, "variable_num_frames_infer", False),
        max_num_frames=(
            4 * (getattr(config, "max_inference_latent_frames", 64) - 1) + 1
            if getattr(config, "variable_num_frames_infer", False)
            else None
        ),
    )
elif args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = ShardedSequentialSampler(dataset, num_replicas=world_size, rank=global_rank)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

print(f"Rank {global_rank}: processing {len(sampler)} samples")

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

latent_shape = list(config.image_or_video_shape)
latent_shape[0] = 1
vertical_hierarchy = getattr(config, "vertical_hierarchy", False)
dynamic_vertical_hierarchy = getattr(config, "dynamic_vertical_hierarchy", False)
variable_num_frames_infer = getattr(config, "variable_num_frames_infer", False)
max_inference_latent_frames = getattr(config, "max_inference_latent_frames", 64)
vertical_leaf_frames = getattr(config, "vertical_leaf_frames", latent_shape[1])
vertical_total_tokens = (
    pipeline.vertical_info["num_tokens"] if vertical_hierarchy and pipeline.vertical_info is not None else latent_shape[1]
)
if vertical_hierarchy and not variable_num_frames_infer:
    if dynamic_vertical_hierarchy:
        pipeline.configure_vertical_runtime(args.num_output_frames)
        vertical_leaf_frames = pipeline.vertical_leaf_frames
        vertical_total_tokens = pipeline.vertical_info["num_tokens"]
    else:
        if args.num_output_frames != vertical_leaf_frames:
            raise ValueError(
                f"Vertical inference expects num_output_frames={vertical_leaf_frames}, got {args.num_output_frames}."
            )
else:
    if args.num_output_frames > latent_shape[1]:
        local_attn_size = getattr(pipeline, "local_attn_size", -1)
        if local_attn_size == -1:
            raise ValueError(
                "num_output_frames exceeds configured latent frames, but local attention is disabled. "
                f"Got num_output_frames={args.num_output_frames}, configured latent frames={latent_shape[1]}. "
                "Set model_kwargs.local_attn_size (e.g., 21) to enable KV cache queue-pop long generation."
            )

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    prompt = batch['prompts'][0]
    output_path = _build_output_path(args.output_folder, prompt, idx)
    if os.path.exists(output_path):
        print('Video has been generated. Pass!')
        continue

    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    prompts = [extended_prompt] if extended_prompt is not None else [prompt]

    initial_pixels = None
    if use_video_metadata:
        assert config.num_frame_per_block == 1, "Frame-level TI2V inference expects num_frame_per_block == 1."
        initial_pixels = batch["frames"][:, :, :1]
    elif args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        initial_pixels = batch["image"].unsqueeze(2)
    elif getattr(config, "condition_first_frame", False):
        raise ValueError("condition_first_frame inference requires either --i2v or data_backend=jsonl_video.")

    current_output_latent_frames = args.num_output_frames
    if use_video_metadata and variable_num_frames_infer:
        current_output_latent_frames = min(
            max_inference_latent_frames,
            _rgb_frames_to_latent_frames(int(batch["num_frames"].item())),
        )
        if not vertical_hierarchy or not dynamic_vertical_hierarchy:
            raise ValueError(
                "variable_num_frames_infer currently requires dynamic vertical inference with jsonl_video input."
            )
        pipeline.configure_vertical_runtime(current_output_latent_frames)
        vertical_leaf_frames = pipeline.vertical_leaf_frames
        vertical_total_tokens = pipeline.vertical_info["num_tokens"]

    if initial_pixels is not None:
        initial_pixels = initial_pixels.to(device=device, dtype=torch.bfloat16)
        initial_latent = pipeline.vae.encode_to_latent(initial_pixels).to(device=device, dtype=torch.bfloat16)
        if vertical_hierarchy:
            noise_num_frames = vertical_total_tokens
        else:
            noise_num_frames = current_output_latent_frames - initial_latent.shape[1]
    else:
        initial_latent = None
        noise_num_frames = vertical_total_tokens if vertical_hierarchy else current_output_latent_frames

    sampled_noise = _make_noise(
        latent_shape=latent_shape,
        num_frames=noise_num_frames,
        device=device,
        dtype=torch.bfloat16,
    )

    # Generate 81 frames
    case_start = time.perf_counter()
    if vertical_hierarchy:
        inference_output = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=False,
            return_vertical_layer_videos=True,
            return_vertical_detail_logs=args.detail_log,
            initial_latent=initial_latent
        )
    else:
        inference_output = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=False,
            initial_latent=initial_latent
        )
    vertical_payload = None
    if vertical_hierarchy:
        video, vertical_payload = inference_output
    else:
        video = inference_output
    inference_seconds = time.perf_counter() - case_start
    video = rearrange(video, 'b t c h w -> b t h w c').cpu()

    # Final output video
    video = (255.0 * video).clamp_(0, 255).to(torch.uint8)

    if vertical_payload is not None:
        layer_output_dir = _build_vertical_layer_output_dir(output_path)
        os.makedirs(layer_output_dir, exist_ok=True)
        for level_index, (level_size, layer_video) in enumerate(
            zip(vertical_payload["level_sizes"], vertical_payload["layer_videos"])
        ):
            layer_video = rearrange(layer_video, 'b t c h w -> b t h w c').cpu()
            layer_video = (255.0 * layer_video).clamp_(0, 255).to(torch.uint8)
            layer_output_path = os.path.join(
                layer_output_dir,
                f"layer_{level_index + 1:02d}_{level_size:02d}_latents.mp4",
            )
            write_video(layer_output_path, layer_video[0], fps=16)

        detail_layer_videos = vertical_payload.get("detail_layer_step_videos")
        if detail_layer_videos is not None:
            detail_output_dir = os.path.join(layer_output_dir, "detail")
            os.makedirs(detail_output_dir, exist_ok=True)
            for level_index, step_videos in enumerate(detail_layer_videos):
                level_size = vertical_payload["level_sizes"][level_index]
                for step_index, step_video in enumerate(step_videos):
                    step_video = rearrange(step_video, 'b t c h w -> b t h w c').cpu()
                    step_video = (255.0 * step_video).clamp_(0, 255).to(torch.uint8)
                    step_output_path = os.path.join(
                        detail_output_dir,
                        f"layer_{level_index + 1:02d}_{level_size:02d}_latents_step_{step_index + 1:02d}.mp4",
                    )
                    write_video(step_output_path, step_video[0], fps=16)

    # Clear VAE cache
    pipeline.vae.clear_cache()

    write_video(output_path, video[0], fps=16)
    total_seconds = time.perf_counter() - case_start
    if local_rank == 0:
        print(
            f"[CausalTiming] idx={idx} "
            f"inference_seconds={inference_seconds:.6f} "
            f"total_seconds={total_seconds:.6f} "
            f"output={output_path}",
            flush=True,
        )
