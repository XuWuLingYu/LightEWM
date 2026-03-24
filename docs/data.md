# Data Guide (Wan2.1 I2V)

This page covers dataset preparation and optional preprocessing.

All commands use:
`lightewm/wanvideo/model_training/train.py`


## 1) Download checkpoint

```bash
# Optional mirror:
# export HF_ENDPOINT=https://hf-mirror.com

hf download alibaba-pai/Wan2.1-Fun-1.3B-InP \
  --local-dir ./checkpoints/Wan2.1-I2V-1.3B
```


## 2) Dataset format

Folder layout:

```text
data/your_dataset/
├── metadata.csv
└── videos/
    ├── 000001.mp4
    ├── 000002.mp4
    └── ...
```

`metadata.csv` minimum columns:
- `video`
- `prompt`

Example:

```csv
video,prompt
videos/000001.mp4,"a robot arm picks up a red cube from the table"
videos/000002.mp4,"open the drawer and place the object inside"
```

Notes:
- `video` is relative to `--dataset_base_path`.
- For Wan2.1-Fun-1.3B-InP training, use `--extra_inputs "input_image,end_image"`.
  The trainer will take first/last frame automatically from each video clip.


## 3) Preprocessing options

These options are applied in preprocessing task (`--task sft:data_process`):

- `--fps`: optional target FPS. Leave empty to keep source FPS.
- `--resize_mode stretch`: direct resize to `(height, width)`.
- `--resize_mode letterbox`: keep aspect ratio, pad black bars.
- `--context_window_short_video_mode drop`: drop short videos (`len < num_frames`).
- `--context_window_short_video_mode repeat_last_frame`: keep short videos by tail-frame padding.
- `--context_window_stride N`: windows start from `0, N, 2N, ...`.
- `--context_window_tail_align`: add one extra tail window if regular windows do not cover the tail.


## 4) Preprocess to cache

Set variables:

```bash
cd /mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/LightEWM

ENV_PREFIX=/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/lightewm

MODEL_PATHS='["checkpoints/Wan2.1-I2V-1.3B/diffusion_pytorch_model.safetensors","checkpoints/Wan2.1-I2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","checkpoints/Wan2.1-I2V-1.3B/Wan2.1_VAE.pth","checkpoints/Wan2.1-I2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]'

DATASET_BASE=data/your_dataset
DATASET_META=data/your_dataset/metadata.csv

# optional
FPS=
RESIZE_MODE=letterbox
CONTEXT_SHORT_MODE=drop
CONTEXT_STRIDE=81
CONTEXT_TAIL_ALIGN=
```

### 4.1 Full cache

```bash
conda run -p "$ENV_PREFIX" \
accelerate launch lightewm/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_BASE" \
  --dataset_metadata_path "$DATASET_META" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_paths "$MODEL_PATHS" \
  --output_path "./models/cache/Wan2.1-I2V-1.3B_full_cache" \
  --trainable_models "dit" \
  --extra_inputs "input_image,end_image" \
  --resize_mode "$RESIZE_MODE" \
  --context_window_short_video_mode "$CONTEXT_SHORT_MODE" \
  --context_window_stride "$CONTEXT_STRIDE" \
  $CONTEXT_TAIL_ALIGN \
  ${FPS:+--fps "$FPS"} \
  --task "sft:data_process"
```

### 4.2 LoRA cache

```bash
conda run -p "$ENV_PREFIX" \
accelerate launch lightewm/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_BASE" \
  --dataset_metadata_path "$DATASET_META" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_paths "$MODEL_PATHS" \
  --output_path "./models/cache/Wan2.1-I2V-1.3B_lora_cache" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,end_image" \
  --resize_mode "$RESIZE_MODE" \
  --context_window_short_video_mode "$CONTEXT_SHORT_MODE" \
  --context_window_stride "$CONTEXT_STRIDE" \
  $CONTEXT_TAIL_ALIGN \
  ${FPS:+--fps "$FPS"} \
  --task "sft:data_process"
```


## 5) Multi-GPU notes

- Preprocessing supports multi-GPU via `accelerate launch`.
- Cache files are written per process and then consumed by `--task sft:train`.
- Use your own `accelerate config` / `--config_file` to control distributed setup.
