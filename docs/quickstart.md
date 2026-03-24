# Training Quickstart (Wan2.1 I2V)

This page contains training commands only.

For dataset preparation and preprocessing, see:
- [Data Guide](./data.md)

All commands use:
`lightewm/wanvideo/model_training/train.py`


## 1) Set variables

```bash
cd /mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/LightEWM

ENV_PREFIX=/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/lightewm

MODEL_PATHS='["checkpoints/Wan2.1-I2V-1.3B/diffusion_pytorch_model.safetensors","checkpoints/Wan2.1-I2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","checkpoints/Wan2.1-I2V-1.3B/Wan2.1_VAE.pth","checkpoints/Wan2.1-I2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]'

DATASET_BASE=data/your_dataset
DATASET_META=data/your_dataset/metadata.csv
```


## 2) Full fine-tuning (direct)

```bash
conda run -p "$ENV_PREFIX" \
accelerate launch lightewm/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_BASE" \
  --dataset_metadata_path "$DATASET_META" \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_paths "$MODEL_PATHS" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-I2V-1.3B_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image,end_image" \
  --use_gradient_checkpointing
```


## 3) LoRA fine-tuning (direct)

```bash
conda run -p "$ENV_PREFIX" \
accelerate launch lightewm/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_BASE" \
  --dataset_metadata_path "$DATASET_META" \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_paths "$MODEL_PATHS" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-I2V-1.3B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,end_image" \
  --use_gradient_checkpointing
```


## 4) Full fine-tuning (from preprocessed cache)

```bash
conda run -p "$ENV_PREFIX" \
accelerate launch lightewm/wanvideo/model_training/train.py \
  --dataset_base_path "./models/cache/Wan2.1-I2V-1.3B_full_cache" \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_paths "$MODEL_PATHS" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-I2V-1.3B_full_cached" \
  --trainable_models "dit" \
  --task "sft:train" \
  --use_gradient_checkpointing
```


## 5) LoRA fine-tuning (from preprocessed cache)

```bash
conda run -p "$ENV_PREFIX" \
accelerate launch lightewm/wanvideo/model_training/train.py \
  --dataset_base_path "./models/cache/Wan2.1-I2V-1.3B_lora_cache" \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_paths "$MODEL_PATHS" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-I2V-1.3B_lora_cached" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --task "sft:train" \
  --use_gradient_checkpointing
```


## 6) Multi-GPU notes

- Both preprocessing and training support multi-GPU via `accelerate launch`.
- Use your own `accelerate config` / `--config_file` to control process count and strategy.
