# Training Quickstart (Wan2.1 1.3B I2V)

This page contains training commands only.

For dataset preparation and preprocessing, see:
- [Data Guide](./data.md)

All commands use:
`lightewm/wanvideo/model_training/train.py`


## 1) Set variables

```bash
cd /mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/LightEWM

MODEL_PATHS='["checkpoints/Wan2.1-I2V-1.3B/diffusion_pytorch_model.safetensors","checkpoints/Wan2.1-I2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","checkpoints/Wan2.1-I2V-1.3B/Wan2.1_VAE.pth","checkpoints/Wan2.1-I2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]'

DATASET_BASE=data/your_dataset
DATASET_META=data/your_dataset/metadata.csv
```


## 2) I2V full fine-tuning (direct)

```bash
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


## 3) I2V full fine-tuning (from preprocessed cache)

```bash
accelerate launch lightewm/wanvideo/model_training/train.py \
  --dataset_base_path "./data/libero_i2v_train/latent_cache" \
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


## 4) Multi-GPU notes

- Both preprocessing and training support multi-GPU via `accelerate launch`.
- Use your own `accelerate config` / `--config_file` to control process count and strategy.
