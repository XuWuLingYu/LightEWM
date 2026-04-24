# LIBERO Example

The commands below use 49-frame clips at 10 FPS. During LIBERO conversion, videos are temporally resampled from 16 FPS to 10 FPS.

For Wan2.1 1.3B and Wan2.1/WoW 14B examples, see `README_others.md`.

# Libero Data Download & Process

## 1) Download LIBERO

```bash
cd /root/to/LightEWM/
mkdir -p data
cd data
hf download \
  yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir ./LIBERO-datasets
cd ..
```

## 2) Convert LIBERO to training CSV + videos

```bash
python scripts/convert_libero_to_csv.py \
  --libero-root data/LIBERO-datasets \
  --output-dir data/libero_i2v_train \
  --suites libero_10,libero_90,libero_goal,libero_object,libero_spatial \
  --source-fps 16 \
  --fps 10 \
  --workers 8 \
  --camera-key agentview_rgb,eye_in_hand_rgb \
  --prompt-source attr_or_filename
```

Outputs:
- `data/libero_i2v_train/metadata.csv`
- `data/libero_i2v_train/videos/*.mp4`

## 3) Generate dense prompts

```bash
# export HF_ENDPOINT=https://hf-mirror.com  # if you are in china

bash scripts/process_dense_prompt.sh \
  --metadata-path data/libero_i2v_train/metadata.csv \
  --output-path data/libero_i2v_train/metadata_dense_prompt.csv
```

This creates:
- `data/libero_i2v_train/metadata_dense_prompt.csv`

The generated file is used by default in the downstream steps:
- original sparse text is preserved in `sparse_prompt`
- generated dense text is written to both `dense_prompt` and `prompt`

# Wan 5B TI2V Train & Infer

## 1) Download checkpoint

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir ./checkpoints/Wan2.2-TI2V-5B

# Robot-pretrained DiT init checkpoint (pretrained on broad robot datasets; recommended for finetuning init)
hf download XuWuLingYu/Wan2.2-5B-Robot \
  --local-dir ./checkpoints/Wan2.2-5B-Robot

# LIBERO-pretrained checkpoint (trained on LIBERO; useful for direct LIBERO inference)
hf download XuWuLingYu/Wan2.2-5B-Libero \
  --local-dir ./checkpoints/Wan2.2-5B-Libero
```

## 2) Build latent cache

```bash
bash scripts/process_cache.sh \
  --config examples/LIBERO/cache_ti2v_5b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv \
  --output-path data/libero_i2v_train/latent_cache_ti2v_5b
```

## 3) Train

Optional W&B setup:

```bash
wandb login
# or disable logging
export WANDB_MODE=disabled
```

```bash
bash scripts/train_full.sh \
  --config examples/LIBERO/train_full_ti2v_5b.yaml \
  --dataset-base-path data/libero_i2v_train/latent_cache_ti2v_5b \
  --ckpt checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors
```

For finetuning, starting from `Wan2.2-5B-Robot` is usually better than starting from the original Wan2.2 TI2V 5B checkpoint.

## 4) Infer

```bash
bash scripts/infer.sh \
  --config examples/LIBERO/infer_ti2v_5b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv
```

Optional checkpoint override:

```bash
bash scripts/infer.sh \
  --config examples/LIBERO/infer_ti2v_5b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv \
  --ckpt /path/to/your/ckpt.safetensors
```

If `--ckpt` is not provided, inference uses the official Wan2.2-TI2V-5B checkpoint from the example config.

## 5) Use Our LIBERO Finetuned 5B Checkpoint

If you want to run LIBERO inference with our finetuned checkpoint instead of the original Wan2.2 TI2V 5B DiT, use:

```bash
bash scripts/infer.sh \
  --config examples/LIBERO/infer_ti2v_5b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv \
  --ckpt checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors
```

This checkpoint was finetuned on the full LIBERO dataset for `85000` steps with `lr=1e-5`, using `49` frames at `10 FPS`.
