# LIBERO Example

This example shows the full Wan2.1-1.3B I2V workflow on LIBERO:
- dataset download
- HDF5 to video/CSV conversion
- dense-prompt generation
- latent-cache preprocessing
- full training
- batch inference

It also includes a separate Wan2.2-TI2V-5B cache/train/infer path. The 5B TI2V cache is not compatible with the 1.3B I2V cache because the pipeline, VAE, and latent shapes are different.

The commands below use 49-frame clips at 10 FPS. During LIBERO conversion, videos are temporally resampled from 16 FPS to 10 FPS.

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

## 2) Download checkpoint

```bash
hf download alibaba-pai/Wan2.1-Fun-1.3B-InP \
  --local-dir ./checkpoints/Wan2.1-I2V-1.3B
```

## 3) Convert LIBERO to training CSV + videos

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

## 4) Generate dense prompts

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

## 5) Build latent cache

```bash
bash scripts/process_cache.sh \
  --config examples/LIBERO/cache.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv \
  --output-path data/libero_i2v_train/latent_cache
```

## 6) Train

Optional W&B setup:

```bash
wandb login
# or disable logging
export WANDB_MODE=disabled
```

```bash
bash scripts/train_full.sh \
  --config examples/LIBERO/train_full.yaml \
  --dataset-base-path data/libero_i2v_train/latent_cache
```

## 7) Infer

```bash
bash scripts/infer.sh \
  --config examples/LIBERO/infer.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv
```

Optional checkpoint override:

```bash
bash scripts/infer.sh \
  --config examples/LIBERO/infer.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv \
  --ckpt /path/to/your/ckpt.safetensors
```

If `--ckpt` is not provided, inference uses the official Wan2.1-Fun-1.3B I2V checkpoint from the example config.

## Wan2.2-TI2V-5B on LIBERO

Use the same converted LIBERO videos and prompts, but switch to the TI2V-5B configs below.

### 5b) Build TI2V-5B latent cache

```bash
bash scripts/process_cache.sh \
  --config examples/LIBERO/cache_ti2v_5b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv \
  --output-path data/libero_i2v_train/latent_cache_ti2v_5b
```

### 6b) Train TI2V-5B

```bash
bash scripts/train_full.sh \
  --config examples/LIBERO/train_full_ti2v_5b.yaml \
  --dataset-base-path data/libero_i2v_train/latent_cache_ti2v_5b
```

### 7b) Infer with TI2V-5B

```bash
bash scripts/infer.sh \
  --config examples/LIBERO/infer_ti2v_5b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv
```
