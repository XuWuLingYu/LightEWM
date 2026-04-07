# LIBERO Other Wan Variants

The 1.3B and 14B examples in this file are provided for reference only.
Their training quality is currently weaker than the Wan2.2 TI2V 5B path, and the parameters here are not guaranteed to be optimal.

The commands below use 49-frame clips at 10 FPS. During LIBERO conversion, videos are temporally resampled from 16 FPS to 10 FPS.

# Wan 1.3B I2V Train & Infer

## 1) Download checkpoint

```bash
hf download alibaba-pai/Wan2.1-Fun-1.3B-InP \
  --local-dir ./checkpoints/Wan2.1-I2V-1.3B
```

## 2) Build latent cache

```bash
bash scripts/process_cache.sh \
  --config examples/LIBERO/cache_1p3b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv \
  --output-path data/libero_i2v_train/latent_cache
```

## 3) Train

```bash
bash scripts/train_full.sh \
  --config examples/LIBERO/train_full_1p3b.yaml \
  --dataset-base-path data/libero_i2v_train/latent_cache
```

## 4) Infer

```bash
bash scripts/infer.sh \
  --config examples/LIBERO/infer_1p3b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv
```

# Wan 14B I2V Train & Infer

This path reuses the same latent cache as Wan 1.3B I2V.

## 1) Download checkpoint

Standard Wan2.1 I2V 14B:

```bash
hf download wan-world/Wan2.1-I2V-14B-480P \
  --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
```

Recommended WoW initialization:

```bash
hf download X-Humanoid/WoW-1-Wan-14B-2M \
  --local-dir ./checkpoints/WoW-1-Wan-14B-2M
```

## 2) Cache

Reuse the Wan 1.3B I2V cache:

```bash
bash scripts/process_cache.sh \
  --config examples/LIBERO/cache_1p3b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv \
  --output-path data/libero_i2v_train/latent_cache
```

## 3) Train

Standard Wan2.1 I2V 14B:

```bash
bash scripts/train_full.sh \
  --config examples/LIBERO/train_full_14b.yaml \
  --dataset-base-path data/libero_i2v_train/latent_cache
```

WoW-1-Wan-14B-2M:

```bash
bash scripts/train_full.sh \
  --config examples/LIBERO/train_full_wow_14b.yaml \
  --dataset-base-path data/libero_i2v_train/latent_cache
```

## 4) Infer

Standard Wan2.1 I2V 14B:

```bash
bash scripts/infer.sh \
  --config examples/LIBERO/infer_14b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv
```

WoW-1-Wan-14B-2M:

```bash
bash scripts/infer.sh \
  --config examples/LIBERO/infer_wow_14b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv
```
