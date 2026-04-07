# CustomDataset Other Wan Variants

The 1.3B and 14B examples in this file are provided for reference only.
Their training quality is currently weaker than the Wan2.2 TI2V 5B path, and the parameters here are not guaranteed to be optimal.

## 1) Dataset layout

```text
data/your_dataset/
├── metadata.csv
└── videos/
    ├── 000001.mp4
    ├── 000002.mp4
    └── ...
```

## 2) Wan 1.3B I2V

### Build latent cache

```bash
bash scripts/process_cache.sh \
  --config examples/CustomDataset/cache_1p3b.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv \
  --output-path data/your_dataset/latent_cache
```

### Train

```bash
bash scripts/train_full.sh \
  --config examples/CustomDataset/train_full_1p3b.yaml \
  --dataset-base-path data/your_dataset/latent_cache
```

### Infer

```bash
bash scripts/infer.sh \
  --config examples/CustomDataset/infer_1p3b.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv
```

## 3) Wan 14B I2V

### Download checkpoints

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

### Cache

Reuse the Wan 1.3B I2V cache:

```bash
bash scripts/process_cache.sh \
  --config examples/CustomDataset/cache_1p3b.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv \
  --output-path data/your_dataset/latent_cache
```

### Train

Standard Wan2.1 I2V 14B:

```bash
bash scripts/train_full.sh \
  --config examples/CustomDataset/train_full_14b.yaml \
  --dataset-base-path data/your_dataset/latent_cache
```

WoW-1-Wan-14B-2M:

```bash
bash scripts/train_full.sh \
  --config examples/CustomDataset/train_full_wow_14b.yaml \
  --dataset-base-path data/your_dataset/latent_cache
```

### Infer

Standard Wan2.1 I2V 14B:

```bash
bash scripts/infer.sh \
  --config examples/CustomDataset/infer_14b.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv
```

WoW-1-Wan-14B-2M:

```bash
bash scripts/infer.sh \
  --config examples/CustomDataset/infer_wow_14b.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv
```
