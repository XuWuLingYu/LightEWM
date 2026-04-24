# CustomDataset Example

This README documents the Wan2.2 TI2V 5B workflow for a custom dataset.
For Wan2.1 1.3B and Wan2.1/WoW 14B examples, see `README_others.md`.
For finetuning initialization, we recommend `XuWuLingYu/Wan2.2-5B-Robot` (https://huggingface.co/XuWuLingYu/Wan2.2-5B-Robot), which is pretrained on broad robot datasets.

## 1) Dataset layout

```text
data/your_dataset/
├── metadata.csv
└── videos/
    ├── 000001.mp4
    ├── 000002.mp4
    └── ...
```

Minimum `metadata.csv` columns:
- `video`
- `prompt`

Recommended optional column:
- `num_frames`

## 2) Optional: generate dense prompts

```bash
bash scripts/process_dense_prompt.sh \
  --metadata-path data/your_dataset/metadata.csv \
  --output-path data/your_dataset/metadata_dense_prompt.csv
```

## 3) Build latent cache

```bash
bash scripts/process_cache.sh \
  --config examples/CustomDataset/cache_ti2v_5b.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv \
  --output-path data/your_dataset/latent_cache_ti2v_5b
```

If you generated dense prompts, switch only the metadata path.

Before training, download the robot-pretrained DiT init checkpoint:

```bash
hf download XuWuLingYu/Wan2.2-5B-Robot \
  --local-dir ./checkpoints/Wan2.2-5B-Robot
```

## 4) Train

```bash
bash scripts/train_full.sh \
  --config examples/CustomDataset/train_full_ti2v_5b.yaml \
  --dataset-base-path data/your_dataset/latent_cache_ti2v_5b \
  --ckpt checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors
```

For custom finetuning, this robot-pretrained initialization often gives better results than starting from the original Wan2.2 TI2V 5B checkpoint.

## 5) Infer

```bash
bash scripts/infer.sh \
  --config examples/CustomDataset/infer_ti2v_5b.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv
```

Optional checkpoint override:

```bash
bash scripts/infer.sh \
  --config examples/CustomDataset/infer_ti2v_5b.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv \
  --ckpt /path/to/your/ckpt.safetensors
```
