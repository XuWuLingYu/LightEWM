# CustomDataset Example

This example shows the expected dataset format and the standard three-stage workflow:
- optional dense-prompt generation
- latent-cache preprocessing
- full training
- batch inference

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

Example:

```csv
video,prompt,num_frames
videos/000001.mp4,"a robot arm picks up a red cube from the table",120
videos/000002.mp4,"open the drawer and place the object inside",97
```

`num_frames` is recommended because cache preprocessing can reuse it directly instead of recounting frames from disk.

## 2) Optional: generate dense prompts

If your `metadata.csv` already exists and you want denser action descriptions before cache generation, run:

```bash
bash scripts/process_dense_prompt.sh \
  --metadata-path data/your_dataset/metadata.csv \
  --output-path data/your_dataset/metadata_dense_prompt.csv
```

This creates:
- `data/your_dataset/metadata_dense_prompt.csv`

The generated file is ready for downstream use:
- original sparse text is preserved in `sparse_prompt`
- generated dense text is written to both `dense_prompt` and `prompt`

If you use dense prompts, update the metadata path in later steps from:
- `data/your_dataset/metadata.csv`

to:
- `data/your_dataset/metadata_dense_prompt.csv`

## 3) Build latent cache

```bash
bash scripts/process_cache.sh \
  --config examples/CustomDataset/cache.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv \
  --output-path data/your_dataset/latent_cache
```

If you generated dense prompts, switch only the metadata path:

```bash
bash scripts/process_cache.sh \
  --config examples/CustomDataset/cache.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata_dense_prompt.csv \
  --output-path data/your_dataset/latent_cache
```

Extra preprocessing overrides can be passed with repeated `--override`, for example:

```bash
bash scripts/process_cache.sh \
  --config examples/CustomDataset/cache.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv \
  --output-path data/your_dataset/latent_cache \
  --override runner.params.fps=12 \
  --override runner.params.resize_mode=stretch \
  --override runner.params.context_window_stride=40
```

## 4) Train

```bash
bash scripts/train_full.sh \
  --config examples/CustomDataset/train_full.yaml \
  --dataset-base-path data/your_dataset/latent_cache
```

Useful overrides:

```bash
bash scripts/train_full.sh \
  --config examples/CustomDataset/train_full.yaml \
  --dataset-base-path data/your_dataset/latent_cache \
  --override runner.params.num_epochs=5 \
  --override runner.params.learning_rate=5e-6
```

## 5) Infer

```bash
bash scripts/infer.sh \
  --config examples/CustomDataset/infer.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv
```

Optional checkpoint override:

```bash
bash scripts/infer.sh \
  --config examples/CustomDataset/infer.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata.csv \
  --ckpt /path/to/your/ckpt.safetensors
```

If you generated dense prompts, switch only the metadata path:

```bash
bash scripts/infer.sh \
  --config examples/CustomDataset/infer.yaml \
  --dataset-base-path data/your_dataset \
  --metadata-path data/your_dataset/metadata_dense_prompt.csv
```

## 6) YAML keys to modify

If you prefer editing the example YAML files directly instead of using script overrides, these are the path-related keys to update:

`examples/CustomDataset/cache.yaml`
- `dataset.params.dataset_base_path`
- `dataset.params.dataset_metadata_path`
- `runner.params.output_path`

`examples/CustomDataset/train_full.yaml`
- `dataset.params.dataset_base_path`

`examples/CustomDataset/infer.yaml`
- `dataset.params.base_path`
- `dataset.params.metadata_path`

You will likely also want to review these non-path keys:
- `runner.params.height`
- `runner.params.width`
- `runner.params.fps`
- `runner.params.infer_kwargs.height`
- `runner.params.infer_kwargs.width`
- `runner.params.wandb_run_name`
