# CustomDataset Example

This example shows the expected dataset format and the standard three-stage workflow:
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

## 2) Build latent cache

```bash
bash examples/CustomDataset/process_cache.sh
```

You can override paths or preprocessing options:

```bash
accelerate launch run.py \
  --config examples/CustomDataset/cache.yaml \
  --overrides \
    dataset.params.dataset_base_path=data/your_dataset \
    dataset.params.dataset_metadata_path=data/your_dataset/metadata.csv \
    runner.params.output_path=./data/your_dataset/latent_cache \
    runner.params.fps=12 \
    runner.params.resize_mode=stretch \
    runner.params.context_window_stride=40
```

## 3) Train

```bash
bash examples/CustomDataset/train_full.sh
```

Useful overrides:

```bash
accelerate launch run.py \
  --config examples/CustomDataset/train_full.yaml \
  --overrides runner.params.num_epochs=5 runner.params.learning_rate=5e-6
```

## 4) Infer

```bash
bash examples/CustomDataset/infer.sh
```

Optional checkpoint override:

```bash
bash examples/CustomDataset/infer.sh --ckpt /path/to/your/ckpt.safetensors
```
