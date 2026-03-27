# Data Guide (Wan2.1-Fun-1.3B I2V)

This page covers:
- dataset format
- preprocessing/cache generation
- config-based launch

## 1) Dataset format

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

Recommended optional column:
- `num_frames` (integer)

Example:

```csv
video,prompt,num_frames
videos/000001.mp4,"a robot arm picks up a red cube from the table",120
videos/000002.mp4,"open the drawer and place the object inside",97
```

Why `num_frames` is recommended:
- During cache preprocessing with context windows, LightEWM reads `num_frames` from metadata if available.
- This avoids re-counting video frames from disk and makes preprocessing faster and more stable on large datasets.
- If `num_frames` is missing, LightEWM will count frames automatically.
- For best correctness, set `num_frames` to match the effective frame count at your preprocessing FPS (`runner.params.fps`).

## 2) Cache preprocessing

Use base config:

```bash
accelerate launch run.py --config configs/wan/i2v/base_cache.yaml
```


## 3) Override preprocess options (including custom dataset CSV)

```bash
accelerate launch run.py \
  --config configs/wan/i2v/base_cache.yaml \
  --overrides \
    dataset.params.dataset_base_path=data/your_dataset \
    dataset.params.dataset_metadata_path=data/your_dataset/metadata.csv \
    runner.params.output_path=./data/your_dataset/latent_cache \
    runner.params.fps=12 \
    runner.params.resize_mode=stretch \
    runner.params.context_window_stride=40
```

Notes:
- `dataset.params.dataset_base_path`: dataset root path.
- `dataset.params.dataset_metadata_path`: metadata file path (CSV/JSON/JSONL).
- `runner.params.output_path`: output latent-cache directory.

Supported preprocess keys in config (`runner.params.*`):
- `fps`
- `resize_mode` (`stretch` or `letterbox`)
- `context_window_short_video_mode` (`drop` or `repeat_last_frame`)
- `context_window_stride`
- `context_window_tail_align`
