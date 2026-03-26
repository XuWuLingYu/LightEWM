# Data Guide (Wan2.1-Fun-1.3B I2V)

This page covers:
- dataset format
- preprocessing/cache generation
- config-based launch

## 1) Download checkpoint

```bash
hf download alibaba-pai/Wan2.1-Fun-1.3B-InP \
  --local-dir ./checkpoints/Wan2.1-I2V-1.3B
```

## 2) Dataset format

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

Example:

```csv
video,prompt
videos/000001.mp4,"a robot arm picks up a red cube from the table"
videos/000002.mp4,"open the drawer and place the object inside"
```

## 3) Cache preprocessing

Use base config:

```bash
accelerate launch run.py --config configs/wan/i2v/base_cache.yaml
```

Use LIBERO config:

```bash
accelerate launch run.py --config configs/libero/cache.yaml
```

## 4) Override preprocess options

```bash
accelerate launch run.py \
  --config configs/libero/cache.yaml \
  --overrides \
    runner.params.fps=12 \
    runner.params.resize_mode=stretch \
    runner.params.context_window_stride=40
```

Supported preprocess keys in config (`runner.params.*`):
- `fps`
- `resize_mode` (`stretch` or `letterbox`)
- `context_window_short_video_mode` (`drop` or `repeat_last_frame`)
- `context_window_stride`
- `context_window_tail_align`

## 5) Helper script

```bash
bash scripts/process_libero_cache.sh
```
