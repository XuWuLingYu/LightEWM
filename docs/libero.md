# LIBERO Dataset Conversion Guide

## 1) Download LIBERO dataset

```bash
mkdir -p data
cd data
hf download \
  yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir ./LIBERO-datasets
cd ..
```

## 2) Convert LIBERO hdf5 to trainable CSV + videos

```bash
python \
  scripts/convert_libero_to_csv.py \
  --libero-root data/LIBERO-datasets \
  --output-dir data/libero_i2v_train \
  --suites libero_10,libero_90,libero_goal,libero_object,libero_spatial \
  --fps 16 \
  --camera-key agentview_rgb,eye_in_hand_rgb \
  --prompt-source attr_or_filename
```

Output:
- `data/libero_i2v_train/metadata.csv`
- `data/libero_i2v_train/videos/*.mp4`

`metadata.csv` is compatible with the training format used in this repo:
- required columns: `video,prompt`
- extra debug columns are included: `source_file,demo_id,camera_key,num_frames`
- prompt normalization:
  - removes task prefix before `SCENE<number>`
  - example: `KITCHEN SCENE10 close the top drawer ...` -> `close the top drawer ...`
- video orientation:
  - frames are rotated 180 degrees by default to fix upside-down + mirrored export

## 3) Optional conversion flags

- Keep only one suite:
```bash
--suites libero_10
```

- Force a specific camera key:
```bash
--camera-key agentview_rgb
```

- Export multiple cameras (agentview + handview):
```bash
--camera-key agentview_rgb,eye_in_hand_rgb
```

- Overwrite existing exported videos:
```bash
--overwrite
```


## 4) Preprocess LIBERO cache (one-click script)

Run from repo root:

```bash
chmod +x scripts/process_libero_cache.sh

# generate cache
bash scripts/process_libero_cache.sh
```

This script uses documented paths and fixed options:
- `REPO_ROOT=/mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/LightEWM`
- `DATASET_BASE=data/libero_i2v_train`
- `FPS=16`
- `RESIZE_MODE=letterbox`
- `CONTEXT_SHORT_MODE=drop`
- `CONTEXT_STRIDE=81`
- `CONTEXT_TAIL_ALIGN=true`

Output cache:
- `./data/libero_i2v_train/latent_cache`

## 5) Start training

Use:
- `docs/quickstart.md` for training commands
- `docs/data.md` for preprocessing options
