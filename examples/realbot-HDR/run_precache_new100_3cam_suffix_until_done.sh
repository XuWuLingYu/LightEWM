#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
source "$ROOT/examples/realbot-HDR/run_env.sh"
DATASET_ROOT=${DATASET_ROOT:-$ROOT/data/realbot_hdr_video_320x384_new100_3cam_61f_suffix10}
MODEL_ROOT=${MODEL_ROOT:-$ROOT/checkpoints}
NUM_WORKERS=${NUM_WORKERS:-4}
TARGET_COUNT=${TARGET_COUNT:-}
MAX_NO_PROGRESS=${MAX_NO_PROGRESS:-3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

if [ -z "$TARGET_COUNT" ]; then
  TARGET_COUNT=$(python3 - <<PY
import csv
from pathlib import Path
p=Path('$DATASET_ROOT')/'metadata_train.csv'
with p.open(newline='', encoding='utf-8') as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)
fi

count_cached() {
  find "$DATASET_ROOT/latent_cache_wan22_ti2v_61f_16lat" -type f -name '*.pt' 2>/dev/null | wc -l
}

mkdir -p "$DATASET_ROOT/latent_cache_wan22_ti2v_61f_16lat"

attempt=0
no_progress=0
prev_count=$(count_cached)
echo "[precache-supervisor] start target=$TARGET_COUNT current=$prev_count workers=$NUM_WORKERS gpus=$CUDA_VISIBLE_DEVICES dataset=$DATASET_ROOT"
while [ "$prev_count" -lt "$TARGET_COUNT" ]; do
  attempt=$((attempt + 1))
  echo "[precache-supervisor] attempt=$attempt current=$prev_count start_time=$(date '+%F %T')"
  set +e
  python3 examples/realbot-HDR/precache_latents.py \
    --dataset-root "$DATASET_ROOT" \
    --model-root "$MODEL_ROOT" \
    --num-workers "$NUM_WORKERS"
  rc=$?
  set -e
  current=$(count_cached)
  echo "[precache-supervisor] attempt=$attempt rc=$rc current=$current prev=$prev_count end_time=$(date '+%F %T')"
  if [ "$current" -le "$prev_count" ]; then
    no_progress=$((no_progress + 1))
    echo "[precache-supervisor] no_progress=$no_progress/$MAX_NO_PROGRESS"
    if [ "$no_progress" -ge "$MAX_NO_PROGRESS" ]; then
      echo "[precache-supervisor] stopping: no progress" >&2
      exit 1
    fi
  else
    no_progress=0
  fi
  prev_count="$current"
  sleep 10
done
echo "[precache-supervisor] complete count=$prev_count"
