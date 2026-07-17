#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
source "$ROOT/examples/realbot-HDR/run_env.sh"

DATASET_ROOT=${DATASET_ROOT:-$ROOT/data/realbot_hdr_video_320x384_z0415_61f_suffix10}
MODEL_ROOT=${MODEL_ROOT:-$ROOT/checkpoints}
GPUS_CSV=${GPUS_CSV:-8,9,10,11,12,13,14,15}
TARGET_COUNT=${TARGET_COUNT:-13040}
MAIN_SHARDS_CSV=${MAIN_SHARDS_CSV:-0,1,2,3}
MAIN_WORLD=${MAIN_WORLD:-4}
WITHIN_WORLD=${WITHIN_WORLD:-2}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-1}
MAX_NO_PROGRESS=${MAX_NO_PROGRESS:-3}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
IFS=',' read -r -a MAIN_SHARDS <<< "$MAIN_SHARDS_CSV"

count_cached() {
  find "$DATASET_ROOT/latent_cache_wan22_ti2v_61f_16lat" -type f -name '*.pt' 2>/dev/null | wc -l
}

mkdir -p "$DATASET_ROOT/latent_cache_wan22_ti2v_61f_16lat"

attempt=0
no_progress=0
prev_count=$(count_cached)
echo "[precache-split] start target=$TARGET_COUNT current=$prev_count gpus=$GPUS_CSV main_shards=$MAIN_SHARDS_CSV"

while [ "$prev_count" -lt "$TARGET_COUNT" ]; do
  attempt=$((attempt + 1))
  echo "[precache-split] attempt=$attempt current=$prev_count start_time=$(date '+%F %T')"
  pids=()
  gpu_i=0
  for main_rank in "${MAIN_SHARDS[@]}"; do
    for within_rank in $(seq 0 $((WITHIN_WORLD - 1))); do
      gpu="${GPUS[$gpu_i]}"
      gpu_i=$((gpu_i + 1))
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        python3 examples/realbot-HDR/precache_latents.py \
          --dataset-root "$DATASET_ROOT" \
          --model-root "$MODEL_ROOT" \
          --num-workers 1 \
          --encode-batch-size "$ENCODE_BATCH_SIZE" \
          --shard-rank "$main_rank" \
          --shard-world-size "$MAIN_WORLD" \
          --within-shard-rank "$within_rank" \
          --within-shard-world-size "$WITHIN_WORLD"
      ) &
      pids+=("$!")
      echo "[precache-split] launched main=$main_rank within=$within_rank gpu=$gpu pid=${pids[-1]}"
      sleep 2
    done
  done

  rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  current=$(count_cached)
  echo "[precache-split] attempt=$attempt rc=$rc current=$current prev=$prev_count end_time=$(date '+%F %T')"
  if [ "$current" -le "$prev_count" ]; then
    no_progress=$((no_progress + 1))
    echo "[precache-split] no_progress=$no_progress/$MAX_NO_PROGRESS"
    if [ "$no_progress" -ge "$MAX_NO_PROGRESS" ]; then
      echo "[precache-split] stopping: no progress" >&2
      exit 1
    fi
  else
    no_progress=0
  fi
  prev_count="$current"
  sleep 10
done

echo "[precache-split] complete count=$prev_count"
