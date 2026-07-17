#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
source "$ROOT/examples/realbot-HDR/run_env.sh"

DATASET_ROOT=${DATASET_ROOT:-$ROOT/data/realbot_hdr_video_320x384_z0415_61f_suffix10}
MODEL_ROOT=${MODEL_ROOT:-$ROOT/checkpoints}
GPUS_CSV=${GPUS_CSV:-8,9,10,11,12,13,14,15}
TARGET_COUNT=${TARGET_COUNT:-13040}
MAX_NO_PROGRESS=${MAX_NO_PROGRESS:-3}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
SHARD_WORLD=${#GPUS[@]}
if [ "$SHARD_WORLD" -lt 1 ]; then
  echo "No GPUs configured" >&2
  exit 1
fi

count_cached() {
  find "$DATASET_ROOT/latent_cache_wan22_ti2v_61f_16lat" -type f -name '*.pt' 2>/dev/null | wc -l
}

attempt=0
no_progress=0
prev_count=$(count_cached)
echo "[precache-sharded] start target=$TARGET_COUNT current=$prev_count shards=$SHARD_WORLD gpus=$GPUS_CSV"

while [ "$prev_count" -lt "$TARGET_COUNT" ]; do
  attempt=$((attempt + 1))
  echo "[precache-sharded] attempt=$attempt current=$prev_count start_time=$(date '+%F %T')"
  pids=()
  for rank in "${!GPUS[@]}"; do
    gpu="${GPUS[$rank]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python3 examples/realbot-HDR/precache_latents.py \
        --dataset-root "$DATASET_ROOT" \
        --model-root "$MODEL_ROOT" \
        --num-workers 1 \
        --shard-rank "$rank" \
        --shard-world-size "$SHARD_WORLD"
    ) &
    pids+=("$!")
    echo "[precache-sharded] launched rank=$rank gpu=$gpu pid=${pids[-1]}"
    sleep 2
  done

  rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  current=$(count_cached)
  echo "[precache-sharded] attempt=$attempt rc=$rc current=$current prev=$prev_count end_time=$(date '+%F %T')"
  if [ "$current" -le "$prev_count" ]; then
    no_progress=$((no_progress + 1))
    echo "[precache-sharded] no_progress=$no_progress/$MAX_NO_PROGRESS"
    if [ "$no_progress" -ge "$MAX_NO_PROGRESS" ]; then
      echo "[precache-sharded] stopping: no progress" >&2
      exit 1
    fi
  else
    no_progress=0
  fi
  prev_count="$current"
  sleep 10
done

echo "[precache-sharded] complete count=$prev_count"
