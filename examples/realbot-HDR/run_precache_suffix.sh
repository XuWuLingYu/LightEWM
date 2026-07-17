#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$ROOT/examples/realbot-HDR/run_env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
python3 examples/realbot-HDR/precache_latents.py \
  --dataset-root $ROOT/data/realbot_hdr_video_320x384_z0415_61f_suffix10 \
  --num-workers 8 \
  "$@"
