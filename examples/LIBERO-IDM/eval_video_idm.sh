#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/envs/lightewm/bin/python}"
VIDEO_CKPT="$REPO_ROOT/checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors"
IDM_CKPT="$REPO_ROOT/checkpoints/LIBERO-IDM/100000.pt"
PROMPT_METADATA_PATH="$REPO_ROOT/data/libero_i2v_train/metadata_dense_prompt.csv"
OUTPUT_DIR="$REPO_ROOT/outputs/libero_video_idm_eval"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-ckpt)
      VIDEO_CKPT="$2"
      shift 2
      ;;
    --idm-ckpt)
      IDM_CKPT="$2"
      shift 2
      ;;
    --prompt-metadata-path)
      PROMPT_METADATA_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

"$PYTHON_BIN" examples/LIBERO-IDM/eval_libero_video_idm.py \
  --video-ckpt "$VIDEO_CKPT" \
  --idm-ckpt "$IDM_CKPT" \
  --prompt-metadata-path "$PROMPT_METADATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --save-debug-media \
  "$@"
