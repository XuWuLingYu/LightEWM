#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

METADATA_PATH="$REPO_ROOT/data/your_dataset_idm/metadata_abs_action.jsonl"
IMAGE_BASE_PATH="$REPO_ROOT/data/your_dataset_idm"
SAVE_DIR="$REPO_ROOT/logs/customdataset_idm"
RUN_NAME="custom_abs_action"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --metadata-path)
      METADATA_PATH="$2"
      shift 2
      ;;
    --image-base-path)
      IMAGE_BASE_PATH="$2"
      shift 2
      ;;
    --save-dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  third_parties/AnyPos/train_metadata_abs_action.py \
  --metadata_path "$METADATA_PATH" \
  --image_base_path "$IMAGE_BASE_PATH" \
  --video_key video \
  --action_key abs_action \
  --model_name direction_aware \
  --learning_rate 1e-4 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --num_workers 8 \
  --eval_interval 2000 \
  --save_interval 2000 \
  --save_dir "$SAVE_DIR" \
  --wandb_project IDM_CustomDataset_abs_action \
  --run_name "$RUN_NAME" \
  "$@"
