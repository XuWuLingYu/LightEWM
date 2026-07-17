#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CKPT=${1:?usage: run_infer_fix5.sh /path/to/checkpoint_model_xxxxxx/model.pt}
shift || true
source "$ROOT/examples/realbot-HDR/run_env.sh"
export LIGHTEWM_RESPECT_OUTPUT_DIR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-8}

# Keep checkpoint_path in the same --overrides group as caller-provided
# overrides. argparse with nargs="*" keeps the last repeated --overrides
# occurrence, so a second --overrides from periodic inference used to drop
# the checkpoint and run base/random weights.
args=("$@")
has_overrides=0
for arg in "${args[@]}"; do
  if [ "$arg" = "--overrides" ]; then
    has_overrides=1
    break
  fi
done

if [ "$has_overrides" -eq 1 ]; then
  python3 run.py --config examples/realbot-HDR/infer_fix5.yaml "${args[@]}" runner.params.checkpoint_path="$CKPT"
else
  python3 run.py --config examples/realbot-HDR/infer_fix5.yaml "${args[@]}" --overrides runner.params.checkpoint_path="$CKPT"
fi
