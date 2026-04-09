#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LIBERO_ROOT="data/LIBERO-datasets"
OUTPUT_DIR="data/libero_idm_abs_action"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --libero-root)
      LIBERO_ROOT="$2"
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

python scripts/convert_libero_to_idm_metadata.py \
  --libero-root "$LIBERO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --suites libero_10,libero_90,libero_goal,libero_object,libero_spatial \
  --camera-key agentview_rgb \
  --target-key ee_states \
  "$@"
