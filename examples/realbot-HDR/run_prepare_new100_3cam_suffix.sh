#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
source "$ROOT/examples/realbot-HDR/run_env.sh"
python3 examples/realbot-HDR/prepare_new100_3cam_suffix_dataset.py "$@"
