#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$ROOT/examples/realbot-HDR/run_env.sh"
python3 examples/realbot-HDR/prepare_dataset.py "$@"
