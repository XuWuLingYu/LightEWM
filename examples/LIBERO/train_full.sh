#!/usr/bin/env bash
set -euo pipefail
accelerate launch run.py --config examples/LIBERO/train_full.yaml "$@"
