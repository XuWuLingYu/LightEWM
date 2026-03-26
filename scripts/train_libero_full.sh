#!/usr/bin/env bash
set -euo pipefail
accelerate launch run.py --config configs/libero/train_full.yaml "$@"
