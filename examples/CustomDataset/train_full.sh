#!/usr/bin/env bash
set -euo pipefail
accelerate launch run.py --config examples/CustomDataset/train_full.yaml "$@"
