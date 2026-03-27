#!/usr/bin/env bash
set -euo pipefail
accelerate launch run.py --config examples/CustomDataset/cache.yaml "$@"
