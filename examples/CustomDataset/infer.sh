#!/usr/bin/env bash
set -euo pipefail
python run.py --config examples/CustomDataset/infer.yaml "$@"
