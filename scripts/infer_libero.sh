#!/usr/bin/env bash
set -euo pipefail
python run.py --config configs/libero/infer.yaml "$@"
