#!/usr/bin/env bash
set -euo pipefail
python run.py --config examples/LIBERO/infer.yaml "$@"
