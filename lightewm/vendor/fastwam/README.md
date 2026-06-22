# Vendored FastWAM

This directory contains the FastWAM runtime used by the
`examples/LIBERO-FASTWAM` entrypoints. It is intentionally copied into the
LightEWM core tree so training and evaluation do not depend on an external
FastWAM checkout.

Included pieces:

- `fastwam/`: model, MoT wrapper, data loaders, schedulers, and runtime code.
- `configs/`: FastWAM Hydra configs used by the runner.
- `scripts/`: train, text-embedding precompute, and accelerate/deepspeed configs.
- `experiments/libero/`: LIBERO rollout evaluator.

The LightEWM runner adds this directory to `PYTHONPATH` before launching the
vendored scripts, so imports still use the upstream package name `fastwam`.
