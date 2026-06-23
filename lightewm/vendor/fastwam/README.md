# Vendored FastWAM

This directory contains the FastWAM runtime used by the
`examples/LIBERO-FASTWAM` entrypoints. It is intentionally copied into the
LightEWM core tree so training and evaluation do not depend on an external
FastWAM checkout.

## Upstream Source

Vendored from `https://github.com/yuantianyuan01/FastWAM` at commit
`45d8e1458921d83f8ad6cf9ce993d371208dabd0` (`Merge pull request #20 from
yuantianyuan01/dev/fix_gpu_oom`).

Path mapping:

- upstream `src/fastwam/` -> `lightewm/vendor/fastwam/fastwam/`
- upstream `configs/` -> `lightewm/vendor/fastwam/configs/`
- upstream `scripts/` -> `lightewm/vendor/fastwam/scripts/`
- upstream `experiments/libero/` -> `lightewm/vendor/fastwam/experiments/libero/`

When refreshing this vendor copy, compare against the pinned upstream commit
and keep local LightEWM integration changes separate from upstream source
updates.

Included pieces:

- `fastwam/`: model, MoT wrapper, data loaders, schedulers, and runtime code.
- `configs/`: FastWAM Hydra configs used by the runner.
- `scripts/`: train, text-embedding precompute, and accelerate/deepspeed configs.
- `experiments/libero/`: LIBERO rollout evaluator.

The LightEWM runner adds this directory to `PYTHONPATH` before launching the
vendored scripts, so imports still use the upstream package name `fastwam`.
