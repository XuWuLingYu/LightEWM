# Fresh L20 Bootstrap Verification

## Bug description

A clean PAI L20 image could not reproduce the four closed-loop benchmark
environments. Sequential clean-room failures were:

- `Unable to locate package cuda-compat-12-8` before the NVIDIA CUDA apt source
  was registered.
- `uv: command not found` and `git: 'lfs' is not a git command` on the base
  image.
- PyTorch extension loading failed at `__nvJitLinkAddData_12_1` because the PAI
  login environment placed PPU CUDA libraries ahead of stock CUDA.
- CuRobo editable metadata failed with `invalid command 'bdist_wheel'`.
- OpenPI prefetched 3.7 GB of checksum-verified wheels but frozen `uv sync`
  followed the original lockfile URLs and tried to download them again.
- OpenPI's server entrypoint failed with `ModuleNotFoundError: No module named
  'pytest'` because upstream imports pytest at runtime but declares it only as
  a dev dependency.

## RED evidence

The regression tests were added one failure at a time before each installer
change. The OpenPI wheel-consumption test initially failed because
`uv pip sync` was absent. The runtime-dependency test initially failed because
`pytest==8.3.5` was absent. The clean L20 logs reproduced every failure string
listed above before the corresponding change.

## GREEN evidence

- `PYTHONPATH=. pytest -q tests/test_aliyun_l20_runtime_script.py`:
  `8 passed`.
- The broader closed-loop unit selection passes with 46 tests, and Ruff,
  `bash -n`, and `git diff --check` pass.
- Clean OpenPI synchronization reports `Resolved 204 packages`, then
  `Checked 204 packages`, using the frozen exported requirements and local
  wheelhouse. The StarWAM and OpenPI policy-server `--help` entrypoints both
  load successfully.
- `lightewm_eval_l20_b` reports NVIDIA L20 46068 MiB, driver 550.163.01, stock
  CUDA 12.4, NVIDIA Vulkan, and both required ray-tracing extensions.
- Official RoboTwin `script/test_render.py` returns `Render Well` through the
  evidence harness.
- RoboLab `BananaInBowlTask` initializes and completes a deterministic five-step
  hold: official success is false, qpos maximum delta is 0.0, object state is
  observed, and the mechanical probe verdict passes.
- The correctness runner lists exactly 5 RoboTwin and 12 RoboLab sample cases,
  and all 56 sampled RoboLab assets pass SHA-256 verification.
