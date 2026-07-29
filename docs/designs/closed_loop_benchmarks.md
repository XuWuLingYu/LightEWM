# Closed-Loop Benchmark Integration

## Goal

Integrate LIBERO, RoboTwin 2.0, and RoboLab without replacing their official
simulators, controllers, success conditions, or native artifacts. LightEWM owns
only the policy contract, benchmark launch description, provenance, and result
normalization.

The first acceptance target is correctness on one NVIDIA L20 48 GB DSW
instance. Large-scale benchmark runs are explicitly out of scope until every
correctness gate below passes.

## Architecture

```text
PolicySpec
  checkpoint identity + observation/action contract + runtime endpoint
        |
BenchmarkSpec
  official repository revision + tasks/settings/seeds + protocol
        |
Official harness command
  simulator/controller/success checker remain benchmark-owned
        |
Native outputs
  benchmark videos, HDF5, logs, result files
        |
ResultNormalizer
  native outputs -> EpisodeRecord JSONL + Summary JSON
        |
ArtifactIndex
  commands, revisions, spec hashes, native and normalized output paths
```

`BackendRunResult` remains the video-generation handoff. Closed-loop evaluation
does not extend that video-shaped class because a simulator episode is not a
generated-video artifact.

## Standalone Boundary

LIBERO, RoboTwin, and RoboLab remain independently runnable. Their official
repositories own simulator construction, episode stepping, controller
semantics, success checks, and native outputs. A standalone run needs the
official repository revision, its locked runtime, assets, and the selected
policy checkpoint; it does not import `lightewm`.

The benchmark overlays only make correctness-gate behavior deterministic or
machine-readable:

- RoboTwin accepts a configurable smoke episode count and receives a copied
  deterministic control-probe policy.
- RoboLab registers only the requested task and records deterministic
  hold/gripper state probes.
- StarWAM and OpenPI remain separate policy runtimes required only by their
  corresponding reference-policy gates.

After `scripts/apply_closed_loop_benchmark_overlays.sh` materializes these
files, RoboTwin and RoboLab have no runtime dependency on the LightEWM checkout.
The StarWAM RoboTwin adapter remains a link to the separately versioned StarWAM
repository because it is model implementation code, not LightEWM glue.

LightEWM starts at the integration boundary: it describes policy and benchmark
contracts, records exact commands and revisions, parses native outputs, writes
normalized episode records, and validates cross-benchmark acceptance gates.
Removing that layer does not stop an official evaluation, but it removes the
common provenance, normalization, and mechanical gate checks.

## Contracts

### PolicySpec

- Stable policy identity and family.
- Checkpoint URI and format.
- Optional backbone, config, statistics, and processor URIs.
- Named observation tensors with dtype, shape, and semantics.
- Action tensor, action chunk size, and execution horizon.
- Local or client/server runtime description.

### BenchmarkSpec

- Benchmark name and immutable repository revision.
- Protocol name and official repository path.
- Tasks, settings, seeds, episode count, and maximum step count.
- Native result parser and official command steps.

### EpisodeRecord

One simulator episode per JSON line:

- benchmark/revision/protocol;
- policy/task/setting/seed/episode;
- success, status, steps, score, and failure reason;
- native video/HDF5/event-log references;
- source record and provenance.

RoboLab's native episode videos, HDF5, event logs, subtask score, failure
reason, and dashboard remain authoritative. LightEWM records paths to them and
does not regenerate equivalent artifacts.

## Correctness Gates

Every gate stores the exact command, source revision, config, log, native
outputs, normalized `episodes.jsonl`, `summary.json`, and `artifact_index.json`.
Deterministic controls additionally store the raw probe and the mechanical
verdict produced by `scripts/analyze_closed_loop_control_probe.py`.
Official commands are launched through
`scripts/run_closed_loop_official_step.py`; this is a thin evidence wrapper
around the benchmark entrypoints, not a replacement harness.

### LIBERO Compatibility Regression

- Normalize an existing OpenVLA/FastWAM official rollout result.
- Preserve task, suite, seed/episode, success, and native video path.
- Verify normalized summary matches the official success counts.

### RoboTwin 2.0

1. **Ground truth**
   - Run `script/test_render.py`; require `Render Well`.
   - Run official `script/collect_data.py` scripted expert.
   - Require `plan_success=true`, `check_success=true`, and a saved trajectory.
2. **Deterministic negative controls**
   - Reuse the same task/config/seed as ground truth.
   - Run hold/zero, wrong-gripper, and reverse-action policies through the
     official evaluation loop.
   - Require every negative episode to report success false.
   - Record initial/final robot qpos and object state. At least one active
     control must change state; hold must remain within a configured tolerance.
3. **Reference policy**
   - StarWAM released RoboTwin MoT checkpoint plus Wan2.2-TI2V-5B backbone and
     released action/state statistics.
   - One stable task, `demo_clean` and `demo_randomized`, 5-10 episodes each,
     unseen instructions, four inference steps, replan horizon 24.
   - Validate observation -> policy server -> 32x14 action chunk -> official
     simulator -> official success checker.
   - Formal expansion is 50 tasks x 2 settings x 100 episodes only after this
     gate passes.

### RoboLab

1. **Ground truth and action path**
   - Run official `examples/run_recorded.py` using bundled recorded data,
     restoring the recorded initial state and actions.
   - Run official `examples/run_gripper_toggle.py`.
   - Require successful replay and visible/action-state gripper change.
   - The official repository bundles a recorded positive trajectory only for
     `RubiksCubeAndBananaTask`. This gate validates the recorded-state replay,
     action, logging, and success-condition paths for that task; it is not a
     positive-oracle check for all 120 benchmark tasks.
   - All-task integration coverage must separately validate every task
     definition and registration, instantiate/reset/step every environment,
     and exercise its termination predicates. A reference-policy rollout can
     provide empirical full-suite coverage, but a failed policy episode does
     not by itself prove that the corresponding task or success predicate is
     incorrect.
2. **Deterministic negative control**
   - Keep official `run_empty.py` native logging, replacing random sampling
     with a deterministic hold action through a small, recorded overlay.
   - Require success false and compare initial/final qpos and object states.
3. **Reference policy**
   - Run the official OpenPI policy server with
     `pi05_droid_jointpos` and the published checkpoint.
   - Validate observation -> RoboLab client -> server -> action chunk ->
     simulator -> success.

## Stratified 10 Percent Correctness Sample

`examples/closed_loop/samples/correctness_10pct.yaml` is the checked sample
contract. It pins the official benchmark revisions, enforces exact sample
counts, and records the rationale and behavioral attributes for each selected
case. `scripts/run_closed_loop_correctness_sample.py` runs the official
entrypoints, checkpoints each case, validates expected-output freshness, and
writes one resumable summary.

The sample contains:

- RoboTwin: 5/50 tasks, each with a fresh official scripted-expert success,
  trajectory, HDF5, scene metadata, and video.
- RoboLab: 12/120 tasks, comprising the repository's one bundled positive
  recorded replay and 11 deterministic hold probes. Every hold case must
  instantiate, reset, step, observe object state, keep robot qpos stationary,
  exercise termination evaluation, and remain unsuccessful.

This is stratified integration evidence, not a claim that all RoboLab success
predicates have a positive oracle. Eleven RoboLab cases still have only a
negative/runability verdict because the official repository does not bundle
positive actions for them.

The accepted run completed 17/17 cases in 51 minutes 30 seconds. RoboTwin
required 0, 0, 1, 0, and 5 failed planning seeds respectively before finding
one successful expert trajectory for `adjust_bottle`, `handover_block`,
`open_microwave`, `stack_blocks_three`, and `scan_object`.

## Environment Lock

```text
Aliyun DSW
  instance: lightewm_eval_l20
  accelerator: 1 x NVIDIA L20 48 GB
  CPU/RAM: 16 / 128 GB
  persistent mount: /mnt/data
  NVIDIA_DRIVER_CAPABILITIES: all
  display: Xorg :99 (headless NVIDIA screen)
  simulator concurrency: 1
  policy batch size: 1
```

Use separate environments/processes:

- `lightewm-policy`: StarWAM/Wan2.2 server.
- `openpi-server`: OpenPI server and JAX/XLA dependencies.
- `robotwin-sim`: Python 3.10 and official RoboTwin dependencies.
- `robolab-sim`: Python 3.11, RoboLab `isaac50`, Isaac Sim 5.0, Isaac Lab
  2.2.0.

On the DSW, repositories, models, locks, and evidence remain on persistent
CPFS under `LIGHTEWM_EVAL_ROOT`. Set
`LIGHTEWM_RUNTIME_ROOT=/tmp/robot-eval` so rebuildable venvs and package caches
use the local NVMe; installing Isaac/OpenPI directly into CPFS is prohibitively
slow because their distributions contain tens of thousands of small files.
The runtime root may be deleted when the instance stops, so the setup script
must remain the source of truth.

For OpenPI start with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`.

`scripts/setup_closed_loop_benchmark_envs.sh` reproduces the four isolated
environments from the pinned official repositories. Run a single component
(`robotwin`, `starwam`, `robolab`, or `openpi`) or `all`.
Fresh PAI L20 images are not assumed to contain the complete toolchain. The
runtime setup installs pinned `uv`, Git LFS, and the NVIDIA CUDA apt source;
replaces the PAI PPU CUDA symlink with stock CUDA 12.4; and removes PPU library
paths before building PyTorch3D and CuRobo. CuRobo's legacy build also requires
`setuptools` and `wheel` to be installed before editable metadata generation.
For RoboLab it materializes and verifies the SHA-256 checksums in
`env/robolab_correctness_sample_assets.sha256`; leaving these Git LFS objects
as pointer files makes task import fail, and the upstream resolver can
misreport that underlying asset error as a missing task class.
On hosts where GitHub is unavailable, OpenPI's pinned LeRobot and dlimp commits
may be mirrored at `${LIGHTEWM_EVAL_ROOT}/git-mirrors/{lerobot,dlimp}`; the
setup script automatically rewrites only those two Git URLs.

OpenPI's large lockfile wheels are downloaded and checksum-verified into a
local wheelhouse. A frozen `uv sync` still follows the registry URLs embedded
in `uv.lock`, so it does not reliably consume those prefetched files. The
installer instead synchronizes the exact frozen `uv export` result with
`uv pip sync` and the local wheelhouse. It then installs the lock's pytest
version explicitly because upstream `gemma_pytorch.py` imports pytest at
policy-server runtime while declaring it only as a development dependency.

The clean-room bootstrap was repeated on `lightewm_eval_l20_b` and accepted
only after the NVIDIA L20/CUDA/Vulkan checks, official RoboTwin render,
RoboLab `BananaInBowlTask` deterministic hold, StarWAM/OpenPI server entrypoint
checks, and the 5+12 correctness-sample manifest validation all passed.

The DSW must expose `/dev/nvidia-modeset` and allow character device
`195:254`. `scripts/setup_aliyun_l20_runtime.sh` starts the NVIDIA Xorg screen,
writes a reusable runtime environment file, and accepts the host only after
`vulkaninfo` reports the L20 plus both ray-tracing extensions.

Reference `.pt` files must pass
`scripts/validate_torch_checkpoint.py` before policy startup. The validator
requires a mapping-style `torch.load` result and records the byte size and
SHA-256 digest. The digest and size should also be stored in `CheckpointSpec`
when the asset is a fixed local file.

## Evidence Ledger

All cloud evidence is stored below
`/mnt/data/zhangyu/robot-eval/evidence`. A gate is not complete until its row
contains command, log, native output, normalized output, and acceptance result.

| Gate | Evidence directory | Current acceptance |
| --- | --- | --- |
| L20 Vulkan RT | `../logs/setup_aliyun_l20_runtime.log`, `../logs/vulkaninfo-550.163.01.log` | passed |
| LIBERO OpenVLA | `libero/openvla` | passed, 79/80 |
| LIBERO FastWAM | `libero/fastwam` | passed, 39/40 |
| RoboTwin render | `robotwin/render` | passed, official `Render Well` |
| RoboTwin ground truth | `robotwin/ground_truth` | passed, official expert 1/1 |
| RoboTwin negative controls | `robotwin/negative_controls` | passed, 0/4 success; all state probes passed |
| RoboTwin StarWAM | `robotwin/reference_starwam` | passed, `demo_clean` 5/5 and `demo_randomized` 5/5 |
| RoboLab replay/toggle | `robolab/ground_truth` | passed, replay 1/1 and gripper action-path probe passed |
| RoboLab deterministic hold | `robolab/negative_hold` | passed, 0/1 success and stationary-qpos probe passed |
| 10% correctness sample | `correctness_sample_10pct_run2` | passed, RoboTwin 5/5 and RoboLab 12/12 |
| RoboLab pi0.5 | `robolab/reference_pi05` | pending |

The accepted StarWAM checkpoint is
`starwam_wan225b_robotwin_mot.pt`, size `12042083065`, SHA-256
`7d5db55758fcd9e4469ec8416d8ca0f046e52a79077e270fece017ed4187faa8`.
An earlier downloader produced a different 10.8 GB raw file that failed
`torch.load` with `invalid load key, '\x1d'`; the invalid file and validation
log remain under `robotwin/reference_starwam` so a same-path asset cannot be
mistaken for the verified checkpoint.

The accepted RoboLab recorded replay ran 648 steps, returned official
`success=true`, and reported maximum recorded-state drift `0.0000` against the
`0.01` threshold. The gripper action-path probe recorded 101 samples, a command
range of `0.785398163`, and an observed gripper-state range of `1.0`. The
deterministic hold control returned official `success=false`, qpos maximum
delta `0.0`, and object-state maximum delta `0.0048403683`.

## Review Gate

After implementation and all cloud gates complete, start an independent Codex
review session with this document as the acceptance contract. The reviewer must
inspect source, tests, commands, logs, native outputs, normalized outputs, and
summaries. Any missing or weak evidence returns to the implementation session
for repair; completion requires no P0/P1 findings.
