# Remote FastWAM-HDR Policy Server

This note documents the LightEWM RoboDojo FastWAM-HDR mode-A remote policy
server. It serves the checkpoint trained in this repository, not the upstream
FastWAM or XPolicyLab policy implementation.

## Files

- `scripts/remote_fastwamhdr_policy_server.py`: TCP policy server.
- `scripts/launch_remote_fastwamhdr_policy_servers.sh`: multi-process launcher,
  one model process per GPU.
- `scripts/dataset_fastwamhdr_policy_client.py`: dataset-backed dummy client for
  open-loop action checks.
- `scripts/mock_fastwamhdr_policy_client.py`: protocol-only dummy client.
- `scripts/test_remote_fastwamhdr_policy_servers.sh`: smoke test wrapper.

## Protocol

The server uses the same length-prefixed socket transport as the remote eval
tunnel:

1. 4-byte big-endian payload length.
2. UTF-8 JSON payload.
3. NumPy arrays are encoded as base64 with dtype and shape metadata.

Supported commands:

- `reset`
- `update_obs`
- `get_action`
- `update_obs_batch`
- `get_action_batch`

For batched inference, call `update_obs_batch(obs_list)` first, then
`get_action_batch(env_idx_list)`. Each returned batch entry is an action chunk
with per-step action dictionaries:

- `left_arm_joint_state`: `(6,)`
- `left_ee_joint_state`: `(1,)`
- `right_arm_joint_state`: `(6,)`
- `right_ee_joint_state`: `(1,)`

## Observation Format

The server expects three RGB camera observations and a 14-D joint state split
into arm and gripper fields.

Accepted image aliases:

- head: `cam_head`, `cam_high`, `head_camera`
- left wrist: `cam_left_wrist`, `left_camera`
- right wrist: `cam_right_wrist`, `right_camera`

Example:

```python
obs = {
    "env_idx": 0,
    "task_instruction": "pick up the object",
    "vision": {
        "cam_head": {"color": head_rgb_uint8},
        "cam_left_wrist": {"color": left_rgb_uint8},
        "cam_right_wrist": {"color": right_rgb_uint8},
    },
    "state": {
        "left_arm_joint_state": left_arm_6,
        "left_ee_joint_state": left_gripper_1,
        "right_arm_joint_state": right_arm_6,
        "right_ee_joint_state": right_gripper_1,
    },
}
```

The server composes the RoboDojo frame internally:

- head camera: resized to `256x320`
- left wrist: resized to `128x160`
- right wrist: resized to `128x160`
- final frame: `384x320`, with the head camera on top and wrists below.

## Environment

Use raw Python on the remote host. Source shell setup and expose the PPU runtime
when launching manually:

```bash
source ~/.bashrc >/dev/null 2>&1 || true
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
```

The launcher already applies these variables.

## Start One Server

```bash
cd /path/to/LightEWM

CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
TOKENIZERS_PARALLELISM=false \
/usr/local/bin/python scripts/remote_fastwamhdr_policy_server.py \
  --host 127.0.0.1 \
  --port 6120 \
  --num-inference-steps 10 \
  --replan-steps 24 \
  --action-horizon 32
```

If `--checkpoint` is omitted, the server uses the newest RoboDojo FastWAM-HDR
`step_030000.pt` checkpoint under `logs/RoboDojo-FASTWAMHDR_train`.

## Start Eight Servers

```bash
cd /path/to/LightEWM

NUM_SERVERS=8 \
GPUS=0,1,2,3,4,5,6,7 \
BASE_PORT=6120 \
scripts/launch_remote_fastwamhdr_policy_servers.sh
```

This starts ports `6120` through `6127`. Each process owns one GPU.

Useful overrides:

```bash
CHECKPOINT=/path/to/step_030000.pt
DATASET_STATS=/path/to/dataset_stats.json
NUM_INFERENCE_STEPS=10
REPLAN_STEPS=24
ACTION_HORIZON=32
MIXED_PRECISION=bf16
LOG_DIR=logs/remote_policy/fastwamhdr
```

## Dataset Dummy Client

Use this to test whether inference is returning plausible actions. The client
reads RoboDojo LeRobot samples, sends the first observation image and state to
the policy server, receives an action chunk, and compares it to dataset ground
truth actions.

```bash
cd /path/to/LightEWM

/usr/local/bin/python scripts/dataset_fastwamhdr_policy_client.py \
  --host 127.0.0.1 \
  --port 6120 \
  --num-samples 8 \
  --batch-size 2 \
  --plot \
  --output-dir logs/eval/RoboDojo-FASTWAMHDR/remote_policy_dataset_check
```

Outputs:

- `summary.json`
- `summary.csv`
- `idx_XXXXXX.json`
- `idx_XXXXXX.npz`
- `idx_XXXXXX.png` action plots when `--plot` is enabled

Metrics are action-space open-loop errors after denormalization.

## Smoke Tests

Protocol-only dummy policy:

```bash
cd /path/to/LightEWM

NUM_SERVERS=2 \
GPUS=0,1 \
BASE_PORT=6180 \
ALLOW_DUMMY_POLICY=true \
BATCH_SIZE=2 \
NUM_SAMPLES=2 \
scripts/test_remote_fastwamhdr_policy_servers.sh
```

Real policy, two GPUs:

```bash
NUM_SERVERS=2 \
GPUS=0,1 \
BASE_PORT=6190 \
NUM_INFERENCE_STEPS=2 \
REPLAN_STEPS=4 \
BATCH_SIZE=2 \
NUM_SAMPLES=2 \
scripts/test_remote_fastwamhdr_policy_servers.sh
```

Real policy, eight GPUs:

```bash
NUM_SERVERS=8 \
GPUS=0,1,2,3,4,5,6,7 \
BASE_PORT=6200 \
NUM_INFERENCE_STEPS=2 \
REPLAN_STEPS=4 \
BATCH_SIZE=2 \
NUM_SAMPLES=2 \
scripts/test_remote_fastwamhdr_policy_servers.sh
```

The eight-GPU smoke test completed successfully at:

```text
logs/remote_policy/fastwamhdr_real_8gpu_batch_smoke_20260626_115153/eval
```

Each server used `batch_size=2`, `num_samples=2`, and returned:

```text
mean_action_l1   = 0.027566718868911266
mean_action_rmse = 0.06775335822113324
mean_action_mse  ~= 0.00459051755
```

No policy server process was left running after the smoke test.
