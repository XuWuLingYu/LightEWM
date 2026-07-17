# RoboDojo FastWAM HDR

This example adapts the LIBERO FastWAM HDR workflow to RoboDojo LeRobot v2.1 data.

It keeps the RoboDojo/FastWAM training choices from XPolicyLab:

- task config: `robotwin_uncond_3cam_384_1e-4`
- three cameras: `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- camera layout: `cam_high` on top at `256x320`, left/right wrist on the bottom at `128x160` each, final frame `384x320`
- action/state: absolute 14-D joint vectors, no delta transform
- normalization: FastWAM z-score using `dataset_stats.json`
- optimizer schedule: AdamW, lr `1e-4`, cosine, 5% warmup, min lr `1e-6`, weight decay `1e-2`
- full run budget: `max_steps=30000`, `save_every=epoch`, `eval_every=500`, bf16, batch size 16 per GPU

It keeps the original LightEWM FastWAM HDR behavior for the extra future frames and attention mask:

- 9 local RGB frames from the normal 33-step / stride-4 FastWAM window
- 4 HDR frames sampled uniformly from after the local window through the episode tail
- 13 RGB frames total, satisfying Wan VAE `T % 4 == 1`
- `fastwam_joint` with `model.action_attend_video=local_clean_first`
- latent precache stores the full 13-frame latent clip and local-frame metadata

## Prepare Data

Register the full dataset and build the 10-task x 10-episode smoke subset. Keep the source path outside git and pass it explicitly:

```bash
python scripts/prepare_robodojo_fastwam_data.py   --source /path/to/RoboDojo_lerobot_v21_video   --xpolicy-stats /path/to/FastWAM/data/robodojo-v21-video/dataset_stats.json   --mode both
```

Smoke output:

```text
data/robodojo_fastwam/robodojo-v21-video-smoke-10task-10ep/
??? dataset_stats.json
??? lerobot/
```

Full output:

```text
data/robodojo_fastwam/robodojo-v21-video/
??? dataset_stats.json
??? lerobot -> /path/to/RoboDojo_lerobot_v21_video
```

## Smoke Gate

Run only this path first. Do not start the full run until the visualization is approved.

```bash
python run.py --config examples/RoboDojo-FASTWAMHDR/precompute_text_smoke.yaml

python run.py --config examples/RoboDojo-FASTWAMHDR/train_smoke.yaml
```

## Full Run

After the smoke visualization is approved:

```bash
python run.py --config examples/RoboDojo-FASTWAMHDR/precompute_text.yaml
python run.py --config examples/RoboDojo-FASTWAMHDR/train.yaml
```

For manual latent precache or visualization, use the relative dataset paths in the YAML files and keep generated `data/`, `logs/`, and checkpoint files untracked.
