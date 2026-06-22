# LIBERO HDR Example

This example adapts LIBERO videos to the Causal-Forcing HDR vertical hierarchy backend. Training is split into two stages: first train the HDR video model, then train the action expert on top of the frozen video checkpoint.

## Prepare Data

Prepare the normal LIBERO video metadata under `./data/libero_i2v_train`. Stage 1 uses:

- `data/libero_i2v_train/metadata_dense_prompt.csv`

For Stage 2, generate the fixed-size action targets:

```bash
python scripts/prepare_libero_hdr_actions.py --overwrite
```

This writes:

- `data/libero_i2v_train/metadata_dense_prompt_hdr_actions_leaf8.csv`
- `data/libero_i2v_train/hdr_actions_leaf8/`
- `data/libero_i2v_train/hdr_actions_leaf8_stats.json`

## Stage 1: Video

Train the HDR video model with dense prompts, 49 video frames, and 13 HDR latent leaves:

```bash
python run.py --config examples/LIBERO-HDR/train.yaml
```

## Stage 2: Action

Set `runner.params.causal_config_overrides.generator_ckpt` in `examples/LIBERO-HDR/train_action.yaml` to a Stage 1 video checkpoint, then train the action expert:

```bash
python run.py --config examples/LIBERO-HDR/train_action.yaml
```

Stage 2 freezes the video model, loads the video checkpoint, initializes the action backbone by FastWAM-style linear interpolation from the loaded video expert, and trains 8 actions per HDR leaf from `metadata_dense_prompt_hdr_actions_leaf8.csv`. The action input encoder and output head are newly initialized.

## FastWAM-Local Joint Branch

`train_video_action_joint_fastwam_local.yaml` is the FastWAM-style local video
and action branch with an HDR tree prepended. It uses the original LIBERO HDF5
controller actions directly: 32 actions, 33 observations, video stride 4, and
9 local RGB frames. The frame shape is `224x448` because each sample is two
separate square views concatenated horizontally: left `agentview_rgb`, right
`eye_in_hand_rgb`. Do not use a single `agentview_rgb` stretched to width 448.

Run:

```bash
python run.py --config examples/LIBERO-HDR/train_video_action_joint_fastwam_local.yaml
```

## Infer

For video inference with a trained HDR checkpoint:

```bash
python run.py --config examples/LIBERO-HDR/infer.yaml \
  --overrides runner.params.checkpoint_path=/path/to/checkpoint_model_xxxxxx/model.pt
```
