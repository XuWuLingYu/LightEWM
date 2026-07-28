# RoboDojo FastWAM HDR

This example uses FastWAM HDR with a locally prepared RoboDojo LeRobot dataset.
The dataset path and action statistics are supplied explicitly so source data
never needs to live in the repository.

## Prepare data

```bash
python scripts/prepare_robodojo_fastwam_data.py \
  --source <source-dataset-directory> \
  --xpolicy-stats <action-statistics-json> \
  --mode both
```

Generate text embeddings after selecting the matching local configuration:

```bash
python run.py --config examples/RoboDojo-FASTWAMHDR/precompute_text.yaml
```

## Precache HDR latents

The episodewise cache preserves both local video context and HDR episode
context. Use the same configuration and HDR overrides that will be used for
training.

```bash
torchrun --standalone --nproc_per_node=<NUM_GPUS> \
  scripts/precache_fastwamhdr_latents_episodewise.py \
  --task <task-config> \
  --data <data-config> \
  --output-dir <cache-directory> \
  --encode-batch-size <batch-size> \
  --timing-report \
  +data.train.hdr_enabled=true
```

For a combined multi-source cache, use
`scripts/precache_mixed_adapter_episodewise.py` and pass each source root and
its action-statistics file explicitly. Generated cache, evaluation, and media
artifacts are ignored by git.

## Train and evaluate

```bash
python run.py --config examples/RoboDojo-FASTWAMHDR/train.yaml
python run.py --config examples/RoboDojo-FASTWAMHDR/eval.yaml
```
