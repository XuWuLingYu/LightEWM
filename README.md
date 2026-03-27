# LightEWM

LightEWM is an open-source training and inference framework for embodied world models.
Current active target: **Wan2.1-Fun-1.3B I2V** with **LIBERO** preprocessing, cache generation, full training, and inference.

## Active Scope

- Wan2.1-Fun-1.3B I2V cache preprocessing (`text + input_image`)
- Wan2.1-Fun-1.3B I2V full training
- Wan2.1-Fun-1.3B I2V LIBERO batch inference

## New Code Layout

```text
LightEWM/
├── lightewm/
│   ├── model/
│   ├── utils/
│   ├── dataset/
│   ├── runner/
│   └── configs/
├── configs/
│   ├── wan/i2v/
│   └── libero/
├── scripts/
├── docs/
└── run.py
```

Legacy `diffsynth` / `wanvideo` trees were moved to `trash_code/` and are no longer active runtime dependencies.

## Quick Start

Install:

```bash
pip install -e .
```

Run by config:

```bash
# LIBERO cache generation
accelerate launch run.py --config configs/libero/cache.yaml

# LIBERO full training
accelerate launch run.py --config configs/libero/train_full.yaml

# LIBERO inference
python run.py --config configs/libero/infer.yaml

# LIBERO inference with a custom finetuned DiT checkpoint (.safetensors)
python run.py --config configs/libero/infer.yaml --ckpt /path/to/your/ckpt.safetensors
```

Run outputs are organized as:

```text
logs/{config_name}/{timestamp}/
```

W&B logging is enabled by default in training configs.

```bash
wandb login
# or disable globally
export WANDB_MODE=disabled
```

## Docs

- [Install](docs/install.md)
- [Data Guide](docs/data.md)
- [Training Quickstart](docs/quickstart.md)
- [LIBERO Guide](docs/libero.md)

## Reference

- DiffSynth-Studio: <https://github.com/modelscope/DiffSynth-Studio>
- Wan documentation: <https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Wan.html>

## License

Apache-2.0
