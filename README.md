# LightEWM

LightEWM is an open-source, out-of-the-box framework for embodied world models and action learning.
It is designed to unify world-model training/inference and embodied policy workflows in one practical codebase.

## Project Status

Early-stage bootstrap.

- [x] Repository scaffold initialized
- [x] Initial DiffSynth-Studio code imported into `lightewm/diffsynth` and `lightewm/wanvideo`
- [ ] Unified LightEWM training/inference interfaces (API + CLI)

## TODO Roadmap

This section is a rolling open-source backlog, not a strict delivery contract.

### Core Framework

- [ ] Define canonical config schema for model, data, training, and inference
- [ ] Build unified runner interfaces for train/eval/infer
- [ ] Standardize experiment logging, checkpoints, and resume behavior
- [ ] Add reproducibility controls (seed, deterministic modes, env capture)

### Wan2.1 (1.3B I2V)

- [ ] Create a clean `Wan2.1 1.3B I2V` training entrypoint
- [ ] Create a clean `Wan2.1 1.3B I2V` inference entrypoint
- [ ] Provide minimal runnable examples (single GPU and multi-GPU)
- [ ] Add model/data validation checks and common failure diagnostics

### Embodied Benchmarks (CALVIN / LIBERO)

- [ ] Define shared dataset adapter interfaces for embodied tasks
- [ ] Add CALVIN data preprocessing and dataloader integration
- [ ] Add LIBERO data preprocessing and dataloader integration
- [ ] Implement train/eval pipelines with benchmark-friendly metrics

### Dev Experience

- [ ] Write a real quickstart in `docs/quickstart.md`
- [ ] Add examples with expected directory layout and command templates
- [ ] Add contribution guide (`CONTRIBUTING.md`) and coding conventions
- [ ] Add issue templates for bug reports and feature requests

### Quality

- [ ] Add smoke tests for key training/inference paths
- [ ] Add CI checks for lint, import sanity, and basic execution
- [ ] Track compatibility matrix (CUDA/PyTorch/driver)

## Repository Layout

```text
LightEWM/
├── lightewm/
│   ├── diffsynth/      # Core diffusion/world-model components
│   └── wanvideo/       # Wan-related training/inference scripts and examples
├── docs/
│   ├── install.md
│   └── quickstart.md
├── data/
├── checkpoints/
├── logs/
└── pyproject.toml
```

## Installation

```bash
git clone https://github.com/XuWuLingYu/LightEWM.git
cd LightEWM
pip install -e .
```

## Notes

- The codebase is under active restructuring.
- Some scripts and arguments are still inherited from DiffSynth-Studio and will be progressively normalized to LightEWM conventions.

## Reference

- Wan documentation: <https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Wan.html>

## License

Apache-2.0
