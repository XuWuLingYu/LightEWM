![LightEWM Banner](assets/LightEWM_banner.jpg)

# LightEWM

**LightEWM: Light Embodied World Model** is an open-source training and inference framework for embodied world models.
Our current focus is **Wan2.1-1.3B I2V**, including **LIBERO** preprocessing, latent-cache generation, full training, and inference.

## 🧭 Roadmap

- [x] Wan2.1-1.3B I2V training and inference support
- [x] LIBERO preprocessing, training, and inference support
- [ ] IDM support for video-to-action learning
- [ ] Causal video model adaptation for Wan 1.3B
- [ ] WAM actor implementation for Wan 1.3B
- [ ] CALVIN preprocessing, training, and inference support
- [ ] Reinforcement learning on top of the world model
 
## 🛠️ Installation

### 1) Environment setup

```bash
git clone https://github.com/XuWuLingYu/LightEWM.git
cd LightEWM
conda create -n lightewm python=3.10
conda activate lightewm
pip install -e .
pip install flash-attn --no-build-isolation
```

### 2) Download checkpoint

```bash
hf download alibaba-pai/Wan2.1-Fun-1.3B-InP \
  --local-dir ./checkpoints/Wan2.1-I2V-1.3B
```

## 🚀 Quick Start on LIBERO

See [LIBERO example](examples/LIBERO/README.md).

## 📦 Run on Custom Dataset

See [CustomDataset example](examples/CustomDataset/README.md).

## 📚 Reference

- DiffSynth-Studio: <https://github.com/modelscope/DiffSynth-Studio>
- Wan documentation: <https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Wan.html>
- WoW World Model: <https://wow-world-model.github.io/>

## 📄 License

Apache-2.0
