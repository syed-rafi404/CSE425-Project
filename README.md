## CSE425 Project

This repository contains code for training a baseline Autoencoder (AE) and a β-VAE on the UIEB dataset.

### Setup

- Python 3.10+ is recommended.
- Create and activate a virtual environment, then install dependencies from `requirements.txt`.

### Dataset

Place images in the following folder structure:

```
data/UIEB/raw/               # training images (PNG/JPG)
data/UIEB/reference/         # optional reference images for evaluation
```

Images are resized to 256x256 during loading.

### Train

- Baseline AE:

```powershell
& ./venv/Scripts/python.exe ./src/train_ae.py
```

- β-VAE:

```powershell
& ./venv/Scripts/python.exe ./src/train_bvae.py
```

Checkpoints and samples are written under `runs/`.

### Config

Basic knobs are in `configs/default.yaml`:

```
batch_size: 16
epochs: 10
beta: 1.0
```

Current scripts use fixed values; update them or load the YAML if you prefer configurable runs.

