### Evaluate + figures

```powershell
& ./venv/Scripts/python.exe ./src/eval.py
& ./venv/Scripts/python.exe ./src/make_figures.py   # loss curves, recon grids, metrics files
& ./venv/Scripts/python.exe ./src/latent_traversal.py # β-VAE latent traversal grid
```

Notes: FID uses pytorch-fid defaults (InceptionV3 pool3=2048); IS uses TorchMetrics InceptionScore with splits=10 on 299×299 RGB inputs.
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

Training logs (CSV) are saved under `runs/logs/` for AE and β-VAE, including epoch-wise train/val losses. Early stopping uses `patience` and `min_delta` from config.

### Config

Basic knobs are in `configs/default.yaml`:

```
batch_size: 16
epochs: 10
beta: 1.0
seed: 42
image_size: 256
learning_rate: 0.001
latent_dim: 128
data_dir: data/UIEB/raw
reference_dir: data/UIEB/reference
reference_resized_dir: data/UIEB/reference_resized
val_split: 0.1
patience: 3
min_delta: 0.0
```

Training and evaluation scripts read values from this YAML. Edit the file to adjust runs.

FID (optional, requires `pytorch-fid` in this environment):

```powershell
& ./venv/Scripts/python.exe -m pytorch_fid ./runs/ae/generated ./data/UIEB/reference_resized --device cuda
& ./venv/Scripts/python.exe -m pytorch_fid ./runs/bvae/generated ./data/UIEB/reference_resized --device cuda
```

### Export images for FID

Use the helper script to export resized real images and generated samples:

```powershell
& ./venv/Scripts/python.exe ./scripts/export_images.py export-val-real --config configs/default.yaml
& ./venv/Scripts/python.exe ./scripts/export_images.py export-gen --model bvae --num 1000 --config configs/default.yaml
```

Then run FID against `data/UIEB/reference_resized` as shown above or simply run:

```powershell
& ./venv/Scripts/python.exe ./src/eval.py
```

Expected saved artifacts for report (under `reports/figures/`):
- ae_loss.png, bvae_loss.png
- ae_recon_grid.png, bvae_recon_grid.png
- bvae_uncertainty.png
- bvae_latent_traversal.png
- ae_metrics.txt, bvae_metrics.txt
```

