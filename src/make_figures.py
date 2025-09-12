import os
import csv
import math
import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image
from data_loader import get_data_loader
from models import Autoencoder, BetaVAE
from utils import load_config, set_seed
from metrics import mse_batch, psnr_from_mse, ssim_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dirs():
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('runs/ae/generated', exist_ok=True)
    os.makedirs('runs/bvae/generated', exist_ok=True)


def save_loss_curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots = [
        ('runs/logs/ae_log.csv', 'AE Loss', 'reports/figures/ae_loss.png'),
        ('runs/logs/bvae_log.csv', 'β-VAE Loss', 'reports/figures/bvae_loss.png'),
    ]
    for path, title, out in plots:
        if not os.path.exists(path):
            continue
        epochs, train, val = [], [], []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                train.append(float(row['train_loss']))
                val.append(float(row['val_loss']))
        plt.figure(figsize=(6,4))
        plt.plot(epochs, train, label='Train')
        plt.plot(epochs, val, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()


def save_recon_grid(model, loader, out_path):
    model.eval()
    images = next(iter(loader))[:8].to(DEVICE)
    with torch.no_grad():
        if isinstance(model, BetaVAE):
            recon, _, _ = model(images)
        else:
            recon = model(images)
    grid = torch.cat([images, recon], dim=0)
    grid = make_grid(grid, nrow=8, normalize=True)
    save_image(grid, out_path)


def compute_full_metrics(model, loader):
    model.eval()
    total_mse = 0.0
    total_ssim = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if isinstance(model, BetaVAE):
                recon, _, _ = model(batch)
            else:
                recon = model(batch)
            total_mse += torch.mean((recon - batch) ** 2).item() * batch.size(0)
            total_ssim += ssim_batch(recon, batch) * batch.size(0)
            n += batch.size(0)
    avg_mse = total_mse / max(1, n)
    avg_psnr = psnr_from_mse(avg_mse)
    avg_ssim = total_ssim / max(1, n)
    return avg_mse, avg_psnr, avg_ssim


def main():
    cfg = load_config('configs/default.yaml', {
        'batch_size': 16,
        'image_size': 256,
        'data_dir': 'data/UIEB/raw',
        'seed': 42,
    })
    set_seed(int(cfg.get('seed', 42)))
    ensure_dirs()

    loader = get_data_loader(cfg.get('data_dir', 'data/UIEB/raw'), batch_size=int(cfg.get('batch_size',16)), image_size=int(cfg.get('image_size',256)), shuffle=False)

    ae = Autoencoder().to(DEVICE)
    bvae = BetaVAE(latent_dim=int(cfg.get('latent_dim', 128))).to(DEVICE)
    ae.load_state_dict(torch.load('runs/ae/ckpts/baseline_ae.pth', map_location=DEVICE))
    bvae.load_state_dict(torch.load('runs/bvae/ckpts/beta_vae.pth', map_location=DEVICE))

    save_recon_grid(ae, loader, 'reports/figures/ae_recon_grid.png')
    save_recon_grid(bvae, loader, 'reports/figures/bvae_recon_grid.png')

    mse, psnr, ssim = compute_full_metrics(ae, loader)
    with open('reports/figures/ae_metrics.txt','w') as f:
        f.write(f"MSE: {mse:.6f}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}\n")

    mse, psnr, ssim = compute_full_metrics(bvae, loader)
    with open('reports/figures/bvae_metrics.txt','w') as f:
        f.write(f"MSE: {mse:.6f}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}\n")

    save_loss_curves()
    print('Figures and metrics saved under reports/figures')

if __name__ == '__main__':
    main()
