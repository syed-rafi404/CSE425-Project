import torch
import os
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from models import Autoencoder, BetaVAE
from data_loader import get_data_loader
from utils import load_config
from math import log10
import subprocess
import sys
import re
import csv
from typing import Optional, Tuple
try:
    from torchmetrics.image.inception import InceptionScore
except Exception:
    InceptionScore = None
from scipy.stats import wilcoxon

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg_eval = load_config('configs/default.yaml', {
    'batch_size': 16,
    'image_size': 256,
    'reference_dir': 'data/UIEB/reference',
    'reference_resized_dir': 'data/UIEB/reference_resized',
    'regenerate': False,
})
IMAGE_SIZE = int(cfg_eval.get('image_size', 256))
BATCH_SIZE = int(cfg_eval.get('batch_size', 16))

def prepare_resized_references(original_dir, resized_dir, image_size):
    if os.path.exists(resized_dir):
        print(f"Resized reference directory already exists: {resized_dir}")
        return
    os.makedirs(resized_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    image_files = [f for f in os.listdir(original_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for file_name in tqdm(image_files, desc="Resizing reference images for FID"):
        img_path = os.path.join(original_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        image.save(os.path.join(resized_dir, file_name))
    print("Finished creating resized reference images.")

def generate_reconstructions(model, data_loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, images in enumerate(tqdm(data_loader, desc=f"Generating reconstructions in {output_dir}")):
            images = images.to(device)
            if isinstance(model, BetaVAE):
                recon_images, _, _ = model(images)
            else:
                recon_images = model(images)
            for j, recon_image in enumerate(recon_images):
                save_image(recon_image, os.path.join(output_dir, f"recon_{i * BATCH_SIZE + j}.png"))

def psnr(mse):
    if mse <= 0:
        return float('inf')
    return 10 * log10(1.0 / mse)

def ssim_batch(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = torch.mean(x, dim=(2, 3), keepdim=True)
    mu_y = torch.mean(y, dim=(2, 3), keepdim=True)
    sigma_x = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True)
    sigma_y = torch.var(y, dim=(2, 3), unbiased=False, keepdim=True)
    sigma_xy = torch.mean((x - mu_x) * (y - mu_y), dim=(2, 3), keepdim=True)
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean().item()

def ssim_per_image(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = torch.mean(x, dim=(2, 3), keepdim=True)
    mu_y = torch.mean(y, dim=(2, 3), keepdim=True)
    sigma_x = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True)
    sigma_y = torch.var(y, dim=(2, 3), unbiased=False, keepdim=True)
    sigma_xy = torch.mean((x - mu_x) * (y - mu_y), dim=(2, 3), keepdim=True)
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    vals = ssim_map.mean(dim=(1, 2, 3))
    return vals.detach().cpu().numpy()

def run_fid(gen_dir, real_dir, device='cuda'):
    try:
        cmd = [sys.executable, '-m', 'pytorch_fid', gen_dir, real_dir, '--device', device]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        out = completed.stdout or ''
        m = re.search(r'FID:\s*([0-9]*\.?[0-9]+)', out)
        if m:
            return float(m.group(1)), out
        return None, out
    except Exception as e:
        return None, str(e)

def run_pytorch_fid(gen_dir, real_dir, device='cuda'):
    return run_fid(gen_dir, real_dir, device)

def compute_inception_score_from_dir(img_dir: str, device: str = 'cuda', batch_size: int = 64, resize_to: int = 299) -> Optional[Tuple[float, float]]:
    if InceptionScore is None:
        return None
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return None
    tf = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.PILToTensor(),
    ])
    metric = InceptionScore(splits=10).to(DEVICE)
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        imgs = []
        for name in batch_files:
            try:
                img = Image.open(os.path.join(img_dir, name)).convert('RGB')
                imgs.append(tf(img))
            except Exception:
                continue
        if not imgs:
            continue
        batch = torch.stack(imgs, dim=0)
        if batch.dtype != torch.uint8:
            batch = batch.to(torch.uint8)
        batch = batch.to(DEVICE)
        metric.update(batch)
    res = metric.compute()
    if isinstance(res, tuple) or isinstance(res, list):
        mean, std = res
    else:
        mean, std = res[0], res[1]
    return float(mean.item()), float(std.item())

def visualize_uncertainty(bvae_model, image_tensor, num_samples=25, output_file="uncertainty_visualization.png"):
    bvae_model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    reconstructions = []
    with torch.no_grad():
        for _ in range(num_samples):
            recon, _, _ = bvae_model(image_tensor)
            reconstructions.append(recon.squeeze(0).cpu())
    reconstructions = torch.stack(reconstructions)
    mean_recon = torch.mean(reconstructions, dim=0)
    variance_map = torch.var(reconstructions, dim=0)
    uncertainty_heatmap = variance_map.mean(dim=0, keepdim=True)
    uncertainty_heatmap = (uncertainty_heatmap - uncertainty_heatmap.min()) / (uncertainty_heatmap.max() - uncertainty_heatmap.min())
    output_grid = torch.cat([image_tensor.cpu().squeeze(0), mean_recon, uncertainty_heatmap.repeat(3, 1, 1)], dim=2)
    save_image(output_grid, output_file, normalize=True)
    print(f"Uncertainty visualization saved to {output_file}")


def main():
    print(f"Using device: {DEVICE}")
    REFERENCE_DIR = cfg_eval.get('reference_dir', 'data/UIEB/reference')
    RESIZED_REFERENCE_DIR = cfg_eval.get('reference_resized_dir', 'data/UIEB/reference_resized')
    prepare_resized_references(REFERENCE_DIR, RESIZED_REFERENCE_DIR, IMAGE_SIZE)
    eval_loader = get_data_loader(RESIZED_REFERENCE_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False)
    ae_model = Autoencoder().to(DEVICE)
    ae_model.load_state_dict(torch.load('runs/ae/ckpts/baseline_ae.pth'))
    bvae_model = BetaVAE().to(DEVICE)
    bvae_model.load_state_dict(torch.load('runs/bvae/ckpts/beta_vae.pth'))
    os.makedirs('runs/ae/generated', exist_ok=True)
    os.makedirs('runs/bvae/generated', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    if bool(cfg_eval.get('regenerate', False)):
        print("Generating reconstructions for baseline Autoencoder...")
        generate_reconstructions(ae_model, eval_loader, DEVICE, 'runs/ae/generated')
        print("Generating reconstructions for Beta-VAE...")
        generate_reconstructions(bvae_model, eval_loader, DEVICE, 'runs/bvae/generated')

    print("\nComputing simple PSNR/SSIM on a batch...")
    sample = next(iter(eval_loader)).to(DEVICE)
    with torch.no_grad():
        ae_out = ae_model(sample)
        bvae_out, _, _ = bvae_model(sample)
    mse_ae = torch.mean((ae_out - sample) ** 2).item()
    mse_bvae = torch.mean((bvae_out - sample) ** 2).item()
    print(f"AE PSNR: {psnr(mse_ae):.2f} dB, SSIM: {ssim_batch(ae_out, sample):.3f}")
    print(f"β-VAE PSNR: {psnr(mse_bvae):.2f} dB, SSIM: {ssim_batch(bvae_out, sample):.3f}")
    print("\nComputing Wilcoxon signed-rank test on per-image SSIM (AE vs β-VAE)...")
    ssim_ae_all = []
    ssim_bvae_all = []
    with torch.no_grad():
        for images in tqdm(eval_loader, desc="SSIM per-image"):
            images = images.to(DEVICE)
            ae_rec = ae_model(images)
            bvae_rec, _, _ = bvae_model(images)
            ssim_ae_all.extend(ssim_per_image(ae_rec, images).tolist())
            ssim_bvae_all.extend(ssim_per_image(bvae_rec, images).tolist())
    try:
        stat, pval = wilcoxon(ssim_ae_all, ssim_bvae_all, alternative='two-sided')
        print(f"Wilcoxon SSIM: stat={stat:.4f}, p-value={pval:.4e}")
    except Exception as e:
        stat, pval = None, None
        print(f"Wilcoxon failed: {e}")
    print("\nReconstructions for FID calculation are saved.")
    os.makedirs('reports', exist_ok=True)
    ae_fid, ae_out = run_pytorch_fid('runs/ae/generated', RESIZED_REFERENCE_DIR, device='cuda' if DEVICE.type == 'cuda' else 'cpu')
    bvae_fid, bvae_out = run_pytorch_fid('runs/bvae/generated', RESIZED_REFERENCE_DIR, device='cuda' if DEVICE.type == 'cuda' else 'cpu')

    def write_metric_file(path, fid_value, raw):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                if fid_value is not None:
                    f.write(f"FID: {fid_value:.4f}\n")
                else:
                    f.write("FID: N/A\n")
                f.write("\nRaw output:\n")
                f.write(raw)
        except Exception:
            pass

    write_metric_file('reports/ae_metrics.txt', ae_fid, ae_out)
    write_metric_file('reports/bvae_metrics.txt', bvae_fid, bvae_out)

    print(f"AE FID: {ae_fid if ae_fid is not None else 'N/A'} | β-VAE FID: {bvae_fid if bvae_fid is not None else 'N/A'}")
    print("\nComputing Inception Score (IS) on generated folders...")
    ae_is = compute_inception_score_from_dir('runs/ae/generated', device='cuda' if DEVICE.type == 'cuda' else 'cpu')
    bvae_is = compute_inception_score_from_dir('runs/bvae/generated', device='cuda' if DEVICE.type == 'cuda' else 'cpu')
    # Write a single clean CSV block per run
    rows = []
    if ae_fid is not None:
        rows.append({'model': 'AE', 'metric': 'FID', 'value': f"{ae_fid:.6f}"})
    if bvae_fid is not None:
        rows.append({'model': 'β‑VAE', 'metric': 'FID', 'value': f"{bvae_fid:.6f}"})
    if ae_is is not None:
        rows.append({'model': 'AE', 'metric': 'IS_mean', 'value': f"{ae_is[0]:.6f}"})
        rows.append({'model': 'AE', 'metric': 'IS_std', 'value': f"{ae_is[1]:.6f}"})
    if bvae_is is not None:
        rows.append({'model': 'β‑VAE', 'metric': 'IS_mean', 'value': f"{bvae_is[0]:.6f}"})
        rows.append({'model': 'β‑VAE', 'metric': 'IS_std', 'value': f"{bvae_is[1]:.6f}"})
    if 'stat' in locals() and 'pval' in locals() and stat is not None and pval is not None:
        rows.append({'model': 'AE_vs_Beta‑VAE', 'metric': 'Wilcoxon_SSIM_stat', 'value': f"{stat:.6f}"})
        rows.append({'model': 'AE_vs_Beta‑VAE', 'metric': 'Wilcoxon_SSIM_p', 'value': f"{pval:.6e}"})
    os.makedirs('reports', exist_ok=True)
    with open('reports/results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)
    summary = (
        f"AE FID={ae_fid if ae_fid is not None else 'N/A'} | "
        f"β-VAE FID={bvae_fid if bvae_fid is not None else 'N/A'} | "
        f"AE IS={(f'{ae_is[0]:.4f}±{ae_is[1]:.4f}' if ae_is else 'N/A')} | "
        f"β-VAE IS={(f'{bvae_is[0]:.4f}±{bvae_is[1]:.4f}' if bvae_is else 'N/A')} | "
        f"Wilcoxon(stat={(f'{stat:.4f}' if stat is not None else 'N/A')}, p={(f'{pval:.2e}' if pval is not None else 'N/A')})"
    )
    print(summary)
    print("\nVisualizing uncertainty for a sample image...")
    sample_images = next(iter(eval_loader))
    single_image = sample_images[0]
    visualize_uncertainty(bvae_model, single_image, output_file='reports/figures/bvae_uncertainty.png')

if __name__ == '__main__':
    main()
