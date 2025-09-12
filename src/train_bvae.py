import os
import torch
from torch import nn, optim
from torchvision.utils import save_image
from data_loader import get_data_loader, get_train_val_loaders
from models import BetaVAE
from utils import load_config, set_seed

def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + beta * kld) / x.size(0)

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config('configs/default.yaml', {
        'batch_size': 16,
        'epochs': 10,
        'beta': 1.0,
        'seed': 42,
        'image_size': 256,
        'learning_rate': 1e-3,
        'data_dir': 'data/UIEB/raw',
        'latent_dim': 128,
    })
    DATA_DIR = cfg.get('data_dir', 'data/UIEB/raw')
    set_seed(int(cfg.get('seed', 42)))
    BATCH_SIZE = int(cfg.get('batch_size', 16))
    IMAGE_SIZE = int(cfg.get('image_size', 256))
    LEARNING_RATE = float(cfg.get('learning_rate', 1e-3))
    EPOCHS = int(cfg.get('epochs', 10))
    BETA = float(cfg.get('beta', 1.0))

    print(f"Using device: {DEVICE}")

    train_loader, val_loader = get_train_val_loaders(DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, val_split=float(cfg.get('val_split', 0.1)))
    model = BetaVAE(latent_dim=int(cfg.get('latent_dim', 128))).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs('runs/bvae/samples', exist_ok=True)
    os.makedirs('runs/bvae/ckpts', exist_ok=True)

    best_val = float('inf')
    patience = int(cfg.get('patience', 3))
    min_delta = float(cfg.get('min_delta', 0.0))
    wait = 0
    import csv
    os.makedirs('runs/logs', exist_ok=True)
    log_path = 'runs/logs/bvae_log.csv'
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for i, images in enumerate(train_loader):
            images = images.to(DEVICE)
            recon_images, mu, log_var = model(images)
            loss = vae_loss_function(recon_images, images, mu, log_var, beta=BETA)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        train_loss = epoch_loss / (len(train_loader.dataset))
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(DEVICE)
                recon_images, mu, log_var = model(images)
                vloss = vae_loss_function(recon_images, images, mu, log_var, beta=BETA)
                val_loss += vloss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        model.train()

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}"])
        print(f'Epoch [{epoch+1}/{EPOCHS}] Train: {train_loss:.4f} | Val: {val_loss:.4f}')

        if val_loss + min_delta < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), 'runs/bvae/ckpts/beta_vae.pth')
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping triggered.')
                break

        sample_images = next(iter(train_loader))[:8].to(DEVICE)
        with torch.no_grad():
            reconstructed_images, _, _ = model(sample_images)
        comparison = torch.cat([sample_images, reconstructed_images])
        save_image(comparison.cpu(), f'runs/bvae/samples/reconstruction_epoch_{epoch+1}.png', nrow=8, normalize=True)

    print("Finished Training β-VAE.")
    print("Best model checkpoint saved to runs/bvae/ckpts/beta_vae.pth")

if __name__ == '__main__':
    main()
