import os
import torch
from torch import nn, optim
from torchvision.utils import save_image
from data_loader import get_data_loader
from models import BetaVAE

def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + beta * kld) / x.size(0)

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = 'data/UIEB/raw'
    BATCH_SIZE = 16
    IMAGE_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    BETA = 1.0

    print(f"Using device: {DEVICE}")

    loader = get_data_loader(DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    model = BetaVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for i, images in enumerate(loader):
            images = images.to(DEVICE)
            recon_images, mu, log_var = model(images)
            loss = vae_loss_function(recon_images, images, mu, log_var, beta=BETA)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}')

        sample_images = next(iter(loader))[:8].to(DEVICE)
        with torch.no_grad():
            reconstructed_images, _, _ = model(sample_images)
        comparison = torch.cat([sample_images, reconstructed_images])
        save_image(comparison.cpu(), f'runs/bvae/samples/reconstruction_epoch_{epoch+1}.png', nrow=8, normalize=True)

    print("Finished Training β-VAE.")
    torch.save(model.state_dict(), 'runs/bvae/ckpts/beta_vae.pth')
    print("Model checkpoint saved to runs/bvae/ckpts/beta_vae.pth")

if __name__ == '__main__':
    main()
