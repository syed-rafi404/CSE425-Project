import os
import torch
from torch import nn, optim
from torchvision.utils import save_image
from data_loader import get_data_loader
from utils import load_config
from models import Autoencoder

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config('configs/default.yaml', {
        'batch_size': 16,
        'epochs': 5,
    })
    DATA_DIR = 'data/UIEB/raw'
    BATCH_SIZE = int(cfg.get('batch_size', 16))
    IMAGE_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = int(cfg.get('epochs', 5))

    print(f"Using device: {DEVICE}")

    loader = get_data_loader(DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    model = Autoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs('runs/ae/samples', exist_ok=True)
    os.makedirs('runs/ae/ckpts', exist_ok=True)

    for epoch in range(EPOCHS):
        for i, images in enumerate(loader):
            images = images.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}')

        sample_images = next(iter(loader)).to(DEVICE)
        with torch.no_grad():
            reconstructed_images = model(sample_images)
        comparison = torch.cat([sample_images[:8], reconstructed_images[:8]])
        save_image(comparison.cpu(), f'runs/ae/samples/reconstruction_epoch_{epoch+1}.png', nrow=8, normalize=True)

    print("Finished Training Baseline AE.")
    torch.save(model.state_dict(), 'runs/ae/ckpts/baseline_ae.pth')
    print("Model checkpoint saved to runs/ae/ckpts/baseline_ae.pth")

if __name__ == '__main__':
    main()
