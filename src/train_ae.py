import os
import torch
from torch import nn, optim
from torchvision.utils import save_image
from data_loader import get_data_loader, get_train_val_loaders
from utils import load_config, set_seed
from models import Autoencoder

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config('configs/default.yaml', {
        'batch_size': 16,
        'epochs': 5,
        'seed': 42,
        'image_size': 256,
        'learning_rate': 1e-3,
        'data_dir': 'data/UIEB/raw',
    })
    DATA_DIR = cfg.get('data_dir', 'data/UIEB/raw')
    set_seed(int(cfg.get('seed', 42)))
    BATCH_SIZE = int(cfg.get('batch_size', 16))
    IMAGE_SIZE = int(cfg.get('image_size', 256))
    LEARNING_RATE = float(cfg.get('learning_rate', 1e-3))
    EPOCHS = int(cfg.get('epochs', 5))

    print(f"Using device: {DEVICE}")

    train_loader, val_loader = get_train_val_loaders(DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, val_split=float(cfg.get('val_split', 0.1)))
    model = Autoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs('runs/ae/samples', exist_ok=True)
    os.makedirs('runs/ae/ckpts', exist_ok=True)

    best_val = float('inf')
    patience = int(cfg.get('patience', 3))
    min_delta = float(cfg.get('min_delta', 0.0))
    wait = 0
    import csv
    os.makedirs('runs/logs', exist_ok=True)
    log_path = 'runs/logs/ae_log.csv'
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for i, images in enumerate(train_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, images)
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
                outputs = model(images)
                vloss = criterion(outputs, images)
                val_loss += vloss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        model.train()

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}"])
        print(f'Epoch [{epoch+1}/{EPOCHS}] Train: {train_loss:.4f} | Val: {val_loss:.4f}')

        if val_loss + min_delta < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), 'runs/ae/ckpts/baseline_ae.pth')
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping triggered.')
                break

        sample_images = next(iter(train_loader)).to(DEVICE)
        with torch.no_grad():
            reconstructed_images = model(sample_images)
        comparison = torch.cat([sample_images[:8], reconstructed_images[:8]])
        save_image(comparison.cpu(), f'runs/ae/samples/reconstruction_epoch_{epoch+1}.png', nrow=8, normalize=True)

    print("Finished Training Baseline AE.")
    print("Best model checkpoint saved to runs/ae/ckpts/baseline_ae.pth")

if __name__ == '__main__':
    main()
