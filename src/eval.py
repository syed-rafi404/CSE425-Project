import torch
import os
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from models import Autoencoder, BetaVAE
from data_loader import get_data_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
BATCH_SIZE = 16

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
    REFERENCE_DIR = 'data/UIEB/reference'
    RESIZED_REFERENCE_DIR = 'data/UIEB/reference_resized'
    prepare_resized_references(REFERENCE_DIR, RESIZED_REFERENCE_DIR, IMAGE_SIZE)
    eval_loader = get_data_loader(RESIZED_REFERENCE_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False)
    ae_model = Autoencoder().to(DEVICE)
    ae_model.load_state_dict(torch.load('runs/ae/ckpts/baseline_ae.pth'))
    bvae_model = BetaVAE().to(DEVICE)
    bvae_model.load_state_dict(torch.load('runs/bvae/ckpts/beta_vae.pth'))
    print("Generating reconstructions for baseline Autoencoder...")
    generate_reconstructions(ae_model, eval_loader, DEVICE, 'runs/ae/generated')
    print("Generating reconstructions for Beta-VAE...")
    generate_reconstructions(bvae_model, eval_loader, DEVICE, 'runs/bvae/generated')
    print("\nReconstructions for FID calculation are saved.")
    print("To calculate FID, run the following commands in your terminal:")
    print(f"python -m pytorch_fid runs/ae/generated {RESIZED_REFERENCE_DIR} --device cuda")
    print(f"python -m pytorch_fid runs/bvae/generated {RESIZED_REFERENCE_DIR} --device cuda")
    print("\nVisualizing uncertainty for a sample image...")
    sample_images = next(iter(eval_loader))
    single_image = sample_images[0]
    visualize_uncertainty(bvae_model, single_image, output_file='reports/figures/bvae_uncertainty.png')

if __name__ == '__main__':
    main()
