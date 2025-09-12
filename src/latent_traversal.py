import os
import torch
from torchvision.utils import save_image, make_grid
from models import BetaVAE
from utils import load_config, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    cfg = load_config('configs/default.yaml', {'latent_dim': 128, 'seed': 42})
    set_seed(int(cfg.get('seed', 42)))
    os.makedirs('reports/figures', exist_ok=True)
    model = BetaVAE(latent_dim=int(cfg.get('latent_dim',128))).to(DEVICE)
    model.load_state_dict(torch.load('runs/bvae/ckpts/beta_vae.pth', map_location=DEVICE))
    model.eval()

    z_dim = int(cfg.get('latent_dim',128))
    steps = 9
    z_base = torch.zeros(1, z_dim, device=DEVICE)
    imgs = []
    with torch.no_grad():
        for d in range(min(8, z_dim)):
            for t in torch.linspace(-2.0, 2.0, steps):
                z = z_base.clone()
                z[0, d] = t
                x = model.decode(z)
                imgs.append(x.squeeze(0).cpu())
    grid = make_grid(torch.stack(imgs), nrow=steps, normalize=True)
    save_image(grid, 'reports/figures/bvae_latent_traversal.png')
    print('Saved traversal to reports/figures/bvae_latent_traversal.png')

if __name__ == '__main__':
    main()
