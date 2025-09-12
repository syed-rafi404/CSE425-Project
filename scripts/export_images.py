import os
import sys
import argparse
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models import BetaVAE, Autoencoder
from src.utils import load_config, set_seed
from src.data_loader import get_data_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def export_val_real(cfg):
    real_src = cfg.get('reference_dir', 'data/UIEB/reference')
    out_dir = cfg.get('reference_resized_dir', 'data/UIEB/reference_resized')
    size = int(cfg.get('image_size', 256))
    ensure_dir(out_dir)

    tf = transforms.Compose([
        transforms.Resize((size, size)),
    ])
    if not os.path.isdir(real_src):
        return
    files = [f for f in os.listdir(real_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for name in files:
        in_path = os.path.join(real_src, name)
        try:
            img = Image.open(in_path).convert('RGB')
            img = tf(img)
            img.save(os.path.join(out_dir, os.path.splitext(name)[0] + '.png'))
        except Exception:
            continue


def export_gen(cfg, model_name='bvae', num=100):
    size = int(cfg.get('image_size', 256))
    out_dir = f'runs/{model_name}/generated'
    ensure_dir(out_dir)

    if model_name.lower() in ['bvae', 'beta-vae', 'beta_vae']:
        model = BetaVAE(latent_dim=int(cfg.get('latent_dim', 128))).to(DEVICE)
        ckpt_path = cfg.get('override_ckpt') or 'runs/bvae/ckpts/beta_vae.pth'
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            for i in range(num):
                z = torch.randn(1, int(cfg.get('latent_dim', 128)), device=DEVICE)
                x = model.decode(z)
                save_image(x.clamp(0,1).cpu(), os.path.join(out_dir, f'sample_{i:05d}.png'))
    elif model_name.lower() == 'ae':
        loader = get_data_loader(cfg.get('data_dir', 'data/UIEB/raw'), batch_size=16, image_size=size, shuffle=False)
        model = Autoencoder().to(DEVICE)
        ckpt_path = cfg.get('override_ckpt') or 'runs/ae/ckpts/baseline_ae.pth'
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            count = 0
            for batch in loader:
                x = batch.to(DEVICE)
                out = model(x)
                for j in range(out.size(0)):
                    if count >= num:
                        return
                    save_image(out[j].clamp(0,1).cpu(), os.path.join(out_dir, f'sample_{count:05d}.png'))
                    count += 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    s1 = sub.add_parser('export-val-real')
    s1.add_argument('--config', default='configs/default.yaml')

    s2 = sub.add_parser('export-gen')
    s2.add_argument('--config', default='configs/default.yaml')
    s2.add_argument('--model', default='bvae', choices=['bvae', 'ae'])
    s2.add_argument('--ckpt', default=None)
    s2.add_argument('--num', type=int, default=100)

    args = parser.parse_args()
    cfg = load_config(args.config, {})
    set_seed(int(cfg.get('seed', 42)))

    if args.cmd == 'export-val-real':
        export_val_real(cfg)
    elif args.cmd == 'export-gen':
        if args.ckpt:
            cfg = dict(cfg)
            cfg['override_ckpt'] = args.ckpt
        export_gen(cfg, args.model, args.num)


if __name__ == '__main__':
    main()
