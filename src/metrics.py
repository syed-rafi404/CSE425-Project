import torch
from math import log10

def mse_batch(x: torch.Tensor, y: torch.Tensor) -> float:
	return torch.mean((x - y) ** 2).item()

def psnr_from_mse(mse: float) -> float:
	if mse <= 0:
		return float('inf')
	return 10 * log10(1.0 / mse)

def ssim_batch(x: torch.Tensor, y: torch.Tensor) -> float:
	C1 = 0.01 ** 2
	C2 = 0.03 ** 2
	mu_x = torch.mean(x, dim=(2, 3), keepdim=True)
	mu_y = torch.mean(y, dim=(2, 3), keepdim=True)
	sigma_x = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True)
	sigma_y = torch.var(y, dim=(2, 3), unbiased=False, keepdim=True)
	sigma_xy = torch.mean((x - mu_x) * (y - mu_y), dim=(2, 3), keepdim=True)
	ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
		(mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
	)
	return ssim_map.mean().item()

