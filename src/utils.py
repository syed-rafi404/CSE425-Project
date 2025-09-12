import os
import random
import numpy as np
import torch

def load_config(path, defaults=None):
	cfg = {} if defaults is None else dict(defaults)
	if not os.path.exists(path):
		return cfg
	try:
		import yaml
		with open(path, 'r', encoding='utf-8') as f:
			data = yaml.safe_load(f) or {}
		if isinstance(data, dict):
			cfg.update(data)
	except Exception:
		return cfg
	return cfg

def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

