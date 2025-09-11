import os

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

