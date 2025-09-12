import sys
import platform

try:
    import torch
except Exception:
    torch = None
try:
    import torchvision
except Exception:
    torchvision = None
try:
    import torchmetrics
except Exception:
    torchmetrics = None
try:
    import scipy
except Exception:
    scipy = None

print(f"Python: {platform.python_version()}")
print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")

if torch is not None:
    print(f"torch: {getattr(torch, '__version__', 'unknown')}")
    cuda_ver = getattr(getattr(torch, 'version', None), 'cuda', None)
    print(f"torch.cuda: {cuda_ver}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            cc = torch.cuda.get_device_capability(idx)
            print(f"gpu: {name} (compute capability {cc[0]}.{cc[1]})")
        except Exception:
            print("gpu: unknown")
else:
    print("torch: not installed")

if torchvision is not None:
    print(f"torchvision: {getattr(torchvision, '__version__', 'unknown')}")
else:
    print("torchvision: not installed")

if torchmetrics is not None:
    print(f"torchmetrics: {getattr(torchmetrics, '__version__', 'unknown')}")
else:
    print("torchmetrics: not installed")

if scipy is not None:
    print(f"scipy: {getattr(scipy, '__version__', 'unknown')}")
else:
    print("scipy: not installed")
