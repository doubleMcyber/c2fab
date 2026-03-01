import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(f"Success! Apple Silicon GPU is active. Tensor: {x}")
else:
    print("MPS device not found. Something is wrong with the PyTorch install.")