import torch, os
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
print("Device 0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")