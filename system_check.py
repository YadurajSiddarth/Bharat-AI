import platform
import torch
import transformers
import datasets

print("=" * 45)
print("       BHARAT-AI SYSTEM DIAGNOSTICS")
print("=" * 45)

print("Python:", platform.python_version())
print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)

print("\nApple GPU (MPS):", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.rand(3, 3, device=device)
    print("\nMPS calculation successful:")
    print(x)
else:
    print("\nMPS unavailable. CPU will be used.")

print("\nBHARAT-AI environment check completed.")
