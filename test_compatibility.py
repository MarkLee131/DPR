import torch

print(torch.__version__)
print(torch.version.cuda)
import torch
# import torchvision

print("PyTorch Version:", torch.__version__)
# print("Torchvision Version:", torchvision.__version__)
print("CUDA Available:", torch.cuda.is_available())
