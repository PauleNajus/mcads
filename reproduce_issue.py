import torchxrayvision as xrv
import torch
import sys

try:
    print("Attempting to load ResNet with weights='resnet50-res512-all'")
    model = xrv.models.ResNet(weights="resnet50-res512-all")
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
