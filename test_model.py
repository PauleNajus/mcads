#!/usr/bin/env python3
"""
Simple test script to verify PyTorch and TorchXRayVision are working
"""
import os
import sys

# Fix PyTorch CPU backend issues BEFORE importing torch
os.environ['MKLDNN_ENABLED'] = '0'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import django

# Add the project directory to Python path
sys.path.insert(0, '/opt/mcads/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mcads_project.settings')
django.setup()

import torch
import torchxrayvision as xrv
import skimage.io
import numpy as np

# Configure PyTorch CPU backend for compatibility
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

def test_model_loading():
    """Test if models can be loaded successfully"""
    print("Testing PyTorch and TorchXRayVision...")
    print(f"PyTorch backend configuration:")
    print(f"  - MKL-DNN enabled: {torch.backends.mkldnn.enabled}")
    print(f"  - Number of threads: {torch.get_num_threads()}")
    
    try:
        # Test CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Test DenseNet model loading
        print("Loading DenseNet model...")
        densenet_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        densenet_model.to(device)
        densenet_model.eval()
        print("✓ DenseNet model loaded successfully")
        
        # Test ResNet model loading
        print("Loading ResNet model...")
        resnet_model = xrv.models.ResNet(weights="resnet50-res512-all")
        resnet_model.to(device)
        resnet_model.eval()
        print("✓ ResNet model loaded successfully")
        
        # Test with dummy data
        print("Testing model inference...")
        dummy_input = torch.randn(1, 1, 224, 224).to(device)
        
        with torch.no_grad():
            densenet_output = densenet_model(dummy_input).cpu()
            print(f"✓ DenseNet inference successful, output shape: {densenet_output.shape}")
            
        dummy_input_512 = torch.randn(1, 1, 512, 512).to(device) 
        with torch.no_grad():
            resnet_output = resnet_model(dummy_input_512).cpu()
            print(f"✓ ResNet inference successful, output shape: {resnet_output.shape}")
        
        print("\n🎉 All tests passed! ML functionality is working properly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1) 