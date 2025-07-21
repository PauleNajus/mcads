#!/usr/bin/env python3
"""
Simple PyTorch test without Django dependencies
"""
import os

# Apply PyTorch fixes
os.environ['MKLDNN_ENABLED'] = '0'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

try:
    import torch
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    
    print("✅ PyTorch imported successfully")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MKL-DNN enabled: {torch.backends.mkldnn.enabled}")
    print(f"Number of threads: {torch.get_num_threads()}")
    
    # Test simple tensor operation
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = x + y
    print(f"✅ Basic tensor operations work: {z.shape}")
    
    # Test neural network operation
    import torch.nn as nn
    layer = nn.Linear(3, 1)
    output = layer(torch.randn(1, 3))
    print(f"✅ Neural network operations work: {output.shape}")
    
    print("\n🎉 PyTorch is working correctly with the fixes!")
    
except Exception as e:
    print(f"❌ PyTorch test failed: {e}")
    import traceback
    traceback.print_exc() 