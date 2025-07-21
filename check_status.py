#!/usr/bin/env python3
"""
Check the status of the MCADS application and PyTorch integration
"""
import os
import sys

# Apply PyTorch fixes
os.environ['MKLDNN_ENABLED'] = '0'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Add project path
sys.path.insert(0, '/opt/mcads/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mcads_project.settings')

def check_pytorch():
    """Check if PyTorch is working"""
    try:
        import torch
        torch.backends.mkldnn.enabled = False
        torch.set_num_threads(1)
        
        print("✅ PyTorch configuration:")
        print(f"  - Version: {torch.__version__}")
        print(f"  - MKL-DNN enabled: {torch.backends.mkldnn.enabled}")
        print(f"  - Number of threads: {torch.get_num_threads()}")
        
        # Test basic operations
        x = torch.randn(2, 3)
        y = x + 1
        print(f"  - Basic operations: ✅ {y.shape}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def check_django():
    """Check if Django is working"""
    try:
        import django
        django.setup()
        
        from django.conf import settings
        print(f"✅ Django {django.get_version()} configured successfully")
        print(f"  - Debug mode: {settings.DEBUG}")
        print(f"  - Database: {settings.DATABASES['default']['ENGINE']}")
        
        return True
    except Exception as e:
        print(f"❌ Django error: {e}")
        return False

def check_xray_models():
    """Check if X-ray models can be loaded"""
    try:
        from xrayapp.utils import load_model
        
        print("Testing model loading...")
        model, resize_dim = load_model('densenet')
        print(f"✅ DenseNet model loaded (resize: {resize_dim})")
        
        # Check if it's a mock model
        is_mock = hasattr(model, 'model_type')
        if is_mock:
            print("  ⚠️  Using mock model (PyTorch compatibility mode)")
        else:
            print("  🎯 Using real AI model")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def main():
    print("🔍 MCADS Application Status Check")
    print("=" * 50)
    
    pytorch_ok = check_pytorch()
    print()
    
    django_ok = check_django()
    print()
    
    models_ok = check_xray_models()
    print()
    
    print("=" * 50)
    if pytorch_ok and django_ok and models_ok:
        print("🎉 All systems working! X-ray analysis should function properly.")
        return 0
    else:
        print("⚠️  Some issues detected. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 