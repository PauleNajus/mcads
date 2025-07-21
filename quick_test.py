#!/usr/bin/env python3
import os
import sys
import django

# Apply PyTorch fixes
os.environ['MKLDNN_ENABLED'] = '0'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Setup Django
sys.path.insert(0, '/opt/mcads/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mcads_project.settings')
django.setup()

from xrayapp.models import XRayImage

print("🔍 Quick X-ray Analysis Fix Verification")
print("="*50)

try:
    # Check if we can load Django models
    xray_count = XRayImage.objects.count()
    print(f"✅ Django working - Found {xray_count} X-ray images in database")
    
    # Check recent processing status
    recent = XRayImage.objects.order_by('-uploaded_at').first()
    if recent:
        print(f"📷 Most recent X-ray: ID {recent.id}")
        print(f"   Status: {recent.processing_status}")
        print(f"   Progress: {recent.progress}%")
    
    # Test basic PyTorch import
    import torch
    print(f"✅ PyTorch {torch.__version__} working")
    print(f"   MKL-DNN disabled: {not torch.backends.mkldnn.enabled}")
    
    print("\n🎉 All basic components are working!")
    print("✅ The threading fix should resolve the 0% stuck issue")
    print("✅ Try uploading a new X-ray image through the web interface")
    
except Exception as e:
    print(f"❌ Error: {e}")
    
print("="*50) 