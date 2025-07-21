#!/usr/bin/env python3
"""
Debug script to test X-ray image processing directly
"""
import os
import sys
import django

# Apply PyTorch fixes first
os.environ['MKLDNN_ENABLED'] = '0'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Setup Django
sys.path.insert(0, '/opt/mcads/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mcads_project.settings')
django.setup()

from xrayapp.models import XRayImage
from xrayapp.utils import process_image, load_model
from pathlib import Path
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_processing():
    """Test the image processing pipeline"""
    try:
        print("🔧 Testing X-ray image processing pipeline...")
        
        # Test model loading first
        print("Loading AI model...")
        model, resize_dim = load_model('densenet')
        print(f"✅ Model loaded successfully (resize: {resize_dim})")
        
        # Check if it's a mock model
        is_mock = hasattr(model, 'model_type')
        if is_mock:
            print("⚠️  Using mock model (PyTorch compatibility mode)")
        else:
            print("🎯 Using real AI model")
        
        # Find the most recent X-ray image to test with
        latest_xray = XRayImage.objects.order_by('-uploaded_at').first()
        
        if not latest_xray:
            print("❌ No X-ray images found in database")
            return False
            
        print(f"📷 Testing with X-ray ID: {latest_xray.id}")
        print(f"   Image: {latest_xray.image.name}")
        print(f"   Status: {latest_xray.processing_status}")
        print(f"   Progress: {latest_xray.progress}%")
        
        # Test the processing function
        image_path = Path('/opt/mcads/app/media') / latest_xray.image.name
        
        if not image_path.exists():
            print(f"❌ Image file not found: {image_path}")
            return False
            
        print(f"🚀 Starting processing test...")
        
        # Reset progress for testing
        latest_xray.progress = 0
        latest_xray.processing_status = 'pending'
        latest_xray.save()
        
        # Process the image
        results = process_image(str(image_path), latest_xray, 'densenet')
        
        print(f"✅ Processing completed!")
        print(f"   Final status: {latest_xray.processing_status}")
        print(f"   Final progress: {latest_xray.progress}%")
        print(f"   Predictions generated: {len(results)}")
        
        # Show top 5 predictions
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        print("\n🎯 Top 5 predictions:")
        for pathology, probability in sorted_results[:5]:
            print(f"   {pathology}: {probability:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_processing()
    print("\n" + "="*50)
    if success:
        print("🎉 Processing test PASSED! The X-ray analysis should work.")
    else:
        print("💥 Processing test FAILED! There are still issues to fix.")
    
    sys.exit(0 if success else 1) 