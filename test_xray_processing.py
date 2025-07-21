#!/usr/bin/env python3
"""
Test X-ray processing to verify 75% hang issue is fixed
"""

import os
import sys
import django
from pathlib import Path
import shutil

# Setup Django
sys.path.insert(0, '/opt/mcads/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mcads_project.settings')
django.setup()

from xrayapp.models import XRayImage
from xrayapp.utils import process_image

def test_processing():
    """Test X-ray processing with a sample image"""
    print("🧪 Testing X-ray processing to verify 75% hang is fixed...")
    
    # Find a test image
    test_images_dir = Path('/opt/mcads/app/tests')
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob('*.jpeg')) + list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png'))
        if test_images:
            test_image = test_images[0]
            print(f"Using test image: {test_image}")
            
            # Test processing without database instance (faster)
            try:
                print("Starting processing test...")
                results = process_image(test_image, None, 'densenet')
                print("✅ Processing completed successfully!")
                print(f"Results: {len(results)} pathologies predicted")
                
                # Show top predictions
                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
                print("Top 5 predictions:")
                for pathology, score in sorted_results:
                    print(f"  {pathology}: {score:.3f}")
                
                return True
                
            except Exception as e:
                print(f"❌ Processing failed: {str(e)}")
                return False
        else:
            print("❌ No test images found in tests/ directory")
            return False
    else:
        print("❌ Tests directory not found")
        return False

if __name__ == "__main__":
    success = test_processing()
    if success:
        print("\n🎉 X-ray processing test PASSED - 75% hang issue is FIXED!")
    else:
        print("\n❌ X-ray processing test FAILED - issue may still exist") 