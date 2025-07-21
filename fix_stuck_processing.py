#!/usr/bin/env python3
"""
MCADS - Fix Stuck X-ray Processing
Diagnoses and fixes X-ray processing stuck at 75% progress
"""

import os
import sys
import django
import torch
import threading
import time
from datetime import datetime, timedelta

# Setup Django
sys.path.insert(0, '/opt/mcads/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mcads_project.settings')
django.setup()

from xrayapp.models import XRayImage
from xrayapp.utils import load_model, process_image
from pathlib import Path

class ProcessingFixer:
    def __init__(self):
        self.timeout_seconds = 120  # 2 minutes timeout for model inference
        
    def find_stuck_records(self):
        """Find X-ray records stuck in processing"""
        print("🔍 Scanning for stuck X-ray processing records...")
        
        # Find records stuck in processing for more than 10 minutes
        stuck_threshold = datetime.now() - timedelta(minutes=10)
        stuck_records = XRayImage.objects.filter(
            processing_status='processing',
            uploaded_at__lt=stuck_threshold
        )
        
        print(f"Found {stuck_records.count()} stuck records")
        
        # Also check for records at exactly 75%
        at_75_percent = XRayImage.objects.filter(
            progress=75,
            processing_status='processing'
        )
        
        print(f"Found {at_75_percent.count()} records stuck at 75%")
        
        return list(stuck_records) + list(at_75_percent)
    
    def diagnose_ml_system(self):
        """Diagnose the ML model loading system"""
        print("\n🧠 Diagnosing ML Model System...")
        
        try:
            # Test model loading
            print("Testing DenseNet model loading...")
            start_time = time.time()
            model, resize_dim = load_model('densenet')
            load_time = time.time() - start_time
            
            print(f"✅ Model loaded in {load_time:.2f} seconds")
            print(f"   Model type: {type(model)}")
            print(f"   Resize dimension: {resize_dim}")
            
            # Check if it's a mock model
            is_mock = hasattr(model, 'model_type')
            if is_mock:
                print("⚠️  Using mock model (this explains the issue!)")
                return False
            else:
                print("✅ Real PyTorch model loaded")
            
            # Test a simple inference
            print("Testing model inference...")
            test_input = torch.randn(1, 1, resize_dim, resize_dim)
            
            start_time = time.time()
            with torch.no_grad():
                test_output = model(test_input)
            inference_time = time.time() - start_time
            
            print(f"✅ Test inference completed in {inference_time:.2f} seconds")
            print(f"   Output shape: {test_output.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ ML system diagnosis failed: {str(e)}")
            return False
    
    def fix_stuck_record(self, xray_record):
        """Fix a single stuck X-ray record"""
        print(f"\n🔧 Fixing X-ray ID {xray_record.id}...")
        
        try:
            # Get the image path
            image_path = Path('/opt/mcads/app/media') / xray_record.image.name
            
            if not image_path.exists():
                print(f"❌ Image file not found: {image_path}")
                xray_record.processing_status = 'error'
                xray_record.save()
                return False
            
            # Reset progress to 70% and try processing again
            xray_record.progress = 70
            xray_record.save()
            
            print(f"Processing image: {image_path}")
            
            # Process with timeout
            result = self.process_with_timeout(image_path, xray_record, 'densenet')
            
            if result:
                print(f"✅ Successfully fixed X-ray ID {xray_record.id}")
                return True
            else:
                print(f"❌ Failed to fix X-ray ID {xray_record.id}")
                xray_record.processing_status = 'error'
                xray_record.save()
                return False
                
        except Exception as e:
            print(f"❌ Error fixing X-ray ID {xray_record.id}: {str(e)}")
            xray_record.processing_status = 'error'
            xray_record.save()
            return False
    
    def process_with_timeout(self, image_path, xray_instance, model_type):
        """Process image with timeout to prevent hanging"""
        result_container = {'result': None, 'error': None}
        
        def target():
            try:
                result_container['result'] = process_image(image_path, xray_instance, model_type)
            except Exception as e:
                result_container['error'] = str(e)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout_seconds)
        
        if thread.is_alive():
            print(f"⚠️  Processing timed out after {self.timeout_seconds} seconds")
            return None
        
        if result_container['error']:
            print(f"❌ Processing error: {result_container['error']}")
            return None
            
        return result_container['result']
    
    def reset_all_stuck_records(self):
        """Reset all stuck records to pending status"""
        print("\n🔄 Resetting all stuck records...")
        
        stuck_records = XRayImage.objects.filter(processing_status='processing')
        count = stuck_records.count()
        
        if count > 0:
            stuck_records.update(
                processing_status='pending',
                progress=0
            )
            print(f"✅ Reset {count} stuck records to pending status")
        else:
            print("ℹ️  No stuck records found")
        
        return count
    
    def optimize_memory(self):
        """Optimize memory usage for PyTorch"""
        print("\n🧹 Optimizing memory usage...")
        
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✅ Cleared CUDA cache")
            
            # Force garbage collection
            import gc
            gc.collect()
            print("✅ Forced garbage collection")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Memory optimization warning: {str(e)}")
            return False
    
    def run_diagnosis_and_fix(self):
        """Run complete diagnosis and fix procedure"""
        print("🔧 MCADS X-ray Processing Diagnostic & Fix Tool")
        print("=" * 60)
        
        # Step 1: Diagnose ML system
        ml_working = self.diagnose_ml_system()
        
        # Step 2: Find stuck records
        stuck_records = self.find_stuck_records()
        
        # Step 3: Optimize memory
        self.optimize_memory()
        
        if not stuck_records:
            print("\n✅ No stuck records found - system appears healthy")
            return True
        
        print(f"\n📋 Found {len(stuck_records)} stuck records to fix")
        
        # Option to reset all or try to fix individually
        if len(stuck_records) > 5:
            print("🔄 Large number of stuck records - resetting all to pending")
            self.reset_all_stuck_records()
        else:
            # Try to fix each record individually
            success_count = 0
            for record in stuck_records:
                if self.fix_stuck_record(record):
                    success_count += 1
            
            print(f"\n📊 Summary: {success_count}/{len(stuck_records)} records fixed successfully")
        
        # Final recommendations
        print("\n💡 Recommendations:")
        if not ml_working:
            print("   • ML models may be using mock functions - check PyTorch installation")
        print("   • Consider restarting Gunicorn workers if memory usage is high")
        print("   • Monitor system resources during processing")
        print("   • Set up processing timeouts to prevent hanging")
        
        return True

def main():
    """Main function"""
    fixer = ProcessingFixer()
    fixer.run_diagnosis_and_fix()

if __name__ == "__main__":
    main() 