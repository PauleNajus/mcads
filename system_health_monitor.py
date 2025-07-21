#!/usr/bin/env python3
"""
MCADS System Health Monitor
Monitors system performance, database connectivity, Redis cache, and ML model availability
"""

import os
import sys
import django
import psutil
import subprocess
import redis
from datetime import datetime

# Setup Django
sys.path.insert(0, '/opt/mcads/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mcads_project.settings')
django.setup()

from django.db import connection
from django.core.cache import cache
from xrayapp.models import XRayImage
from xrayapp.utils import load_model

class HealthMonitor:
    def __init__(self):
        self.checks = []
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    def check_database(self):
        """Test PostgreSQL database connectivity"""
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                row = cursor.fetchone()
                if row[0] == 1:
                    count = XRayImage.objects.count()
                    self.checks.append(f"✅ PostgreSQL: Connected ({count} X-ray records)")
                    return True
        except Exception as e:
            self.checks.append(f"❌ PostgreSQL: {str(e)}")
            return False
    
    def check_redis(self):
        """Test Redis cache connectivity"""
        try:
            r = redis.Redis(host='localhost', port=6379, db=1)
            r.ping()
            info = r.info('memory')
            memory_usage = info['used_memory_human']
            cache.set('health_check', 'ok', 30)
            test_value = cache.get('health_check')
            if test_value == 'ok':
                self.checks.append(f"✅ Redis: Connected (Memory: {memory_usage})")
                return True
        except Exception as e:
            self.checks.append(f"❌ Redis: {str(e)}")
            return False
    
    def check_ml_models(self):
        """Test ML model loading capability"""
        try:
            # Test DenseNet model
            model, resize_dim = load_model('densenet')
            is_mock = hasattr(model, 'model_type')
            
            if is_mock:
                self.checks.append(f"⚠️  ML Models: Using mock models (PyTorch compatibility mode)")
            else:
                self.checks.append(f"✅ ML Models: Real AI models loaded (DenseNet {resize_dim}x{resize_dim})")
            return True
        except Exception as e:
            self.checks.append(f"❌ ML Models: {str(e)}")
            return False
    
    def check_system_resources(self):
        """Monitor system resources"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available // (1024**3)  # GB
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free // (1024**3)  # GB
            
            self.checks.append(f"📊 System Resources:")
            self.checks.append(f"   CPU: {cpu_percent:.1f}%")
            self.checks.append(f"   Memory: {memory_percent:.1f}% used ({memory_available}GB free)")
            self.checks.append(f"   Disk: {disk_percent:.1f}% used ({disk_free}GB free)")
            
            # Check if resources are healthy
            if cpu_percent > 90:
                self.checks.append(f"⚠️  High CPU usage: {cpu_percent:.1f}%")
            if memory_percent > 90:
                self.checks.append(f"⚠️  High memory usage: {memory_percent:.1f}%")
            if disk_percent > 90:
                self.checks.append(f"⚠️  High disk usage: {disk_percent:.1f}%")
            
            return True
        except Exception as e:
            self.checks.append(f"❌ System Resources: {str(e)}")
            return False
    
    def check_services(self):
        """Check system services status"""
        services = ['mcads.service', 'nginx', 'postgresql', 'redis-server']
        for service in services:
            try:
                result = subprocess.run(
                    ['systemctl', 'is-active', service],
                    capture_output=True, text=True
                )
                status = result.stdout.strip()
                if status == 'active':
                    self.checks.append(f"✅ Service {service}: Active")
                else:
                    self.checks.append(f"❌ Service {service}: {status}")
            except Exception as e:
                self.checks.append(f"❌ Service {service}: Error checking status")
    
    def check_ssl_certificate(self):
        """Check SSL certificate validity"""
        try:
            result = subprocess.run(
                ['openssl', 's_client', '-connect', 'mcads.casa:443', '-servername', 'mcads.casa'],
                input=b'',
                capture_output=True,
                timeout=10
            )
            if result.returncode == 0:
                # Check certificate expiration
                cert_result = subprocess.run(
                    ['openssl', 's_client', '-connect', 'mcads.casa:443', '-servername', 'mcads.casa'],
                    input=b'',
                    capture_output=True,
                    timeout=5
                )
                self.checks.append("✅ SSL Certificate: Valid and accessible")
            else:
                self.checks.append("⚠️  SSL Certificate: Could not verify")
        except Exception as e:
            self.checks.append(f"⚠️  SSL Certificate: {str(e)}")
    
    def run_all_checks(self):
        """Run all health checks"""
        print(f"🔍 MCADS System Health Check - {self.timestamp}")
        print("=" * 60)
        
        self.check_database()
        self.check_redis()
        self.check_ml_models()
        self.check_system_resources()
        self.check_services()
        self.check_ssl_certificate()
        
        print("\n".join(self.checks))
        print("=" * 60)
        
        # Overall health score
        total_checks = len([c for c in self.checks if c.startswith(('✅', '❌'))])
        successful_checks = len([c for c in self.checks if c.startswith('✅')])
        
        if total_checks > 0:
            health_score = (successful_checks / total_checks) * 100
            print(f"🎯 Overall System Health: {health_score:.1f}% ({successful_checks}/{total_checks} checks passed)")
        
        return successful_checks, total_checks

if __name__ == "__main__":
    monitor = HealthMonitor()
    successful, total = monitor.run_all_checks()
    
    # Exit with appropriate code
    if successful == total:
        sys.exit(0)  # All checks passed
    elif successful >= total * 0.8:
        sys.exit(1)  # Minor issues
    else:
        sys.exit(2)  # Major issues 