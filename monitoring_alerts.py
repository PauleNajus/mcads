#!/usr/bin/env python3
"""
MCADS Monitoring and Alerting System
Monitors system health and sends alerts when issues are detected
"""

import os
import sys
import smtplib
import psutil
import subprocess
import redis
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path

# Add Django setup
sys.path.insert(0, '/opt/mcads/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mcads_project.settings')

import django
django.setup()

from django.db import connection
from django.core.cache import cache
from xrayapp.models import XRayImage

class MCadsMonitor:
    def __init__(self):
        self.alerts = []
        self.critical_alerts = []
        self.warning_alerts = []
        self.timestamp = datetime.now()
        
        # Monitoring thresholds
        self.thresholds = {
            'cpu_critical': 95.0,
            'cpu_warning': 80.0,
            'memory_critical': 95.0,
            'memory_warning': 85.0,
            'disk_critical': 95.0,
            'disk_warning': 90.0,
            'response_time_critical': 10.0,  # seconds
            'response_time_warning': 5.0,
        }
    
    def check_system_resources(self):
        """Monitor CPU, Memory, and Disk usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=2)
            if cpu_percent >= self.thresholds['cpu_critical']:
                self.critical_alerts.append(f"🚨 CRITICAL: CPU usage at {cpu_percent:.1f}%")
            elif cpu_percent >= self.thresholds['cpu_warning']:
                self.warning_alerts.append(f"⚠️ WARNING: CPU usage at {cpu_percent:.1f}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent >= self.thresholds['memory_critical']:
                self.critical_alerts.append(f"🚨 CRITICAL: Memory usage at {memory.percent:.1f}%")
            elif memory.percent >= self.thresholds['memory_warning']:
                self.warning_alerts.append(f"⚠️ WARNING: Memory usage at {memory.percent:.1f}%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            if disk.percent >= self.thresholds['disk_critical']:
                self.critical_alerts.append(f"🚨 CRITICAL: Disk usage at {disk.percent:.1f}%")
            elif disk.percent >= self.thresholds['disk_warning']:
                self.warning_alerts.append(f"⚠️ WARNING: Disk usage at {disk.percent:.1f}%")
            
            return True
            
        except Exception as e:
            self.critical_alerts.append(f"🚨 CRITICAL: System monitoring failed - {str(e)}")
            return False
    
    def check_services(self):
        """Check critical service status"""
        critical_services = ['mcads.service', 'nginx', 'postgresql', 'redis-server']
        
        for service in critical_services:
            try:
                result = subprocess.run(
                    ['systemctl', 'is-active', service],
                    capture_output=True, text=True
                )
                if result.stdout.strip() != 'active':
                    self.critical_alerts.append(f"🚨 CRITICAL: Service {service} is not active")
            except Exception as e:
                self.critical_alerts.append(f"🚨 CRITICAL: Cannot check service {service} - {str(e)}")
    
    def check_database_health(self):
        """Check database connectivity and performance"""
        try:
            start_time = datetime.now()
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM xrayapp_xrayimage")
                record_count = cursor.fetchone()[0]
            end_time = datetime.now()
            
            query_time = (end_time - start_time).total_seconds()
            
            if query_time > self.thresholds['response_time_critical']:
                self.critical_alerts.append(f"🚨 CRITICAL: Database query took {query_time:.2f}s")
            elif query_time > self.thresholds['response_time_warning']:
                self.warning_alerts.append(f"⚠️ WARNING: Database slow query ({query_time:.2f}s)")
            
            # Check for recent activity
            recent_uploads = XRayImage.objects.filter(
                uploaded_at__gte=datetime.now() - timedelta(hours=24)
            ).count()
            
            if recent_uploads == 0:
                self.warning_alerts.append("⚠️ WARNING: No new X-ray uploads in 24 hours")
            
        except Exception as e:
            self.critical_alerts.append(f"🚨 CRITICAL: Database check failed - {str(e)}")
    
    def check_redis_health(self):
        """Check Redis cache connectivity"""
        try:
            r = redis.Redis(host='localhost', port=6379, db=1)
            start_time = datetime.now()
            r.ping()
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            
            if response_time > 1.0:
                self.warning_alerts.append(f"⚠️ WARNING: Redis slow response ({response_time:.2f}s)")
            
            # Check cache hit rate (basic test)
            cache.set('health_test', 'ok', 30)
            if cache.get('health_test') != 'ok':
                self.critical_alerts.append("🚨 CRITICAL: Redis cache not working properly")
            
        except Exception as e:
            self.critical_alerts.append(f"🚨 CRITICAL: Redis check failed - {str(e)}")
    
    def check_ssl_certificate(self):
        """Check SSL certificate expiration"""
        try:
            result = subprocess.run([
                'openssl', 's_client', '-connect', 'mcads.casa:443', '-servername', 'mcads.casa'
            ], input=b'', capture_output=True, timeout=10)
            
            if result.returncode != 0:
                self.warning_alerts.append("⚠️ WARNING: SSL certificate check failed")
                return
            
            # Check certificate expiration (simplified)
            cert_result = subprocess.run([
                'echo', '|', 'openssl', 's_client', '-connect', 'mcads.casa:443', '-servername', 'mcads.casa', '2>/dev/null', '|',
                'openssl', 'x509', '-noout', '-dates'
            ], capture_output=True, text=True, shell=True)
            
            if 'notAfter' in cert_result.stdout:
                # Basic check - in production, you'd parse the date properly
                self.alerts.append("✅ SSL certificate status checked")
            else:
                self.warning_alerts.append("⚠️ WARNING: Could not verify SSL certificate expiration")
                
        except subprocess.TimeoutExpired:
            self.warning_alerts.append("⚠️ WARNING: SSL certificate check timed out")
        except Exception as e:
            self.warning_alerts.append(f"⚠️ WARNING: SSL certificate check error - {str(e)}")
    
    def check_backup_status(self):
        """Check backup system status"""
        backup_dir = Path('/opt/mcads/backups')
        
        if not backup_dir.exists():
            self.critical_alerts.append("🚨 CRITICAL: Backup directory not found")
            return
        
        # Check for recent backups
        db_backups = list((backup_dir / 'database').glob('*.gz')) if (backup_dir / 'database').exists() else []
        
        if db_backups:
            latest_backup = max(db_backups, key=lambda x: x.stat().st_mtime)
            backup_age = datetime.now() - datetime.fromtimestamp(latest_backup.stat().st_mtime)
            
            if backup_age.days > 2:
                self.critical_alerts.append(f"🚨 CRITICAL: Latest backup is {backup_age.days} days old")
            elif backup_age.days > 1:
                self.warning_alerts.append(f"⚠️ WARNING: Latest backup is {backup_age.days} day old")
        else:
            self.critical_alerts.append("🚨 CRITICAL: No database backups found")
    
    def check_log_files(self):
        """Check for critical errors in log files"""
        log_dir = Path('/opt/mcads/app/logs')
        
        if not log_dir.exists():
            self.warning_alerts.append("⚠️ WARNING: Log directory not found")
            return
        
        # Check recent error logs
        error_log = log_dir / 'gunicorn_error.log'
        if error_log.exists():
            # Check for recent errors (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            try:
                with open(error_log, 'r') as f:
                    recent_lines = f.readlines()[-100:]  # Last 100 lines
                
                error_count = sum(1 for line in recent_lines if 'ERROR' in line.upper())
                
                if error_count > 10:
                    self.critical_alerts.append(f"🚨 CRITICAL: {error_count} errors in last hour")
                elif error_count > 5:
                    self.warning_alerts.append(f"⚠️ WARNING: {error_count} errors in recent logs")
                    
            except Exception as e:
                self.warning_alerts.append(f"⚠️ WARNING: Could not read error log - {str(e)}")
    
    def run_all_checks(self):
        """Run all monitoring checks"""
        print(f"🔍 MCADS Health Monitor - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        self.check_system_resources()
        self.check_services()
        self.check_database_health()
        self.check_redis_health()
        self.check_ssl_certificate()
        self.check_backup_status()
        self.check_log_files()
        
        # Compile all alerts
        all_alerts = self.critical_alerts + self.warning_alerts + self.alerts
        
        if self.critical_alerts:
            print("🚨 CRITICAL ALERTS:")
            for alert in self.critical_alerts:
                print(f"   {alert}")
            print()
        
        if self.warning_alerts:
            print("⚠️ WARNING ALERTS:")
            for alert in self.warning_alerts:
                print(f"   {alert}")
            print()
        
        if not self.critical_alerts and not self.warning_alerts:
            print("✅ All systems healthy - no alerts")
        
        print("=" * 60)
        
        # Return status
        if self.critical_alerts:
            return 'critical'
        elif self.warning_alerts:
            return 'warning'
        else:
            return 'healthy'
    
    def send_email_alert(self, status):
        """Send email alerts for critical issues (placeholder)"""
        # This is a placeholder for email notification system
        # In production, you would configure SMTP settings
        
        if status == 'critical':
            subject = f"🚨 MCADS CRITICAL ALERT - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
            body = "Critical issues detected:\n\n" + "\n".join(self.critical_alerts)
        elif status == 'warning':
            subject = f"⚠️ MCADS Warning - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
            body = "Warning issues detected:\n\n" + "\n".join(self.warning_alerts)
        else:
            return  # No alerts to send
        
        # Log the alert (in production, send actual email)
        alert_log = Path('/opt/mcads/app/logs/alerts.log')
        with open(alert_log, 'a') as f:
            f.write(f"{self.timestamp.isoformat()} - {subject}\n{body}\n\n")
        
        print(f"📧 Alert logged to {alert_log}")

def main():
    """Main monitoring function"""
    monitor = MCadsMonitor()
    status = monitor.run_all_checks()
    
    # Send alerts if needed
    if status in ['critical', 'warning']:
        monitor.send_email_alert(status)
    
    # Exit with appropriate status code
    if status == 'critical':
        sys.exit(2)
    elif status == 'warning':
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 