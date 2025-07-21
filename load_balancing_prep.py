#!/usr/bin/env python3
"""
MCADS Load Balancing Preparation
Prepares the system for horizontal scaling and load balancing
"""

import os
import sys
import subprocess
from pathlib import Path

class LoadBalancingPrep:
    def __init__(self):
        self.nginx_sites_dir = Path('/etc/nginx/sites-available')
        self.nginx_enabled_dir = Path('/etc/nginx/sites-enabled')
        
    def create_load_balancer_config(self):
        """Create nginx load balancer configuration template"""
        lb_config = """
# MCADS Load Balancer Configuration
# For multi-server deployment

upstream mcads_backend {
    # Primary server (current)
    server 127.0.0.1:8000 weight=3 max_fails=3 fail_timeout=30s;
    
    # Additional servers (add when scaling)
    # server 10.0.0.2:8000 weight=2 max_fails=3 fail_timeout=30s;
    # server 10.0.0.3:8000 weight=2 max_fails=3 fail_timeout=30s;
    
    # Health check and session persistence
    least_conn;
    keepalive 32;
}

# Rate limiting for API endpoints
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;

server {
    listen 80;
    listen [::]:80;
    server_name mcads.casa www.mcads.casa;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name mcads.casa www.mcads.casa;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/mcads.casa/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mcads.casa/privkey.pem;
    
    # SSL Security Headers
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Client upload limits for medical images
    client_max_body_size 50M;
    client_body_timeout 60s;
    client_header_timeout 60s;
    
    # Rate limiting
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://mcads_backend;
        include /etc/nginx/proxy_params;
    }
    
    location /upload/ {
        limit_req zone=upload burst=5 nodelay;
        proxy_pass http://mcads_backend;
        include /etc/nginx/proxy_params;
    }
    
    # Static files (served directly by nginx)
    location /static/ {
        alias /opt/mcads/app/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        gzip_static on;
    }
    
    # Media files (X-ray images)
    location /media/ {
        alias /opt/mcads/app/media/;
        expires 7d;
        add_header Cache-Control "private";
        
        # Security for medical images
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
    }
    
    # Health check endpoint
    location /health/ {
        access_log off;
        proxy_pass http://mcads_backend;
        include /etc/nginx/proxy_params;
        proxy_read_timeout 10s;
    }
    
    # Main application
    location / {
        proxy_pass http://mcads_backend;
        include /etc/nginx/proxy_params;
        
        # WebSocket support for future real-time features
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Session affinity (if needed)
        # ip_hash;
    }
    
    # Error pages
    error_page 502 503 504 /maintenance.html;
    location = /maintenance.html {
        root /opt/mcads/app/static/;
        internal;
    }
}

# Server status monitoring (internal only)
server {
    listen 127.0.0.1:8080;
    server_name localhost;
    
    location /nginx_status {
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        deny all;
    }
}
"""
        
        lb_config_file = self.nginx_sites_dir / 'mcads_loadbalancer'
        with open(lb_config_file, 'w') as f:
            f.write(lb_config)
        
        print(f"✅ Load balancer configuration created: {lb_config_file}")
        return lb_config_file
    
    def create_proxy_params(self):
        """Create optimized proxy parameters"""
        proxy_params = """
# Optimized proxy parameters for MCADS load balancing

proxy_set_header Host $http_host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;

# Timeouts optimized for medical image processing
proxy_connect_timeout 60s;
proxy_send_timeout 300s;
proxy_read_timeout 300s;

# Buffer sizes for large medical images
proxy_buffering on;
proxy_buffer_size 8k;
proxy_buffers 16 8k;
proxy_busy_buffers_size 16k;

# Connection keep-alive
proxy_http_version 1.1;
proxy_set_header Connection "";

# Hide backend server information
proxy_hide_header X-Powered-By;
proxy_hide_header Server;
"""
        
        proxy_params_file = Path('/etc/nginx/proxy_params')
        with open(proxy_params_file, 'w') as f:
            f.write(proxy_params)
        
        print(f"✅ Proxy parameters created: {proxy_params_file}")
    
    def create_maintenance_page(self):
        """Create maintenance page for scheduled downtime"""
        maintenance_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCADS - Scheduled Maintenance</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            color: white;
        }
        .container {
            text-align: center;
            max-width: 600px;
            padding: 40px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .logo {
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #fff;
        }
        .message {
            font-size: 24px;
            margin-bottom: 30px;
        }
        .details {
            font-size: 16px;
            opacity: 0.9;
            line-height: 1.6;
        }
        .icon {
            font-size: 64px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">🔧</div>
        <div class="logo">MCADS</div>
        <div class="message">Scheduled Maintenance</div>
        <div class="details">
            <p>We're performing scheduled maintenance to improve your experience.</p>
            <p>The system will be back online shortly.</p>
            <p>For urgent medical cases, please contact your healthcare provider directly.</p>
            <p><strong>Expected downtime:</strong> 15-30 minutes</p>
        </div>
    </div>
</body>
</html>
"""
        
        static_dir = Path('/opt/mcads/app/static')
        static_dir.mkdir(exist_ok=True)
        maintenance_file = static_dir / 'maintenance.html'
        
        with open(maintenance_file, 'w') as f:
            f.write(maintenance_html)
        
        print(f"✅ Maintenance page created: {maintenance_file}")
    
    def create_scaling_guide(self):
        """Create documentation for horizontal scaling"""
        scaling_guide = """
# MCADS Horizontal Scaling Guide

## Overview
This guide explains how to scale MCADS across multiple servers for high availability and load distribution.

## Prerequisites
- Additional Ubuntu 22.04 servers
- Shared database server (PostgreSQL)
- Shared Redis cache server
- Shared file storage (NFS/GlusterFS/AWS EFS)

## Scaling Steps

### 1. Prepare Additional Servers
```bash
# On each new server
sudo apt update && sudo apt upgrade -y
sudo apt install python3.10-venv nginx

# Clone application
git clone <repository> /opt/mcads/app
cd /opt/mcads/app

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Shared Services

#### PostgreSQL (Shared Database)
```bash
# Update .env on all servers
DB_HOST=<shared_postgres_server_ip>
DB_NAME=mcads_db
DB_USER=mcads_user
DB_PASSWORD=mcads_secure_password_2024
```

#### Redis (Shared Cache)
```bash
# Update settings.py on all servers
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://<shared_redis_server_ip>:6379/1',
    }
}
```

#### Shared Media Storage
```bash
# Mount shared storage for media files
sudo mount -t nfs <nfs_server>:/opt/mcads/shared/media /opt/mcads/app/media
```

### 3. Configure Load Balancer
```bash
# Enable load balancer configuration
sudo cp /etc/nginx/sites-available/mcads_loadbalancer /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/mcads  # Remove single-server config
sudo nginx -t && sudo systemctl reload nginx
```

### 4. Update Upstream Servers
Edit `/etc/nginx/sites-available/mcads_loadbalancer`:
```nginx
upstream mcads_backend {
    server 10.0.0.1:8000 weight=3;  # Primary server
    server 10.0.0.2:8000 weight=2;  # Secondary server
    server 10.0.0.3:8000 weight=2;  # Tertiary server
}
```

### 5. Session Management
For multi-server deployments, ensure session storage is shared:
```python
# In settings.py
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'  # Redis cache
```

### 6. Static Files Distribution
```bash
# Sync static files to all servers
rsync -av /opt/mcads/app/staticfiles/ server2:/opt/mcads/app/staticfiles/
rsync -av /opt/mcads/app/staticfiles/ server3:/opt/mcads/app/staticfiles/
```

## Monitoring Multi-Server Setup
- Use the monitoring_alerts.py script on each server
- Implement centralized logging (ELK stack or similar)
- Monitor load balancer metrics via nginx status module

## Health Checks
Configure health check endpoints:
```python
# In urls.py
urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health_check'),
]
```

## Backup Strategy for Scaled Environment
- Database: Single point backup from shared PostgreSQL
- Media files: Backup shared storage
- Configuration: Backup from primary server only

## Performance Tuning
- Increase Gunicorn workers: `workers = 2 * CPU_cores + 1`
- Optimize PostgreSQL connections: Use connection pooling
- Redis memory optimization: Configure appropriate maxmemory settings
- Nginx caching: Enable proxy caching for static content

## Security Considerations
- Use internal network for inter-server communication
- Firewall rules to restrict access between servers
- SSL termination at load balancer
- Shared secrets management (consider HashiCorp Vault)
"""
        
        guide_file = Path('/opt/mcads/app/SCALING_GUIDE.md')
        with open(guide_file, 'w') as f:
            f.write(scaling_guide)
        
        print(f"✅ Scaling guide created: {guide_file}")
    
    def run_preparation(self):
        """Run all load balancing preparation steps"""
        print("🚀 Preparing MCADS for Load Balancing and Scaling")
        print("=" * 60)
        
        try:
            self.create_load_balancer_config()
            self.create_proxy_params()
            self.create_maintenance_page()
            self.create_scaling_guide()
            
            print("\n✅ Load balancing preparation completed!")
            print("\nNext steps:")
            print("1. When ready to scale, enable the load balancer config:")
            print("   sudo ln -sf /etc/nginx/sites-available/mcads_loadbalancer /etc/nginx/sites-enabled/")
            print("2. Add additional servers to the upstream configuration")
            print("3. Test with: sudo nginx -t && sudo systemctl reload nginx")
            print("4. Monitor with: curl http://localhost:8080/nginx_status")
            
        except Exception as e:
            print(f"❌ Error during preparation: {str(e)}")
            return False
        
        return True

def main():
    """Main function for load balancing preparation"""
    prep = LoadBalancingPrep()
    success = prep.run_preparation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 