
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
DB_PASSWORD=<your_secure_password>
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
- Use external monitoring (Prometheus, ELK); legacy local scripts removed
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
