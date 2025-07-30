# MCADS Docker Deployment Guide

This guide provides comprehensive instructions for containerizing and deploying the MCADS X-ray Analysis System using Docker as a complete backup solution for your VPS system.

## 🎯 Overview

The Docker setup includes all system components:
- **Django Application** - X-ray analysis web application with ML capabilities
- **PostgreSQL Database** - Primary data storage
- **Redis** - Caching and Celery task queue
- **Celery Worker** - Async task processing for ML operations
- **Celery Beat** - Scheduled task management
- **Nginx** - Reverse proxy and static file serving

## 📋 Prerequisites

### System Requirements
- Docker Engine 20.0+ 
- Docker Compose 2.0+
- 4GB+ RAM (for ML processing)
- 20GB+ disk space
- Linux/macOS/Windows with WSL2

### Installation
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## 🚀 Quick Start

### 1. One-Command Deployment
```bash
./scripts/deploy.sh
```

### 2. Manual Deployment
```bash
# Create directories
mkdir -p logs media staticfiles data_exports backups

# Build and start services
docker-compose build
docker-compose up -d

# Check status
docker-compose ps
```

## 📁 Project Structure

```
/opt/mcads/app/
├── docker-compose.yml          # Main orchestration file
├── Dockerfile                  # Django application container
├── docker-entrypoint.sh        # Application initialization script
├── nginx/                      # Nginx configuration
│   ├── nginx.conf             # Main nginx config
│   └── mcads.conf             # Application-specific config
├── scripts/                    # Management scripts
│   ├── deploy.sh              # Deployment script
│   ├── backup.sh              # Backup script
│   └── restore.sh             # Restore script
├── backups/                    # Backup storage
├── logs/                       # Application logs
├── media/                      # User uploads
└── staticfiles/               # Static assets
```

## 🔧 Configuration

### Environment Variables
Key environment variables are defined in `docker-compose.yml`:
- `SECRET_KEY` - Django secret key
- `DEBUG` - Debug mode (False for production)
- `DB_*` - Database connection settings
- `ALLOWED_HOSTS` - Permitted host names
- `REDIS_URL` - Redis connection string

### Database Configuration
PostgreSQL is configured with:
- Database: `mcads_db`
- User: `mcads_user`
- Password: `mcads_secure_password_2024`
- Port: `5432` (exposed)

### Redis Configuration
Redis is configured with:
- Port: `6379` (exposed)
- Persistence enabled
- Used for Celery and caching

## 🏃‍♂️ Running the Application

### Start Services
```bash
docker-compose up -d
```

### View Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs web
docker-compose logs db
docker-compose logs celery
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

## 💾 Backup and Restore

### Create Backup
```bash
./scripts/backup.sh
```
Creates timestamped backup including:
- PostgreSQL database dump
- Media files
- Static files
- Configuration files
- Application logs

### Restore from Backup
```bash
# List available backups
./scripts/restore.sh

# Restore specific backup
./scripts/restore.sh 20240101_120000
```

### Backup Storage
- Backups stored in `./backups/`
- Automatic cleanup (keeps last 10 backups)
- Compressed tar.gz format
- Include metadata and restore instructions

## 🔍 Health Monitoring

### Health Checks
All services include health checks:
- **Database**: PostgreSQL connectivity
- **Redis**: Redis ping response
- **Web**: HTTP health endpoint
- **Nginx**: Service availability

### Service Status
```bash
# Check all services
docker-compose ps

# Service health details
docker-compose exec web curl http://localhost:8000/health/
```

## 🛠 Management Commands

### Django Commands
```bash
# Run Django management commands
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py collectstatic
docker-compose exec web python manage.py createsuperuser

# Access Django shell
docker-compose exec web python manage.py shell
```

### Database Access
```bash
# PostgreSQL shell
docker-compose exec db psql -U mcads_user -d mcads_db

# Database backup
docker-compose exec db pg_dump -U mcads_user mcads_db > backup.sql
```

### Redis Access
```bash
# Redis CLI
docker-compose exec redis redis-cli
```

## 🔐 Security Considerations

### Production Security
1. **Change Default Passwords**
   ```bash
   # Update in docker-compose.yml
   POSTGRES_PASSWORD=your_secure_password
   SECRET_KEY=your_secure_secret_key
   ```

2. **Enable HTTPS**
   - Add SSL certificates to `./ssl/`
   - Update nginx configuration
   - Use secure environment variables

3. **Network Security**
   - Remove exposed ports for internal services
   - Use Docker networks for service communication
   - Implement firewall rules

### Data Protection
- Regular automated backups
- Encrypted backup storage
- Database access logging
- File permission controls

## 🚨 Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check logs
docker-compose logs

# Check resource usage
docker stats

# Rebuild images
docker-compose build --no-cache
```

#### Database Connection Errors
```bash
# Check database status
docker-compose exec db pg_isready -U mcads_user

# Reset database
docker-compose down -v
docker-compose up -d
```

#### Memory Issues
```bash
# Monitor resource usage
docker stats

# Adjust worker settings in gunicorn_config.py
workers = 1  # Reduce for low memory
```

#### Permission Errors
```bash
# Fix file permissions
sudo chown -R $USER:$USER ./media ./logs ./staticfiles

# Check container user
docker-compose exec web id
```

## 📊 Performance Optimization

### Resource Limits
Add to `docker-compose.yml`:
```yaml
services:
  web:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### Volume Optimization
- Use named volumes for persistent data
- Regular cleanup of unused images/containers
- Monitor disk usage

## 🔄 Migration from Current System

### Data Migration
1. **Export Current Data**
   ```bash
   # From current system
   sudo -u postgres pg_dump mcads_db > current_backup.sql
   tar -czf current_media.tar.gz ./media
   ```

2. **Import to Docker**
   ```bash
   # After Docker deployment
   docker-compose exec -T db psql -U mcads_user mcads_db < current_backup.sql
   tar -xzf current_media.tar.gz
   ```

### Service Migration
1. Stop current services
2. Deploy Docker containers
3. Import data
4. Update DNS/proxy configuration
5. Test thoroughly

## 📞 Support

### Log Locations
- Application logs: `./logs/`
- Nginx logs: `docker-compose logs nginx`
- Database logs: `docker-compose logs db`
- Container logs: `docker-compose logs <service>`

### Monitoring Commands
```bash
# Real-time logs
docker-compose logs -f

# Resource usage
docker stats

# Service health
docker-compose ps
```

## 🎉 Success Indicators

After deployment, verify:
- ✅ All services showing "Up" status
- ✅ Web application accessible at http://localhost
- ✅ Admin panel accessible at http://localhost/admin
- ✅ Health check endpoint responding
- ✅ File uploads working
- ✅ ML processing functional
- ✅ Celery tasks processing
- ✅ Backup/restore working

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Django Docker Best Practices](https://docs.djangoproject.com/en/stable/howto/deployment/)
- [PostgreSQL Docker Guide](https://hub.docker.com/_/postgres)

---

**Your MCADS system is now fully containerized and ready for production deployment or backup restoration!** 🚀