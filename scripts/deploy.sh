#!/bin/bash
# MCADS Docker Deployment Script
set -e

echo "🚀 MCADS Docker Deployment Setup"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs media staticfiles data_exports backups

# Set permissions
echo "🔒 Setting proper permissions..."
chmod 755 logs media staticfiles data_exports backups
chmod +x scripts/*.sh docker-entrypoint.sh

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

services=("db" "redis" "web" "nginx")
for service in "${services[@]}"; do
    if docker-compose ps | grep -q "${service}.*Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service is not running"
    fi
done

# Test application response
echo "🌐 Testing application..."
if curl -f http://localhost/health/ > /dev/null 2>&1; then
    echo "✅ Application is responding"
else
    echo "⚠️  Application health check failed - checking logs..."
    docker-compose logs --tail=20 web
fi

echo ""
echo "🎉 MCADS Docker deployment completed!"
echo ""
echo "📋 Access Information:"
echo "   🌐 Web Application: http://localhost"
echo "   🔧 Admin Panel: http://localhost/admin"
echo "   📊 Database: localhost:5432"
echo "   🔴 Redis: localhost:6379"
echo ""
echo "📚 Management Commands:"
echo "   📁 Create backup: ./scripts/backup.sh"
echo "   📁 Restore backup: ./scripts/restore.sh <backup_name>"
echo "   🔍 View logs: docker-compose logs"
echo "   🛑 Stop services: docker-compose down"
echo "   🔄 Restart services: docker-compose restart"
echo ""
echo "🔐 Default Admin Credentials:"
echo "   Username: admin"
echo "   Password: admin123"
echo "   (⚠️  Change these in production!)"