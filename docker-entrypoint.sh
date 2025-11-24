#!/bin/bash
set -e

echo "Starting MCADS Docker initialization..."

# Export critical environment variables to ensure they persist through worker restarts
export USE_CELERY="${USE_CELERY:-1}"
export REDIS_URL="${REDIS_URL}"
export DB_HOST="${DB_HOST}"
export DB_PORT="${DB_PORT}"
export DB_USER="${DB_USER}"
export DB_PASSWORD="${DB_PASSWORD}"
export DB_NAME="${DB_NAME}"

echo "Environment variables exported (USE_CELERY=$USE_CELERY)"

# Wait for database to be ready
echo "Waiting for database to be ready..."
while ! pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; do
    echo "Database is unavailable - sleeping"
    sleep 2
done
echo "Database is ready!"

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
while ! redis-cli -h redis ping > /dev/null 2>&1; do
    echo "Redis is unavailable - sleeping"
    sleep 2
done
echo "Redis is ready!"

# Run database migrations
echo "Running database migrations..."
python manage.py migrate --noinput

# Create superuser if it doesn't exist
echo "Creating superuser if needed..."
python manage.py shell << EOF
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@mcads.casa', 'admin123')
    print('Superuser created successfully')
else:
    print('Superuser already exists')
EOF

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

# Create necessary directories
mkdir -p /app/logs /app/media /app/data_exports /app/backups

echo "MCADS initialization completed successfully!"

# Execute the main command
exec "$@"