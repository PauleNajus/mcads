#!/bin/bash
# MCADS Docker Backup Script
set -e

BACKUP_DIR="./backups"
DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="mcads_backup_${DATE}"

echo "Starting MCADS backup: ${BACKUP_NAME}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup PostgreSQL database
echo "Backing up PostgreSQL database..."
docker-compose exec -T db pg_dump -U mcads_user mcads_db > "${BACKUP_DIR}/${BACKUP_NAME}/database.sql"

# Backup media files
echo "Backing up media files..."
if [ -d "./media" ]; then
  tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/media.tar.gz" ./media
else
  echo "No media directory found"
fi

# Backup static files (collected)
echo "Backing up static files..."
if [ -d "./staticfiles" ]; then
  tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/staticfiles.tar.gz" ./staticfiles
else
  echo "No staticfiles directory found"
fi

# Backup configuration files
echo "Backing up configuration files..."
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/config.tar.gz" \
    ./docker-compose.yml \
    ./Dockerfile \
    ./nginx/ \
    ./scripts/ \
    ./.env \
    ./docker-entrypoint.sh

# Backup logs
echo "Backing up logs..."
if [ -d "./logs" ]; then
  tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/logs.tar.gz" ./logs
else
  echo "No logs directory found"
fi

# Create backup info file
cat > "${BACKUP_DIR}/${BACKUP_NAME}/backup_info.txt" << EOF
MCADS Backup Information
========================
Backup Date: $(date)
Backup Name: ${BACKUP_NAME}
System Version: $(docker-compose exec -T web python manage.py --version)
Components:
- PostgreSQL Database: database.sql
- Media Files: media.tar.gz
- Static Files: staticfiles.tar.gz
- Configuration: config.tar.gz
- Logs: logs.tar.gz

Restore Command:
./scripts/restore.sh ${BACKUP_NAME}
EOF

# Create compressed backup archive
echo "Creating compressed backup archive..."
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}/"
rm -rf "${BACKUP_NAME}/"
cd ..

echo "Backup completed successfully: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo "Backup size: $(du -h ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz | cut -f1)"

# Clean up old backups (keep last 10)
echo "Cleaning up old backups..."
ls -t ${BACKUP_DIR}/mcads_backup_*.tar.gz | tail -n +11 | xargs -r rm

echo "Backup process completed!"