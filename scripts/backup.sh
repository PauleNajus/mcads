#!/bin/bash
# MCADS Docker Backup Script
set -euo pipefail

# Prefer the modern Docker Compose plugin (`docker compose`). Fall back to legacy `docker-compose` if present.
if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
elif docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
else
  echo "❌ Docker Compose is not available."
  exit 1
fi

# Compose file selection:
# - Always use docker-compose.yml
# - If TLS certs exist, enable docker-compose.prod.yml (binds 443 + TLS config)
COMPOSE_FILES=(-f docker-compose.yml)
if [[ -f docker-compose.prod.yml && -f ssl/fullchain.pem && -f ssl/privkey.pem ]]; then
  COMPOSE_FILES+=(-f docker-compose.prod.yml)
fi

# Load local env overrides if present (kept out of git).
# This keeps the script in sync with docker-compose variables without hardcoding.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Resolve DB identity with sensible defaults (matches docker-compose.yml).
DB_NAME="${DB_NAME:-${POSTGRES_DB:-mcads_db}}"
DB_USER="${DB_USER:-${POSTGRES_USER:-mcads_user}}"

BACKUP_DIR="./backups"
DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="mcads_backup_${DATE}"

echo "Starting MCADS backup: ${BACKUP_NAME}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup PostgreSQL database
echo "Backing up PostgreSQL database..."
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" exec -T db pg_dump -U "${DB_USER}" "${DB_NAME}" > "${BACKUP_DIR}/${BACKUP_NAME}/database.sql"

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
 System Version: $("${COMPOSE[@]}" "${COMPOSE_FILES[@]}" exec -T web python manage.py --version)
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