#!/bin/bash
# MCADS Docker Restore Script
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
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Resolve DB identity with sensible defaults (matches docker-compose.yml).
DB_NAME="${DB_NAME:-${POSTGRES_DB:-mcads_db}}"
DB_USER="${DB_USER:-${POSTGRES_USER:-mcads_user}}"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_name>"
    echo "Available backups:"
    ls -1 ./backups/mcads_backup_*.tar.gz 2>/dev/null | sed 's/.*mcads_backup_\(.*\)\.tar\.gz/\1/' || echo "No backups found"
    exit 1
fi

BACKUP_NAME="$1"
BACKUP_DIR="./backups"
BACKUP_FILE="${BACKUP_DIR}/mcads_backup_${BACKUP_NAME}.tar.gz"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "Starting MCADS restore from backup: $BACKUP_NAME"

# Stop services
echo "Stopping Docker services..."
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" down

# Extract backup
echo "Extracting backup..."
cd "$BACKUP_DIR"
tar -xzf "mcads_backup_${BACKUP_NAME}.tar.gz"
cd ..

# Restore database
echo "Restoring PostgreSQL database..."
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" up -d db
sleep 10  # Wait for database to be ready

# Drop and recreate database
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" exec -T db psql -U "${DB_USER}" -d postgres -c "DROP DATABASE IF EXISTS ${DB_NAME};"
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" exec -T db psql -U "${DB_USER}" -d postgres -c "CREATE DATABASE ${DB_NAME};"

# Restore database data
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" exec -T db psql -U "${DB_USER}" -d "${DB_NAME}" < "${BACKUP_DIR}/mcads_backup_${BACKUP_NAME}/database.sql"

# Restore media files
echo "Restoring media files..."
rm -rf ./media
tar -xzf "${BACKUP_DIR}/mcads_backup_${BACKUP_NAME}/media.tar.gz"

# Restore static files
echo "Restoring static files..."
rm -rf ./staticfiles
tar -xzf "${BACKUP_DIR}/mcads_backup_${BACKUP_NAME}/staticfiles.tar.gz"

# Restore logs
echo "Restoring logs..."
rm -rf ./logs
tar -xzf "${BACKUP_DIR}/mcads_backup_${BACKUP_NAME}/logs.tar.gz"

# Clean up extracted backup directory
rm -rf "${BACKUP_DIR}/mcads_backup_${BACKUP_NAME}/"

# Start all services
echo "Starting Docker services..."
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Verify restore
echo "Verifying restore..."
if curl -f http://localhost:8000/health/ > /dev/null 2>&1; then
    echo "✓ Application is responding"
else
    echo "⚠ Application health check failed"
fi

echo "Restore completed successfully!"
echo "Access your application at: http://localhost"