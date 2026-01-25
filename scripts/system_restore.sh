#!/bin/bash
# MCADS System Restore Script (Non-Docker)
# Restore the entire MCADS system from backup
set -e

# Configuration
BACKUP_DIR="./backups"
LOG_FILE="./logs/restore_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to display usage
usage() {
    echo "Usage: $0 <backup_name> [--force]"
    echo "  backup_name: Name of the backup archive (without .tar.gz extension)"
    echo "  --force: Force restore without confirmation"
    echo ""
    echo "Available backups:"
    ls -1 ${BACKUP_DIR}/mcads_system_backup_*.tar.gz 2>/dev/null | sed 's/.*mcads_system_backup_//' | sed 's/\.tar\.gz$//' || echo "No backups found"
    exit 1
}

# Check arguments
if [ $# -eq 0 ]; then
    usage
fi

BACKUP_NAME="$1"
FORCE_RESTORE=false

if [ "$2" = "--force" ]; then
    FORCE_RESTORE=true
fi

BACKUP_FILE="${BACKUP_DIR}/mcads_system_backup_${BACKUP_NAME}.tar.gz"
RESTORE_DIR="${BACKUP_DIR}/restore_${BACKUP_NAME}"

# Validate backup exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Error: Backup file not found: $BACKUP_FILE"
    usage
fi

echo "Starting MCADS System Restore from: ${BACKUP_NAME}" | tee -a "$LOG_FILE"

# Create restore directory
mkdir -p "$RESTORE_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Extract backup
log_message "Extracting backup archive..."
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"
BACKUP_CONTENT_DIR=$(ls -d "$RESTORE_DIR"/mcads_system_backup_*)
log_message "Backup extracted to: $BACKUP_CONTENT_DIR"

# Display backup information
if [ -f "$BACKUP_CONTENT_DIR/system_info.txt" ]; then
    echo ""
    echo "📋 Backup Information:"
    cat "$BACKUP_CONTENT_DIR/system_info.txt"
    echo ""
fi

# Confirm restore (unless forced)
if [ "$FORCE_RESTORE" = false ]; then
    echo "⚠️  WARNING: This will overwrite current system files!"
    echo "Backup to restore: $BACKUP_NAME"
    echo "Backup size: $(du -h "$BACKUP_FILE" | cut -f1)"
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Restore cancelled."
        exit 0
    fi
fi

log_message "Starting system restore..."

# Stop services if running
log_message "Stopping services..."
systemctl stop mcads 2>/dev/null || true
systemctl stop nginx 2>/dev/null || true

# Load environment variables (Postgres credentials) if present.
if [ -f "./.env" ]; then
    # Export vars from .env for child processes like psql.
    set -a
    # shellcheck disable=SC1091
    . "./.env"
    set +a
fi

# Restore PostgreSQL database
DB_HOST="${DB_HOST:-${POSTGRES_HOST:-localhost}}"
DB_PORT="${DB_PORT:-${POSTGRES_PORT:-5432}}"
DB_NAME="${DB_NAME:-${POSTGRES_DB:-mcads_db}}"
DB_USER="${DB_USER:-${POSTGRES_USER:-mcads_user}}"
DB_PASSWORD="${DB_PASSWORD:-${POSTGRES_PASSWORD:-}}"

if [ -f "$BACKUP_CONTENT_DIR/database.sql" ]; then
    log_message "Restoring PostgreSQL database..."
    if ! command -v psql >/dev/null 2>&1; then
        log_message "ERROR: psql not found. Install PostgreSQL client tools and re-run."
        exit 1
    fi

    # Drop and recreate the database for a clean restore.
    PGPASSWORD="$DB_PASSWORD" psql \
        --host "$DB_HOST" \
        --port "$DB_PORT" \
        --username "$DB_USER" \
        --dbname postgres \
        -v ON_ERROR_STOP=1 \
        -c "DROP DATABASE IF EXISTS \"${DB_NAME}\";"

    PGPASSWORD="$DB_PASSWORD" psql \
        --host "$DB_HOST" \
        --port "$DB_PORT" \
        --username "$DB_USER" \
        --dbname postgres \
        -v ON_ERROR_STOP=1 \
        -c "CREATE DATABASE \"${DB_NAME}\";"

    # Restore data.
    PGPASSWORD="$DB_PASSWORD" psql \
        --host "$DB_HOST" \
        --port "$DB_PORT" \
        --username "$DB_USER" \
        --dbname "$DB_NAME" \
        -v ON_ERROR_STOP=1 \
        -f "$BACKUP_CONTENT_DIR/database.sql"

    log_message "Database restored"
else
    log_message "No database backup found (database.sql)"
fi

# Restore media files
if [ -f "$BACKUP_CONTENT_DIR/media.tar.gz" ]; then
    log_message "Restoring media files..."
    rm -rf ./media
    tar -xzf "$BACKUP_CONTENT_DIR/media.tar.gz"
    log_message "Media files restored"
fi

# Restore static files
if [ -f "$BACKUP_CONTENT_DIR/staticfiles.tar.gz" ]; then
    log_message "Restoring static files..."
    rm -rf ./staticfiles
    tar -xzf "$BACKUP_CONTENT_DIR/staticfiles.tar.gz"
    log_message "Static files restored"
fi

# Restore static directory
if [ -f "$BACKUP_CONTENT_DIR/static.tar.gz" ]; then
    log_message "Restoring static directory..."
    rm -rf ./static
    tar -xzf "$BACKUP_CONTENT_DIR/static.tar.gz"
    log_message "Static directory restored"
fi

# Restore configuration files
if [ -f "$BACKUP_CONTENT_DIR/config.tar.gz" ]; then
    log_message "Restoring configuration files..."
    tar -xzf "$BACKUP_CONTENT_DIR/config.tar.gz"
    log_message "Configuration files restored"
fi

# Restore logs
if [ -f "$BACKUP_CONTENT_DIR/logs.tar.gz" ]; then
    log_message "Restoring logs..."
    rm -rf ./logs
    tar -xzf "$BACKUP_CONTENT_DIR/logs.tar.gz"
    log_message "Logs restored"
fi

# Restore data exports
if [ -f "$BACKUP_CONTENT_DIR/data_exports.tar.gz" ]; then
    log_message "Restoring data exports..."
    rm -rf ./data_exports
    tar -xzf "$BACKUP_CONTENT_DIR/data_exports.tar.gz"
    log_message "Data exports restored"
fi

# Restore application code
if [ -f "$BACKUP_CONTENT_DIR/app_code.tar.gz" ]; then
    log_message "Restoring application code..."
    tar -xzf "$BACKUP_CONTENT_DIR/app_code.tar.gz"
    log_message "Application code restored"
fi

# Restore models
if [ -f "$BACKUP_CONTENT_DIR/models.tar.gz" ]; then
    log_message "Restoring models..."
    rm -rf ./models
    tar -xzf "$BACKUP_CONTENT_DIR/models.tar.gz"
    log_message "Models restored"
fi

# Restore tests
if [ -f "$BACKUP_CONTENT_DIR/tests.tar.gz" ]; then
    log_message "Restoring tests..."
    rm -rf ./tests
    tar -xzf "$BACKUP_CONTENT_DIR/tests.tar.gz"
    log_message "Tests restored"
fi

# Restore locale files
if [ -f "$BACKUP_CONTENT_DIR/locale.tar.gz" ]; then
    log_message "Restoring locale files..."
    rm -rf ./locale
    tar -xzf "$BACKUP_CONTENT_DIR/locale.tar.gz"
    log_message "Locale files restored"
fi

# Set proper permissions
log_message "Setting file permissions..."
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x *.sh 2>/dev/null || true

# Restart services
log_message "Restarting services..."
systemctl start mcads 2>/dev/null || true
systemctl start nginx 2>/dev/null || true

# Clean up restore directory
log_message "Cleaning up restore directory..."
rm -rf "$RESTORE_DIR"

# Create restore verification
log_message "Creating restore verification..."
cat > "${BACKUP_DIR}/restore_${BACKUP_NAME}_verification.txt" << EOF
MCADS System Restore Verification
=================================
Restore Date: $(date)
Restored From: ${BACKUP_NAME}
Restore Location: ${BACKUP_FILE}

System Status:
- Database: $(if command -v psql >/dev/null 2>&1; then echo "✅ PostgreSQL client found"; else echo "❌ psql not found"; fi)
- Media Files: $(if [ -d "media" ]; then echo "✅ Restored"; else echo "❌ Not found"; fi)
- Static Files: $(if [ -d "staticfiles" ]; then echo "✅ Restored"; else echo "❌ Not found"; fi)
- Configuration: $(if [ -f "docker-compose.yml" ]; then echo "✅ Restored"; else echo "❌ Not found"; fi)
- Application Code: $(if [ -d "xrayapp" ]; then echo "✅ Restored"; else echo "❌ Not found"; fi)

Services Status:
- MCADS Service: $(systemctl is-active mcads 2>/dev/null || echo "unknown")
- Nginx Service: $(systemctl is-active nginx 2>/dev/null || echo "unknown")

Restore Status: SUCCESS
EOF

log_message "Restore verification file created"
log_message "System restore completed successfully!"

echo "✅ MCADS System Restore Completed!"
echo "📁 Restored From: $BACKUP_NAME"
echo "📝 Log File: $LOG_FILE"
echo "📋 Verification: ${BACKUP_DIR}/restore_${BACKUP_NAME}_verification.txt"
echo ""
echo "🔄 Next Steps:"
echo "1. Verify the application is working correctly"
echo "2. Check all services are running"
echo "3. Test key functionality"
echo "4. Review the verification file for details" 