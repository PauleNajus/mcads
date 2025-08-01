#!/bin/bash
# MCADS System Backup Script (Non-Docker)
# Comprehensive backup of the entire MCADS system
set -e

# Configuration
BACKUP_DIR="./backups"
DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="mcads_system_backup_${DATE}"
LOG_FILE="./logs/backup_${DATE}.log"

echo "Starting MCADS System Backup: ${BACKUP_NAME}" | tee -a "$LOG_FILE"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Backup SQLite database (if exists)
if [ -f "db.sqlite3" ]; then
    log_message "Backing up SQLite database..."
    cp db.sqlite3 "${BACKUP_DIR}/${BACKUP_NAME}/db.sqlite3"
    log_message "Database backup completed"
else
    log_message "No SQLite database found"
fi

# Backup media files
log_message "Backing up media files..."
if [ -d "./media" ]; then
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/media.tar.gz" ./media
    log_message "Media files backup completed"
else
    log_message "No media directory found"
fi

# Backup static files
log_message "Backing up static files..."
if [ -d "./staticfiles" ]; then
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/staticfiles.tar.gz" ./staticfiles
    log_message "Static files backup completed"
else
    log_message "No staticfiles directory found"
fi

# Backup static directory
log_message "Backing up static directory..."
if [ -d "./static" ]; then
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/static.tar.gz" ./static
    log_message "Static directory backup completed"
else
    log_message "No static directory found"
fi

# Backup configuration files
log_message "Backing up configuration files..."
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/config.tar.gz" \
    ./docker-compose.yml \
    ./Dockerfile \
    ./nginx/ \
    ./scripts/ \
    ./.env \
    ./docker-entrypoint.sh \
    ./gunicorn_config.py \
    ./mcads_nginx.conf \
    ./mcads.service \
    ./requirements.txt \
    ./pyrightconfig.json \
    ./.stylelintrc.json \
    ./.dockerignore \
    ./.gitignore

log_message "Configuration files backup completed"

# Backup logs
log_message "Backing up logs..."
if [ -d "./logs" ]; then
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/logs.tar.gz" ./logs
    log_message "Logs backup completed"
else
    log_message "No logs directory found"
fi

# Backup data exports
log_message "Backing up data exports..."
if [ -d "./data_exports" ]; then
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/data_exports.tar.gz" ./data_exports
    log_message "Data exports backup completed"
else
    log_message "No data_exports directory found"
fi

# Backup Python application code
log_message "Backing up application code..."
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/app_code.tar.gz" \
    ./xrayapp/ \
    ./mcads_project/ \
    ./manage.py \
    ./backup_system.py \
    ./system_health_monitor.py \
    ./monitoring_alerts.py \
    ./load_balancing_prep.py \
    ./debug_processing.py \
    ./emergency_settings_patch.py

log_message "Application code backup completed"

# Backup models directory
log_message "Backing up models..."
if [ -d "./models" ]; then
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/models.tar.gz" ./models
    log_message "Models backup completed"
else
    log_message "No models directory found"
fi

# Backup tests
log_message "Backing up tests..."
if [ -d "./tests" ]; then
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/tests.tar.gz" ./tests
    log_message "Tests backup completed"
else
    log_message "No tests directory found"
fi

# Backup locale files
log_message "Backing up locale files..."
if [ -d "./locale" ]; then
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/locale.tar.gz" ./locale
    log_message "Locale files backup completed"
else
    log_message "No locale directory found"
fi

# Create system information file
log_message "Creating system information..."
cat > "${BACKUP_DIR}/${BACKUP_NAME}/system_info.txt" << EOF
MCADS System Backup Information
===============================
Backup Date: $(date)
Backup Name: ${BACKUP_NAME}
System: $(uname -a)
Python Version: $(python3 --version 2>/dev/null || echo "Python not found")
Disk Usage: $(df -h . | tail -1)
Memory Usage: $(free -h | grep Mem)

Components Backed Up:
- SQLite Database: db.sqlite3
- Media Files: media.tar.gz
- Static Files: staticfiles.tar.gz
- Static Directory: static.tar.gz
- Configuration: config.tar.gz
- Logs: logs.tar.gz
- Data Exports: data_exports.tar.gz
- Application Code: app_code.tar.gz
- Models: models.tar.gz
- Tests: tests.tar.gz
- Locale: locale.tar.gz

Restore Instructions:
1. Extract the backup archive
2. Copy files to appropriate locations
3. Restore database if needed
4. Restart services

Backup Location: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz
EOF

log_message "System information file created"

# Create compressed backup archive
log_message "Creating compressed backup archive..."
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}/"
rm -rf "${BACKUP_NAME}/"
cd ..

BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -f1)
log_message "Backup completed successfully: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
log_message "Backup size: ${BACKUP_SIZE}"

# Clean up old backups (keep last 10)
log_message "Cleaning up old backups..."
ls -t ${BACKUP_DIR}/mcads_system_backup_*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm

# Create backup verification file
log_message "Creating backup verification..."
cat > "${BACKUP_DIR}/${BACKUP_NAME}_verification.txt" << EOF
Backup Verification Report
==========================
Backup Name: ${BACKUP_NAME}
Backup Date: $(date)
Backup Size: ${BACKUP_SIZE}
Backup Location: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz

Archive Contents:
$(tar -tzf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | head -20)

Total Files in Archive:
$(tar -tzf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | wc -l)

Backup Status: SUCCESS
EOF

log_message "Backup verification file created"
log_message "Backup process completed successfully!"

echo "✅ MCADS System Backup Completed!"
echo "📁 Backup Location: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo "📊 Backup Size: ${BACKUP_SIZE}"
echo "📝 Log File: ${LOG_FILE}" 