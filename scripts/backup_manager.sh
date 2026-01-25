#!/bin/bash
# MCADS Backup Manager
# Interactive backup management system
set -e

# Configuration
BACKUP_DIR="./backups"
SCRIPT_DIR="./scripts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to display menu
show_menu() {
    clear
    echo "=========================================="
    echo "           MCADS Backup Manager"
    echo "=========================================="
    echo ""
    echo "1. Create System Backup"
    echo "2. List Available Backups"
    echo "3. Restore from Backup"
    echo "4. Show Backup Details"
    echo "5. Clean Old Backups"
    echo "6. Show System Status"
    echo "7. Exit"
    echo ""
    echo "=========================================="
}

# Function to create backup
create_backup() {
    print_info "Creating MCADS system backup..."
    if [ -f "$SCRIPT_DIR/system_backup.sh" ]; then
        "$SCRIPT_DIR/system_backup.sh"
        print_status "Backup completed successfully!"
    else
        print_error "Backup script not found: $SCRIPT_DIR/system_backup.sh"
        return 1
    fi
}

# Function to list backups
list_backups() {
    print_info "Available MCADS backups:"
    echo ""
    
    if [ ! -d "$BACKUP_DIR" ]; then
        print_warning "No backups directory found"
        return
    fi
    
    local backups=$(ls -1 ${BACKUP_DIR}/mcads_system_backup_*.tar.gz 2>/dev/null | sort -r)
    
    if [ -z "$backups" ]; then
        print_warning "No system backups found"
    else
        echo "System Backups:"
        echo "=============="
        for backup in $backups; do
            local name=$(basename "$backup" .tar.gz)
            local size=$(du -h "$backup" | cut -f1)
            local date=$(stat -c %y "$backup" | cut -d' ' -f1,2)
            echo "📁 $name"
            echo "   Size: $size | Date: $date"
            echo ""
        done
    fi
    
    # Also show Docker backups if they exist
    local docker_backups=$(ls -1 ${BACKUP_DIR}/mcads_backup_*.tar.gz 2>/dev/null | sort -r)
    if [ -n "$docker_backups" ]; then
        echo "Docker Backups:"
        echo "=============="
        for backup in $docker_backups; do
            local name=$(basename "$backup" .tar.gz)
            local size=$(du -h "$backup" | cut -f1)
            local date=$(stat -c %y "$backup" | cut -d' ' -f1,2)
            echo "🐳 $name"
            echo "   Size: $size | Date: $date"
            echo ""
        done
    fi
}

# Function to restore backup
restore_backup() {
    print_info "Available backups for restoration:"
    echo ""
    
    local backups=$(ls -1 ${BACKUP_DIR}/mcads_system_backup_*.tar.gz 2>/dev/null | sort -r)
    
    if [ -z "$backups" ]; then
        print_error "No system backups available for restoration"
        return 1
    fi
    
    echo "Select backup to restore:"
    echo "========================"
    local i=1
    for backup in $backups; do
        local name=$(basename "$backup" .tar.gz | sed 's/mcads_system_backup_//')
        local size=$(du -h "$backup" | cut -f1)
        local date=$(stat -c %y "$backup" | cut -d' ' -f1,2)
        echo "$i. $name ($size) - $date"
        i=$((i+1))
    done
    echo ""
    
    read -p "Enter backup number (or 'cancel' to abort): " choice
    
    if [ "$choice" = "cancel" ]; then
        print_info "Restore cancelled"
        return 0
    fi
    
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt $((i-1)) ]; then
        print_error "Invalid selection"
        return 1
    fi
    
    local selected_backup=$(echo "$backups" | sed -n "${choice}p")
    local backup_name=$(basename "$selected_backup" .tar.gz | sed 's/mcads_system_backup_//')
    
    print_warning "You are about to restore from: $backup_name"
    print_warning "This will overwrite current system files!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        if [ -f "$SCRIPT_DIR/system_restore.sh" ]; then
            "$SCRIPT_DIR/system_restore.sh" "$backup_name"
            print_status "Restore completed successfully!"
        else
            print_error "Restore script not found: $SCRIPT_DIR/system_restore.sh"
            return 1
        fi
    else
        print_info "Restore cancelled"
    fi
}

# Function to show backup details
show_backup_details() {
    print_info "Available backups for details:"
    echo ""
    
    local backups=$(ls -1 ${BACKUP_DIR}/mcads_system_backup_*.tar.gz 2>/dev/null | sort -r)
    
    if [ -z "$backups" ]; then
        print_error "No system backups available"
        return 1
    fi
    
    echo "Select backup for details:"
    echo "========================="
    local i=1
    for backup in $backups; do
        local name=$(basename "$backup" .tar.gz | sed 's/mcads_system_backup_//')
        local size=$(du -h "$backup" | cut -f1)
        local date=$(stat -c %y "$backup" | cut -d' ' -f1,2)
        echo "$i. $name ($size) - $date"
        i=$((i+1))
    done
    echo ""
    
    read -p "Enter backup number (or 'cancel' to abort): " choice
    
    if [ "$choice" = "cancel" ]; then
        return 0
    fi
    
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt $((i-1)) ]; then
        print_error "Invalid selection"
        return 1
    fi
    
    local selected_backup=$(echo "$backups" | sed -n "${choice}p")
    local backup_name=$(basename "$selected_backup" .tar.gz | sed 's/mcads_system_backup_//')
    local verification_file="${BACKUP_DIR}/mcads_system_backup_${backup_name}_verification.txt"
    
    echo ""
    echo "Backup Details for: $backup_name"
    echo "================================="
    echo ""
    
    if [ -f "$verification_file" ]; then
        cat "$verification_file"
    else
        print_warning "No verification file found for this backup"
    fi
    
    echo ""
    print_info "Archive contents:"
    tar -tzf "$selected_backup" | head -20
    echo "..."
    echo "Total files: $(tar -tzf "$selected_backup" | wc -l)"
}

# Function to clean old backups
clean_old_backups() {
    print_info "Current backup retention policy: Keep last 10 backups"
    echo ""
    
    local system_backups=$(ls -1 ${BACKUP_DIR}/mcads_system_backup_*.tar.gz 2>/dev/null | wc -l)
    local docker_backups=$(ls -1 ${BACKUP_DIR}/mcads_backup_*.tar.gz 2>/dev/null | wc -l)
    
    echo "Current backups:"
    echo "- System backups: $system_backups"
    echo "- Docker backups: $docker_backups"
    echo ""
    
    if [ "$system_backups" -gt 10 ]; then
        local to_delete=$((system_backups - 10))
        print_warning "Found $to_delete old system backups to clean"
        read -p "Delete old system backups? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            ls -t ${BACKUP_DIR}/mcads_system_backup_*.tar.gz | tail -n +11 | xargs -r rm
            print_status "Old system backups cleaned"
        fi
    else
        print_info "No old system backups to clean"
    fi
    
    if [ "$docker_backups" -gt 10 ]; then
        local to_delete=$((docker_backups - 10))
        print_warning "Found $to_delete old Docker backups to clean"
        read -p "Delete old Docker backups? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            ls -t ${BACKUP_DIR}/mcads_backup_*.tar.gz | tail -n +11 | xargs -r rm
            print_status "Old Docker backups cleaned"
        fi
    else
        print_info "No old Docker backups to clean"
    fi
}

# Function to show system status
show_system_status() {
    print_info "MCADS System Status:"
    echo ""
    
    echo "System Information:"
    echo "=================="
    echo "OS: $(uname -a)"
    echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
    echo "Disk Usage: $(df -h . | tail -1)"
    echo "Memory: $(free -h | grep Mem)"
    echo ""
    
    echo "Application Status:"
    echo "=================="
    echo "Database: $(if command -v psql >/dev/null 2>&1; then echo "✅ PostgreSQL client found"; else echo "❌ psql not found"; fi)"
    echo "Media Files: $(if [ -d "media" ]; then echo "✅ Found"; else echo "❌ Not found"; fi)"
    echo "Static Files: $(if [ -d "staticfiles" ]; then echo "✅ Found"; else echo "❌ Not found"; fi)"
    echo "Application Code: $(if [ -d "xrayapp" ]; then echo "✅ Found"; else echo "❌ Not found"; fi)"
    echo ""
    
    echo "Service Status:"
    echo "==============="
    echo "MCADS Service: $(systemctl is-active mcads 2>/dev/null || echo 'unknown')"
    echo "Nginx Service: $(systemctl is-active nginx 2>/dev/null || echo 'unknown')"
    echo ""
    
    echo "Backup Status:"
    echo "=============="
    local backup_count=$(ls -1 ${BACKUP_DIR}/mcads_system_backup_*.tar.gz 2>/dev/null | wc -l)
    echo "System Backups: $backup_count"
    if [ "$backup_count" -gt 0 ]; then
        local latest_backup=$(ls -t ${BACKUP_DIR}/mcads_system_backup_*.tar.gz | head -1)
        local latest_date=$(stat -c %y "$latest_backup" | cut -d' ' -f1,2)
        echo "Latest Backup: $latest_date"
    fi
}

# Main menu loop
while true; do
    show_menu
    read -p "Select an option (1-7): " choice
    
    case $choice in
        1)
            create_backup
            ;;
        2)
            list_backups
            ;;
        3)
            restore_backup
            ;;
        4)
            show_backup_details
            ;;
        5)
            clean_old_backups
            ;;
        6)
            show_system_status
            ;;
        7)
            print_info "Exiting MCADS Backup Manager"
            exit 0
            ;;
        *)
            print_error "Invalid option. Please select 1-7."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done 