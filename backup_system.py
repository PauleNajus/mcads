#!/usr/bin/env python3
"""
MCADS Automated Backup System
Handles PostgreSQL database backups and media file archival with retention policies
"""

import os
import sys
import subprocess
import shutil
import gzip
import datetime
import logging
from pathlib import Path
import tarfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/mcads/app/logs/backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MCadsBackupSystem:
    def __init__(self):
        self.base_backup_dir = Path('/opt/mcads/backups')
        self.db_backup_dir = self.base_backup_dir / 'database'
        self.media_backup_dir = self.base_backup_dir / 'media'
        self.app_backup_dir = self.base_backup_dir / 'application'
        
        # Database configuration
        self.db_name = 'mcads_db'
        self.db_user = 'mcads_user'
        self.db_host = 'localhost'
        self.db_port = '5432'
        
        # Retention policy (days)
        self.daily_retention = 7
        self.weekly_retention = 30
        self.monthly_retention = 365
        
        # Create backup directories
        self._create_directories()
    
    def _create_directories(self):
        """Create backup directory structure"""
        for directory in [self.db_backup_dir, self.media_backup_dir, self.app_backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info("Backup directories initialized")
    
    def backup_database(self):
        """Create PostgreSQL database backup"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"mcads_db_{timestamp}.sql"
        backup_path = self.db_backup_dir / backup_filename
        
        try:
            # Create database dump
            cmd = [
                'pg_dump',
                '-h', self.db_host,
                '-p', self.db_port,
                '-U', self.db_user,
                '-d', self.db_name,
                '--no-password',
                '-f', str(backup_path)
            ]
            
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = 'mcads_secure_password_2024'
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Compress the backup
                compressed_path = backup_path.with_suffix('.sql.gz')
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove uncompressed file
                backup_path.unlink()
                
                # Get file size
                size_mb = compressed_path.stat().st_size / (1024 * 1024)
                logger.info(f"Database backup created: {compressed_path.name} ({size_mb:.2f} MB)")
                return compressed_path
            else:
                logger.error(f"Database backup failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Database backup error: {str(e)}")
            return None
    
    def backup_media_files(self):
        """Create backup of media files (X-ray images)"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"mcads_media_{timestamp}.tar.gz"
        backup_path = self.media_backup_dir / backup_filename
        
        media_source = Path('/opt/mcads/app/media')
        
        if not media_source.exists():
            logger.warning("Media directory not found, skipping media backup")
            return None
        
        try:
            with tarfile.open(backup_path, 'w:gz') as tar:
                tar.add(media_source, arcname='media')
            
            size_mb = backup_path.stat().st_size / (1024 * 1024)
            logger.info(f"Media backup created: {backup_filename} ({size_mb:.2f} MB)")
            return backup_path
            
        except Exception as e:
            logger.error(f"Media backup error: {str(e)}")
            return None
    
    def backup_application_config(self):
        """Create backup of application configuration"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"mcads_config_{timestamp}.tar.gz"
        backup_path = self.app_backup_dir / backup_filename
        
        config_files = [
            '/opt/mcads/app/.env',
            '/opt/mcads/app/mcads_project/settings.py',
            '/opt/mcads/app/gunicorn_config.py',
            '/opt/mcads/app/mcads_nginx.conf',
            '/opt/mcads/app/mcads.service',
            '/etc/nginx/sites-available/mcads',
            '/etc/systemd/system/mcads.service'
        ]
        
        try:
            with tarfile.open(backup_path, 'w:gz') as tar:
                for config_file in config_files:
                    if os.path.exists(config_file):
                        tar.add(config_file, arcname=os.path.basename(config_file))
            
            size_kb = backup_path.stat().st_size / 1024
            logger.info(f"Configuration backup created: {backup_filename} ({size_kb:.2f} KB)")
            return backup_path
            
        except Exception as e:
            logger.error(f"Configuration backup error: {str(e)}")
            return None
    
    def cleanup_old_backups(self):
        """Remove old backups according to retention policy"""
        now = datetime.datetime.now()
        
        for backup_dir, backup_type in [
            (self.db_backup_dir, 'database'),
            (self.media_backup_dir, 'media'),
            (self.app_backup_dir, 'configuration')
        ]:
            if not backup_dir.exists():
                continue
                
            removed_count = 0
            for backup_file in backup_dir.glob('*'):
                if not backup_file.is_file():
                    continue
                
                # Get file age
                file_age = now - datetime.datetime.fromtimestamp(backup_file.stat().st_mtime)
                
                # Apply retention policy
                should_remove = False
                if file_age.days > self.monthly_retention:
                    should_remove = True
                elif file_age.days > self.weekly_retention and not self._is_weekly_backup(backup_file, now):
                    should_remove = True
                elif file_age.days > self.daily_retention and not self._is_monthly_backup(backup_file, now):
                    should_remove = True
                
                if should_remove:
                    backup_file.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old {backup_type} backups")
    
    def _is_weekly_backup(self, backup_file, now):
        """Check if backup should be kept as weekly backup (Sunday backups)"""
        file_time = datetime.datetime.fromtimestamp(backup_file.stat().st_mtime)
        return file_time.weekday() == 6  # Sunday
    
    def _is_monthly_backup(self, backup_file, now):
        """Check if backup should be kept as monthly backup (first of month)"""
        file_time = datetime.datetime.fromtimestamp(backup_file.stat().st_mtime)
        return file_time.day == 1
    
    def create_full_backup(self):
        """Create a complete backup of all components"""
        logger.info("Starting full MCADS backup...")
        
        db_backup = self.backup_database()
        media_backup = self.backup_media_files()
        config_backup = self.backup_application_config()
        
        # Cleanup old backups
        self.cleanup_old_backups()
        
        # Generate backup report
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'database_backup': str(db_backup) if db_backup else 'FAILED',
            'media_backup': str(media_backup) if media_backup else 'FAILED',
            'config_backup': str(config_backup) if config_backup else 'FAILED'
        }
        
        success_count = sum(1 for v in report.values() if v != 'FAILED' and v != report['timestamp'])
        total_count = len(report) - 1  # Exclude timestamp
        
        if success_count == total_count:
            logger.info("✅ Full backup completed successfully")
            return True
        else:
            logger.warning(f"⚠️ Backup completed with issues ({success_count}/{total_count} successful)")
            return False
    
    def restore_database(self, backup_file):
        """Restore database from backup file"""
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        try:
            # Extract if compressed
            if backup_path.suffix == '.gz':
                temp_file = backup_path.with_suffix('')
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(temp_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                sql_file = temp_file
            else:
                sql_file = backup_path
            
            # Restore database
            cmd = [
                'psql',
                '-h', self.db_host,
                '-p', self.db_port,
                '-U', self.db_user,
                '-d', self.db_name,
                '-f', str(sql_file)
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = 'mcads_secure_password_2024'
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            # Cleanup temp file if created
            if backup_path.suffix == '.gz' and temp_file.exists():
                temp_file.unlink()
            
            if result.returncode == 0:
                logger.info(f"Database restored successfully from {backup_file}")
                return True
            else:
                logger.error(f"Database restore failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Database restore error: {str(e)}")
            return False
    
    def get_backup_status(self):
        """Get current backup status and statistics"""
        status = {
            'backup_directory': str(self.base_backup_dir),
            'database_backups': len(list(self.db_backup_dir.glob('*.gz'))) if self.db_backup_dir.exists() else 0,
            'media_backups': len(list(self.media_backup_dir.glob('*.tar.gz'))) if self.media_backup_dir.exists() else 0,
            'config_backups': len(list(self.app_backup_dir.glob('*.tar.gz'))) if self.app_backup_dir.exists() else 0,
        }
        
        # Calculate total backup size
        total_size = 0
        if self.base_backup_dir.exists():
            for item in self.base_backup_dir.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        
        status['total_size_mb'] = total_size / (1024 * 1024)
        
        return status

def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python backup_system.py <command>")
        print("Commands: backup, restore <backup_file>, status")
        sys.exit(1)
    
    backup_system = MCadsBackupSystem()
    command = sys.argv[1]
    
    if command == 'backup':
        success = backup_system.create_full_backup()
        sys.exit(0 if success else 1)
    
    elif command == 'restore' and len(sys.argv) == 3:
        backup_file = sys.argv[2]
        success = backup_system.restore_database(backup_file)
        sys.exit(0 if success else 1)
    
    elif command == 'status':
        status = backup_system.get_backup_status()
        print("📦 MCADS Backup Status:")
        print(f"   Directory: {status['backup_directory']}")
        print(f"   Database backups: {status['database_backups']}")
        print(f"   Media backups: {status['media_backups']}")
        print(f"   Config backups: {status['config_backups']}")
        print(f"   Total size: {status['total_size_mb']:.2f} MB")
    
    else:
        print("Invalid command")
        sys.exit(1)

if __name__ == "__main__":
    main() 