from __future__ import annotations

import os
from typing import Any

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from xrayapp.models import XRayImage, PredictionHistory


class Command(BaseCommand):
    help = 'Migrate existing XRayImage and PredictionHistory records to assign them to admin user'

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--admin-username",
            default=os.environ.get("MCADS_MIGRATION_ADMIN_USERNAME", "admin"),
            help="Username to assign orphaned records to (default: env MCADS_MIGRATION_ADMIN_USERNAME or 'admin')",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            admin_username: str = str(options.get("admin_username") or "admin")
            admin_user = User.objects.get(username=admin_username)
            
            # Update XRayImage records without a user
            xray_count = XRayImage.objects.filter(user__isnull=True).update(user=admin_user)
            self.stdout.write(self.style.SUCCESS(f'Updated {xray_count} XRayImage records'))
            
            # Update PredictionHistory records without a user
            history_count = PredictionHistory.objects.filter(user__isnull=True).update(user=admin_user)
            self.stdout.write(self.style.SUCCESS(f'Updated {history_count} PredictionHistory records'))
            
            self.stdout.write(self.style.SUCCESS('Migration completed successfully'))
            
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR('Admin user not found. Please ensure the target user exists.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during migration: {e}')) 