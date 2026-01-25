from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from xrayapp.models import XRayImage, PredictionHistory


class Command(BaseCommand):
    help = 'Migrate existing XRayImage and PredictionHistory records to assign them to admin user'

    def handle(self, *args, **kwargs):
        try:
            # Get the admin user
            admin_user = User.objects.get(username='admin')
            
            # Update XRayImage records without a user
            xray_count = XRayImage.objects.filter(user__isnull=True).update(user=admin_user)
            self.stdout.write(self.style.SUCCESS(f'Updated {xray_count} XRayImage records'))
            
            # Update PredictionHistory records without a user
            history_count = PredictionHistory.objects.filter(user__isnull=True).update(user=admin_user)
            self.stdout.write(self.style.SUCCESS(f'Updated {history_count} PredictionHistory records'))
            
            self.stdout.write(self.style.SUCCESS('Migration completed successfully'))
            
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR('Admin user not found. Please ensure the admin user exists.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during migration: {str(e)}')) 