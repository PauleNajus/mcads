"""
Management command to populate severity_level for existing PredictionHistory records
"""
from django.core.management.base import BaseCommand
from xrayapp.models import PredictionHistory


class Command(BaseCommand):
    help = 'Populate severity_level for existing PredictionHistory records that have NULL severity_level'

    def handle(self, *args, **kwargs):
        # Get all records without severity_level
        records_to_update = PredictionHistory.objects.filter(severity_level__isnull=True)
        total_count = records_to_update.count()
        
        if total_count == 0:
            self.stdout.write(self.style.SUCCESS('No records need updating. All records have severity_level set.'))
            return
        
        self.stdout.write(f'Found {total_count} records without severity_level. Updating...')
        
        updated_count = 0
        failed_count = 0
        
        # Update records in batches
        for record in records_to_update:
            try:
                # Calculate severity level using the model's property method
                calculated_severity = record.calculate_severity_level
                
                if calculated_severity is not None:
                    record.severity_level = calculated_severity
                    record.save(update_fields=['severity_level'])
                    updated_count += 1
                    
                    # Print progress every 100 records
                    if updated_count % 100 == 0:
                        self.stdout.write(f'Updated {updated_count}/{total_count} records...')
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                self.stdout.write(self.style.ERROR(f'Error updating record {record.id}: {str(e)}'))
        
        # Print summary
        self.stdout.write(self.style.SUCCESS(f'\nCompleted!'))
        self.stdout.write(self.style.SUCCESS(f'Successfully updated: {updated_count} records'))
        if failed_count > 0:
            self.stdout.write(self.style.WARNING(f'Failed to update: {failed_count} records'))

