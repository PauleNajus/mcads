"""
Management command to populate severity_level for existing PredictionHistory records
"""
from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand
from django.db import transaction

from xrayapp.models import PATHOLOGY_FIELDS, PredictionHistory


class Command(BaseCommand):
    help = 'Populate severity_level for existing PredictionHistory records that have NULL severity_level'

    def handle(self, *args: Any, **kwargs: Any) -> None:
        # Get all records without severity_level (load only fields we need).
        records_to_update = (
            PredictionHistory.objects
            .filter(severity_level__isnull=True)
            .only("pk", *PATHOLOGY_FIELDS)
        )
        total_count = records_to_update.count()
        
        if total_count == 0:
            self.stdout.write(self.style.SUCCESS('No records need updating. All records have severity_level set.'))
            return
        
        self.stdout.write(f'Found {total_count} records without severity_level. Updating...')
        
        failed_count = 0

        # Bulk update for speed: avoids one UPDATE per row.
        batch_size = 500
        updated_count = 0
        pending: list[PredictionHistory] = []

        with transaction.atomic():
            for record in records_to_update.iterator(chunk_size=batch_size):
                try:
                    calculated_severity = record.calculate_severity_level
                    if calculated_severity is None:
                        failed_count += 1
                        continue
                    record.severity_level = calculated_severity
                    pending.append(record)
                except Exception as e:
                    failed_count += 1
                    self.stdout.write(self.style.ERROR(f'Error updating record {record.pk}: {e}'))

                if len(pending) >= batch_size:
                    PredictionHistory.objects.bulk_update(pending, ["severity_level"], batch_size=batch_size)
                    updated_count += len(pending)
                    pending.clear()
                    self.stdout.write(f'Updated {updated_count}/{total_count} records...')

            if pending:
                PredictionHistory.objects.bulk_update(pending, ["severity_level"], batch_size=batch_size)
                updated_count += len(pending)
        
        # Print summary
        self.stdout.write(self.style.SUCCESS(f'\nCompleted!'))
        self.stdout.write(self.style.SUCCESS(f'Successfully updated: {updated_count} records'))
        if failed_count > 0:
            self.stdout.write(self.style.WARNING(f'Failed to update: {failed_count} records'))

