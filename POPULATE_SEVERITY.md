# Populate Severity Levels for Existing Records

## Overview
A management command has been created to populate the `severity_level` field for existing `PredictionHistory` records in the database.

## Why This Is Needed
The sorting functionality for "Severity" requires the `severity_level` field to be populated. New records will automatically have this field calculated and saved, but existing records in the database may have NULL values.

## How to Run

### If using Docker (recommended):
```bash
cd /opt/mcads/app
docker exec mcads_web python manage.py populate_severity_levels
```

### If using virtual environment:
```bash
cd /opt/mcads/app
source venv/bin/activate  # Activate your virtual environment
python manage.py populate_severity_levels
```

### If running directly:
```bash
cd /opt/mcads/app
python3 manage.py populate_severity_levels
```

## What It Does
- Finds all `PredictionHistory` records with NULL `severity_level`
- Calculates severity based on average pathology probabilities:
  - **Level 1** (Insignificant findings): 0-19% average
  - **Level 2** (Moderate findings): 20-30% average  
  - **Level 3** (Significant findings): 31-100% average
- Updates records in batches with progress reporting
- Provides a summary of successfully updated and failed records

## Expected Output
```
Found 15 records without severity_level. Updating...
Updated 100/15 records...
Completed!
Successfully updated: 15 records
```

## After Running
Once this command is run, the severity sorting functionality will work correctly for all records.

## Future Records
All new `PredictionHistory` records created after this code update will automatically have `severity_level` calculated and saved, so this command only needs to be run once.

