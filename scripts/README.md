# Utility Scripts

This directory contains utility scripts for deployment, monitoring, and debugging.

## Scripts

### `deploy_monitoring.py`
Sets up Prefect deployment for automated model monitoring with email notifications.

**Usage:**
```bash
uv run python scripts/deploy_monitoring.py
```

### `inspect_s3_parquet.py`
Utility script to inspect and validate parquet files stored in S3.

**Usage:**
```bash
uv run python scripts/inspect_s3_parquet.py
```

### `test_mlflow_setup.py`
Tests MLflow server connectivity and basic functionality.

**Usage:**
```bash
uv run python scripts/test_mlflow_setup.py
```

### `prefect_email_notification.txt`
Configuration template for setting up email notifications in Prefect.

## Notes

- Make sure your environment variables are properly configured before running these scripts
- These scripts are meant for development and deployment assistance
- Check the individual script files for specific requirements and usage instructions