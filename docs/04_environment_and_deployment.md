# Environment Configuration and Deployment

## Overview

This document describes the environment variables, configuration, and deployment
considerations necessary to run the AirWatch MLOps pipeline seamlessly. Proper setup
ensures secure, scalable, and reproducible execution of data ingestion, model training,
inference, and monitoring workflows.

______________________________________________________________________

## Environment Variables

The project uses a `.env` file for configuration management. The following environment
variables are essential:

| Variable                          | Description                                                | Required/Optional | Notes                                 |
| --------------------------------- | ---------------------------------------------------------- | ----------------- | ------------------------------------- |
| `EPA_AQS_EMAIL`                   | Email registered with EPA AQS API                          | Required          | Used for API authentication           |
| `EPA_AQS_API_KEY`                 | API key for EPA AQS API                                    | Required          | Used for API authentication           |
| `AWS_REGION`                      | AWS region for all S3 operations                           | Optional          | Defaults to `us-east-1`               |
| `AWS_ACCESS_KEY_ID`               | AWS IAM access key ID                                      | Required          | For programmatic AWS access           |
| `AWS_SECRET_ACCESS_KEY`           | AWS IAM secret access key                                  | Required          | For programmatic AWS access           |
| `S3_DATA_BUCKET_NAME`             | S3 bucket for storing raw, processed data, and predictions | Required          | Bucket must have correct policies     |
| `S3_MLFLOW_ARTIFACTS_BUCKET_NAME` | S3 bucket for MLflow model artifacts                       | Required          | Separate bucket recommended           |
| `MLFLOW_TRACKING_URI`             | URL of MLflow tracking server                              | Optional          | Defaults to local server              |
| `MLFLOW_S3_ENDPOINT_URL`          | S3-compatible endpoint URL for MLflow artifacts            | Optional          | Usually defaults to AWS S3 endpoint   |
| `MLFLOW_ARTIFACT_LOCATION`        | S3 path prefix for storing MLflow artifacts                | Optional          | Example: `s3://your-bucket/artifacts` |
| `PREFECT_API_URL`                 | URL for Prefect server API                                 | Optional          | If using Prefect Cloud or Server      |

______________________________________________________________________

## Secure Credential Management

- **Never commit `.env` files containing secrets** to version control.
- Use secret management solutions such as AWS Secrets Manager or HashiCorp Vault in
  production.
- For local development, store secrets in `.env` and include `.env` in `.gitignore`.

______________________________________________________________________

## Deployment Considerations

### Local Development

- Install dependencies via the preferred Python package manager (e.g., `uv`).
- Configure AWS CLI or export AWS credentials in the environment.
- Run MLflow tracking server locally for experiment tracking.
- Start Prefect server or use Prefect Cloud for orchestration.

### Cloud or Production Environment

- Use managed MLflow tracking (e.g., Databricks, AWS Sagemaker) or dedicated MLflow
  server.
- Store secrets securely with IAM roles or environment variables injected by container
  orchestrators.
- Ensure S3 buckets have correct policies for read/write access.
- Deploy Prefect flows as scheduled jobs or event-driven triggers in orchestration
  platforms.

______________________________________________________________________

## Folder Structure and File Locations

├── .env.example ├── flows/ │ ├── data_ingestion.py │ ├── data_transformation.py │ ├──
model_training.py │ ├── model_selector.py │ ├── inference_data_preparation.py │ ├──
model_inference.py │ └── model_monitoring.py ├── tests/ │ ├── test_data_ingestion.py │
├── test_model_training.py │ └── ... ├── README.md └── requirements.txt or
pyproject.toml

______________________________________________________________________

## Best Practices

- Use separate buckets or prefixes for raw data, processed data, models, and predictions
  to keep data organized.
- Implement IAM least privilege access for AWS credentials.
- Regularly rotate API keys and AWS credentials.
- Use Prefect's retry and scheduling capabilities to handle transient failures and
  automate workflows.
- Log comprehensively for easier debugging and monitoring.

______________________________________________________________________

## Summary

This configuration and deployment guide ensures the AirWatch MLOps pipeline runs
smoothly across local and cloud environments, enabling robust, secure, and scalable air
quality prediction workflows.
