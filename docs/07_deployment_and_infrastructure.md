# Deployment & Infrastructure Setup

## Overview

This document describes the infrastructure and deployment strategy for the AirWatch
MLOps pipeline, covering cloud resources, orchestration, artifact storage, and access
management.

______________________________________________________________________

## Cloud Infrastructure

### AWS Services Used

- **S3 (Simple Storage Service):** Used as the primary data lake for storing raw data,
  processed data, model artifacts, predictions, and monitoring reports.

- **IAM (Identity and Access Management):** Manages secure access to AWS resources. IAM
  roles and policies are configured to enforce least privilege for pipeline components.

- **(Optional) EC2 or ECS:** For hosting components such as MLflow tracking server,
  Prefect server, or batch prediction jobs, if not running locally or on managed
  services.

______________________________________________________________________

## Storage Organization

S3 buckets and key prefixes are organized as follows:

| Bucket                            | Purpose                             | Key Prefix Examples                            |
| --------------------------------- | ----------------------------------- | ---------------------------------------------- |
| `S3_DATA_BUCKET_NAME`             | Raw and processed data, predictions | `raw_data/`, `processed_data/`, `predictions/` |
| `S3_MLFLOW_ARTIFACTS_BUCKET_NAME` | MLflow experiment artifacts         | `artifacts/models/`                            |
| `Monitoring Reports Bucket`       | Evidently reports and dashboards    | `monitoring_reports/data_drift_report_*/`      |

______________________________________________________________________

## MLflow Tracking Server

- Hosted locally or on cloud infrastructure.
- Uses the configured S3 bucket for artifact storage.
- Tracks experiments, metrics, parameters, and artifacts.
- Model registry enables versioning and staged deployment.

______________________________________________________________________

## Prefect Orchestration

- Prefect server or Prefect Cloud manages workflow scheduling and monitoring.
- Flows are defined to automate data ingestion, model training, inference, and
  monitoring.
- Supports retries, logging, and alerts for robustness.

______________________________________________________________________

## Access and Security

- AWS credentials are provided via environment variables or AWS IAM roles.
- Prefect and MLflow credentials are managed securely using `.env` files and secrets
  managers.
- Network security policies ensure restricted access to APIs and servers.

______________________________________________________________________

## Deployment Process

- Code and infrastructure configurations are version-controlled.
- Automated CI/CD pipelines handle testing, packaging, and deployment.
- Models are automatically registered in MLflow and artifacts pushed to S3.
- Batch prediction and monitoring flows are scheduled or triggered as needed.

______________________________________________________________________

## Scalability and Maintenance

- Infrastructure supports scaling storage and compute resources independently.
- Modular code and workflows allow easy extension and updates.
- Monitoring alerts trigger on anomalies or failures for proactive maintenance.

______________________________________________________________________

## Summary

The deployment and infrastructure setup ensures a scalable, secure, and manageable
environment for running the AirWatch MLOps pipeline. It leverages AWS for storage and
compute, MLflow for model management, and Prefect for orchestration.

______________________________________________________________________
