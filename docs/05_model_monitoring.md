# Model Monitoring & Data Drift Detection

## Overview

Effective model monitoring is critical for maintaining prediction quality and reliability in production. This document outlines the monitoring setup in the AirWatch MLOps pipeline, covering data drift detection, model performance tracking, and automated alerting.

---

## Monitoring Goals

- Detect changes in input data distributions (data drift) that could degrade model performance
- Track key regression metrics on new prediction batches
- Generate human-readable reports and dashboards for quick assessment
- Enable alerts or automated triggers on detected anomalies or performance degradation

---

## Tools & Technologies

| Component    | Purpose                                  |
|--------------|------------------------------------------|
| **Evidently AI** | Data drift detection and performance reports |
| **Prefect**       | Scheduling and orchestrating monitoring workflows |
| **AWS S3**        | Storage of monitoring reports and historical data |
| **MLflow**        | Storing experiment metrics and model versions |

---

## Data Drift Detection

- **Features Monitored:**  
  The system monitors distribution changes for features like `latitude`, `longitude`, `year`, `month`, `day_of_week`, `day_of_year`, and `is_weekend`.

- **Methodology:**  
  Evidently AI uses statistical tests and distribution comparisons (e.g., Kolmogorov-Smirnov test) between the training baseline and incoming data batches.

- **Reports:**  
  Generated reports include data drift dashboards highlighting which features exhibit significant drift and summary metrics.

- **Storage:**  
  Reports are stored in S3 under `monitoring_reports/` with timestamps for historical tracking.

---

## Model Performance Monitoring

- **Metrics Tracked:**  
  RMSE, MAE, R² on validation and test sets, plus metrics on live prediction feedback if available.

- **Comparison:**  
  Current model performance is compared against baseline and previous runs to detect degradation.

- **Reporting:**  
  Evidently AI generates regression performance reports updated with new evaluation data.

---

## Alerting and Automation

- **Automated Alerts:**  
  Integration hooks can send alerts (e.g., email, Slack) when drift exceeds thresholds or performance drops below acceptable limits.

- **Prefect Orchestration:**  
  Monitoring flows run on schedule, fetch latest data and predictions, run Evidently analysis, store reports, and trigger alerts.

---

## Configuration

### Environment Variables Used

| Variable                     | Purpose                              |
|------------------------------|------------------------------------|
| `S3_DATA_BUCKET_NAME`         | Bucket for storing reports          |
| `MLFLOW_TRACKING_URI`         | Access to experiment metrics        |
| `PREFECT_API_URL`             | Prefect server for scheduling       |

---

## Best Practices

- Regularly schedule monitoring flows to keep tabs on model health.
- Use versioning for model and data to identify when drift or degradation started.
- Combine monitoring with retraining pipelines to enable automated model updates.
- Keep monitoring logs and reports centralized for audit and compliance.

---

## Summary

The monitoring system in AirWatch MLOps ensures that deployed models remain reliable by tracking data quality and model predictions continuously, empowering teams to act promptly on anomalies or drift.
