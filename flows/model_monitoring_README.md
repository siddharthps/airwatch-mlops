# Model Monitoring for Air Quality Predictions

This section details the implementation of a robust model monitoring pipeline within the MLOps project for Chicago Air Quality Predictions. The primary goal of model monitoring is to detect and alert on changes in data distributions (data drift) or degradation in model performance over time, ensuring the predictive accuracy and reliability of the deployed model.

## 🎯 Purpose

In MLOps, models deployed in production can suffer from various issues over time, leading to reduced performance. These issues often stem from:
* **Data Drift:** Changes in the characteristics of incoming data (features, target variable) compared to the data the model was trained on.
* **Concept Drift:** Changes in the relationship between input features and the target variable.
* **Model Performance Degradation:** The model's predictions become less accurate due to the above drifts or other factors.

This monitoring pipeline is designed to automatically detect such issues by comparing current operational data and model predictions against historical reference data.

## 🛠️ Technologies Used

* **Prefect:** Orchestration framework for defining, scheduling, and executing data workflows and managing notifications.
* **Evidently AI (v0.6.7):** An open-source library used for generating interactive data quality, data drift, and model performance reports. Version `0.6.7` was specifically used due to API compatibility requirements.
* **AWS S3:** Cloud storage for storing raw data, processed data, model predictions, and the generated Evidently AI reports.
* **Boto3:** Python SDK for interacting with AWS services.
* **Python:** The primary programming language for implementing the monitoring logic.

## ✨ Core Functionality Achieved

This model monitoring pipeline provides the following key capabilities:

1.  **Automated Data Loading:**
    * Loads historical (reference) air quality data from an S3 bucket.
    * Loads the latest model predictions (current data) for the target year (e.g., 2025) from S3.

2.  **Comprehensive Report Generation with Evidently AI:**
    * Generates two distinct and insightful HTML reports:
        * **Data Drift Report:** Compares the distribution of input features and the target variable between the historical reference data and the current operational data. This helps identify shifts in data characteristics.
        * **Regression Performance Report:** Evaluates the performance of the model on the current prediction data by comparing actual values (`arithmetic_mean`) with predicted values (`predicted_arithmetic_mean`).
    * Reports are generated with specific column mappings (`evidently.model_profile.sections.ColumnMapping`) tailored for Evidently AI `v0.6.7`.

3.  **S3 Storage for Reports:**
    * All generated Evidently AI HTML reports are automatically saved to a designated `monitoring_reports` folder within the S3 bucket, ensuring persistent storage and easy access.

4.  **Automated Scheduling with Prefect:**
    * The entire monitoring flow (`model_monitoring_flow`) is deployed as a Prefect Deployment.
    * It is configured to run automatically on a defined schedule (e.g., every 24 hours, or via a cron expression), eliminating the need for manual triggering.

5.  **Automated Alerting for Data Drift:**
    * An `EmailNotification` block is configured within Prefect to send alerts.
    * The `model_monitoring_flow` is designed to explicitly raise an exception if significant data drift is detected by Evidently AI.
    * This exception causes the Prefect flow run to transition to a `FAILED` state, triggering an automated email notification (using dummy email details for demonstration purposes) to alert stakeholders about potential data integrity or model performance issues.

## 🚀 How it Works (High-Level Flow)

1.  A Prefect server is running, managing deployments and flow runs.
2.  The `Air-Quality-Monitoring-Deployment` is configured with a schedule (e.g., daily).
3.  At the scheduled time, Prefect initiates a flow run for `model_monitoring_flow`.
4.  The flow loads historical data (2009-2024) and the latest 2025 predictions from S3.
5.  Evidently AI analyzes these datasets to generate:
    * A Data Drift report (comparing historical data vs. current data features/target).
    * A Regression Performance report (evaluating current predictions against current actuals).
6.  Both reports are saved as HTML files back to the S3 bucket.
7.  The flow checks the Data Drift report's findings. If data drift is detected, the flow deliberately fails.
8.  Prefect captures the `FAILED` state and triggers the configured email notification.
9.  Stakeholders receive an alert (in a real scenario) prompting them to review the detailed Evidently reports in S3.

## ⚙️ Setup and Prerequisites

To run and utilize this monitoring setup:

1.  **AWS Account & S3 Bucket:**
    * An AWS account with an S3 bucket (e.g., `air-quality-mlops-data-chicago-2025`).
    * The bucket must contain:
        * Historical processed data (e.g., `processed_data/pm25_daily/pm25_daily_cleaned_2009_2024.parquet`).
        * Latest model predictions (e.g., `predictions/pm25_daily/pm25_predictions_2025_*.parquet`).
2.  **AWS CLI Configured:** Ensure your AWS Command Line Interface is installed and configured with credentials that have read/write access to your S3 bucket.
3.  **Python Environment:**
    * Python 3.8+
    * Virtual environment activated (`.venv\Scripts\activate`).
    * Required packages installed:
        ```bash
        uv pip install prefect evidently==0.6.7 boto3 pandas python-dotenv
        ```
4.  **Prefect Server Running:**
    * In a dedicated terminal, start your local Prefect server:
        ```bash
        prefect dev start
        ```
5.  **Evidently Email Notification Block:**
    * Create an `EmailNotification` block in your Prefect environment, named `model-monitoring-email-alert`, using the CLI or UI. Remember to use dummy SMTP details for non-real email sending:
        ```bash
        prefect block create email-notification --name "model-monitoring-email-alert" \
            --field "sender_email=your_dummy_sender@example.com" \
            --field "sender_password=your_dummy_password" \
            --field "smtp_server=smtp.example.com" \
            --field "smtp_port=587" \
            --field "smtp_use_tls=true" \
            --field "recipient_email=recipient@example.com" \
            --field "subject=Air Quality Model Monitoring Alert!" \
            --field "body=Attention: Air quality model monitoring flow status: {state_name} - {flow_run_name}. More details at {flow_run_url}"
        ```

## ▶️ Usage

1.  **Ensure Prefect Server is Running:** (`prefect dev start` in a separate terminal).
2.  **Activate your Python virtual environment.**
3.  **Apply the Prefect Deployment:**
    ```bash
    python deploy_monitoring.py
    ```
    This registers the scheduled flow with the Prefect server.
4.  **Monitor in Prefect UI:** Visit `http://127.0.0.1:4200` to observe your deployment under "Deployments" and upcoming "Flow Runs".
5.  **Access Reports:** Once runs complete, download the `.html` reports from your S3 bucket using the AWS Console or AWS CLI:
    ```bash
    aws s3 cp s3://air-quality-mlops-data-chicago-2025/monitoring_reports/data_drift_report_2025_YYYYMMDD_HHMMSS.html .
    aws s3 cp s3://air-quality-mlops-data-chicago-2025/monitoring_reports/regression_performance_report_2025_YYYYMMDD_HHMMSS.html .
    ```
    *(Replace `YYYYMMDD_HHMMSS` with the actual timestamp from your flow run logs).*

## ⏭️ Future Enhancements

* **Real Email Service Integration:** Replace dummy SMTP details with actual credentials for live email alerts.
* **More Granular Alerts:** Implement custom Evidently Tests and link them to more specific alert conditions (e.g., alert only if drift in `arithmetic_mean` exceeds a certain threshold).
* **Dashboards:** Integrate key monitoring metrics into a dedicated dashboard (e.g., using Grafana, Power BI, Tableau) for a consolidated, visual overview.
* **Automated Retraining Trigger:** If drift or performance degradation is severe, automatically trigger a model retraining pipeline.
* **Categorical Features:** If applicable, explicitly define categorical features in the `ColumnMapping` for more detailed analysis by Evidently.
* **Timezone Handling:** Refine timezone handling for schedules and data if operating across multiple timezones.