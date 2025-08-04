# Air Quality Prediction MLOps Project – Overview

## 🎯 Project Objective

The goal of this project is to build a robust MLOps pipeline to predict PM2.5 air
quality values using environmental data from the U.S. Environmental Protection Agency
(EPA) AQS API. The project encompasses end-to-end machine learning workflows, including
data ingestion, transformation, training, deployment, monitoring, and alerting.

______________________________________________________________________

## 🧱 Project Architecture

This project is structured using the following modular components:

- **Data Ingestion**: Periodically pulls raw data from the AQS API.
- **Data Transformation**: Cleans and prepares the data, readying it for modeling.
- **Model Training**: Trains and logs machine learning models using MLflow.
- **Model Selection**: Selects the best model based on validation metrics and uploads it
  to S3.
- **Batch Prediction**: Periodically fetches fresh data, runs predictions, and stores
  the outputs.
- **Monitoring**: Uses Evidently AI to detect data drift and monitor regression metrics.
- **Orchestration**: Entire workflow is coordinated using Prefect flows.
- **Deployment**: No live API deployed — instead, batch inference pipelines are
  scheduled for prediction.

______________________________________________________________________

## ⚙️ Tech Stack

| Component            | Tool/Service                  |
| -------------------- | ----------------------------- |
| Orchestration        | Prefect                       |
| Data Source          | EPA AQS API                   |
| Model Tracking       | MLflow                        |
| Artifact Storage     | AWS S3                        |
| Monitoring           | Evidently AI                  |
| Data Processing      | pandas, scikit-learn          |
| Model Training       | XGBoost                       |
| Environment Handling | `.env` files, `python-dotenv` |
| CI/CD (optional)     | Manual (can be extended)      |

______________________________________________________________________

## 🔁 High-Level Pipeline Flow

```mermaid
flowchart TD
    A[Fetch EPA AQS Data] --> B[Transform Data]
    B --> C[Train Multiple Models]
    C --> D[Evaluate + Select Best Model]
    D --> E[Log Model with MLflow]
    E --> F[Upload to S3]
    F --> G[Batch Inference on New Data]
    G --> H[Monitor Predictions with Evidently]

Repository Structure

.
├── flows/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_training.py
│   ├── model_selector.py
│   ├── batch_prediction.py
│   └── model_monitoring.py
├── data/
│   └── (artifacts, raw data, predictions)
├── .env.example
├── README.md
└── requirements.txt (implied)

📈 ML Objective

    Target Variable: PM2.5 concentration

    Model Type: Regression

    Model Candidates: XGBoost, Decision Tree Regressor, Linear Regression

    Selection Metric: R² score (on validation set)

🗝️ Security & Secrets

Sensitive credentials (e.g., AWS keys, API tokens) are managed through a .env file and not committed to the repo.
```
