# Model Inference Pipeline: Batch Prediction of PM2.5 Levels

## Overview

This module orchestrates the batch prediction workflow for air quality (PM2.5) levels
using the best trained model. It integrates:

- Data preparation by invoking the inference data preparation flow
- Loading the best model stored in an S3 bucket via MLflow artifact path
- Generating predictions on the processed input features
- Saving the predictions back to S3 with timestamped keys for traceability

______________________________________________________________________

## Key Components

### 1. Data Preparation

The pipeline starts by preparing inference data using the **inference data preparation
flow**, which fetches and processes new raw EPA AQS data for the target year. This
ensures the prediction inputs align with the model’s expected feature set.

### 2. Model Loading

- The best-performing model is loaded directly from S3 using MLflow's `pyfunc`
  interface.
- This dynamic loading supports flexibility in model versioning and deployment without
  needing to embed model files locally.
- Error handling ensures failures in loading are logged and raised appropriately.

### 3. Prediction Generation

- The loaded model predicts PM2.5 values on the processed features.
- Ensures the presence of required features (`latitude`, `longitude`, `year`, `month`,
  `day_of_week`, `day_of_year`, `is_weekend`) before prediction.
- Appends predictions to the original DataFrame in a new column:
  `predicted_arithmetic_mean`.
- Handles empty data gracefully by skipping prediction.

### 4. Saving Predictions

- Predictions are saved as Parquet files to a designated S3 bucket.
- File naming includes the prediction year and a timestamp to avoid overwriting.
- Logs successful uploads or any errors during S3 operations.

______________________________________________________________________

## Prefect Flow: `model_batch_prediction_flow`

- This is the main orchestrator flow, coordinating all tasks sequentially.
- Defaults to the current year if no year is specified.
- Returns the final DataFrame with predictions for downstream use or further monitoring.

______________________________________________________________________

## Logging and Error Handling

- Detailed logging tracks each step's progress and outcome.
- Tasks include retries on failure with backoff.
- Exceptions propagate with full traceback for troubleshooting.

______________________________________________________________________

## Integration Notes

- This flow depends on the inference data preparation flow for data consistency.
- The S3 bucket names and MLflow artifact paths are configurable via environment
  variables.
- Designed to be run on-demand or scheduled within a workflow scheduler such as Prefect
  Cloud or Prefect Server.

______________________________________________________________________

## Summary

The model inference pipeline provides a robust and maintainable batch prediction process
that integrates with the full MLOps lifecycle — from data acquisition to storing
actionable predictions — ready to support operational monitoring and decision making.
