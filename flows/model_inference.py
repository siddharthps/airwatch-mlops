"""
Model inference module for batch prediction of PM2.5 levels.

This module orchestrates the batch prediction process by:
1. Preparing new data for inference
2. Loading the best model from S3
3. Generating and saving predictions
"""

import io
import os
import traceback
from datetime import datetime

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

# Prefect imports
from prefect import flow, task, get_run_logger

# Import the inference data preparation flow/tasks
from flows.inference_data_preparation import inference_data_preparation_flow

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration from .env ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET_NAME")  # Your main data bucket
S3_MLFLOW_ARTIFACTS_BUCKET = os.getenv("S3_MLFLOW_ARTIFACTS_BUCKET_NAME")

# THIS IS THE CORRECTED PATH BASED ON YOUR MLFLOW UI ARTIFACT STRUCTURE
BEST_MODEL_S3_PATH = (
    "s3://mlflow-artifacts-chicago-2025/artifacts/models/"
    "m-1099ed93db9743d59baf0ede76be06c8/artifacts"
)

# Set boto3 default config globally
boto3_config = Config(region_name=AWS_REGION, s3={'addressing_style': 'virtual'})
boto3.setup_default_session(region_name=AWS_REGION)
s3_client = boto3.client('s3', config=boto3_config)


@task
def load_model_from_s3(s3_path: str):
    """
    Load a model from a specified S3 path (MLflow artifact).

    Args:
        s3_path: S3 path to the model artifact

    Returns:
        Loaded MLflow model

    Raises:
        Exception: If model loading fails
    """
    logger = get_run_logger()
    logger.info("Loading model from S3 path: %s", s3_path)

    try:
        # Import here to avoid global MLflow setup if not needed elsewhere
        import mlflow.pyfunc  # pylint: disable=import-outside-toplevel

        model = mlflow.pyfunc.load_model(s3_path)
        logger.info("Successfully loaded model from %s", s3_path)
        return model
    except Exception as e:
        logger.error("Failed to load model from S3 path '%s': %s", s3_path, e)
        traceback.print_exc()
        raise


@task
def load_processed_inference_data_from_s3(
    bucket_name: str, key_prefix: str, file_name: str
) -> pd.DataFrame:
    """
    Load processed inference data from S3.

    Args:
        bucket_name: S3 bucket name
        key_prefix: S3 key prefix
        file_name: Name of the file (without extension)

    Returns:
        DataFrame containing the processed inference data

    Raises:
        Exception: If data loading fails
    """
    logger = get_run_logger()
    s3_key = f"{key_prefix}/{file_name}.parquet"
    logger.info("📥 Loading processed inference data from s3://%s/%s",
                bucket_name, s3_key)

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        df = pd.read_parquet(io.BytesIO(response['Body'].read()))
        logger.info("Loaded %d records for inference.", len(df))
        return df
    except Exception as e:
        logger.error("Failed to load processed inference data from S3: %s", e)
        traceback.print_exc()
        raise


@task
def generate_predictions(model, data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions using the loaded model on the transformed data.

    Args:
        model: Loaded MLflow model
        data_df: DataFrame containing features for prediction

    Returns:
        DataFrame with original data plus predictions

    Raises:
        ValueError: If required features are missing
        Exception: If prediction fails
    """
    logger = get_run_logger()

    if data_df.empty:
        logger.warning("Input DataFrame for prediction is empty, "
                      "returning empty DataFrame.")
        return pd.DataFrame()

    # Define the exact features the model was trained on
    # This must match `split_data` in `model_training.py`
    features_for_prediction = [
        'latitude', 'longitude', 'year', 'month', 'day_of_week', 'day_of_year',
        'is_weekend'
    ]

    # Ensure the input DataFrame contains all required features
    missing_features = [f for f in features_for_prediction
                       if f not in data_df.columns]
    if missing_features:
        error_msg = f"Missing required features for prediction: {missing_features}"
        logger.error("%s", error_msg)
        raise ValueError(error_msg)

    # Select only the features the model expects
    x_predict = data_df[features_for_prediction]

    logger.info("Generating predictions for %d records.", len(x_predict))
    logger.info("Features used for prediction: %s", features_for_prediction)

    try:
        predictions = model.predict(x_predict)
        # Add predictions to a new DataFrame or to the original data_df
        predictions_df = data_df.copy()
        predictions_df['predicted_arithmetic_mean'] = predictions
        logger.info("Predictions generated successfully.")
        return predictions_df
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        traceback.print_exc()
        raise


@task(retries=3, retry_delay_seconds=10)
def save_predictions_to_s3(df_predictions: pd.DataFrame, bucket_name: str,
                          target_year: int):
    """
    Save the DataFrame with predictions to S3.

    Args:
        df_predictions: DataFrame containing predictions
        bucket_name: S3 bucket name
        target_year: Year for which predictions were made

    Raises:
        Exception: If saving to S3 fails
    """
    logger = get_run_logger()

    if df_predictions.empty:
        logger.warning("Predictions DataFrame is empty, skipping save to S3.")
        return

    predictions_s3_prefix = "predictions/pm25_daily"
    # Use current date to ensure unique file name for each run
    current_date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"pm25_predictions_{target_year}_{current_date_str}.parquet"
    full_s3_key = f"{predictions_s3_prefix}/{file_name}"

    logger.info("📤 Saving predictions to s3://%s/%s", bucket_name, full_s3_key)

    try:
        buffer = io.BytesIO()
        df_predictions.to_parquet(buffer, index=False)
        buffer.seek(0)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=full_s3_key,
            Body=buffer.getvalue()
        )
        logger.info("Successfully saved predictions to s3://%s/%s",
                   bucket_name, full_s3_key)
    except Exception as e:
        logger.error("Failed to save predictions to S3: %s", e)
        traceback.print_exc()
        raise


@flow(name="Model Batch Prediction Flow")
def model_batch_prediction_flow(inference_year: int = None):
    """
    Orchestrate the entire batch prediction process.

    This flow:
    1. Prepares new data for inference
    2. Loads the best model from S3
    3. Generates and saves predictions

    Args:
        inference_year: Year for which to generate predictions.
                       Defaults to current year if None.

    Returns:
        DataFrame containing predictions, or None if no data available
    """
    logger = get_run_logger()
    logger.info("Starting Model Batch Prediction Flow.")

    if inference_year is None:
        inference_year = datetime.now().year
        logger.info("No inference_year specified. Defaulting to current year: %d",
                   inference_year)
    else:
        logger.info("Running prediction for year: %d", inference_year)

    # 1. Prepare new data for inference
    processed_inference_df_future = inference_data_preparation_flow(
        target_year=inference_year,
        return_state=True  # Required to get the result of the flow
    )
    processed_inference_df = processed_inference_df_future.result()

    if processed_inference_df.empty:
        logger.warning("No data available for inference after preparation. "
                      "Aborting prediction.")
        return None

    # 2. Load the best model from S3
    model = load_model_from_s3(BEST_MODEL_S3_PATH)

    # 3. Generate predictions
    predictions_df = generate_predictions(model, processed_inference_df)

    # 4. Save predictions to S3
    save_predictions_to_s3(predictions_df, S3_DATA_BUCKET, inference_year)

    logger.info("Model Batch Prediction Flow finished.")
    return predictions_df


if __name__ == "__main__":
    # Example: Run batch prediction for 2025 data
    model_batch_prediction_flow(inference_year=2025)
    # To run for current year: model_batch_prediction_flow()
    