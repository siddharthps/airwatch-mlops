"""
Air Quality Model Training Pipeline

This module provides functionality for training machine learning models
to predict air quality metrics using processed PM2.5 data.
"""

import io
import math
import os
from typing import Any

import boto3
from botocore.config import Config
from dotenv import load_dotenv
import mlflow
from mlflow.exceptions import MlflowException
import mlflow.sklearn
import pandas as pd
from prefect import flow, get_run_logger, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

load_dotenv()

# Load environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET_NAME")
S3_MLFLOW_ARTIFACTS_BUCKET = os.getenv("S3_MLFLOW_ARTIFACTS_BUCKET_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
MLFLOW_ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_LOCATION")

# Set boto3 default config to ensure correct region and addressing style
BOTO3_CONFIG = Config(region_name=AWS_REGION, s3={"addressing_style": "virtual"})

# Setup boto3 session and clients globally
boto3.setup_default_session(region_name=AWS_REGION)
s3_client = boto3.client("s3", config=BOTO3_CONFIG)

# Set MLflow tracking URI globally for this script
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@task(retries=3, retry_delay_seconds=10)
def load_processed_data_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame:
    """
    Load processed data from S3 bucket.

    Args:
        bucket_name: Name of the S3 bucket
        file_key: Key of the file in the S3 bucket

    Returns:
        DataFrame containing the loaded data

    Raises:
        Exception: If data loading fails
    """
    logger = get_run_logger()
    logger.info("📥 Loading data from s3://%s/%s", bucket_name, file_key)
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        data_frame = pd.read_parquet(io.BytesIO(response["Body"].read()))
        logger.info(
            "✅ Loaded %d records. Shape: %s", len(data_frame), data_frame.shape
        )
        return data_frame
    except Exception as exc:
        logger.exception("❌ Failed to load data from S3")
        raise exc


@task
def split_data(
    data_frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    features: list[str] | None = None,
    target: str = "arithmetic_mean",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/validation/test sets.

    Args:
        data_frame: Input DataFrame
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility
        features: List of feature columns to use
        target: Target column name

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger = get_run_logger()
    logger.info("📊 Splitting data into train/val/test sets")
    logger.info("Available columns: %s", data_frame.columns.tolist())

    if data_frame.empty:
        logger.warning("⚠️ DataFrame is empty")
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=float)
        return empty_df, empty_df, empty_df, empty_series, empty_series, empty_series

    # Default features (removed 'aqi' to prevent data leakage with 'arithmetic_mean' target)
    default_features = [
        "latitude",
        "longitude",
        "year",
        "month",
        "day_of_week",
        "day_of_year",
        "is_weekend",
    ]

    features = features or default_features
    # Ensure selected features actually exist in the DataFrame
    features = [feature for feature in features if feature in data_frame.columns]

    logger.info("Selected features: %s", features)

    if not features or target not in data_frame.columns:
        available_cols = data_frame.columns.tolist()
        raise ValueError(
            f"Valid features or target '{target}' not found in DataFrame. "
            f"Available columns: {available_cols}"
        )

    df_clean = data_frame[[*features, target]].dropna()
    removed_rows = len(data_frame) - len(df_clean)
    logger.info(
        "Cleaned data shape: %s (removed %d rows with NaN)",
        df_clean.shape,
        removed_rows,
    )

    feature_matrix = df_clean[features]
    target_vector = df_clean[target]

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        feature_matrix, target_vector, test_size=test_size, random_state=random_state
    )

    # 0.25 of train_val set means 0.25 * (1-test_size) of total data.
    # If test_size=0.2, then 0.25 * 0.8 = 0.2, so val_size is 20% of total
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.25, random_state=random_state
    )

    logger.info(
        "   - Train: %d | Val: %d | Test: %d", len(x_train), len(x_val), len(x_test)
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


@task
def train_and_log_model(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Train a model and log results to MLflow.

    Args:
        model_name: Name of the model
        model: Sklearn-compatible model instance
        x_train: Training features
        y_train: Training targets
        x_val: Validation features
        y_val: Validation targets
        x_test: Test features
        y_test: Test targets

    Returns:
        Dictionary containing model results
    """
    logger = get_run_logger()

    # Set MLflow experiment with artifact location
    try:
        mlflow.create_experiment(
            "air_quality_model_training", artifact_location=MLFLOW_ARTIFACT_LOCATION
        )
    except MlflowException as mlflow_exc:
        # Experiment probably exists, log if it's not the "already exists" error
        if "already exists" not in str(mlflow_exc):
            logger.warning("MLflow experiment creation warning: %s", mlflow_exc)

    mlflow.set_experiment("air_quality_model_training")

    with mlflow.start_run(run_name=model_name) as run:
        logger.info("🚀 Training %s...", model_name)
        model.fit(x_train, y_train)

        preds_val = model.predict(x_val)
        rmse_val = math.sqrt(mean_squared_error(y_val, preds_val))

        preds_test = model.predict(x_test)
        rmse_test = math.sqrt(mean_squared_error(y_test, preds_test))

        # Log params if available (e.g., for RandomForest, XGBoost)
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("val_rmse", rmse_val)
        mlflow.log_metric("test_rmse", rmse_test)

        logger.info(
            "✅ %s trained. Val RMSE: %.4f, Test RMSE: %.4f",
            model_name,
            rmse_val,
            rmse_test,
        )

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "val_rmse": rmse_val,
            "test_rmse": rmse_test,
        }


@flow(name="Model Training Flow")
def train_models() -> list[dict[str, Any]]:
    """
    Main flow for training multiple models.

    Returns:
        List of dictionaries containing results for each model
    """
    logger = get_run_logger()
    logger.info("Starting Model Training Flow.")

    # Data file key inside the data bucket
    data_file_key = "processed_data/pm25_daily/pm25_daily_cleaned_2009_2024.parquet"
    data_frame = load_processed_data_from_s3(S3_DATA_BUCKET, data_file_key)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(data_frame)

    models = [
        ("LinearRegression", LinearRegression()),
        (
            "RandomForest",
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        ),
        ("XGBoost", XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ]

    results = []
    for name, model in models:
        result = train_and_log_model(
            name, model, x_train, y_train, x_val, y_val, x_test, y_test
        )
        results.append(result)

    logger.info("All models trained. Results:")
    for result in results:
        logger.info(result)

    return results


if __name__ == "__main__":
    # The MLflow environment variables are picked up by mlflow client automatically
    # when you run this script. No need to explicitly set os.environ here if .env is loaded.
    training_results = train_models()
    print("✅ Training complete. Results printed in Prefect logs and above.")
