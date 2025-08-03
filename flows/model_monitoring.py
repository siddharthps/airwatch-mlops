# flows/model_monitoring.py
"""
Model monitoring flow using Evidently AI for data drift and performance monitoring.

This module provides functionality to:
1. Load historical reference data from S3
2. Load current prediction data from S3
3. Generate drift and performance reports using Evidently AI
4. Save reports to S3 and alert on data drift
"""

import io
import os
from datetime import datetime
from typing import Optional

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

# Evidently imports for version 0.6.7
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, RegressionPreset
from evidently.report import Report

load_dotenv()

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET_NAME")

# AWS S3 client setup
boto3_config = Config(
    region_name=AWS_REGION,
    s3={'addressing_style': 'virtual'}
)
s3_client = boto3.client('s3', config=boto3_config)


@task
def load_historical_data_from_s3(
    bucket_name: str,
    key_prefix: str = "processed_data/pm25_daily",
    file_name: str = "pm25_daily_cleaned_2009_2024.parquet"
) -> pd.DataFrame:
    """
    Load historical reference data from S3.

    Args:
        bucket_name: S3 bucket name
        key_prefix: S3 key prefix path
        file_name: Name of the parquet file

    Returns:
        DataFrame containing historical data

    Raises:
        Exception: If loading from S3 fails
    """
    logger = get_run_logger()
    s3_key = f"{key_prefix}/{file_name}"
    logger.info("Loading historical (reference) data from s3://%s/%s",
                bucket_name, s3_key)

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        df = pd.read_parquet(io.BytesIO(response['Body'].read()))
        logger.info("Successfully loaded %d records of historical data.", len(df))
        return df
    except Exception as e:
        logger.error("Failed to load historical data from S3: %s", e)
        raise


@task
def load_predictions_from_s3(
    bucket_name: str,
    key_prefix: str = "predictions/pm25_daily",
    target_year: int = datetime.now().year
) -> pd.DataFrame:
    """
    Load latest prediction data from S3 for a given year.

    Args:
        bucket_name: S3 bucket name
        key_prefix: S3 key prefix for predictions
        target_year: Year to search for predictions

    Returns:
        DataFrame containing prediction data

    Raises:
        Exception: If loading from S3 fails
    """
    logger = get_run_logger()
    prefix_to_list = f"{key_prefix}/pm25_predictions_{target_year}_"
    logger.info("Listing objects in s3://%s/%s to find latest predictions.",
                bucket_name, prefix_to_list)

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix_to_list)

        latest_file = None
        latest_time = None

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    if latest_file is None or obj['LastModified'] > latest_time:
                        latest_time = obj['LastModified']
                        latest_file = obj['Key']

        if latest_file:
            logger.info("Found latest prediction file: %s", latest_file)
            response = s3_client.get_object(Bucket=bucket_name, Key=latest_file)
            df = pd.read_parquet(io.BytesIO(response['Body'].read()))
            logger.info("Successfully loaded %d predictions from %s.",
                       len(df), latest_file)
            return df
        else:
            logger.warning(
                "No prediction files found for year %d under s3://%s/%s",
                target_year, bucket_name, prefix_to_list
            )
            return pd.DataFrame()

    except Exception as e:
        logger.error("Failed to load predictions from S3: %s", e)
        raise


@task
def save_evidently_report_to_s3(report: Report, bucket_name: str, s3_key: str):
    """
    Save Evidently report as HTML to S3.

    Args:
        report: Evidently Report object
        bucket_name: S3 bucket name
        s3_key: S3 key path for the report

    Raises:
        Exception: If saving to S3 fails
    """
    logger = get_run_logger()
    logger.info("Saving Evidently report to s3://%s/%s", bucket_name, s3_key)

    try:
        html_report = report.get_html()
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=html_report.encode('utf-8')
        )
        logger.info("Successfully saved Evidently report to s3://%s/%s",
                   bucket_name, s3_key)
    except Exception as e:
        logger.error("Failed to save Evidently report to S3: %s", e)
        raise


def validate_required_columns(data: pd.DataFrame, required_columns: list,
                            data_type: str) -> list:
    """
    Validate that required columns exist in the data.

    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        data_type: Description of data type for logging

    Returns:
        List of missing columns
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        try:
            logger = get_run_logger()
            logger.error("Missing required columns in %s data: %s",
                        data_type, missing_columns)
        except Exception:
            # Fallback for testing without Prefect context
            print(f"Missing required columns in {data_type} data: {missing_columns}")
    return missing_columns


def create_data_drift_report(reference_data: pd.DataFrame,
                           current_data: pd.DataFrame,
                           target_year: int) -> tuple:
    """
    Create data drift and quality report.

    Args:
        reference_data: Historical reference data
        current_data: Current prediction data
        target_year: Year for report naming

    Returns:
        Tuple of (report, report_path)
    """
    try:
        logger = get_run_logger()
    except Exception:
        # Fallback for testing without Prefect context
        import logging
        logger = logging.getLogger(__name__)

    # Column mapping for data drift report
    data_drift_column_mapping = ColumnMapping(
        target="arithmetic_mean",
        numerical_features=[
            'latitude', 'longitude', 'year', 'month',
            'day_of_week', 'day_of_year', 'is_weekend'
        ]
    )

    data_drift_report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])

    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=data_drift_column_mapping
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = (f"monitoring_reports/"
                  f"data_drift_report_{target_year}_{timestamp}.html")

    logger.info("Generated Data Drift Report: %s", report_path)
    return data_drift_report, report_path


def create_regression_performance_report(current_data: pd.DataFrame,
                                       target_year: int) -> tuple:
    """
    Create regression performance report.

    Args:
        current_data: Current prediction data with both targets and predictions
        target_year: Year for report naming

    Returns:
        Tuple of (report, report_path)
    """
    try:
        logger = get_run_logger()
    except Exception:
        # Fallback for testing without Prefect context
        import logging
        logger = logging.getLogger(__name__)

    # Column mapping for regression performance report
    regression_column_mapping = ColumnMapping(
        target="arithmetic_mean",
        prediction="predicted_arithmetic_mean",
        numerical_features=[
            'latitude', 'longitude', 'year', 'month',
            'day_of_week', 'day_of_year', 'is_weekend'
        ]
    )

    regression_performance_report = Report(metrics=[
        RegressionPreset(),
        DataQualityPreset()
    ])

    # Using current_data as both reference and current for self-comparison
    regression_performance_report.run(
        reference_data=current_data,
        current_data=current_data,
        column_mapping=regression_column_mapping
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = (f"monitoring_reports/"
                  f"regression_performance_report_{target_year}_{timestamp}.html")

    logger.info("Generated Regression Performance Report: %s", report_path)
    return regression_performance_report, report_path


def check_data_drift(data_drift_report: Report) -> bool:
    """
    Check if data drift is detected in the report.

    Args:
        data_drift_report: Evidently data drift report

    Returns:
        True if data drift is detected, False otherwise
    """
    try:
        logger = get_run_logger()
    except Exception:
        # Fallback for testing without Prefect context
        import logging
        logger = logging.getLogger(__name__)

    try:
        data_drift_result = data_drift_report.as_dict()
        
        # Validate the expected structure
        if not data_drift_result or "metrics" not in data_drift_result:
            raise ValueError("Invalid report structure: missing 'metrics' key")
        
        if not data_drift_result["metrics"] or len(data_drift_result["metrics"]) == 0:
            raise ValueError("Invalid report structure: empty 'metrics' list")
        
        if "result" not in data_drift_result["metrics"][0]:
            raise ValueError("Invalid report structure: missing 'result' key in metrics")
        
        if "dataset_drift" not in data_drift_result["metrics"][0]["result"]:
            raise ValueError("Invalid report structure: missing 'dataset_drift' key in result")

        if data_drift_result["metrics"][0]["result"]["dataset_drift"]:
            logger.error("ALERT: Data drift detected! Action required.")
            return True
        else:
            logger.info("No significant data drift detected.")
            return False
    except Exception as e:
        logger.error("Error checking data drift: %s", e)
        raise

@flow(name="Model Monitoring Flow with Evidently")
def model_monitoring_flow(target_year: Optional[int] = None):
    """
    Main model monitoring flow that generates drift and performance reports.

    Args:
        target_year: Year to monitor. Defaults to current year if None.

    Raises:
        ValueError: If data drift is detected
        Exception: For other processing errors
    """
    logger = get_run_logger()
    logger.info("Starting Model Monitoring Flow.")

    # Set target year
    if target_year is None:
        target_year = datetime.now().year
        logger.info("No target_year specified. Defaulting to current year: %d",
                   target_year)
    else:
        logger.info("Running monitoring for data from year: %d", target_year)

    # Load data
    reference_data = load_historical_data_from_s3(S3_DATA_BUCKET)
    if reference_data.empty:
        logger.error("Reference data is empty. Cannot perform monitoring.")
        return

    current_data = load_predictions_from_s3(S3_DATA_BUCKET, target_year=target_year)
    if current_data.empty:
        logger.error("Current prediction data is empty. Cannot perform monitoring.")
        return

    # Log data information
    logger.info("Reference data columns: %s", reference_data.columns.tolist())
    logger.info("Current data columns: %s", current_data.columns.tolist())

    # Validate required columns
    prediction_columns = ['predicted_arithmetic_mean', 'arithmetic_mean']
    feature_columns = [
        'latitude', 'longitude', 'year', 'month',
        'day_of_week', 'day_of_year', 'is_weekend'
    ]

    missing_prediction_cols = validate_required_columns(
        current_data, prediction_columns, "current prediction"
    )
    missing_feature_cols = validate_required_columns(
        current_data, feature_columns, "current feature"
    )

    if missing_prediction_cols or missing_feature_cols:
        logger.error("Cannot proceed with monitoring due to missing columns.")
        return

    logger.info("Generating Evidently AI reports...")

    try:
        # Generate Data Drift Report
        data_drift_report, data_drift_path = create_data_drift_report(
            reference_data, current_data, target_year
        )
        save_evidently_report_to_s3(
            data_drift_report, S3_DATA_BUCKET, data_drift_path
        )

        # Generate Regression Performance Report
        regression_report, regression_path = create_regression_performance_report(
            current_data, target_year
        )
        save_evidently_report_to_s3(
            regression_report, S3_DATA_BUCKET, regression_path
        )

        logger.info("Evidently AI reports generated and saved to S3.")

        # Check for data drift and raise alert if detected
        if check_data_drift(data_drift_report):
            raise ValueError(
                "Data drift detected! Action required: Review data drift report."
            )

    except Exception as e:
        logger.error("Error generating or saving Evidently reports: %s", e)
        raise

    logger.info("Model Monitoring Flow finished successfully.")


if __name__ == "__main__":
    model_monitoring_flow(target_year=2025)
