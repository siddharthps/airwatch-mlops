"""
EPA Air Quality Data Transformation Pipeline

This module provides functions to load raw air quality data from S3,
transform it through cleaning and feature engineering, and save the
processed data back to S3 using Prefect for orchestration.
"""

import io
import os
import traceback

import pandas as pd
import boto3
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger

# --- Load environment variables from .env file ---
load_dotenv()


@task(retries=3, retry_delay_seconds=10)
def load_raw_data_from_s3(bucket_name: str, key_prefix: str, file_name: str) -> pd.DataFrame:
    """
    Loads raw Parquet data from S3 into a Pandas DataFrame.
    """
    logger = get_run_logger()
    s3_key = f"{key_prefix}/{file_name}.parquet"
    logger.info("📥 Loading raw data from s3://%s/%s", bucket_name, s3_key)

    try:
        # Get boto3 S3 client directly
        s3_client = boto3.client('s3')

        # Download the object
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)

        # Read into Pandas DataFrame directly from bytes
        df = pd.read_parquet(io.BytesIO(response['Body'].read()))

        logger.info("Successfully loaded %d records from %s.", len(df), s3_key)
        logger.info("Raw data shape: %s", df.shape)
        logger.info("Columns in raw DataFrame: %s", df.columns.tolist())
        return df
    except Exception as e:
        logger.error("ERROR loading data from S3: %s", e)
        traceback.print_exc()
        raise


@task
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning, type conversions, and feature engineering.
    """
    logger = get_run_logger()

    if df.empty:
        logger.warning("Input DataFrame is empty, returning empty DataFrame.")
        return pd.DataFrame()

    logger.info("✨ Starting data transformation on %d records.", len(df))

    # --- Data Cleaning & Type Conversion ---
    if 'date_local' in df.columns:
        df['date_local'] = pd.to_datetime(df['date_local'], errors='coerce')
        before_drop = len(df)
        df.dropna(subset=['date_local'], inplace=True)
        logger.info(
            "   - Converted 'date_local' to datetime. Dropped %d rows with invalid dates.",
            before_drop - len(df)
        )
    else:
        logger.warning("   - 'date_local' column not found for datetime conversion.")

    numeric_cols = ['arithmetic_mean', 'first_max_value', 'aqi']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isnull().sum()
            logger.info("   - Converted '%s' to numeric. NaNs: %d", col, nan_count)
        else:
            logger.warning("   - Column '%s' not found for numeric conversion.", col)

    initial_rows = len(df)
    df.dropna(subset=[col for col in numeric_cols if col in df.columns], inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        logger.info("   - Dropped %d rows with missing critical numeric values.", dropped_rows)

    # --- Feature Engineering ---
    if 'date_local' in df.columns:
        df['year'] = df['date_local'].dt.year
        df['month'] = df['date_local'].dt.month
        df['day_of_week'] = df['date_local'].dt.dayofweek
        df['day_of_year'] = df['date_local'].dt.dayofyear
        logger.info("   - Added year, month, day_of_week, day_of_year features.")

    if 'day_of_week' in df.columns:
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        logger.info("   - Added 'is_weekend' feature.")

    selected_columns = [
        'date_local', 'arithmetic_mean', 'first_max_value', 'aqi',
        'latitude', 'longitude', 'cbsa_code', 'state', 'county', 'city',
        'year', 'month', 'day_of_week', 'day_of_year', 'is_weekend'
    ]

    df = df[[col for col in selected_columns if col in df.columns]]
    logger.info("   - Selected %d columns.", len(df.columns))

    logger.info("✨ Data transformation complete. Final shape: %s", df.shape)
    logger.info("\nFirst 5 rows of transformed data:\n%s", df.head())

    return df


@task(retries=3, retry_delay_seconds=10)
def write_transformed_data_to_s3(df: pd.DataFrame, bucket_name: str,
                                key_prefix: str, file_name: str):
    """
    Writes a Pandas DataFrame to S3 in Parquet format.
    """
    logger = get_run_logger()

    if df.empty:
        logger.warning("Transformed DataFrame is empty, skipping export to S3.")
        return

    full_s3_key = f"{key_prefix}/{file_name}.parquet"
    logger.info("📤 Uploading transformed data to s3://%s/%s", bucket_name, full_s3_key)

    try:
        s3_client = boto3.client('s3')

        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_bytes = parquet_buffer.getvalue()

        s3_client.put_object(Bucket=bucket_name, Key=full_s3_key, Body=parquet_bytes)

        logger.info("Successfully wrote transformed data to s3://%s/%s",
                   bucket_name, full_s3_key)
    except Exception as e:
        logger.error("ERROR during S3 upload of transformed data: %s", e)
        traceback.print_exc()
        raise


@flow(name="EPA AQS Data Transformation to S3")
def air_quality_transformation_flow(
    input_bucket_name: str,
    input_key_prefix: str,
    input_file_name: str,
    output_bucket_name: str,
    output_key_prefix: str,
    output_file_name: str,
):
    """
    Orchestrates loading raw data, transforming it, and saving to S3.
    """
    logger = get_run_logger()
    logger.info("Starting EPA AQS Data Transformation Flow.")

    raw_df = load_raw_data_from_s3(
        bucket_name=input_bucket_name,
        key_prefix=input_key_prefix,
        file_name=input_file_name,
    )

    transformed_df = transform_data(df=raw_df)

    write_transformed_data_to_s3(
        df=transformed_df,
        bucket_name=output_bucket_name,
        key_prefix=output_key_prefix,
        file_name=output_file_name,
    )

    logger.info("EPA AQS Data Transformation Flow finished.")


if __name__ == "__main__":
    input_bucket = os.getenv("S3_DATA_BUCKET_NAME")
    output_bucket = os.getenv("S3_DATA_BUCKET_NAME")  # Usually same bucket but can differ

    if not input_bucket or not output_bucket:
        raise ValueError("S3_DATA_BUCKET_NAME environment variable must be set in .env")

    air_quality_transformation_flow(
        input_bucket_name=input_bucket,
        input_key_prefix="raw_data/pm25_daily",
        input_file_name="pm25_daily_2009_2024",
        output_bucket_name=output_bucket,
        output_key_prefix="processed_data/pm25_daily",
        output_file_name="pm25_daily_cleaned_2009_2024",
    )
