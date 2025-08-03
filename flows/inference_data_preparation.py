"""
EPA AQS Inference Data Preparation Module

This module provides functionality to fetch, transform, and store EPA AQS data
for inference purposes. It includes tasks for data retrieval, cleaning, feature
engineering, and S3 storage using Prefect workflows.
"""

from datetime import datetime
import io
import os
import traceback

import boto3
from botocore.config import Config
from dotenv import load_dotenv
import pandas as pd
from prefect import flow, get_run_logger, task
import requests

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration from .env ---
PM25_PARAMETER_CODE = "88101"
CBSA_CODE = "16980"  # Chicago-Naperville-Elgin, IL-IN-WI

# API Call sleep time
SLEEP_TIME_SECONDS = 6

# Get S3 bucket name from environment variables
S3_DATA_BUCKET_NAME = os.getenv("S3_DATA_BUCKET_NAME")

# Validate essential environment variable
if not S3_DATA_BUCKET_NAME:
    raise ValueError("S3_DATA_BUCKET_NAME environment variable not set in .env")

# Set boto3 default config globally for this script
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
boto3_config = Config(region_name=AWS_REGION, s3={"addressing_style": "virtual"})
boto3.setup_default_session(region_name=AWS_REGION)
s3_client = boto3.client("s3", config=boto3_config)


@task(retries=3, retry_delay_seconds=10)
def fetch_epa_aqs_data_for_inference(
    email: str,
    api_key: str,
    target_year: int,  # Changed to target_year for specific inference year
    param_code: str,
    cbsa_code: str,
) -> pd.DataFrame:
    """
    Fetches daily PM2.5 data for a given CBSA from the EPA AQS API for a
    specific target year.
    Returns the data as a Pandas DataFrame.
    """
    logger = get_run_logger()
    all_data_frames = []

    bdate = f"{target_year}0101"
    edate = f"{target_year}1231"  # Fetch for the whole year

    url = (
        f"https://aqs.epa.gov/data/api/dailyData/byCBSA?"
        f"email={email}&key={api_key}&param={param_code}&bdate={bdate}"
        f"&edate={edate}&cbsa={cbsa_code}"
    )

    logger.info("Fetching data for %s from %s", target_year, url)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            if not data:
                logger.info(
                    "API response for %s is an empty list, no data.", target_year
                )
            else:
                df = pd.DataFrame(data)
                logger.info(
                    "Successfully fetched %s records for %s (list response).",
                    len(df),
                    target_year,
                )
                all_data_frames.append(df)

        elif isinstance(data, dict):
            header = data.get("Header")
            if (
                header
                and isinstance(header, list)
                and header[0].get("status") == "No data meets your criteria"
            ):
                logger.info("No data found for %s.", target_year)
            elif "Data" in data and isinstance(data["Data"], list):
                df = pd.DataFrame(data["Data"])
                logger.info(
                    "Successfully fetched %s records for %s (dict response).",
                    len(df),
                    target_year,
                )
                all_data_frames.append(df)
            else:
                logger.warning(
                    "API response for %s is a dict but has no valid 'Data' key: %s",
                    target_year,
                    data,
                )
        else:
            logger.warning(
                "Unexpected API response format for %s: Expected dict or list, "
                "got %s: %s",
                target_year,
                type(data),
                data,
            )

    except requests.exceptions.RequestException as e:
        logger.error("HTTP Error fetching data for %s: %s", target_year, e)
    except ValueError as e:
        logger.error("JSON Decoding Error for %s: %s", target_year, e)
    except Exception as e:
        logger.error("An unexpected error occurred for %s: %s", target_year, e)
        raise

    # No sleep needed if only fetching for one year in this task

    if not all_data_frames:
        logger.warning("No data was fetched. Returning empty DataFrame.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data_frames, ignore_index=True)

    if "date_local" in combined_df.columns:
        combined_df["date_local"] = pd.to_datetime(combined_df["date_local"])

    logger.info("Combined raw data shape for inference: %s", combined_df.shape)
    logger.info(
        "Columns in raw DataFrame for inference: %s", combined_df.columns.tolist()
    )
    logger.info(
        "First 5 rows of combined raw data for inference:\n%s", str(combined_df.head())
    )

    return combined_df


@task
def transform_data_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning, type conversions, and feature engineering for
    inference data. This must precisely match the transformation logic used
    during model training.
    """
    logger = get_run_logger()

    if df.empty:
        logger.warning(
            "Input DataFrame for transformation is empty, returning empty DataFrame."
        )
        return pd.DataFrame()

    logger.info("Starting data transformation for inference on %s records.", len(df))

    # --- Data Cleaning & Type Conversion ---
    # Ensure 'date_local' is datetime
    if "date_local" in df.columns:
        df["date_local"] = pd.to_datetime(df["date_local"], errors="coerce")
        initial_len = len(df)
        df.dropna(subset=["date_local"], inplace=True)
        if len(df) < initial_len:
            logger.info(
                "Dropped %s rows with invalid 'date_local'.", initial_len - len(df)
            )
    else:
        logger.warning("'date_local' column not found, cannot convert to datetime.")
        return pd.DataFrame()  # Essential for time-based features

    # Ensure geographic coordinates are numeric
    geo_cols = ["latitude", "longitude"]
    for col in geo_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                logger.warning("Found NaNs in '%s' after numeric conversion.", col)
        else:
            logger.warning("Column '%s' not found for numeric conversion.", col)

    # Drop rows with NaNs in essential features *for prediction*
    # The features used for prediction are:
    # 'latitude', 'longitude', 'year', 'month', 'day_of_week',
    # 'day_of_year', 'is_weekend'
    # We also keep 'date_local' and identifier columns for context in
    # predictions output.
    features_for_model = ["latitude", "longitude"]  # These must exist and be valid

    # Identify non-numeric and potentially missing columns
    required_cols_check = [*features_for_model, "date_local"]
    for col in required_cols_check:
        if col not in df.columns or df[col].isnull().all():
            logger.error(
                "Critical column '%s' is missing or entirely NaN. "
                "Cannot proceed with transformation.",
                col,
            )
            return pd.DataFrame()  # Return empty if critical columns are missing

    initial_rows = len(df)
    # Only drop NaNs for the actual features the model will use (excluding target)
    df.dropna(subset=features_for_model, inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        logger.info(
            "Dropped %s rows with missing critical feature values "
            "(latitude/longitude).",
            dropped_rows,
        )

    if df.empty:
        logger.warning(
            "DataFrame became empty after dropping NaNs for essential "
            "features. Returning empty."
        )
        return pd.DataFrame()

    # --- Feature Engineering ---
    df["year"] = df["date_local"].dt.year
    df["month"] = df["date_local"].dt.month
    df["day_of_week"] = df["date_local"].dt.dayofweek
    df["day_of_year"] = df["date_local"].dt.dayofyear
    logger.info("Added year, month, day_of_week, day_of_year features.")

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    logger.info("Added 'is_weekend' feature.")

    # Select columns relevant for output (features + identifiers)
    # 'arithmetic_mean', 'first_max_value', 'aqi' will NOT be present in *new*
    # inference data or will be the values we are trying to predict, so we don't
    # include them for the *model's input X*.
    # But we keep them if they exist in the raw data, for contextual output.
    columns_for_output = [
        "date_local",
        "latitude",
        "longitude",
        "cbsa_code",
        "state",
        "county",
        "city",  # Identifiers
        "year",
        "month",
        "day_of_week",
        "day_of_year",
        "is_weekend",  # Features
    ]
    # Add any original data columns that might still be useful for context
    optional_cols = ["arithmetic_mean", "first_max_value", "aqi"]
    for col in optional_cols:
        if col in df.columns and col not in columns_for_output:
            columns_for_output.append(col)

    df = df[[col for col in columns_for_output if col in df.columns]]
    logger.info(
        "Selected %s columns for processed inference data output.", len(df.columns)
    )

    logger.info("Data transformation for inference complete. Final shape: %s", df.shape)
    logger.info("First 5 rows of transformed inference data:\n%s", str(df.head()))

    return df


@task(retries=3, retry_delay_seconds=10)
def write_processed_inference_data_to_s3(
    df: pd.DataFrame, bucket_name: str, key_prefix: str, file_name: str
):
    """
    Writes a Pandas DataFrame (processed for inference) to S3 in Parquet format.
    """
    logger = get_run_logger()

    if df.empty:
        logger.warning("Processed inference DataFrame is empty, skipping export to S3.")
        return

    full_s3_key = f"{key_prefix}/{file_name}.parquet"
    logger.info(
        "Uploading processed inference data to s3://%s/%s", bucket_name, full_s3_key
    )

    try:
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_bytes = parquet_buffer.getvalue()

        s3_client.put_object(Bucket=bucket_name, Key=full_s3_key, Body=parquet_bytes)

        logger.info(
            "Successfully wrote processed inference data to s3://%s/%s",
            bucket_name,
            full_s3_key,
        )
    except Exception as e:
        logger.error("ERROR during S3 upload of processed inference data: %s", e)
        traceback.print_exc()
        raise


@flow(name="EPA AQS Inference Data Preparation Flow")
def inference_data_preparation_flow(target_year: int | None = None):
    """
    Orchestrates fetching new raw data, transforming it for inference,
    and saving to S3.
    """
    logger = get_run_logger()

    if target_year is None:
        target_year = datetime.now().year
        logger.info(
            "No target_year specified. Defaulting to current year: %s", target_year
        )
    else:
        logger.info("Preparing inference data for year: %s", target_year)

    epa_aqs_email = os.getenv("EPA_AQS_EMAIL")
    epa_aqs_api_key = os.getenv("EPA_AQS_API_KEY")

    if not epa_aqs_email or not epa_aqs_api_key:
        raise ValueError(
            "EPA_AQS_EMAIL and EPA_AQS_API_KEY environment variables "
            "must be set in .env"
        )

    raw_df_for_inference = fetch_epa_aqs_data_for_inference(
        email=epa_aqs_email,
        api_key=epa_aqs_api_key,
        target_year=target_year,
        param_code=PM25_PARAMETER_CODE,
        cbsa_code=CBSA_CODE,
    )

    processed_df_for_inference = transform_data_for_inference(df=raw_df_for_inference)

    # Define S3 keys for the processed inference data
    output_prefix = "processed_data/pm25_daily_inference"
    output_file = f"pm25_daily_cleaned_{target_year}_inference"

    write_processed_inference_data_to_s3(
        df=processed_df_for_inference,
        bucket_name=S3_DATA_BUCKET_NAME,
        key_prefix=output_prefix,
        file_name=output_file,
    )

    logger.info("EPA AQS Inference Data Preparation Flow finished.")
    return processed_df_for_inference  # Return for direct use in prediction flow


if __name__ == "__main__":
    # Example: Prepare 2025 data for inference
    inference_data_preparation_flow(target_year=2025)
    # To run for current year: inference_data_preparation_flow()
