import io
import os

import boto3
from dotenv import load_dotenv
import pandas as pd
import pyarrow.parquet as pq

# Load environment variables (assuming your .env is configured)
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET_NAME")

s3_client = boto3.client('s3', region_name=AWS_REGION)

def inspect_parquet_schema_from_s3(bucket_name, s3_key):
    """
    Inspects and prints the schema of a Parquet file directly from S3.
    """
    print(f"\n--- Inspecting schema for s3://{bucket_name}/{s3_key} ---")
    try:
        # Get object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        parquet_bytes = io.BytesIO(response['Body'].read())

        # Read schema using pyarrow
        parquet_file = pq.ParquetFile(parquet_bytes)
        schema = parquet_file.schema.to_arrow_schema()

        print("Column Names and Data Types:")
        for field in schema:
            print(f"- {field.name}: {field.type}")

        # Optionally, read a small sample to see data
        # df_sample = pd.read_parquet(io.BytesIO(response['Body'].read()), n_rows=5) # n_rows is for fastparquet
        # For pyarrow, you can read the whole thing and then take head:
        parquet_bytes.seek(0) # Reset buffer position
        df_sample = pd.read_parquet(parquet_bytes)
        print("\nFirst 5 rows (sample data):")
        print(df_sample.head())


    except Exception as e:
        print(f"Error inspecting Parquet file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Verify the historical data file
    historical_data_key = "processed_data/pm25_daily/pm25_daily_cleaned_2009_2024.parquet"
    inspect_parquet_schema_from_s3(S3_DATA_BUCKET, historical_data_key)

    # Verify the current inference data file (with actuals)
    current_inference_data_key = "processed_data/pm25_daily_inference/pm25_daily_cleaned_2025_inference.parquet"
    inspect_parquet_schema_from_s3(S3_DATA_BUCKET, current_inference_data_key)

    # Verify the predictions file
    # This needs to find the latest one, similar to the task in your flow
    # For quick check, you can directly specify the file you just saw with `aws s3 ls`
    predictions_data_key = "predictions/pm25_daily/pm25_predictions_2025_20250802_155758.parquet"
    inspect_parquet_schema_from_s3(S3_DATA_BUCKET, predictions_data_key)
