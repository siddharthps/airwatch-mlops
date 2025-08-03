"""
Prefect flow to fetch EPA AQS PM2.5 data and upload it to S3 using the Prefect AWS block.
"""

import os
import io
import time
import traceback
import requests
import pandas as pd

from dotenv import load_dotenv

# Prefect imports
from prefect import flow, task
from prefect_aws import S3Bucket

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration from .env ---
PM25_PARAMETER_CODE = "88101"
CBSA_CODE = "16980"  # Chicago-Naperville-Elgin, IL-IN-WI

START_YEAR = 2009
END_YEAR = 2024
SLEEP_TIME_SECONDS = 6

# Get bucket names from environment variables
OUTPUT_S3_BUCKET_NAME = os.getenv("S3_DATA_BUCKET_NAME")
S3_KEY_PREFIX = "raw_data/pm25_daily"

# Validate essential environment variables
if not OUTPUT_S3_BUCKET_NAME:
    raise ValueError("S3_DATA_BUCKET_NAME environment variable not set in .env")


@task
def fetch_epa_aqs_data(
    email: str,
    api_key: str,
    start_year: int,
    end_year: int,
    param_code: str,
    cbsa_code: str,
    sleep_time: int
) -> pd.DataFrame:
    """
    Fetches daily PM2.5 data for a given CBSA from the EPA AQS API for multiple years.
    Returns the combined data as a Pandas DataFrame.
    """
    all_data_frames = []

    for year in range(start_year, end_year + 1):
        bdate = f"{year}0101"
        edate = f"{year}1231"

        url = (
            "https://aqs.epa.gov/data/api/dailyData/byCBSA"
            f"?email={email}&key={api_key}&param={param_code}"
            f"&bdate={bdate}&edate={edate}&cbsa={cbsa_code}"
        )

        print(f"Fetching data for {year} from {url}")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            print(f"DEBUG: type(data) = {type(data)}")
            print(f"DEBUG: data preview = {str(data)[:500]}")

            if isinstance(data, list):
                if not data:
                    print(f"API response for {year} is an empty list, no data.")
                    continue
                df = pd.DataFrame(data)
                print(f"Successfully fetched {len(df)} records for {year} (list response).")
                all_data_frames.append(df)

            elif isinstance(data, dict):
                header = data.get("Header")
                if (
                    header
                    and isinstance(header, list)
                    and header[0].get("status") == "No data meets your criteria"
                    ):
                    print(f"No data found for {year}.")
                    continue

                if "Data" in data and isinstance(data["Data"], list):
                    df = pd.DataFrame(data["Data"])
                    print(f"Successfully fetched {len(df)} records for {year} (dict response).")
                    all_data_frames.append(df)
                else:
                    print(f"API response for {year} is a dict but has no valid 'Data' key: {data}")

            else:
                print(
                    f"Unexpected API response format for {year}: "
                    f"Expected dict or list, got {type(data)}: {data}"
                    )

        except requests.exceptions.RequestException as e:
            print(f"HTTP Error fetching data for {year}: {e}")
        except ValueError as e:
            print(f"JSON Decoding Error for {year}: {e}")
            print(f"Raw response text: {response.text}")
        except Exception as e:
            print(f"An unexpected error occurred for {year}: {e}")
            raise   # re-raises the exception after logging

        if year < end_year:
            print(f"Waiting for {sleep_time} seconds before next year's request...")
            time.sleep(sleep_time)

    if not all_data_frames:
        print("No data was fetched across all years. Returning empty DataFrame.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data_frames, ignore_index=True)

    if 'date_local' in combined_df.columns:
        combined_df['date_local'] = pd.to_datetime(combined_df['date_local'])

    print(f"Combined raw data shape: {combined_df.shape}")
    print(f"Columns in raw DataFrame: {combined_df.columns.tolist()}")
    print("\nFirst 5 rows of combined raw data:")
    print(combined_df.head())

    return combined_df


@task
def write_data_to_s3(df: pd.DataFrame, bucket_block_name: str, key_prefix: str, file_name: str):
    """
    Uploads a Pandas DataFrame to an S3 bucket using a Prefect S3Bucket block.
    """
    if df.empty:
        print("DataFrame is empty, skipping export to S3.")
        return

    full_s3_key = f"{key_prefix}/{file_name}.parquet"
    print(f"📤 Uploading to s3://{bucket_block_name}/{full_s3_key}")

    try:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        s3_block = S3Bucket.load(bucket_block_name)
        s3_block.upload_from_file_object(buffer, full_s3_key)

        print(f"Successfully uploaded to s3://{bucket_block_name}/{full_s3_key}")
    except Exception as e:
        print(f"ERROR during S3 upload: {e}")
        traceback.print_exc()
        raise


@flow(name="EPA AQS Data Ingestion to S3")
def air_quality_ingestion_flow():
    """
    Orchestrates fetching EPA AQS data and writing it to S3.
    """
    print(f"Starting EPA AQS Data Ingestion Flow for years {START_YEAR}-{END_YEAR}")

    epa_aqs_email = os.getenv("EPA_AQS_EMAIL")
    epa_aqs_api_key = os.getenv("EPA_AQS_API_KEY")

    if not epa_aqs_email or not epa_aqs_api_key:
        raise ValueError(
            "EPA_AQS_EMAIL and EPA_AQS_API_KEY environment variables must be set in .env"
            )

    df = fetch_epa_aqs_data(
        email=epa_aqs_email,
        api_key=epa_aqs_api_key,
        start_year=START_YEAR,
        end_year=END_YEAR,
        param_code=PM25_PARAMETER_CODE,
        cbsa_code=CBSA_CODE,
        sleep_time=SLEEP_TIME_SECONDS
    )

    write_data_to_s3(
        df=df,
        bucket_block_name=OUTPUT_S3_BUCKET_NAME,
        key_prefix=S3_KEY_PREFIX,
        file_name=f"pm25_daily_{START_YEAR}_{END_YEAR}"
    )

    print("EPA AQS Data Ingestion Flow finished.")


if __name__ == "__main__":
    air_quality_ingestion_flow()
