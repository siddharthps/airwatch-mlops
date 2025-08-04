# Data Pipeline: Ingestion, Transformation, and Storage

## Overview

This module handles the acquisition, cleaning, and preparation of EPA AQS air quality
data to make it ready for model training and inference. The pipeline uses Prefect flows
and tasks to orchestrate these steps and stores processed data in AWS S3.

______________________________________________________________________

## 1. Data Ingestion

- **Source:** EPA AQS API — provides daily PM2.5 measurements by Core-Based Statistical
  Areas (CBSA).
- **Parameters:** PM2.5 parameter code (`88101`), CBSA code for Chicago (`16980`).
- **API Authentication:** Requires user email and API key, stored in environment
  variables.
- **Fetching Logic:** Data for a target year is requested, parsed from JSON, and
  combined into a Pandas DataFrame.
- **Error Handling:** Retries and detailed logging are implemented with Prefect's retry
  mechanism and Python exceptions.

______________________________________________________________________

## 2. Data Transformation

- **Cleaning:**
  - Converts `date_local` column to datetime.
  - Ensures latitude and longitude are numeric.
  - Drops rows with missing or invalid essential data.
- **Feature Engineering:**
  - Extracts temporal features from `date_local`: `year`, `month`, `day_of_week`,
    `day_of_year`.
  - Adds `is_weekend` binary indicator.
- **Output Columns:**
  - Includes geographic identifiers (`latitude`, `longitude`, `cbsa_code`, `state`,
    `county`, `city`).
  - Includes engineered features required by the model.

______________________________________________________________________

## 3. S3 Storage

- Processed data is saved in **Parquet** format to an S3 bucket.
- Bucket name and region are configured via environment variables.
- Uses boto3 client with region-specific configuration.
- Uploads data under a specific prefix for inference data, e.g.,
  `processed_data/pm25_daily_inference/pm25_daily_cleaned_2025_inference.parquet`.

______________________________________________________________________

## 4. Prefect Flow: `inference_data_preparation_flow`

- Combines fetching, transformation, and storage in a single flow.
- Defaults to current year if no year is provided.
- Returns the processed DataFrame for downstream consumption.
- Includes retries and logging for robustness.

Logging & Monitoring

```
Each step logs detailed information using Prefect's logger.

Errors trigger retries with exponential backoff.

Validation on critical columns ensures data integrity.
```
