EPA AQS Data Transformation Pipeline
A data pipeline that loads, transforms, and saves air quality data from S3 using Prefect orchestration.

Overview
This script processes Environmental Protection Agency (EPA) Air Quality Surveillance (AQS) data. It:

Loads raw data from S3 in Parquet format
Performs data cleaning and transformation
Saves transformed data back to S3 in Parquet format
The pipeline is designed to process daily PM2.5 air quality data with timestamps, numeric values, and geographic metadata.

Key Components
1. Data Loading
load_raw_data_from_s3()
Connects to AWS S3 using boto3
Loads Parquet files from specified S3 path
Validates data structure and logs metadata
2. Data Transformation
transform_data()
Converts date columns to datetime format
Converts numeric columns to float type
Handles missing values
Adds derived features (year, month, day of week, etc.)
Selects final set of relevant columns
3. Data Saving
write_transformed_data_to_s3()
Converts DataFrame to Parquet format
Uploads to S3 with error handling
Maintains original data structure with added features
4. Orchestration
air_quality_transformation_flow()
Coordinates data loading, transformation, and saving
Uses Prefect for workflow management
Includes retry logic for error handling
Usage
Environment Variables
S3_DATA_BUCKET_NAME: Name of S3 bucket containing data
(Optional) S3_DATA_BUCKET_NAME_OUTPUT: Name of output bucket (can be different from input)
Execution
if __name__ == "__main__":
    input_bucket = os.getenv("S3_DATA_BUCKET_NAME")
    output_bucket = os.getenv("S3_DATA_BUCKET_NAME")  # Usually same bucket but can differ

    air_quality_transformation_flow(
        input_bucket=input_bucket,
        input_prefix="raw_data/pm25_daily",
        input_file="pm25_daily_2009_2024",
        output_bucket=output_bucket,
        output_prefix="processed_data/pm25_daily",
        output_file="pm25_daily_cleaned_2009_2024",
    )
Data Flow
S3 Raw Data (pm25_daily_2009_2024.parquet)
   ↓
[Load] → [Transform] → [Save to S3]
   ↓
S3 Processed Data (pm25_daily_cleaned_2009_2024.parquet)
Features
Retry Logic: 3 retries with 10-second delay for S3 operations
Data Validation: Logs data shape, column names, and null counts
Feature Engineering: Adds temporal features (year, month, day of week)
Data Quality Checks: Handles missing values and invalid date formats
Version Control: Preserves original data structure with added features
Notes
This pipeline processes daily air quality data with timestamps
The transformed data includes geographic metadata (latitude, longitude, etc.)
The workflow is designed for batch processing of environmental data
The code uses Prefect for workflow orchestration and error handling
Dependencies
pandas
boto3
prefect
python-dotenv
io
traceback
This pipeline provides a robust, reusable solution for processing environmental data with clear data quality checks and error handling.