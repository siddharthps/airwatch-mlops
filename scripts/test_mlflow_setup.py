#!/usr/bin/env python3
"""
Simple script to test MLflow setup and S3 connectivity.
Run this before running the full model training pipeline.
"""

import os
import sys

import boto3
from dotenv import load_dotenv
import mlflow

# Load environment variables
load_dotenv()


def test_mlflow_connection():
    """Test MLflow tracking server connection."""
    print("🧪 Testing MLflow connection...")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    print(f"MLflow Tracking URI: {mlflow_uri}")

    try:
        mlflow.set_tracking_uri(mlflow_uri)

        # Try to create a test experiment
        experiment_name = "test_connection"
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created test experiment: {experiment_name} (ID: {experiment_id})")

        # Try to start a run
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            print(f"✅ Created test run: {run.info.run_id}")

        # Clean up - delete the test experiment
        mlflow.delete_experiment(experiment_id)
        print("✅ Cleaned up test experiment")

        return True

    except Exception as e:
        print(f"❌ MLflow connection failed: {e}")
        return False


def test_s3_connection():
    """Test S3 connectivity."""
    print("\n🧪 Testing S3 connection...")

    bucket_name = os.getenv("S3_DATA_BUCKET_NAME")
    print(f"S3 Data Bucket: {bucket_name}")

    try:
        s3_client = boto3.client("s3")

        # Test bucket access
        response = s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ Successfully connected to bucket: {bucket_name}")

        # List some objects
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=5)
        if "Contents" in response:
            print(f"✅ Found {len(response['Contents'])} objects in bucket")
            for obj in response["Contents"][:3]:
                print(f"   - {obj['Key']}")
        else:
            print("Bucket is empty")

        return True

    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        return False


def test_data_availability():
    """Test if processed data is available in S3."""
    print("\n🧪 Testing processed data availability...")

    bucket_name = os.getenv("S3_DATA_BUCKET_NAME")
    data_key = "processed_data/pm25_daily/pm25_daily_cleaned_2009_2024.parquet"

    try:
        s3_client = boto3.client("s3")
        response = s3_client.head_object(Bucket=bucket_name, Key=data_key)

        size_mb = response["ContentLength"] / (1024 * 1024)
        print(f"✅ Found processed data file: {data_key}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Last Modified: {response['LastModified']}")

        return True

    except Exception as e:
        print(f"❌ Processed data not found: {e}")
        print(f"   Expected location: s3://{bucket_name}/{data_key}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing MLOps setup...\n")

    # Check environment variables
    required_vars = [
        "S3_DATA_BUCKET_NAME",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_ARTIFACT_LOCATION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]

    print("🧪 Checking environment variables...")
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Don't print sensitive values
            if "KEY" in var or "SECRET" in var:
                print(f"✅ {var}: ***")
            else:
                print(f"✅ {var}: {value}")
        else:
            missing_vars.append(var)
            print(f"❌ {var}: Not set")

    if missing_vars:
        print(f"\n❌ Missing required environment variables: {missing_vars}")
        return False

    # Run tests
    tests_passed = 0
    total_tests = 3

    if test_s3_connection():
        tests_passed += 1

    if test_data_availability():
        tests_passed += 1

    if test_mlflow_connection():
        tests_passed += 1

    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Ready to run model training.")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before running model training.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
