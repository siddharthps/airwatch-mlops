# flows/model_selector.py
"""
Model selection and upload flow for air quality ML models.

This module provides functionality to:
1. Find the best non-overfit model from MLflow experiments
2. Download model artifacts from MLflow
3. Upload selected model and metadata to S3
4. Generate and store model summary information
"""

import json
import logging
import os
from pathlib import Path

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError, NoCredentialsError
import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task

# Configuration constants
EXPERIMENT_NAME = "air_quality_model_training"
S3_BUCKET = "air-quality-mlops-data-chicago-2025"
S3_PREFIX = "selected_models"
ARTIFACT_PATH = "model"
MLFLOW_URI = "http://localhost:5000"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# MLflow configuration
mlflow.set_tracking_uri(MLFLOW_URI)


@task
def get_best_non_overfit_run(experiment_name: str) -> Run | None:
    """
    Find the best performing non-overfit model run from MLflow experiment.

    Args:
        experiment_name: Name of the MLflow experiment to search

    Returns:
        Best Run object if found, None otherwise

    Raises:
        MlflowException: If there are issues accessing MLflow
    """
    task_logger = get_run_logger()
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            task_logger.error("Experiment '%s' not found.", experiment_name)
            return None

        runs = client.search_runs(
            [experiment.experiment_id], order_by=["metrics.test_rmse ASC"]
        )

        # Filter for valid (non-overfit) runs
        valid_runs = [
            run
            for run in runs
            if (
                run.data.metrics.get("val_rmse", 0) > 1e-4
                and "test_rmse" in run.data.metrics
            )
        ]

        if valid_runs:
            task_logger.info("Selected best non-overfit model.")
            return valid_runs[0]
        elif runs:
            task_logger.warning(
                "All models appear overfit. Falling back to best available model."
            )
            return runs[0]
        else:
            task_logger.error("No runs found.")
            return None

    except MlflowException as e:
        task_logger.error("MLflow error while searching for runs: %s", e)
        raise


@task
def download_model_artifacts(run_id: str, artifact_path: str) -> Path | None:
    """
    Download model artifacts from MLflow.

    Args:
        run_id: MLflow run ID
        artifact_path: Path to the model artifact

    Returns:
        Path to downloaded model directory, None if failed

    Raises:
        MlflowException: If artifact download fails
    """
    task_logger = get_run_logger()

    try:
        model_dir = Path(download_artifacts(run_id=run_id, artifact_path=artifact_path))
        task_logger.info("Successfully downloaded model artifacts to %s", model_dir)
        return model_dir
    except MlflowException as e:
        task_logger.error("Failed to download model artifacts: %s", e)
        raise


@task
def upload_files_to_s3(model_dir: Path, run_id: str, bucket: str, prefix: str) -> bool:
    """
    Upload model files to S3.

    Args:
        model_dir: Local directory containing model files
        run_id: MLflow run ID for S3 key construction
        bucket: S3 bucket name
        prefix: S3 key prefix

    Returns:
        True if all files uploaded successfully, False otherwise

    Raises:
        ClientError: If S3 operations fail
        NoCredentialsError: If AWS credentials are not available
    """
    task_logger = get_run_logger()

    # Check if directory exists
    if not model_dir.exists() or not model_dir.is_dir():
        task_logger.error("Model directory does not exist: %s", model_dir)
        return False

    s3_client = boto3.client("s3")
    upload_successful = True

    try:
        # Upload model files
        for root, _, files in os.walk(model_dir):
            for file in files:
                local_path = Path(root) / file
                relative_path = local_path.relative_to(model_dir)
                s3_key = f"{prefix}/{run_id}/{relative_path}".replace("\\", "/")

                try:
                    s3_client.upload_file(str(local_path), bucket, s3_key)
                    task_logger.info("Uploaded %s", s3_key)
                except (ClientError, S3UploadFailedError) as e:
                    task_logger.error("Failed to upload %s: %s", s3_key, e)
                    upload_successful = False
                except OSError as e:
                    task_logger.error("File system error uploading %s: %s", s3_key, e)
                    upload_successful = False

        return upload_successful

    except NoCredentialsError as e:
        task_logger.error("AWS credentials not available: %s", e)
        raise


@task
def create_and_upload_model_summary(
    run: Run, model_dir: Path, bucket: str, prefix: str
) -> bool:
    """
    Create model summary JSON and upload to S3.

    Args:
        run: MLflow Run object containing metadata
        model_dir: Local model directory
        bucket: S3 bucket name
        prefix: S3 key prefix

    Returns:
        True if summary created and uploaded successfully, False otherwise

    Raises:
        ClientError: If S3 operations fail
        OSError: If file operations fail
    """
    task_logger = get_run_logger()
    run_id = run.info.run_id

    # Generate summary data
    summary = {
        "run_id": run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }

    summary_file = model_dir / "model_summary.json"

    try:
        # Write summary to file
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Upload to S3
        s3_client = boto3.client("s3")
        s3_key = f"{prefix}/{run_id}/model_summary.json"
        s3_client.upload_file(str(summary_file), bucket, s3_key)

        task_logger.info("Uploaded model_summary.json to %s", s3_key)
        return True

    except OSError as e:
        task_logger.error("Failed to write summary file: %s", e)
        return False
    except (ClientError, S3UploadFailedError) as e:
        task_logger.error("Failed to upload model_summary.json: %s", e)
        return False
    except (TypeError, ValueError) as e:
        task_logger.error("Data serialization error: %s", e)
        return False


@task
def upload_model_and_summary_to_s3(run: Run, bucket: str, prefix: str) -> bool:
    """
    Download model artifacts and upload to S3 with summary.

    Args:
        run: MLflow Run object
        bucket: S3 bucket name
        prefix: S3 key prefix

    Returns:
        True if all operations successful, False otherwise
    """
    task_logger = get_run_logger()
    run_id = run.info.run_id

    try:
        # Download model artifacts
        model_dir = download_model_artifacts(run_id, ARTIFACT_PATH)
        if model_dir is None:
            return False

        # Upload model files
        files_uploaded = upload_files_to_s3(model_dir, run_id, bucket, prefix)

        # Create and upload summary
        summary_uploaded = create_and_upload_model_summary(
            run, model_dir, bucket, prefix
        )

        upload_successful = files_uploaded and summary_uploaded
        if upload_successful:
            task_logger.info(
                "Successfully uploaded model and summary for run %s", run_id
            )
        else:
            task_logger.warning("Partial failure uploading model for run %s", run_id)

        return upload_successful

    except (MlflowException, ClientError, NoCredentialsError, OSError) as e:
        task_logger.error("Failed to upload model and summary: %s", e)
        return False


@flow(name="Model Selection and Upload Flow")
def model_selection_flow() -> bool:
    """
    Main flow to select best model and upload to S3.

    Returns:
        True if model selection and upload successful, False otherwise
    """
    task_logger = get_run_logger()
    task_logger.info("Starting model selection flow")

    try:
        # Get best model run
        best_run = get_best_non_overfit_run(EXPERIMENT_NAME)

        if best_run:
            task_logger.info(
                "Found best model run: %s with test RMSE: %.4f",
                best_run.info.run_id,
                best_run.data.metrics.get("test_rmse", 0),
            )

            # Upload model and summary
            upload_successful = upload_model_and_summary_to_s3(
                best_run, S3_BUCKET, S3_PREFIX
            )

            if upload_successful:
                task_logger.info("Model selection flow completed successfully")
                return True
            else:
                task_logger.error("Failed to upload model to S3")
                return False
        else:
            task_logger.error("No suitable model found. Skipping upload.")
            return False

    except (MlflowException, ClientError, NoCredentialsError, OSError) as e:
        task_logger.error("Model selection flow failed: %s", e)
        return False


if __name__ == "__main__":
    import sys

    FLOW_SUCEESS = model_selection_flow()
    sys.exit(0 if FLOW_SUCEESS else 1)
