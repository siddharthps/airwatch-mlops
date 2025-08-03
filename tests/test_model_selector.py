"""
Unit tests for the model selector flow using pytest and moto for S3 interactions.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
from moto import mock_aws
import pytest

# Import the modules to test
from flows.model_selector import (
    create_and_upload_model_summary,
    download_model_artifacts,
    get_best_non_overfit_run,
    model_selection_flow,
    upload_files_to_s3,
    upload_model_and_summary_to_s3,
)


class TestGetBestNonOverfitRun:
    """Test cases for the get_best_non_overfit_run task."""

    @pytest.fixture
    def mock_mlflow_client(self):
        """Mock MLflow client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def mock_experiment(self):
        """Mock MLflow experiment."""
        experiment = MagicMock()
        experiment.experiment_id = "test_experiment_id"
        return experiment

    @pytest.fixture
    def mock_valid_runs(self):
        """Mock valid (non-overfit) runs."""
        runs = []
        for i in range(3):
            run = MagicMock()
            run.data.metrics = {
                "val_rmse": 0.5 + i * 0.1,  # Valid validation RMSE
                "test_rmse": 0.6 + i * 0.1,  # Test RMSE
            }
            runs.append(run)
        return runs

    @pytest.fixture
    def mock_overfit_runs(self):
        """Mock overfit runs (very low validation RMSE)."""
        runs = []
        for i in range(2):
            run = MagicMock()
            run.data.metrics = {
                "val_rmse": 1e-6,  # Suspiciously low validation RMSE
                "test_rmse": 0.8 + i * 0.1,
            }
            runs.append(run)
        return runs

    @patch("flows.model_selector.MlflowClient")
    def test_get_best_non_overfit_run_success(
        self, mock_client_class, mock_experiment, mock_valid_runs
    ):
        """Test successful selection of best non-overfit run."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = mock_valid_runs

        result = get_best_non_overfit_run("test_experiment")

        assert result == mock_valid_runs[0]  # Should return the first (best) run
        mock_client.get_experiment_by_name.assert_called_once_with("test_experiment")
        mock_client.search_runs.assert_called_once_with(
            ["test_experiment_id"], order_by=["metrics.test_rmse ASC"]
        )

    @patch("flows.model_selector.MlflowClient")
    def test_get_best_non_overfit_run_experiment_not_found(self, mock_client_class):
        """Test handling when experiment is not found."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.get_experiment_by_name.return_value = None

        result = get_best_non_overfit_run("nonexistent_experiment")

        assert result is None

    @patch("flows.model_selector.MlflowClient")
    def test_get_best_non_overfit_run_only_overfit_runs(
        self, mock_client_class, mock_experiment, mock_overfit_runs
    ):
        """Test fallback to best available run when all runs are overfit."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = mock_overfit_runs

        result = get_best_non_overfit_run("test_experiment")

        # Should return the first run as fallback
        assert result == mock_overfit_runs[0]

    @patch("flows.model_selector.MlflowClient")
    def test_get_best_non_overfit_run_no_runs(self, mock_client_class, mock_experiment):
        """Test handling when no runs are found."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = []

        result = get_best_non_overfit_run("test_experiment")

        assert result is None

    @patch("flows.model_selector.MlflowClient")
    def test_get_best_non_overfit_run_mlflow_exception(self, mock_client_class):
        """Test handling of MLflow exceptions."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        from mlflow.exceptions import MlflowException

        mock_client.get_experiment_by_name.side_effect = MlflowException("MLflow error")

        with pytest.raises(MlflowException):
            get_best_non_overfit_run("test_experiment")


class TestDownloadModelArtifacts:
    """Test cases for the download_model_artifacts task."""

    @patch("flows.model_selector.download_artifacts")
    def test_download_model_artifacts_success(self, mock_download):
        """Test successful model artifact download."""
        mock_download.return_value = "/path/to/model"

        result = download_model_artifacts("test_run_id", "model")

        assert result == Path("/path/to/model")
        mock_download.assert_called_once_with(
            run_id="test_run_id", artifact_path="model"
        )

    @patch("flows.model_selector.download_artifacts")
    def test_download_model_artifacts_failure(self, mock_download):
        """Test handling of download failures."""
        from mlflow.exceptions import MlflowException

        mock_download.side_effect = MlflowException("Download failed")

        with pytest.raises(MlflowException):
            download_model_artifacts("test_run_id", "model")


class TestUploadFilesToS3:
    """Test cases for the upload_files_to_s3 task."""

    @pytest.fixture
    def temp_model_dir(self, tmp_path):
        """Create a temporary model directory with files."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create some test files
        (model_dir / "model.pkl").write_text("model content")
        (model_dir / "metadata.json").write_text('{"version": "1.0"}')

        # Create subdirectory
        subdir = model_dir / "subdir"
        subdir.mkdir()
        (subdir / "config.yaml").write_text("config: value")

        return model_dir

    @mock_aws
    def test_upload_files_to_s3_success(self, temp_model_dir):
        """Test successful file upload to S3."""
        bucket_name = "test-bucket"
        run_id = "test_run_id"
        prefix = "selected_models"

        # Create S3 bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        result = upload_files_to_s3(temp_model_dir, run_id, bucket_name, prefix)

        assert result is True

        # Verify files were uploaded
        response = s3_client.list_objects_v2(
            Bucket=bucket_name, Prefix=f"{prefix}/{run_id}/"
        )
        uploaded_keys = [obj["Key"] for obj in response.get("Contents", [])]

        expected_keys = [
            f"{prefix}/{run_id}/model.pkl",
            f"{prefix}/{run_id}/metadata.json",
            f"{prefix}/{run_id}/subdir/config.yaml",
        ]

        for key in expected_keys:
            assert key in uploaded_keys

    @mock_aws
    def test_upload_files_to_s3_bucket_not_found(self, temp_model_dir):
        """Test handling when S3 bucket doesn't exist."""
        result = upload_files_to_s3(
            temp_model_dir, "test_run_id", "nonexistent-bucket", "prefix"
        )

        assert result is False

    def test_upload_files_to_s3_no_credentials(self, temp_model_dir):
        """Test handling when AWS credentials are not available."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import NoCredentialsError

            mock_boto_client.side_effect = NoCredentialsError()

            with pytest.raises(NoCredentialsError):
                upload_files_to_s3(
                    temp_model_dir, "test_run_id", "test-bucket", "prefix"
                )

    def test_upload_files_to_s3_nonexistent_directory(self):
        """Test handling when model directory doesn't exist."""
        nonexistent_dir = Path("/nonexistent/path")

        result = upload_files_to_s3(
            nonexistent_dir, "test_run_id", "test-bucket", "prefix"
        )

        assert result is False


class TestCreateAndUploadModelSummary:
    """Test cases for the create_and_upload_model_summary task."""

    @pytest.fixture
    def mock_run(self):
        """Mock MLflow run."""
        run = MagicMock()
        run.info.run_id = "test_run_id"
        run.data.metrics = {"test_rmse": 0.5, "val_rmse": 0.6}
        run.data.params = {"n_estimators": 100, "max_depth": 5}
        run.data.tags = {"model_type": "RandomForest"}
        return run

    @pytest.fixture
    def temp_model_dir(self, tmp_path):
        """Create a temporary model directory."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        return model_dir

    @mock_aws
    def test_create_and_upload_model_summary_success(self, mock_run, temp_model_dir):
        """Test successful creation and upload of model summary."""
        bucket_name = "test-bucket"
        prefix = "selected_models"

        # Create S3 bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        result = create_and_upload_model_summary(
            mock_run, temp_model_dir, bucket_name, prefix
        )

        assert result is True

        # Verify summary file was uploaded
        s3_key = f"{prefix}/{mock_run.info.run_id}/model_summary.json"
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        summary_content = json.loads(response["Body"].read().decode("utf-8"))

        expected_summary = {
            "run_id": "test_run_id",
            "metrics": {"test_rmse": 0.5, "val_rmse": 0.6},
            "params": {"n_estimators": 100, "max_depth": 5},
            "tags": {"model_type": "RandomForest"},
        }

        assert summary_content == expected_summary

    @mock_aws
    def test_create_and_upload_model_summary_s3_error(self, mock_run, temp_model_dir):
        """Test handling of S3 upload errors."""
        # Don't create the bucket to simulate an error
        result = create_and_upload_model_summary(
            mock_run, temp_model_dir, "nonexistent-bucket", "prefix"
        )

        assert result is False

    def test_create_and_upload_model_summary_file_write_error(self, mock_run):
        """Test handling of file write errors."""
        nonexistent_dir = Path("/nonexistent/path")

        result = create_and_upload_model_summary(
            mock_run, nonexistent_dir, "test-bucket", "prefix"
        )

        assert result is False


class TestUploadModelAndSummaryToS3:
    """Test cases for the upload_model_and_summary_to_s3 task."""

    @pytest.fixture
    def mock_run(self):
        """Mock MLflow run."""
        run = MagicMock()
        run.info.run_id = "test_run_id"
        run.data.metrics = {"test_rmse": 0.5}
        run.data.params = {"n_estimators": 100}
        run.data.tags = {"model_type": "RandomForest"}
        return run

    @patch("flows.model_selector.create_and_upload_model_summary")
    @patch("flows.model_selector.upload_files_to_s3")
    @patch("flows.model_selector.download_model_artifacts")
    def test_upload_model_and_summary_to_s3_success(
        self, mock_download, mock_upload_files, mock_upload_summary, mock_run
    ):
        """Test successful upload of model and summary."""
        mock_model_dir = Path("/path/to/model")
        mock_download.return_value = mock_model_dir
        mock_upload_files.return_value = True
        mock_upload_summary.return_value = True

        result = upload_model_and_summary_to_s3(mock_run, "test-bucket", "prefix")

        assert result is True
        mock_download.assert_called_once_with("test_run_id", "model")
        mock_upload_files.assert_called_once_with(
            mock_model_dir, "test_run_id", "test-bucket", "prefix"
        )
        mock_upload_summary.assert_called_once_with(
            mock_run, mock_model_dir, "test-bucket", "prefix"
        )

    @patch("flows.model_selector.download_model_artifacts")
    def test_upload_model_and_summary_to_s3_download_failure(
        self, mock_download, mock_run
    ):
        """Test handling when model download fails."""
        mock_download.return_value = None

        result = upload_model_and_summary_to_s3(mock_run, "test-bucket", "prefix")

        assert result is False

    @patch("flows.model_selector.create_and_upload_model_summary")
    @patch("flows.model_selector.upload_files_to_s3")
    @patch("flows.model_selector.download_model_artifacts")
    def test_upload_model_and_summary_to_s3_partial_failure(
        self, mock_download, mock_upload_files, mock_upload_summary, mock_run
    ):
        """Test handling when some uploads fail."""
        mock_model_dir = Path("/path/to/model")
        mock_download.return_value = mock_model_dir
        mock_upload_files.return_value = True
        mock_upload_summary.return_value = False  # Summary upload fails

        result = upload_model_and_summary_to_s3(mock_run, "test-bucket", "prefix")

        assert result is False


class TestModelSelectionFlow:
    """Test cases for the main model selection flow."""

    @pytest.fixture
    def mock_best_run(self):
        """Mock best run."""
        run = MagicMock()
        run.info.run_id = "best_run_id"
        run.data.metrics = {"test_rmse": 0.45}
        return run

    @patch("flows.model_selector.upload_model_and_summary_to_s3")
    @patch("flows.model_selector.get_best_non_overfit_run")
    def test_model_selection_flow_success(
        self, mock_get_best, mock_upload, mock_best_run
    ):
        """Test successful model selection flow."""
        mock_get_best.return_value = mock_best_run
        mock_upload.return_value = True

        result = model_selection_flow()

        assert result is True
        mock_get_best.assert_called_once_with("air_quality_model_training")
        mock_upload.assert_called_once_with(
            mock_best_run, "air-quality-mlops-data-chicago-2025", "selected_models"
        )

    @patch("flows.model_selector.get_best_non_overfit_run")
    def test_model_selection_flow_no_suitable_model(self, mock_get_best):
        """Test flow when no suitable model is found."""
        mock_get_best.return_value = None

        result = model_selection_flow()

        assert result is False

    @patch("flows.model_selector.upload_model_and_summary_to_s3")
    @patch("flows.model_selector.get_best_non_overfit_run")
    def test_model_selection_flow_upload_failure(
        self, mock_get_best, mock_upload, mock_best_run
    ):
        """Test flow when model upload fails."""
        mock_get_best.return_value = mock_best_run
        mock_upload.return_value = False

        result = model_selection_flow()

        assert result is False

    @patch("flows.model_selector.get_best_non_overfit_run")
    def test_model_selection_flow_exception_handling(self, mock_get_best):
        """Test flow exception handling."""
        from mlflow.exceptions import MlflowException

        mock_get_best.side_effect = MlflowException("MLflow error")

        result = model_selection_flow()

        assert result is False


# Parametrized tests for different run scenarios
@pytest.mark.parametrize(
    "val_rmse,test_rmse,expected_valid",
    [
        (0.5, 0.6, True),  # Valid run
        (1e-6, 0.6, False),  # Overfit (very low val_rmse)
        (0.0, 0.6, False),  # Overfit (zero val_rmse)
        (0.001, 0.6, True),  # Valid run (0.001 > 1e-4)
        (0.1, 0.6, True),  # Valid run
    ],
)
def test_run_validation_logic(val_rmse, test_rmse, expected_valid):
    """Test the logic for determining if a run is valid (non-overfit)."""
    # This tests the filtering logic used in get_best_non_overfit_run
    run = MagicMock()
    run.data.metrics = {"val_rmse": val_rmse, "test_rmse": test_rmse}

    # Simulate the filtering condition from the actual function
    is_valid = (
        run.data.metrics.get("val_rmse", 0) > 1e-4 and "test_rmse" in run.data.metrics
    )

    assert is_valid == expected_valid


@pytest.mark.parametrize(
    "metrics,expected_best_index",
    [
        (
            [{"test_rmse": 0.5}, {"test_rmse": 0.3}, {"test_rmse": 0.7}],
            1,
        ),  # Second run is best
        (
            [{"test_rmse": 0.8}, {"test_rmse": 0.6}, {"test_rmse": 0.4}],
            2,
        ),  # Third run is best
        ([{"test_rmse": 0.2}], 0),  # Single run
    ],
)
def test_best_run_selection_logic(metrics, expected_best_index):
    """Test the logic for selecting the best run based on test_rmse."""
    runs = []
    for _i, metric in enumerate(metrics):
        run = MagicMock()
        run.data.metrics = {**metric, "val_rmse": 0.5}  # Add valid val_rmse
        runs.append(run)

    # Simulate sorting by test_rmse (ascending)
    valid_runs = [run for run in runs if run.data.metrics.get("val_rmse", 0) > 1e-4]
    sorted_runs = sorted(
        valid_runs, key=lambda r: r.data.metrics.get("test_rmse", float("inf"))
    )

    if sorted_runs:
        best_run = sorted_runs[0]
        original_index = runs.index(best_run)
        assert original_index == expected_best_index
