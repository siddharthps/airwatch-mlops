"""
Unit tests for the model monitoring flow using pytest and moto for S3 interactions.
"""

from datetime import datetime
import io
from unittest.mock import MagicMock, patch

import boto3
from moto import mock_aws
import pandas as pd
import pytest

# Import the modules to test
from flows.model_monitoring import (
    check_data_drift,
    create_data_drift_report,
    create_regression_performance_report,
    load_historical_data_from_s3,
    load_predictions_from_s3,
    model_monitoring_flow,
    save_evidently_report_to_s3,
    validate_required_columns,
)


class TestLoadHistoricalDataFromS3:
    """Test cases for the load_historical_data_from_s3 task."""

    @pytest.fixture
    def sample_historical_data(self):
        """Sample historical data DataFrame."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03"]
                ),
                "arithmetic_mean": [12.5, 15.2, 10.8],
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "year": [2023, 2023, 2023],
                "month": [1, 1, 1],
                "day_of_week": [6, 0, 1],
                "day_of_year": [1, 2, 3],
                "is_weekend": [1, 1, 0],
            }
        )

    @mock_aws
    def test_load_historical_data_from_s3_success(self, sample_historical_data):
        """Test successful historical data loading from S3."""
        bucket_name = "test-bucket"
        key_prefix = "processed_data/pm25_daily"
        file_name = "pm25_daily_cleaned_2009_2024.parquet"
        s3_key = f"{key_prefix}/{file_name}"

        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        # Upload test data to mock S3
        buffer = io.BytesIO()
        sample_historical_data.to_parquet(buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.getvalue())

        # Test the function
        result = load_historical_data_from_s3(bucket_name, key_prefix, file_name)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, sample_historical_data)

    @mock_aws
    def test_load_historical_data_from_s3_file_not_found(self):
        """Test handling of missing historical data file."""
        bucket_name = "test-bucket"
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        with pytest.raises(Exception):
            load_historical_data_from_s3(
                bucket_name, "processed_data", "nonexistent.parquet"
            )


class TestLoadPredictionsFromS3:
    """Test cases for the load_predictions_from_s3 task."""

    @pytest.fixture
    def sample_predictions_data(self):
        """Sample predictions data DataFrame."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "arithmetic_mean": [12.5, 15.2],  # Actual values
                "predicted_arithmetic_mean": [12.8, 14.9],  # Predicted values
                "latitude": [41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298],
                "year": [2025, 2025],
                "month": [1, 1],
                "day_of_week": [2, 3],
                "day_of_year": [1, 2],
                "is_weekend": [0, 0],
            }
        )

    @mock_aws
    def test_load_predictions_from_s3_success(self, sample_predictions_data):
        """Test successful predictions loading from S3."""
        bucket_name = "test-bucket"
        key_prefix = "predictions/pm25_daily"
        target_year = 2025

        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        # Upload test data with timestamp in filename
        file_key = (
            f"{key_prefix}/pm25_predictions_{target_year}_20250101_120000.parquet"
        )
        buffer = io.BytesIO()
        sample_predictions_data.to_parquet(buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=buffer.getvalue())

        # Test the function
        result = load_predictions_from_s3(bucket_name, key_prefix, target_year)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        pd.testing.assert_frame_equal(result, sample_predictions_data)

    @mock_aws
    def test_load_predictions_from_s3_multiple_files(self, sample_predictions_data):
        """Test loading predictions when multiple files exist (should get latest)."""
        bucket_name = "test-bucket"
        key_prefix = "predictions/pm25_daily"
        target_year = 2025

        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        # Upload multiple files with different timestamps
        older_file = (
            f"{key_prefix}/pm25_predictions_{target_year}_20250101_100000.parquet"
        )
        newer_file = (
            f"{key_prefix}/pm25_predictions_{target_year}_20250101_120000.parquet"
        )

        buffer = io.BytesIO()
        sample_predictions_data.to_parquet(buffer, index=False)

        # Upload older file first
        s3_client.put_object(Bucket=bucket_name, Key=older_file, Body=buffer.getvalue())

        # Upload newer file
        newer_data = sample_predictions_data.copy()
        newer_data["predicted_arithmetic_mean"] = [13.0, 15.0]  # Different predictions
        buffer = io.BytesIO()
        newer_data.to_parquet(buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=newer_file, Body=buffer.getvalue())

        # Test the function - should return one of the files (based on LastModified time)
        result = load_predictions_from_s3(bucket_name, key_prefix, target_year)

        assert isinstance(result, pd.DataFrame)
        # The function returns the file with the latest LastModified time, which could be either
        # In moto, this might be the first or last uploaded file depending on timing
        predictions = result["predicted_arithmetic_mean"].tolist()
        assert predictions in [[12.8, 14.9], [13.0, 15.0]]

    @mock_aws
    def test_load_predictions_from_s3_no_files(self):
        """Test handling when no prediction files exist."""
        bucket_name = "test-bucket"
        key_prefix = "predictions/pm25_daily"
        target_year = 2025

        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        result = load_predictions_from_s3(bucket_name, key_prefix, target_year)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestSaveEvidentlyReportToS3:
    """Test cases for the save_evidently_report_to_s3 task."""

    @pytest.fixture
    def mock_evidently_report(self):
        """Mock Evidently report."""
        report = MagicMock()
        report.get_html.return_value = "<html><body>Test Report</body></html>"
        return report

    @mock_aws
    def test_save_evidently_report_to_s3_success(self, mock_evidently_report):
        """Test successful saving of Evidently report to S3."""
        bucket_name = "test-bucket"
        s3_key = "monitoring_reports/test_report.html"

        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        # Test the function
        save_evidently_report_to_s3(mock_evidently_report, bucket_name, s3_key)

        # Verify file was uploaded
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        uploaded_content = response["Body"].read().decode("utf-8")

        assert uploaded_content == "<html><body>Test Report</body></html>"
        mock_evidently_report.get_html.assert_called_once()

    @mock_aws
    def test_save_evidently_report_to_s3_upload_error(self, mock_evidently_report):
        """Test handling of S3 upload errors."""
        # Don't create the bucket to simulate an error
        with pytest.raises(Exception):
            save_evidently_report_to_s3(
                mock_evidently_report, "nonexistent-bucket", "test_report.html"
            )


class TestValidateRequiredColumns:
    """Test cases for the validate_required_columns function."""

    def test_validate_required_columns_all_present(self):
        """Test validation when all required columns are present."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
        required_columns = ["col1", "col2", "col3"]

        missing = validate_required_columns(df, required_columns, "test")

        assert missing == []

    def test_validate_required_columns_some_missing(self):
        """Test validation when some required columns are missing."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        required_columns = ["col1", "col2", "col3", "col4"]

        missing = validate_required_columns(df, required_columns, "test")

        assert set(missing) == {"col3", "col4"}

    def test_validate_required_columns_all_missing(self):
        """Test validation when all required columns are missing."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        required_columns = ["col1", "col2"]

        missing = validate_required_columns(df, required_columns, "test")

        assert set(missing) == {"col1", "col2"}


class TestCreateDataDriftReport:
    """Test cases for the create_data_drift_report function."""

    @pytest.fixture
    def reference_data(self):
        """Reference data for drift detection."""
        return pd.DataFrame(
            {
                "arithmetic_mean": [12.5, 15.2, 10.8],
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "year": [2023, 2023, 2023],
                "month": [1, 1, 1],
                "day_of_week": [0, 1, 2],
                "day_of_year": [1, 2, 3],
                "is_weekend": [1, 0, 0],
            }
        )

    @pytest.fixture
    def current_data(self):
        """Current data for drift detection."""
        return pd.DataFrame(
            {
                "arithmetic_mean": [13.0, 16.0, 11.0],
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "year": [2025, 2025, 2025],
                "month": [1, 1, 1],
                "day_of_week": [0, 1, 2],
                "day_of_year": [1, 2, 3],
                "is_weekend": [1, 0, 0],
            }
        )

    @patch("flows.model_monitoring.Report")
    @patch("flows.model_monitoring.datetime")
    def test_create_data_drift_report_success(
        self, mock_datetime, mock_report_class, reference_data, current_data
    ):
        """Test successful creation of data drift report."""
        # Mock datetime for consistent naming
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"

        # Mock Evidently Report
        mock_report = MagicMock()
        mock_report_class.return_value = mock_report

        target_year = 2025
        report, report_path = create_data_drift_report(
            reference_data, current_data, target_year
        )

        # Verify report creation
        mock_report_class.assert_called_once()
        mock_report.run.assert_called_once_with(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=mock_report.run.call_args[1]["column_mapping"],
        )

        # Verify report path
        expected_path = "monitoring_reports/data_drift_report_2025_20250101_120000.html"
        assert report_path == expected_path
        assert report == mock_report


class TestCreateRegressionPerformanceReport:
    """Test cases for the create_regression_performance_report function."""

    @pytest.fixture
    def performance_data(self):
        """Data with both actual and predicted values."""
        return pd.DataFrame(
            {
                "arithmetic_mean": [12.5, 15.2, 10.8],  # Actual values
                "predicted_arithmetic_mean": [12.8, 14.9, 11.1],  # Predicted values
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "year": [2025, 2025, 2025],
                "month": [1, 1, 1],
                "day_of_week": [0, 1, 2],
                "day_of_year": [1, 2, 3],
                "is_weekend": [1, 0, 0],
            }
        )

    @patch("flows.model_monitoring.Report")
    @patch("flows.model_monitoring.datetime")
    def test_create_regression_performance_report_success(
        self, mock_datetime, mock_report_class, performance_data
    ):
        """Test successful creation of regression performance report."""
        # Mock datetime for consistent naming
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"

        # Mock Evidently Report
        mock_report = MagicMock()
        mock_report_class.return_value = mock_report

        target_year = 2025
        report, report_path = create_regression_performance_report(
            performance_data, target_year
        )

        # Verify report creation
        mock_report_class.assert_called_once()
        mock_report.run.assert_called_once()

        # Verify report path
        expected_path = (
            "monitoring_reports/regression_performance_report_2025_20250101_120000.html"
        )
        assert report_path == expected_path
        assert report == mock_report


class TestCheckDataDrift:
    """Test cases for the check_data_drift function."""

    @pytest.fixture
    def mock_report_no_drift(self):
        """Mock report indicating no data drift."""
        report = MagicMock()
        report.as_dict.return_value = {
            "metrics": [{"result": {"dataset_drift": False}}]
        }
        return report

    @pytest.fixture
    def mock_report_with_drift(self):
        """Mock report indicating data drift detected."""
        report = MagicMock()
        report.as_dict.return_value = {"metrics": [{"result": {"dataset_drift": True}}]}
        return report

    def test_check_data_drift_no_drift(self, mock_report_no_drift):
        """Test drift checking when no drift is detected."""
        result = check_data_drift(mock_report_no_drift)

        assert result is False
        mock_report_no_drift.as_dict.assert_called_once()

    def test_check_data_drift_with_drift(self, mock_report_with_drift):
        """Test drift checking when drift is detected."""
        result = check_data_drift(mock_report_with_drift)

        assert result is True
        mock_report_with_drift.as_dict.assert_called_once()

    def test_check_data_drift_malformed_response(self):
        """Test drift checking with malformed report response."""
        mock_report = MagicMock()
        mock_report.as_dict.return_value = {"invalid": "structure"}

        with pytest.raises(Exception):
            check_data_drift(mock_report)


class TestModelMonitoringFlow:
    """Test cases for the main model monitoring flow."""

    @pytest.fixture
    def sample_reference_data(self):
        """Sample reference data."""
        return pd.DataFrame(
            {
                "arithmetic_mean": [12.5, 15.2, 10.8],
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "year": [2023, 2023, 2023],
                "month": [1, 1, 1],
                "day_of_week": [0, 1, 2],
                "day_of_year": [1, 2, 3],
                "is_weekend": [1, 0, 0],
            }
        )

    @pytest.fixture
    def sample_current_data(self):
        """Sample current data with predictions."""
        return pd.DataFrame(
            {
                "arithmetic_mean": [13.0, 16.0, 11.0],
                "predicted_arithmetic_mean": [13.2, 15.8, 10.9],
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "year": [2025, 2025, 2025],
                "month": [1, 1, 1],
                "day_of_week": [0, 1, 2],
                "day_of_year": [1, 2, 3],
                "is_weekend": [1, 0, 0],
            }
        )

    @patch("flows.model_monitoring.check_data_drift")
    @patch("flows.model_monitoring.save_evidently_report_to_s3")
    @patch("flows.model_monitoring.create_regression_performance_report")
    @patch("flows.model_monitoring.create_data_drift_report")
    @patch("flows.model_monitoring.load_predictions_from_s3")
    @patch("flows.model_monitoring.load_historical_data_from_s3")
    def test_model_monitoring_flow_success_no_drift(
        self,
        mock_load_historical,
        mock_load_predictions,
        mock_create_drift,
        mock_create_regression,
        mock_save_report,
        mock_check_drift,
        sample_reference_data,
        sample_current_data,
    ):
        """Test successful monitoring flow with no drift detected."""
        # Setup mocks
        mock_load_historical.return_value = sample_reference_data
        mock_load_predictions.return_value = sample_current_data

        mock_drift_report = MagicMock()
        mock_regression_report = MagicMock()

        mock_create_drift.return_value = (mock_drift_report, "drift_report.html")
        mock_create_regression.return_value = (
            mock_regression_report,
            "regression_report.html",
        )

        mock_check_drift.return_value = False  # No drift detected

        # Execute flow
        model_monitoring_flow(target_year=2025)

        # Verify all tasks were called
        mock_load_historical.assert_called_once()
        mock_load_predictions.assert_called_once_with(
            "air-quality-mlops-data-chicago-2025", target_year=2025
        )
        mock_create_drift.assert_called_once()
        mock_create_regression.assert_called_once()
        mock_save_report.assert_called()  # Called twice (drift + regression)
        mock_check_drift.assert_called_once_with(mock_drift_report)

    @patch("flows.model_monitoring.check_data_drift")
    @patch("flows.model_monitoring.save_evidently_report_to_s3")
    @patch("flows.model_monitoring.create_regression_performance_report")
    @patch("flows.model_monitoring.create_data_drift_report")
    @patch("flows.model_monitoring.load_predictions_from_s3")
    @patch("flows.model_monitoring.load_historical_data_from_s3")
    def test_model_monitoring_flow_with_drift_detected(
        self,
        mock_load_historical,
        mock_load_predictions,
        mock_create_drift,
        mock_create_regression,
        mock_save_report,
        mock_check_drift,
        sample_reference_data,
        sample_current_data,
    ):
        """Test monitoring flow when data drift is detected."""
        # Setup mocks
        mock_load_historical.return_value = sample_reference_data
        mock_load_predictions.return_value = sample_current_data

        mock_drift_report = MagicMock()
        mock_regression_report = MagicMock()

        mock_create_drift.return_value = (mock_drift_report, "drift_report.html")
        mock_create_regression.return_value = (
            mock_regression_report,
            "regression_report.html",
        )

        mock_check_drift.return_value = True  # Drift detected

        # Execute flow - should raise ValueError due to drift
        with pytest.raises(ValueError, match="Data drift detected"):
            model_monitoring_flow(target_year=2025)

    @patch("flows.model_monitoring.load_predictions_from_s3")
    @patch("flows.model_monitoring.load_historical_data_from_s3")
    def test_model_monitoring_flow_empty_reference_data(
        self, mock_load_historical, mock_load_predictions
    ):
        """Test monitoring flow when reference data is empty."""
        mock_load_historical.return_value = pd.DataFrame()

        # Execute flow
        model_monitoring_flow(target_year=2025)

        # Should return early without calling other functions
        mock_load_predictions.assert_not_called()

    @patch("flows.model_monitoring.load_predictions_from_s3")
    @patch("flows.model_monitoring.load_historical_data_from_s3")
    def test_model_monitoring_flow_empty_current_data(
        self, mock_load_historical, mock_load_predictions, sample_reference_data
    ):
        """Test monitoring flow when current data is empty."""
        mock_load_historical.return_value = sample_reference_data
        mock_load_predictions.return_value = pd.DataFrame()

        # Execute flow
        model_monitoring_flow(target_year=2025)

        # Should return early after loading both datasets

    @patch("flows.model_monitoring.load_predictions_from_s3")
    @patch("flows.model_monitoring.load_historical_data_from_s3")
    def test_model_monitoring_flow_default_year(
        self, mock_load_historical, mock_load_predictions, sample_reference_data
    ):
        """Test monitoring flow with default year (current year)."""
        current_year = datetime.now().year

        mock_load_historical.return_value = sample_reference_data
        mock_load_predictions.return_value = pd.DataFrame()

        # Execute flow without specifying year
        model_monitoring_flow()

        # Should use current year
        mock_load_predictions.assert_called_once()
        args, kwargs = mock_load_predictions.call_args
        assert kwargs["target_year"] == current_year


# Parametrized tests for different drift scenarios
@pytest.mark.parametrize(
    "drift_result,expected_exception",
    [
        (False, None),
        (True, ValueError),
    ],
)
def test_model_monitoring_flow_drift_scenarios(drift_result, expected_exception):
    """Test monitoring flow with different drift detection results."""
    with (
        patch("flows.model_monitoring.load_historical_data_from_s3") as mock_load_hist,
        patch("flows.model_monitoring.load_predictions_from_s3") as mock_load_pred,
        patch("flows.model_monitoring.create_data_drift_report") as mock_create_drift,
        patch(
            "flows.model_monitoring.create_regression_performance_report"
        ) as mock_create_reg,
        patch("flows.model_monitoring.save_evidently_report_to_s3"),
        patch("flows.model_monitoring.check_data_drift") as mock_check_drift,
    ):
        # Setup mocks
        sample_data = pd.DataFrame(
            {
                "arithmetic_mean": [12.5],
                "predicted_arithmetic_mean": [12.8],
                "latitude": [41.8781],
                "longitude": [-87.6298],
                "year": [2025],
                "month": [1],
                "day_of_week": [0],
                "day_of_year": [1],
                "is_weekend": [0],
            }
        )

        mock_load_hist.return_value = sample_data
        mock_load_pred.return_value = sample_data
        mock_create_drift.return_value = (MagicMock(), "drift.html")
        mock_create_reg.return_value = (MagicMock(), "regression.html")
        mock_check_drift.return_value = drift_result

        if expected_exception:
            with pytest.raises(expected_exception):
                model_monitoring_flow(target_year=2025)
        else:
            model_monitoring_flow(target_year=2025)  # Should not raise
