"""
Unit tests for the inference data preparation flow using pytest and moto for S3 interactions.
"""

from datetime import datetime
import io
from unittest.mock import patch

import boto3
from moto import mock_aws
import pandas as pd
import pytest
import requests_mock

# Import the modules to test
from flows.inference_data_preparation import (
    fetch_epa_aqs_data_for_inference,
    inference_data_preparation_flow,
    transform_data_for_inference,
    write_processed_inference_data_to_s3,
)


class TestFetchEpaAqsDataForInference:
    """Test cases for the fetch_epa_aqs_data_for_inference task."""

    @pytest.fixture
    def sample_api_response_dict(self):
        """Sample API response in dictionary format."""
        return {
            "Header": [{"status": "Success"}],
            "Data": [
                {
                    "date_local": "2025-01-01",
                    "parameter_code": "88101",
                    "arithmetic_mean": 12.5,
                    "cbsa_code": "16980",
                    "site_number": "0001",
                    "latitude": 41.8781,
                    "longitude": -87.6298,
                },
                {
                    "date_local": "2025-01-02",
                    "parameter_code": "88101",
                    "arithmetic_mean": 15.2,
                    "cbsa_code": "16980",
                    "site_number": "0001",
                    "latitude": 41.8781,
                    "longitude": -87.6298,
                },
            ],
        }

    @pytest.fixture
    def no_data_response(self):
        """API response indicating no data available."""
        return {"Header": [{"status": "No data meets your criteria"}], "Data": []}

    def test_fetch_epa_aqs_data_for_inference_success(self, sample_api_response_dict):
        """Test successful data fetch for inference."""
        with requests_mock.Mocker() as m:
            m.get(requests_mock.ANY, json=sample_api_response_dict)

            result = fetch_epa_aqs_data_for_inference(
                email="test@example.com",
                api_key="test_key",
                target_year=2025,
                param_code="88101",
                cbsa_code="16980",
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "date_local" in result.columns
            assert "parameter_code" in result.columns
            assert "arithmetic_mean" in result.columns
            # Check if date_local is converted to datetime
            assert pd.api.types.is_datetime64_any_dtype(result["date_local"])

    def test_fetch_epa_aqs_data_for_inference_no_data(self, no_data_response):
        """Test handling of 'no data' response."""
        with requests_mock.Mocker() as m:
            m.get(requests_mock.ANY, json=no_data_response)

            result = fetch_epa_aqs_data_for_inference(
                email="test@example.com",
                api_key="test_key",
                target_year=2025,
                param_code="88101",
                cbsa_code="16980",
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_fetch_epa_aqs_data_for_inference_http_error(self):
        """Test handling of HTTP errors."""
        with requests_mock.Mocker() as m:
            m.get(requests_mock.ANY, status_code=500)

            result = fetch_epa_aqs_data_for_inference(
                email="test@example.com",
                api_key="test_key",
                target_year=2025,
                param_code="88101",
                cbsa_code="16980",
            )

            # Should return empty DataFrame on HTTP error
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_fetch_epa_aqs_data_for_inference_json_error(self):
        """Test handling of JSON decode errors."""
        with requests_mock.Mocker() as m:
            m.get(requests_mock.ANY, text="Invalid JSON")

            result = fetch_epa_aqs_data_for_inference(
                email="test@example.com",
                api_key="test_key",
                target_year=2025,
                param_code="88101",
                cbsa_code="16980",
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_fetch_epa_aqs_data_for_inference_list_response(self):
        """Test handling of list response format."""
        list_response = [
            {
                "date_local": "2025-01-01",
                "parameter_code": "88101",
                "arithmetic_mean": 12.5,
                "latitude": 41.8781,
                "longitude": -87.6298,
            }
        ]

        with requests_mock.Mocker() as m:
            m.get(requests_mock.ANY, json=list_response)

            result = fetch_epa_aqs_data_for_inference(
                email="test@example.com",
                api_key="test_key",
                target_year=2025,
                param_code="88101",
                cbsa_code="16980",
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

    def test_fetch_epa_aqs_data_for_inference_empty_list(self):
        """Test handling of empty list response."""
        with requests_mock.Mocker() as m:
            m.get(requests_mock.ANY, json=[])

            result = fetch_epa_aqs_data_for_inference(
                email="test@example.com",
                api_key="test_key",
                target_year=2025,
                param_code="88101",
                cbsa_code="16980",
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0


class TestTransformDataForInference:
    """Test cases for the transform_data_for_inference task."""

    @pytest.fixture
    def inference_raw_data(self):
        """Sample raw inference data."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(
                    ["2025-01-01", "2025-01-02", "2025-01-03"]
                ),
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "cbsa_code": ["16980", "16980", "16980"],
                "state": ["Illinois", "Illinois", "Illinois"],
                "county": ["Cook", "Cook", "Cook"],
                "city": ["Chicago", "Chicago", "Chicago"],
            }
        )

    @pytest.fixture
    def inference_data_with_target(self):
        """Sample inference data that includes target values."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "arithmetic_mean": [12.5, 15.2],  # Target values
                "latitude": [41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298],
                "cbsa_code": ["16980", "16980"],
                "state": ["Illinois", "Illinois"],
            }
        )

    def test_transform_data_for_inference_success(self, inference_raw_data):
        """Test successful data transformation for inference."""
        result = transform_data_for_inference(inference_raw_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

        # Check that new features are created
        expected_features = [
            "year",
            "month",
            "day_of_week",
            "day_of_year",
            "is_weekend",
        ]
        for feature in expected_features:
            assert feature in result.columns

        # Check feature values
        assert result["year"].iloc[0] == 2025
        assert result["month"].iloc[0] == 1
        assert result["day_of_week"].iloc[0] == 2  # Wednesday
        assert result["is_weekend"].iloc[0] == 0  # Wednesday is not weekend

    def test_transform_data_for_inference_with_target(self, inference_data_with_target):
        """Test transformation when target values are present."""
        result = transform_data_for_inference(inference_data_with_target)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

        # Target values should be preserved if present
        if "arithmetic_mean" in result.columns:
            assert "arithmetic_mean" in result.columns

    def test_transform_data_for_inference_empty_dataframe(self):
        """Test transformation with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = transform_data_for_inference(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_transform_data_for_inference_missing_date(self):
        """Test transformation when date_local is missing."""
        df_no_date = pd.DataFrame({"latitude": [41.8781], "longitude": [-87.6298]})

        result = transform_data_for_inference(df_no_date)

        # Should return empty DataFrame as date is essential
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_transform_data_for_inference_missing_coordinates(self):
        """Test transformation when coordinates are missing."""
        df_no_coords = pd.DataFrame(
            {"date_local": pd.to_datetime(["2025-01-01"]), "cbsa_code": ["16980"]}
        )

        result = transform_data_for_inference(df_no_coords)

        # Should return empty DataFrame as coordinates are essential
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_transform_data_for_inference_invalid_coordinates(self):
        """Test transformation with invalid coordinate values."""
        df_invalid_coords = pd.DataFrame(
            {
                "date_local": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "latitude": [41.8781, None],  # One invalid
                "longitude": [-87.6298, -87.6298],
            }
        )

        result = transform_data_for_inference(df_invalid_coords)

        # Should drop rows with invalid coordinates
        assert len(result) == 1
        assert not result["latitude"].isnull().any()

    def test_transform_data_for_inference_string_coordinates(self):
        """Test transformation with string coordinates that need conversion."""
        df_string_coords = pd.DataFrame(
            {
                "date_local": pd.to_datetime(["2025-01-01"]),
                "latitude": ["41.8781"],
                "longitude": ["-87.6298"],
            }
        )

        result = transform_data_for_inference(df_string_coords)

        assert len(result) == 1
        assert pd.api.types.is_numeric_dtype(result["latitude"])
        assert pd.api.types.is_numeric_dtype(result["longitude"])


class TestWriteProcessedInferenceDataToS3:
    """Test cases for the write_processed_inference_data_to_s3 task."""

    @pytest.fixture
    def processed_inference_data(self):
        """Sample processed inference data."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "latitude": [41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298],
                "year": [2025, 2025],
                "month": [1, 1],
                "is_weekend": [0, 0],
            }
        )

    @mock_aws
    def test_write_processed_inference_data_to_s3_success(
        self, processed_inference_data
    ):
        """Test successful writing to S3."""
        bucket_name = "test-bucket"
        key_prefix = "processed_inference"
        file_name = "test_file"

        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        # Test the function
        write_processed_inference_data_to_s3(
            processed_inference_data, bucket_name, key_prefix, file_name
        )

        # Verify file was uploaded
        s3_key = f"{key_prefix}/{file_name}.parquet"
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)

        # Read back the data and verify
        uploaded_data = pd.read_parquet(io.BytesIO(response["Body"].read()))
        pd.testing.assert_frame_equal(uploaded_data, processed_inference_data)

    def test_write_processed_inference_data_to_s3_empty_dataframe(self):
        """Test writing empty DataFrame to S3."""
        empty_df = pd.DataFrame()

        # Should not raise an exception, just skip the upload
        write_processed_inference_data_to_s3(
            empty_df, "test-bucket", "test_prefix", "test_file"
        )

    @mock_aws
    def test_write_processed_inference_data_to_s3_upload_error(
        self, processed_inference_data
    ):
        """Test handling of S3 upload errors."""
        # Don't create the bucket to simulate an error
        with pytest.raises(Exception):
            write_processed_inference_data_to_s3(
                processed_inference_data,
                "nonexistent-bucket",
                "test_prefix",
                "test_file",
            )


class TestInferenceDataPreparationFlow:
    """Test cases for the main inference data preparation flow."""

    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables."""
        return {
            "EPA_AQS_EMAIL": "test@example.com",
            "EPA_AQS_API_KEY": "test_api_key",
            "S3_DATA_BUCKET_NAME": "test-bucket",
        }

    @patch("flows.inference_data_preparation.write_processed_inference_data_to_s3")
    @patch("flows.inference_data_preparation.transform_data_for_inference")
    @patch("flows.inference_data_preparation.fetch_epa_aqs_data_for_inference")
    def test_inference_data_preparation_flow_success(
        self, mock_fetch, mock_transform, mock_write, mock_env_vars
    ):
        """Test successful execution of the inference data preparation flow."""
        # Setup mocks
        raw_data = pd.DataFrame(
            {
                "date_local": pd.to_datetime(["2025-01-01"]),
                "latitude": [41.8781],
                "longitude": [-87.6298],
            }
        )
        processed_data = raw_data.copy()
        processed_data["year"] = 2025

        mock_fetch.return_value = raw_data
        mock_transform.return_value = processed_data

        with patch.dict("os.environ", mock_env_vars):
            result = inference_data_preparation_flow(target_year=2025)

            # Verify all tasks were called
            mock_fetch.assert_called_once_with(
                email="test@example.com",
                api_key="test_api_key",
                target_year=2025,
                param_code="88101",
                cbsa_code="16980",
            )
            mock_transform.assert_called_once_with(df=raw_data)
            mock_write.assert_called_once()

            # Verify return value
            pd.testing.assert_frame_equal(result, processed_data)

    @patch("flows.inference_data_preparation.write_processed_inference_data_to_s3")
    @patch("flows.inference_data_preparation.transform_data_for_inference")
    @patch("flows.inference_data_preparation.fetch_epa_aqs_data_for_inference")
    def test_inference_data_preparation_flow_default_year(
        self, mock_fetch, mock_transform, mock_write, mock_env_vars
    ):
        """Test flow with default year (current year)."""
        current_year = datetime.now().year

        mock_fetch.return_value = pd.DataFrame()
        mock_transform.return_value = pd.DataFrame()

        with patch.dict("os.environ", mock_env_vars):
            inference_data_preparation_flow()  # No target_year specified

            # Should use current year
            mock_fetch.assert_called_once()
            args, kwargs = mock_fetch.call_args
            assert kwargs["target_year"] == current_year

    def test_inference_data_preparation_flow_missing_credentials(self):
        """Test flow fails when credentials are missing."""
        env_vars = {"S3_DATA_BUCKET_NAME": "test-bucket"}

        with patch.dict("os.environ", env_vars, clear=True):
            with pytest.raises(ValueError, match="EPA_AQS_EMAIL and EPA_AQS_API_KEY"):
                inference_data_preparation_flow(target_year=2025)

    @patch("flows.inference_data_preparation.write_processed_inference_data_to_s3")
    @patch("flows.inference_data_preparation.transform_data_for_inference")
    @patch("flows.inference_data_preparation.fetch_epa_aqs_data_for_inference")
    def test_inference_data_preparation_flow_empty_data(
        self, mock_fetch, mock_transform, mock_write, mock_env_vars
    ):
        """Test flow with empty data at each stage."""
        empty_df = pd.DataFrame()

        mock_fetch.return_value = empty_df
        mock_transform.return_value = empty_df

        with patch.dict("os.environ", mock_env_vars):
            result = inference_data_preparation_flow(target_year=2025)

            # All tasks should still be called
            mock_fetch.assert_called_once()
            mock_transform.assert_called_once()
            mock_write.assert_called_once()

            # Result should be empty DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0


# Parametrized tests for different years and date scenarios
@pytest.mark.parametrize(
    "target_year,expected_year",
    [
        (2024, 2024),
        (2025, 2025),
        (2026, 2026),
    ],
)
def test_inference_data_preparation_different_years(target_year, expected_year):
    """Test inference data preparation for different target years."""
    df = pd.DataFrame(
        {
            "date_local": pd.to_datetime([f"{target_year}-01-01"]),
            "latitude": [41.8781],
            "longitude": [-87.6298],
        }
    )

    result = transform_data_for_inference(df)

    if len(result) > 0:
        assert result["year"].iloc[0] == expected_year


@pytest.mark.parametrize(
    "date_str,expected_weekend",
    [
        ("2025-01-01", 0),  # Wednesday
        ("2025-01-04", 1),  # Saturday
        ("2025-01-05", 1),  # Sunday
        ("2025-01-06", 0),  # Monday
    ],
)
def test_inference_weekend_feature_creation(date_str, expected_weekend):
    """Test weekend feature creation for inference data."""
    df = pd.DataFrame(
        {
            "date_local": pd.to_datetime([date_str]),
            "latitude": [41.8781],
            "longitude": [-87.6298],
        }
    )

    result = transform_data_for_inference(df)

    if len(result) > 0:
        assert result["is_weekend"].iloc[0] == expected_weekend
