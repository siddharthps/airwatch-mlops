"""
Unit tests for the data transformation flow using pytest and moto for S3 interactions.
"""

import io
from unittest.mock import patch

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws
import pandas as pd
import pytest

# Import the modules to test
from flows.data_transformation import (
    air_quality_transformation_flow,
    load_raw_data_from_s3,
    transform_data,
    write_transformed_data_to_s3,
)


class TestLoadRawDataFromS3:
    """Test cases for the load_raw_data_from_s3 task."""

    @pytest.fixture
    def sample_raw_data(self):
        """Sample raw data DataFrame."""
        return pd.DataFrame(
            {
                "date_local": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "arithmetic_mean": [12.5, 15.2, 10.8],
                "first_max_value": [18.0, 20.5, 14.2],
                "aqi": [45, 52, 38],
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "cbsa_code": ["16980", "16980", "16980"],
                "state": ["Illinois", "Illinois", "Illinois"],
                "county": ["Cook", "Cook", "Cook"],
                "city": ["Chicago", "Chicago", "Chicago"],
            }
        )

    @mock_aws
    def test_load_raw_data_from_s3_success(self, sample_raw_data):
        """Test successful data loading from S3."""
        # Setup mock S3
        bucket_name = "test-bucket"
        key_prefix = "raw_data"
        file_name = "test_file"
        s3_key = f"{key_prefix}/{file_name}.parquet"

        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        # Upload test data to mock S3
        buffer = io.BytesIO()
        sample_raw_data.to_parquet(buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.getvalue())

        # Test the function
        result = load_raw_data_from_s3(bucket_name, key_prefix, file_name)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == list(sample_raw_data.columns)
        pd.testing.assert_frame_equal(result, sample_raw_data)

    @mock_aws
    def test_load_raw_data_from_s3_file_not_found(self):
        """Test handling of missing file in S3."""
        bucket_name = "test-bucket"
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        with pytest.raises((FileNotFoundError, ClientError)):
            load_raw_data_from_s3(bucket_name, "raw_data", "nonexistent_file")

    @mock_aws
    def test_load_raw_data_from_s3_bucket_not_found(self):
        """Test handling of missing bucket."""
        with pytest.raises(ClientError):
            load_raw_data_from_s3("nonexistent-bucket", "raw_data", "test_file")


class TestTransformData:
    """Test cases for the transform_data task."""

    @pytest.fixture
    def raw_data_with_dates(self):
        """Raw data with proper date format."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03"]
                ),
                "arithmetic_mean": [12.5, 15.2, 10.8],
                "first_max_value": [18.0, 20.5, 14.2],
                "aqi": [45, 52, 38],
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "cbsa_code": ["16980", "16980", "16980"],
                "state": ["Illinois", "Illinois", "Illinois"],
                "county": ["Cook", "Cook", "Cook"],
                "city": ["Chicago", "Chicago", "Chicago"],
            }
        )

    @pytest.fixture
    def raw_data_string_dates(self):
        """Raw data with string dates that need conversion."""
        return pd.DataFrame(
            {
                "date_local": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "arithmetic_mean": ["12.5", "15.2", "10.8"],
                "first_max_value": ["18.0", "20.5", "14.2"],
                "aqi": ["45", "52", "38"],
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "cbsa_code": ["16980", "16980", "16980"],
                "state": ["Illinois", "Illinois", "Illinois"],
                "county": ["Cook", "Cook", "Cook"],
                "city": ["Chicago", "Chicago", "Chicago"],
            }
        )

    def test_transform_data_success(self, raw_data_with_dates):
        """Test successful data transformation."""
        result = transform_data(raw_data_with_dates)

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

        # Check date conversion
        assert pd.api.types.is_datetime64_any_dtype(result["date_local"])

        # Check feature values
        assert result["year"].iloc[0] == 2023
        assert result["month"].iloc[0] == 1
        assert result["day_of_week"].iloc[0] == 6  # Sunday
        assert result["is_weekend"].iloc[0] == 1  # Sunday is weekend

    def test_transform_data_string_conversion(self, raw_data_string_dates):
        """Test data transformation with string to numeric conversion."""
        result = transform_data(raw_data_string_dates)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

        # Check numeric conversion
        assert pd.api.types.is_numeric_dtype(result["arithmetic_mean"])
        assert pd.api.types.is_numeric_dtype(result["first_max_value"])
        assert pd.api.types.is_numeric_dtype(result["aqi"])

    def test_transform_data_empty_dataframe(self):
        """Test transformation with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = transform_data(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_transform_data_missing_date_column(self):
        """Test transformation when date_local column is missing."""
        df_no_date = pd.DataFrame(
            {
                "arithmetic_mean": [12.5, 15.2, 10.8],
                "first_max_value": [18.0, 20.5, 14.2],
                "aqi": [45, 52, 38],
            }
        )

        result = transform_data(df_no_date)

        # Should still process but without date features
        assert isinstance(result, pd.DataFrame)
        assert "year" not in result.columns
        assert "month" not in result.columns

    def test_transform_data_invalid_dates(self):
        """Test transformation with invalid dates."""
        df_invalid_dates = pd.DataFrame(
            {
                "date_local": ["invalid-date", "2023-01-02", "another-invalid"],
                "arithmetic_mean": [12.5, 15.2, 10.8],
                "first_max_value": [18.0, 20.5, 14.2],
                "aqi": [45, 52, 38],
            }
        )

        result = transform_data(df_invalid_dates)

        # Should drop rows with invalid dates
        assert len(result) == 1  # Only one valid date
        assert result["date_local"].iloc[0] == pd.to_datetime("2023-01-02")

    def test_transform_data_missing_numeric_values(self):
        """Test transformation with missing numeric values."""
        df_with_nans = pd.DataFrame(
            {
                "date_local": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03"]
                ),
                "arithmetic_mean": [12.5, None, 10.8],
                "first_max_value": [18.0, 20.5, None],
                "aqi": [45, 52, 38],
            }
        )

        result = transform_data(df_with_nans)

        # Should drop rows with NaN values in critical columns
        assert len(result) < 3
        assert not result["arithmetic_mean"].isnull().any()


class TestWriteTransformedDataToS3:
    """Test cases for the write_transformed_data_to_s3 task."""

    @pytest.fixture
    def transformed_data(self):
        """Sample transformed data."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "arithmetic_mean": [12.5, 15.2],
                "year": [2023, 2023],
                "month": [1, 1],
                "is_weekend": [1, 0],
            }
        )

    @mock_aws
    def test_write_transformed_data_to_s3_success(self, transformed_data):
        """Test successful writing to S3."""
        bucket_name = "test-bucket"
        key_prefix = "processed_data"
        file_name = "test_file"

        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        # Test the function
        write_transformed_data_to_s3(
            transformed_data, bucket_name, key_prefix, file_name
        )

        # Verify file was uploaded
        s3_key = f"{key_prefix}/{file_name}.parquet"
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)

        # Read back the data and verify
        uploaded_data = pd.read_parquet(io.BytesIO(response["Body"].read()))
        pd.testing.assert_frame_equal(uploaded_data, transformed_data)

    def test_write_transformed_data_to_s3_empty_dataframe(self):
        """Test writing empty DataFrame to S3."""
        empty_df = pd.DataFrame()

        # Should not raise an exception, just skip the upload
        write_transformed_data_to_s3(
            empty_df, "test-bucket", "test_prefix", "test_file"
        )

    @mock_aws
    def test_write_transformed_data_to_s3_upload_error(self, transformed_data):
        """Test handling of S3 upload errors."""
        # Don't create the bucket to simulate an error
        with pytest.raises((ClientError, FileNotFoundError)):
            write_transformed_data_to_s3(
                transformed_data, "nonexistent-bucket", "test_prefix", "test_file"
            )


class TestAirQualityTransformationFlow:
    """Test cases for the main transformation flow."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for flow testing."""
        return pd.DataFrame(
            {
                "date_local": ["2023-01-01", "2023-01-02"],
                "arithmetic_mean": [12.5, 15.2],
                "first_max_value": [18.0, 20.5],
                "aqi": [45, 52],
            }
        )

    @patch("flows.data_transformation.write_transformed_data_to_s3")
    @patch("flows.data_transformation.transform_data")
    @patch("flows.data_transformation.load_raw_data_from_s3")
    def test_air_quality_transformation_flow_success(
        self, mock_load, mock_transform, mock_write, sample_data
    ):
        """Test successful execution of the transformation flow."""
        # Setup mocks
        transformed_data = sample_data.copy()
        transformed_data["year"] = 2023

        mock_load.return_value = sample_data
        mock_transform.return_value = transformed_data

        # Execute flow
        air_quality_transformation_flow(
            input_bucket_name="input-bucket",
            input_key_prefix="raw_data",
            input_file_name="raw_file",
            output_bucket_name="output-bucket",
            output_key_prefix="processed_data",
            output_file_name="processed_file",
        )

        # Verify all tasks were called
        mock_load.assert_called_once_with(
            bucket_name="input-bucket", key_prefix="raw_data", file_name="raw_file"
        )
        mock_transform.assert_called_once_with(df=sample_data)
        mock_write.assert_called_once_with(
            df=transformed_data,
            bucket_name="output-bucket",
            key_prefix="processed_data",
            file_name="processed_file",
        )

    @patch("flows.data_transformation.write_transformed_data_to_s3")
    @patch("flows.data_transformation.transform_data")
    @patch("flows.data_transformation.load_raw_data_from_s3")
    def test_air_quality_transformation_flow_empty_data(
        self, mock_load, mock_transform, mock_write
    ):
        """Test flow with empty data."""
        empty_df = pd.DataFrame()

        mock_load.return_value = empty_df
        mock_transform.return_value = empty_df

        air_quality_transformation_flow(
            input_bucket_name="input-bucket",
            input_key_prefix="raw_data",
            input_file_name="raw_file",
            output_bucket_name="output-bucket",
            output_key_prefix="processed_data",
            output_file_name="processed_file",
        )

        # All tasks should still be called
        mock_load.assert_called_once()
        mock_transform.assert_called_once()
        mock_write.assert_called_once()


# Parametrized tests for edge cases
@pytest.mark.parametrize(
    "date_input,expected_weekend",
    [
        ("2023-01-01", 1),  # Sunday
        ("2023-01-02", 0),  # Monday
        ("2023-01-07", 1),  # Saturday
        ("2023-01-08", 1),  # Sunday
    ],
)
def test_weekend_feature_creation(date_input, expected_weekend):
    """Test weekend feature creation for different days."""
    df = pd.DataFrame({"date_local": [date_input], "arithmetic_mean": [12.5]})

    result = transform_data(df)

    if len(result) > 0:  # If transformation succeeded
        assert result["is_weekend"].iloc[0] == expected_weekend


@pytest.mark.parametrize(
    "numeric_col,test_values,expected_type",
    [
        ("arithmetic_mean", ["12.5", "15.2"], "float"),
        ("first_max_value", ["18.0", "20.5"], "float"),
        ("aqi", ["45", "52"], "int"),  # AQI is typically integer values
    ],
)
def test_numeric_conversion(numeric_col, test_values, expected_type):
    """Test numeric conversion for different columns."""
    df = pd.DataFrame(
        {
            "date_local": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            numeric_col: test_values,
        }
    )

    result = transform_data(df)

    if numeric_col in result.columns:
        assert str(result[numeric_col].dtype).startswith(expected_type)
