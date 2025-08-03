"""
Unit tests for the model training flow using pytest and moto for S3 interactions.
"""

import io
from unittest.mock import MagicMock, patch

import boto3
from moto import mock_aws
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Import the modules to test
from flows.model_training import (
    load_processed_data_from_s3,
    split_data,
    train_and_log_model,
    train_models,
)


class TestLoadProcessedDataFromS3:
    """Test cases for the load_processed_data_from_s3 task."""

    @pytest.fixture
    def sample_processed_data(self):
        """Sample processed training data."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03"]
                ),
                "arithmetic_mean": [12.5, 15.2, 10.8],  # Target variable
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "year": [2023, 2023, 2023],
                "month": [1, 1, 1],
                "day_of_week": [6, 0, 1],
                "day_of_year": [1, 2, 3],
                "is_weekend": [1, 1, 0],
                "first_max_value": [18.0, 20.5, 14.2],
                "aqi": [45, 52, 38],
            }
        )

    @mock_aws
    def test_load_processed_data_from_s3_success(self, sample_processed_data):
        """Test successful data loading from S3."""
        bucket_name = "test-bucket"
        file_key = "processed_data/pm25_daily/pm25_daily_cleaned_2009_2024.parquet"

        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        # Upload test data to mock S3
        buffer = io.BytesIO()
        sample_processed_data.to_parquet(buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=buffer.getvalue())

        # Test the function
        result = load_processed_data_from_s3(bucket_name, file_key)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, sample_processed_data)

    @mock_aws
    def test_load_processed_data_from_s3_file_not_found(self):
        """Test handling of missing file in S3."""
        bucket_name = "test-bucket"
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)

        with pytest.raises(Exception):
            load_processed_data_from_s3(bucket_name, "nonexistent/file.parquet")

    @mock_aws
    def test_load_processed_data_from_s3_bucket_not_found(self):
        """Test handling of missing bucket."""
        with pytest.raises(Exception):
            load_processed_data_from_s3("nonexistent-bucket", "test/file.parquet")


class TestSplitData:
    """Test cases for the split_data task."""

    @pytest.fixture
    def training_data_complete(self):
        """Complete training data with all required features."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-05",
                        "2023-01-06",
                        "2023-01-07",
                        "2023-01-08",
                        "2023-01-09",
                        "2023-01-10",
                    ]
                ),
                "arithmetic_mean": [
                    12.5,
                    15.2,
                    10.8,
                    13.1,
                    14.7,
                    11.3,
                    16.2,
                    12.9,
                    15.8,
                    13.5,
                ],
                "latitude": [41.8781] * 10,
                "longitude": [-87.6298] * 10,
                "year": [2023] * 10,
                "month": [1] * 10,
                "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
                "day_of_year": list(range(1, 11)),
                "is_weekend": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                "first_max_value": [
                    18.0,
                    20.5,
                    14.2,
                    17.1,
                    19.3,
                    15.8,
                    21.2,
                    16.9,
                    20.1,
                    17.5,
                ],
                "aqi": [45, 52, 38, 47, 51, 40, 56, 46, 54, 48],
            }
        )

    @pytest.fixture
    def training_data_missing_features(self):
        """Training data missing some features."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-05",
                    ]
                ),
                "arithmetic_mean": [12.5, 15.2, 10.8, 14.1, 16.3],
                "latitude": [41.8781, 41.8781, 41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298, -87.6298, -87.6298],
                # Missing: year, month, day_of_week, day_of_year, is_weekend
            }
        )

    def test_split_data_success(self, training_data_complete):
        """Test successful data splitting."""
        x_train, x_val, x_test, y_train, y_val, y_test = split_data(
            training_data_complete, test_size=0.2, random_state=42
        )

        # Check that all splits are DataFrames/Series
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_val, pd.DataFrame)
        assert isinstance(x_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_val, pd.Series)
        assert isinstance(y_test, pd.Series)

        # Check split sizes (approximately)
        total_size = len(training_data_complete)
        test_size = len(x_test)
        val_size = len(x_val)
        train_size = len(x_train)

        assert test_size == int(total_size * 0.2)  # 20% for test
        assert train_size + val_size + test_size == total_size

        # Check that features are correctly selected
        expected_features = [
            "latitude",
            "longitude",
            "year",
            "month",
            "day_of_week",
            "day_of_year",
            "is_weekend",
        ]
        assert list(x_train.columns) == expected_features
        assert list(x_val.columns) == expected_features
        assert list(x_test.columns) == expected_features

        # Check that target is correct
        assert y_train.name == "arithmetic_mean"
        assert y_val.name == "arithmetic_mean"
        assert y_test.name == "arithmetic_mean"

    def test_split_data_custom_features(self, training_data_complete):
        """Test data splitting with custom feature selection."""
        custom_features = ["latitude", "longitude", "year"]

        x_train, x_val, x_test, y_train, y_val, y_test = split_data(
            training_data_complete,
            features=custom_features,
            test_size=0.3,
            random_state=42,
        )

        assert list(x_train.columns) == custom_features
        assert list(x_val.columns) == custom_features
        assert list(x_test.columns) == custom_features

    def test_split_data_custom_target(self, training_data_complete):
        """Test data splitting with custom target variable."""
        x_train, x_val, x_test, y_train, y_val, y_test = split_data(
            training_data_complete, target="first_max_value", random_state=42
        )

        assert y_train.name == "first_max_value"
        assert y_val.name == "first_max_value"
        assert y_test.name == "first_max_value"

    def test_split_data_empty_dataframe(self):
        """Test data splitting with empty DataFrame."""
        empty_df = pd.DataFrame()

        x_train, x_val, x_test, y_train, y_val, y_test = split_data(empty_df)

        assert len(x_train) == 0
        assert len(x_val) == 0
        assert len(x_test) == 0
        assert len(y_train) == 0
        assert len(y_val) == 0
        assert len(y_test) == 0

    def test_split_data_missing_target(self, training_data_complete):
        """Test data splitting when target column is missing."""
        df_no_target = training_data_complete.drop("arithmetic_mean", axis=1)

        with pytest.raises(ValueError, match="Valid features or target"):
            split_data(df_no_target)

    def test_split_data_missing_features(self, training_data_missing_features):
        """Test data splitting when some features are missing."""
        x_train, x_val, x_test, y_train, y_val, y_test = split_data(
            training_data_missing_features
        )

        # Should only include available features
        available_features = ["latitude", "longitude"]
        assert list(x_train.columns) == available_features

    def test_split_data_with_nan_values(self, training_data_complete):
        """Test data splitting with NaN values."""
        # Introduce some NaN values
        df_with_nans = training_data_complete.copy()
        df_with_nans.loc[0, "arithmetic_mean"] = None
        df_with_nans.loc[1, "latitude"] = None

        x_train, x_val, x_test, y_train, y_val, y_test = split_data(df_with_nans)

        # Should have fewer rows after dropping NaNs
        total_after_split = len(x_train) + len(x_val) + len(x_test)
        assert total_after_split < len(training_data_complete)

        # No NaN values should remain
        assert not x_train.isnull().any().any()
        assert not y_train.isnull().any()

    def test_split_data_reproducibility(self, training_data_complete):
        """Test that data splitting is reproducible with same random state."""
        x_train1, x_val1, x_test1, y_train1, y_val1, y_test1 = split_data(
            training_data_complete, random_state=42
        )
        x_train2, x_val2, x_test2, y_train2, y_val2, y_test2 = split_data(
            training_data_complete, random_state=42
        )

        pd.testing.assert_frame_equal(x_train1, x_train2)
        pd.testing.assert_frame_equal(x_val1, x_val2)
        pd.testing.assert_frame_equal(x_test1, x_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_val1, y_val2)
        pd.testing.assert_series_equal(y_test1, y_test2)


class TestTrainAndLogModel:
    """Test cases for the train_and_log_model task."""

    @pytest.fixture
    def sample_training_data(self):
        """Sample training data splits."""
        x_train = pd.DataFrame(
            {
                "latitude": [41.8781, 41.8781, 41.8781],
                "longitude": [-87.6298, -87.6298, -87.6298],
                "year": [2023, 2023, 2023],
                "month": [1, 1, 1],
                "day_of_week": [0, 1, 2],
                "day_of_year": [1, 2, 3],
                "is_weekend": [0, 0, 0],
            }
        )
        y_train = pd.Series([12.5, 15.2, 10.8], name="arithmetic_mean")

        x_val = pd.DataFrame(
            {
                "latitude": [41.8781],
                "longitude": [-87.6298],
                "year": [2023],
                "month": [1],
                "day_of_week": [3],
                "day_of_year": [4],
                "is_weekend": [0],
            }
        )
        y_val = pd.Series([13.1], name="arithmetic_mean")

        x_test = pd.DataFrame(
            {
                "latitude": [41.8781],
                "longitude": [-87.6298],
                "year": [2023],
                "month": [1],
                "day_of_week": [4],
                "day_of_year": [5],
                "is_weekend": [0],
            }
        )
        y_test = pd.Series([14.7], name="arithmetic_mean")

        return x_train, x_val, x_test, y_train, y_val, y_test

    @patch("flows.model_training.mlflow")
    def test_train_and_log_model_success(self, mock_mlflow, sample_training_data):
        """Test successful model training and logging."""
        x_train, x_val, x_test, y_train, y_val, y_test = sample_training_data

        # Mock MLflow context manager
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Create a simple model
        model = LinearRegression()

        result = train_and_log_model(
            model_name="TestModel",
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert result["model_name"] == "TestModel"
        assert result["run_id"] == "test_run_id"
        assert "val_rmse" in result
        assert "test_rmse" in result
        assert isinstance(result["val_rmse"], float)
        assert isinstance(result["test_rmse"], float)

        # Verify MLflow calls
        mock_mlflow.start_run.assert_called_once_with(run_name="TestModel")
        mock_mlflow.log_param.assert_called()
        mock_mlflow.log_metric.assert_called()
        mock_mlflow.sklearn.log_model.assert_called_once_with(
            model, artifact_path="model"
        )

    @patch("flows.model_training.mlflow")
    def test_train_and_log_model_random_forest(self, mock_mlflow, sample_training_data):
        """Test training with RandomForest model."""
        x_train, x_val, x_test, y_train, y_val, y_test = sample_training_data

        mock_run = MagicMock()
        mock_run.info.run_id = "rf_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        model = RandomForestRegressor(n_estimators=10, random_state=42)

        result = train_and_log_model(
            model_name="RandomForest",
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )

        assert result["model_name"] == "RandomForest"
        assert result["run_id"] == "rf_run_id"

        # Verify that model parameters were logged
        mock_mlflow.log_params.assert_called_once()

    @patch("flows.model_training.mlflow")
    def test_train_and_log_model_experiment_creation(
        self, mock_mlflow, sample_training_data
    ):
        """Test that MLflow experiment is created/set correctly."""
        x_train, x_val, x_test, y_train, y_val, y_test = sample_training_data

        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        model = LinearRegression()

        train_and_log_model(
            model_name="TestModel",
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )

        # Verify experiment setup
        mock_mlflow.create_experiment.assert_called()
        mock_mlflow.set_experiment.assert_called_with("air_quality_model_training")

    @patch("flows.model_training.mlflow")
    def test_train_and_log_model_existing_experiment(
        self, mock_mlflow, sample_training_data
    ):
        """Test handling when experiment already exists."""
        x_train, x_val, x_test, y_train, y_val, y_test = sample_training_data

        # Mock experiment already exists exception
        from mlflow.exceptions import MlflowException

        mock_mlflow.create_experiment.side_effect = MlflowException("already exists")

        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        model = LinearRegression()

        # Should not raise an exception
        result = train_and_log_model(
            model_name="TestModel",
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )

        assert isinstance(result, dict)


class TestTrainModels:
    """Test cases for the main train_models flow."""

    @pytest.fixture
    def sample_complete_data(self):
        """Complete sample data for training multiple models."""
        return pd.DataFrame(
            {
                "date_local": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-05",
                        "2023-01-06",
                        "2023-01-07",
                        "2023-01-08",
                        "2023-01-09",
                        "2023-01-10",
                    ]
                ),
                "arithmetic_mean": [
                    12.5,
                    15.2,
                    10.8,
                    13.1,
                    14.7,
                    11.3,
                    16.2,
                    12.9,
                    15.8,
                    13.5,
                ],
                "latitude": [41.8781] * 10,
                "longitude": [-87.6298] * 10,
                "year": [2023] * 10,
                "month": [1] * 10,
                "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
                "day_of_year": list(range(1, 11)),
                "is_weekend": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            }
        )

    @patch("flows.model_training.train_and_log_model")
    @patch("flows.model_training.split_data")
    @patch("flows.model_training.load_processed_data_from_s3")
    def test_train_models_success(
        self, mock_load_data, mock_split_data, mock_train_model, sample_complete_data
    ):
        """Test successful training of multiple models."""
        # Setup mocks
        mock_load_data.return_value = sample_complete_data

        # Mock split_data to return appropriate splits
        x_train = sample_complete_data[
            [
                "latitude",
                "longitude",
                "year",
                "month",
                "day_of_week",
                "day_of_year",
                "is_weekend",
            ]
        ].iloc[:6]
        y_train = sample_complete_data["arithmetic_mean"].iloc[:6]
        x_val = sample_complete_data[
            [
                "latitude",
                "longitude",
                "year",
                "month",
                "day_of_week",
                "day_of_year",
                "is_weekend",
            ]
        ].iloc[6:8]
        y_val = sample_complete_data["arithmetic_mean"].iloc[6:8]
        x_test = sample_complete_data[
            [
                "latitude",
                "longitude",
                "year",
                "month",
                "day_of_week",
                "day_of_year",
                "is_weekend",
            ]
        ].iloc[8:]
        y_test = sample_complete_data["arithmetic_mean"].iloc[8:]

        mock_split_data.return_value = (x_train, x_val, x_test, y_train, y_val, y_test)

        # Mock train_and_log_model to return different results for each model
        mock_train_model.side_effect = [
            {
                "model_name": "LinearRegression",
                "run_id": "lr_run",
                "val_rmse": 0.5,
                "test_rmse": 0.6,
            },
            {
                "model_name": "RandomForest",
                "run_id": "rf_run",
                "val_rmse": 0.4,
                "test_rmse": 0.5,
            },
            {
                "model_name": "XGBoost",
                "run_id": "xgb_run",
                "val_rmse": 0.3,
                "test_rmse": 0.4,
            },
        ]

        # Execute the flow
        results = train_models()

        # Verify results
        assert isinstance(results, list)
        assert len(results) == 3

        # Verify each model was trained
        model_names = [result["model_name"] for result in results]
        assert "LinearRegression" in model_names
        assert "RandomForest" in model_names
        assert "XGBoost" in model_names

        # Verify all functions were called
        mock_load_data.assert_called_once()
        mock_split_data.assert_called_once()
        assert mock_train_model.call_count == 3

    @patch("flows.model_training.train_and_log_model")
    @patch("flows.model_training.split_data")
    @patch("flows.model_training.load_processed_data_from_s3")
    def test_train_models_data_loading_error(
        self, mock_load_data, mock_split_data, mock_train_model
    ):
        """Test handling of data loading errors."""
        mock_load_data.side_effect = Exception("S3 loading failed")

        with pytest.raises(Exception, match="S3 loading failed"):
            train_models()

        # Other functions should not be called
        mock_split_data.assert_not_called()
        mock_train_model.assert_not_called()

    @patch("flows.model_training.train_and_log_model")
    @patch("flows.model_training.split_data")
    @patch("flows.model_training.load_processed_data_from_s3")
    def test_train_models_split_error(
        self, mock_load_data, mock_split_data, mock_train_model, sample_complete_data
    ):
        """Test handling of data splitting errors."""
        mock_load_data.return_value = sample_complete_data
        mock_split_data.side_effect = ValueError("Invalid data for splitting")

        with pytest.raises(ValueError, match="Invalid data for splitting"):
            train_models()

        mock_train_model.assert_not_called()

    @patch("flows.model_training.train_and_log_model")
    @patch("flows.model_training.split_data")
    @patch("flows.model_training.load_processed_data_from_s3")
    def test_train_models_partial_training_failure(
        self, mock_load_data, mock_split_data, mock_train_model, sample_complete_data
    ):
        """Test handling when some models fail to train."""
        # Setup mocks
        mock_load_data.return_value = sample_complete_data

        x_train = sample_complete_data[
            [
                "latitude",
                "longitude",
                "year",
                "month",
                "day_of_week",
                "day_of_year",
                "is_weekend",
            ]
        ].iloc[:6]
        y_train = sample_complete_data["arithmetic_mean"].iloc[:6]
        x_val = sample_complete_data[
            [
                "latitude",
                "longitude",
                "year",
                "month",
                "day_of_week",
                "day_of_year",
                "is_weekend",
            ]
        ].iloc[6:8]
        y_val = sample_complete_data["arithmetic_mean"].iloc[6:8]
        x_test = sample_complete_data[
            [
                "latitude",
                "longitude",
                "year",
                "month",
                "day_of_week",
                "day_of_year",
                "is_weekend",
            ]
        ].iloc[8:]
        y_test = sample_complete_data["arithmetic_mean"].iloc[8:]

        mock_split_data.return_value = (x_train, x_val, x_test, y_train, y_val, y_test)

        # First model succeeds, second fails, third succeeds
        mock_train_model.side_effect = [
            {
                "model_name": "LinearRegression",
                "run_id": "lr_run",
                "val_rmse": 0.5,
                "test_rmse": 0.6,
            },
            Exception("RandomForest training failed"),
            {
                "model_name": "XGBoost",
                "run_id": "xgb_run",
                "val_rmse": 0.3,
                "test_rmse": 0.4,
            },
        ]

        # Should raise exception on second model
        with pytest.raises(Exception, match="RandomForest training failed"):
            train_models()


# Parametrized tests for different data scenarios
@pytest.mark.parametrize(
    "test_size,expected_test_ratio",
    [
        (0.1, 0.1),
        (0.2, 0.2),
        (0.3, 0.3),
    ],
)
def test_split_data_different_test_sizes(test_size, expected_test_ratio):
    """Test data splitting with different test sizes."""
    # Create a larger dataset for meaningful splits
    data = pd.DataFrame(
        {
            "arithmetic_mean": list(range(100)),
            "latitude": [41.8781] * 100,
            "longitude": [-87.6298] * 100,
            "year": [2023] * 100,
            "month": [1] * 100,
            "day_of_week": [i % 7 for i in range(100)],
            "day_of_year": list(range(1, 101)),
            "is_weekend": [i % 7 >= 5 for i in range(100)],
        }
    )

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        data, test_size=test_size, random_state=42
    )

    total_size = len(data)
    test_size_actual = len(x_test)

    # Allow for small rounding differences
    assert abs(test_size_actual / total_size - expected_test_ratio) < 0.02


@pytest.mark.parametrize("target_column", ["arithmetic_mean", "first_max_value", "aqi"])
def test_split_data_different_targets(target_column):
    """Test data splitting with different target columns."""
    data = pd.DataFrame(
        {
            "arithmetic_mean": [12.5, 15.2, 10.8, 13.1, 14.7],
            "first_max_value": [18.0, 20.5, 14.2, 17.1, 19.3],
            "aqi": [45, 52, 38, 47, 51],
            "latitude": [41.8781] * 5,
            "longitude": [-87.6298] * 5,
            "year": [2023] * 5,
            "month": [1] * 5,
            "day_of_week": [0, 1, 2, 3, 4],
            "day_of_year": [1, 2, 3, 4, 5],
            "is_weekend": [0, 0, 0, 0, 0],
        }
    )

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        data, target=target_column, random_state=42
    )

    assert y_train.name == target_column
    assert y_val.name == target_column
    assert y_test.name == target_column
