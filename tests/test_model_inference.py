"""
Unit tests for the model inference flow using pytest and moto for S3 interactions.
"""

import io
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, Mock
from moto import mock_aws
import boto3
from datetime import datetime

# Import the modules to test
from flows.model_inference import (
    load_model_from_s3,
    load_processed_inference_data_from_s3,
    generate_predictions,
    save_predictions_to_s3,
    model_batch_prediction_flow
)


class TestLoadModelFromS3:
    """Test cases for the load_model_from_s3 task."""

    def test_load_model_from_s3_success(self):
        """Test successful model loading from S3."""
        mock_model = MagicMock()
        
        # Mock the entire mlflow module in sys.modules before the function runs
        with patch.dict('sys.modules'):
            # Create mock mlflow module structure
            mock_mlflow = MagicMock()
            mock_pyfunc = MagicMock()
            mock_pyfunc.load_model.return_value = mock_model
            mock_mlflow.pyfunc = mock_pyfunc
            
            # Add to sys.modules so the import inside the function finds it
            import sys
            sys.modules['mlflow'] = mock_mlflow
            sys.modules['mlflow.pyfunc'] = mock_pyfunc
            
            s3_path = "s3://test-bucket/model/path"
            result = load_model_from_s3(s3_path)
            
            mock_pyfunc.load_model.assert_called_once_with(s3_path)
            assert result == mock_model

    def test_load_model_from_s3_failure(self):
        """Test handling of model loading failure."""
        # Mock the entire mlflow module in sys.modules before the function runs
        with patch.dict('sys.modules'):
            # Create mock mlflow module structure that raises an exception
            mock_mlflow = MagicMock()
            mock_pyfunc = MagicMock()
            mock_pyfunc.load_model.side_effect = Exception("Model loading failed")
            mock_mlflow.pyfunc = mock_pyfunc
            
            # Add to sys.modules so the import inside the function finds it
            import sys
            sys.modules['mlflow'] = mock_mlflow
            sys.modules['mlflow.pyfunc'] = mock_pyfunc
            
            s3_path = "s3://test-bucket/model/path"
            with pytest.raises(Exception, match="Model loading failed"):
                load_model_from_s3(s3_path)


class TestLoadProcessedInferenceDataFromS3:
    """Test cases for the load_processed_inference_data_from_s3 task."""

    @pytest.fixture
    def sample_inference_data(self):
        """Sample processed inference data."""
        return pd.DataFrame({
            'date_local': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'latitude': [41.8781, 41.8781],
            'longitude': [-87.6298, -87.6298],
            'year': [2025, 2025],
            'month': [1, 1],
            'day_of_week': [2, 3],
            'day_of_year': [1, 2],
            'is_weekend': [0, 0]
        })

    @mock_aws
    def test_load_processed_inference_data_from_s3_success(self, sample_inference_data):
        """Test successful data loading from S3."""
        bucket_name = "test-bucket"
        key_prefix = "processed_inference"
        file_name = "test_file"
        s3_key = f"{key_prefix}/{file_name}.parquet"
        
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket=bucket_name)
        
        # Upload test data to mock S3
        buffer = io.BytesIO()
        sample_inference_data.to_parquet(buffer, index=False)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=buffer.getvalue()
        )
        
        # Test the function
        result = load_processed_inference_data_from_s3(bucket_name, key_prefix, file_name)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        pd.testing.assert_frame_equal(result, sample_inference_data)

    @mock_aws
    def test_load_processed_inference_data_from_s3_file_not_found(self):
        """Test handling of missing file in S3."""
        bucket_name = "test-bucket"
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket=bucket_name)
        
        with pytest.raises(Exception):
            load_processed_inference_data_from_s3(
                bucket_name, "processed_inference", "nonexistent_file"
            )

    @mock_aws
    def test_load_processed_inference_data_from_s3_bucket_not_found(self):
        """Test handling of missing bucket."""
        with pytest.raises(Exception):
            load_processed_inference_data_from_s3(
                "nonexistent-bucket", "processed_inference", "test_file"
            )


class TestGeneratePredictions:
    """Test cases for the generate_predictions task."""

    @pytest.fixture
    def mock_model(self):
        """Mock MLflow model."""
        model = MagicMock()
        model.predict.return_value = [12.5, 15.2, 10.8]
        return model

    @pytest.fixture
    def inference_data_complete(self):
        """Complete inference data with all required features."""
        return pd.DataFrame({
            'date_local': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
            'latitude': [41.8781, 41.8781, 41.8781],
            'longitude': [-87.6298, -87.6298, -87.6298],
            'year': [2025, 2025, 2025],
            'month': [1, 1, 1],
            'day_of_week': [2, 3, 4],
            'day_of_year': [1, 2, 3],
            'is_weekend': [0, 0, 0],
            'cbsa_code': ['16980', '16980', '16980']
        })

    @pytest.fixture
    def inference_data_missing_features(self):
        """Inference data missing required features."""
        return pd.DataFrame({
            'date_local': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'latitude': [41.8781, 41.8781],
            'longitude': [-87.6298, -87.6298],
            # Missing: year, month, day_of_week, day_of_year, is_weekend
        })

    def test_generate_predictions_success(self, mock_model, inference_data_complete):
        """Test successful prediction generation."""
        result = generate_predictions(mock_model, inference_data_complete)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'predicted_arithmetic_mean' in result.columns
        
        # Verify model was called with correct features
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args[0][0]
        expected_features = [
            'latitude', 'longitude', 'year', 'month', 'day_of_week', 
            'day_of_year', 'is_weekend'
        ]
        assert list(call_args.columns) == expected_features
        
        # Verify predictions are added to the result
        assert result['predicted_arithmetic_mean'].tolist() == [12.5, 15.2, 10.8]

    def test_generate_predictions_empty_dataframe(self, mock_model):
        """Test prediction generation with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = generate_predictions(mock_model, empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        mock_model.predict.assert_not_called()

    def test_generate_predictions_missing_features(self, mock_model, inference_data_missing_features):
        """Test prediction generation with missing required features."""
        with pytest.raises(ValueError, match="Missing required features"):
            generate_predictions(mock_model, inference_data_missing_features)

    def test_generate_predictions_model_error(self, inference_data_complete):
        """Test handling of model prediction errors."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        
        with pytest.raises(Exception, match="Prediction failed"):
            generate_predictions(mock_model, inference_data_complete)

    def test_generate_predictions_preserves_original_data(self, mock_model, inference_data_complete):
        """Test that original data is preserved in predictions DataFrame."""
        result = generate_predictions(mock_model, inference_data_complete)
        
        # All original columns should be preserved
        for col in inference_data_complete.columns:
            assert col in result.columns
            pd.testing.assert_series_equal(
                result[col], inference_data_complete[col], check_names=False
            )


class TestSavePredictionsToS3:
    """Test cases for the save_predictions_to_s3 task."""

    @pytest.fixture
    def predictions_data(self):
        """Sample predictions data."""
        return pd.DataFrame({
            'date_local': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'latitude': [41.8781, 41.8781],
            'longitude': [-87.6298, -87.6298],
            'predicted_arithmetic_mean': [12.5, 15.2]
        })

    @mock_aws
    @patch('flows.model_inference.datetime')
    def test_save_predictions_to_s3_success(self, mock_datetime, predictions_data):
        """Test successful saving of predictions to S3."""
        # Mock datetime for consistent file naming
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        
        bucket_name = "test-bucket"
        target_year = 2025
        
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket=bucket_name)
        
        # Test the function
        save_predictions_to_s3(predictions_data, bucket_name, target_year)
        
        # Verify file was uploaded with correct naming
        expected_key = "predictions/pm25_daily/pm25_predictions_2025_20250101_120000.parquet"
        response = s3_client.get_object(Bucket=bucket_name, Key=expected_key)
        
        # Read back the data and verify
        uploaded_data = pd.read_parquet(io.BytesIO(response['Body'].read()))
        pd.testing.assert_frame_equal(uploaded_data, predictions_data)

    def test_save_predictions_to_s3_empty_dataframe(self):
        """Test saving empty DataFrame to S3."""
        empty_df = pd.DataFrame()
        
        # Should not raise an exception, just skip the upload
        save_predictions_to_s3(empty_df, "test-bucket", 2025)

    @mock_aws
    def test_save_predictions_to_s3_upload_error(self, predictions_data):
        """Test handling of S3 upload errors."""
        # Don't create the bucket to simulate an error
        with pytest.raises(Exception):
            save_predictions_to_s3(predictions_data, "nonexistent-bucket", 2025)


class TestModelBatchPredictionFlow:
    """Test cases for the main model batch prediction flow."""

    @pytest.fixture
    def mock_model(self):
        """Mock MLflow model."""
        model = MagicMock()
        model.predict.return_value = [12.5, 15.2]
        return model

    @pytest.fixture
    def sample_processed_data(self):
        """Sample processed inference data."""
        return pd.DataFrame({
            'date_local': pd.to_datetime(['2025-01-01', '2025-01-02']),
            'latitude': [41.8781, 41.8781],
            'longitude': [-87.6298, -87.6298],
            'year': [2025, 2025],
            'month': [1, 1],
            'day_of_week': [2, 3],
            'day_of_year': [1, 2],
            'is_weekend': [0, 0]
        })

    @patch('flows.model_inference.save_predictions_to_s3')
    @patch('flows.model_inference.generate_predictions')
    @patch('flows.model_inference.load_model_from_s3')
    @patch('flows.model_inference.inference_data_preparation_flow')
    def test_model_batch_prediction_flow_success(
        self, mock_prep_flow, mock_load_model, mock_generate, mock_save,
        mock_model, sample_processed_data
    ):
        """Test successful execution of the batch prediction flow."""
        # Setup mocks
        mock_prep_flow_future = MagicMock()
        mock_prep_flow_future.result.return_value = sample_processed_data
        mock_prep_flow.return_value = mock_prep_flow_future
        
        mock_load_model.return_value = mock_model
        
        predictions_data = sample_processed_data.copy()
        predictions_data['predicted_arithmetic_mean'] = [12.5, 15.2]
        mock_generate.return_value = predictions_data
        
        # Execute flow
        result = model_batch_prediction_flow(inference_year=2025)
        
        # Verify all tasks were called
        mock_prep_flow.assert_called_once_with(target_year=2025, return_state=True)
        mock_load_model.assert_called_once()
        mock_generate.assert_called_once_with(mock_model, sample_processed_data)
        mock_save.assert_called_once()
        
        # Verify return value
        pd.testing.assert_frame_equal(result, predictions_data)

    @patch('flows.model_inference.save_predictions_to_s3')
    @patch('flows.model_inference.generate_predictions')
    @patch('flows.model_inference.load_model_from_s3')
    @patch('flows.model_inference.inference_data_preparation_flow')
    def test_model_batch_prediction_flow_default_year(
        self, mock_prep_flow, mock_load_model, mock_generate, mock_save,
        mock_model, sample_processed_data
    ):
        """Test flow with default year (current year)."""
        current_year = datetime.now().year
        
        mock_prep_flow_future = MagicMock()
        mock_prep_flow_future.result.return_value = sample_processed_data
        mock_prep_flow.return_value = mock_prep_flow_future
        
        mock_load_model.return_value = mock_model
        mock_generate.return_value = sample_processed_data
        
        # Execute flow without specifying year
        model_batch_prediction_flow()
        
        # Should use current year
        mock_prep_flow.assert_called_once_with(target_year=current_year, return_state=True)

    @patch('flows.model_inference.inference_data_preparation_flow')
    def test_model_batch_prediction_flow_empty_data(self, mock_prep_flow):
        """Test flow when no data is available for inference."""
        # Setup mock to return empty DataFrame
        mock_prep_flow_future = MagicMock()
        mock_prep_flow_future.result.return_value = pd.DataFrame()
        mock_prep_flow.return_value = mock_prep_flow_future
        
        result = model_batch_prediction_flow(inference_year=2025)
        
        # Should return None when no data available
        assert result is None

    @patch('flows.model_inference.save_predictions_to_s3')
    @patch('flows.model_inference.generate_predictions')
    @patch('flows.model_inference.load_model_from_s3')
    @patch('flows.model_inference.inference_data_preparation_flow')
    def test_model_batch_prediction_flow_model_loading_error(
        self, mock_prep_flow, mock_load_model, mock_generate, mock_save,
        sample_processed_data
    ):
        """Test flow when model loading fails."""
        mock_prep_flow_future = MagicMock()
        mock_prep_flow_future.result.return_value = sample_processed_data
        mock_prep_flow.return_value = mock_prep_flow_future
        
        mock_load_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            model_batch_prediction_flow(inference_year=2025)

    @patch('flows.model_inference.save_predictions_to_s3')
    @patch('flows.model_inference.generate_predictions')
    @patch('flows.model_inference.load_model_from_s3')
    @patch('flows.model_inference.inference_data_preparation_flow')
    def test_model_batch_prediction_flow_prediction_error(
        self, mock_prep_flow, mock_load_model, mock_generate, mock_save,
        mock_model, sample_processed_data
    ):
        """Test flow when prediction generation fails."""
        mock_prep_flow_future = MagicMock()
        mock_prep_flow_future.result.return_value = sample_processed_data
        mock_prep_flow.return_value = mock_prep_flow_future
        
        mock_load_model.return_value = mock_model
        mock_generate.side_effect = Exception("Prediction failed")
        
        with pytest.raises(Exception, match="Prediction failed"):
            model_batch_prediction_flow(inference_year=2025)


# Parametrized tests for different prediction scenarios
@pytest.mark.parametrize("prediction_values,expected_length", [
    ([12.5], 1),
    ([12.5, 15.2], 2),
    ([12.5, 15.2, 10.8, 18.3], 4),
])
def test_generate_predictions_different_sizes(prediction_values, expected_length):
    """Test prediction generation with different data sizes."""
    mock_model = MagicMock()
    mock_model.predict.return_value = prediction_values
    
    # Create data with the expected length
    data = pd.DataFrame({
        'latitude': [41.8781] * expected_length,
        'longitude': [-87.6298] * expected_length,
        'year': [2025] * expected_length,
        'month': [1] * expected_length,
        'day_of_week': [0] * expected_length,
        'day_of_year': [1] * expected_length,
        'is_weekend': [0] * expected_length
    })
    
    result = generate_predictions(mock_model, data)
    
    assert len(result) == expected_length
    assert result['predicted_arithmetic_mean'].tolist() == prediction_values


@pytest.mark.parametrize("missing_feature", [
    'latitude', 'longitude', 'year', 'month', 'day_of_week', 'day_of_year', 'is_weekend'
])
def test_generate_predictions_missing_individual_features(missing_feature):
    """Test prediction generation fails when individual required features are missing."""
    mock_model = MagicMock()
    
    # Create complete feature set
    features = {
        'latitude': [41.8781],
        'longitude': [-87.6298],
        'year': [2025],
        'month': [1],
        'day_of_week': [0],
        'day_of_year': [1],
        'is_weekend': [0]
    }
    
    # Remove one feature
    del features[missing_feature]
    data = pd.DataFrame(features)
    
    with pytest.raises(ValueError, match="Missing required features"):
        generate_predictions(mock_model, data)