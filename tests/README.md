# Tests for Air Quality MLOps Project

This directory contains unit tests for the air quality MLOps project.

## Test Files

- `test_data_ingestion.py` - Comprehensive unit tests for the data ingestion flow
- `test_data_transformation.py` - Unit tests for the data transformation pipeline
- `test_inference_data_preparation.py` - Unit tests for inference data preparation
- `test_model_inference.py` - Unit tests for model inference and batch prediction
- `test_model_monitoring.py` - Unit tests for model monitoring with Evidently AI
- `test_model_selector.py` - Unit tests for model selection and S3 upload
- `test_model_training.py` - Unit tests for model training pipeline

## Running Tests

### Prerequisites

Make sure you have installed the test dependencies:

```bash
pip install -r requirements.txt
```

### Running All Tests

```bash
# From the project root directory
pytest tests/

# Or run with verbose output
pytest tests/ -v
```

### Running Specific Test Files

```bash
# Run only data ingestion tests
pytest tests/test_data_ingestion.py -v

# Run data transformation tests
pytest tests/test_data_transformation.py -v

# Run inference data preparation tests
pytest tests/test_inference_data_preparation.py -v

# Run model inference tests
pytest tests/test_model_inference.py -v

# Run model monitoring tests
pytest tests/test_model_monitoring.py -v

# Run model selector tests
pytest tests/test_model_selector.py -v

# Run model training tests
pytest tests/test_model_training.py -v
```

### Running Specific Test Classes or Methods

```bash
# Run only tests for fetch_epa_aqs_data function
pytest tests/test_data_ingestion.py::TestFetchEpaAqsData -v

# Run a specific test method
pytest tests/test_data_ingestion.py::TestFetchEpaAqsData::test_fetch_epa_aqs_data_dict_response_success -v
```

### Using the Test Runner Script

```bash
# Simple way to run data ingestion tests
python run_tests.py
```

## Test Coverage

The tests provide comprehensive coverage across all MLOps pipeline components:

### Data Ingestion (`test_data_ingestion.py`):
- ✅ EPA AQS API data fetching (dict/list responses, errors, timeouts)
- ✅ S3 uploads using Prefect blocks
- ✅ Environment variable validation
- ✅ Multi-year data processing
- ✅ Error handling and retry logic

### Data Transformation (`test_data_transformation.py`):
- ✅ S3 data loading and saving
- ✅ Data cleaning and type conversions
- ✅ Feature engineering (date features, weekend detection)
- ✅ Missing data handling
- ✅ End-to-end transformation pipeline

### Inference Data Preparation (`test_inference_data_preparation.py`):
- ✅ Single-year API data fetching for inference
- ✅ Inference-specific data transformation
- ✅ Feature consistency with training data
- ✅ S3 storage of processed inference data

### Model Inference (`test_model_inference.py`):
- ✅ MLflow model loading from S3
- ✅ Batch prediction generation
- ✅ Feature validation and error handling
- ✅ Prediction results storage
- ✅ Integration with data preparation flow

### Model Monitoring (`test_model_monitoring.py`):
- ✅ Historical and current data loading
- ✅ Evidently AI report generation
- ✅ Data drift detection and alerting
- ✅ Regression performance monitoring
- ✅ S3 report storage

### Model Selection (`test_model_selector.py`):
- ✅ MLflow experiment querying
- ✅ Best model selection (non-overfit filtering)
- ✅ Model artifact download and upload
- ✅ Model metadata and summary generation
- ✅ S3 model storage

### Model Training (`test_model_training.py`):
- ✅ Training data loading from S3
- ✅ Data splitting (train/validation/test)
- ✅ Multiple model training (LinearRegression, RandomForest, XGBoost)
- ✅ MLflow experiment tracking
- ✅ Model evaluation and metrics logging

## Test Technologies Used

- **pytest** - Main testing framework with fixtures and parametrized tests
- **requests-mock** - For mocking HTTP requests to EPA AQS API
- **moto** - For mocking AWS S3 services (lightweight localstack alternative)
- **unittest.mock** - For mocking Prefect, MLflow, and other external dependencies
- **pandas** - For DataFrame testing utilities and data validation
- **boto3** - For S3 client testing with moto
- **scikit-learn** - For model testing utilities

## Test Structure

Tests are organized into classes by functionality across all modules:

### Data Pipeline Tests:
- `TestFetchEpaAqsData` - API data fetching
- `TestTransformData` - Data cleaning and feature engineering
- `TestLoadRawDataFromS3` / `TestWriteDataToS3` - S3 operations
- `TestAirQualityIngestionFlow` / `TestAirQualityTransformationFlow` - End-to-end pipelines

### ML Pipeline Tests:
- `TestLoadProcessedDataFromS3` - Training data loading
- `TestSplitData` - Data splitting for ML
- `TestTrainAndLogModel` - Model training and MLflow logging
- `TestTrainModels` - Multi-model training orchestration

### Inference Tests:
- `TestFetchEpaAqsDataForInference` - Inference data fetching
- `TestTransformDataForInference` - Inference data transformation
- `TestGeneratePredictions` - Model prediction generation
- `TestModelBatchPredictionFlow` - End-to-end inference pipeline

### Model Management Tests:
- `TestGetBestNonOverfitRun` - Model selection logic
- `TestDownloadModelArtifacts` / `TestUploadFilesToS3` - Model artifact management
- `TestModelSelectionFlow` - Model selection and deployment

### Monitoring Tests:
- `TestLoadHistoricalData` / `TestLoadPredictionsFromS3` - Monitoring data loading
- `TestCreateDataDriftReport` / `TestCreateRegressionPerformanceReport` - Report generation
- `TestCheckDataDrift` - Drift detection logic
- `TestModelMonitoringFlow` - End-to-end monitoring pipeline

## Mocking Strategy

### API Calls
- Uses `requests-mock` to mock EPA AQS API responses
- Tests various response formats and error conditions
- No actual HTTP requests are made during testing

### S3 Operations
- Uses `moto` library to mock S3 operations locally
- Provides a lightweight alternative to localstack for unit testing
- Tests S3 upload/download functionality without requiring AWS credentials
- Supports both direct boto3 and Prefect S3Bucket operations

### MLflow Components
- Uses `unittest.mock` to mock MLflow tracking and model operations
- Mocks experiment creation, run logging, and model artifact handling
- Isolates ML pipeline logic from MLflow infrastructure

### Prefect Components
- Uses `unittest.mock` to mock Prefect S3Bucket blocks and flow execution
- Isolates the business logic from Prefect infrastructure
- Tests task orchestration without requiring Prefect server

### External Libraries
- Mocks Evidently AI report generation for monitoring tests
- Mocks scikit-learn model training and prediction for faster test execution
- Uses dependency injection patterns for better testability

## Environment Variables for Testing

The tests mock all required environment variables:
- `EPA_AQS_EMAIL` - EPA AQS API email
- `EPA_AQS_API_KEY` - EPA AQS API key
- `S3_DATA_BUCKET_NAME` - Main data storage bucket
- `S3_MLFLOW_ARTIFACTS_BUCKET_NAME` - MLflow artifacts bucket
- `MLFLOW_TRACKING_URI` - MLflow tracking server URI
- `AWS_REGION` - AWS region for S3 operations

No real credentials or infrastructure are needed for running the tests.

## Adding New Tests

When adding new tests:

1. Follow the existing naming convention (`test_*`)
2. Use appropriate fixtures for test data and mock objects
3. Mock external dependencies (APIs, S3, MLflow, etc.)
4. Include both success and failure scenarios
5. Add docstrings explaining what each test validates
6. Use parametrized tests for testing multiple scenarios
7. Organize tests into logical classes by functionality
8. Ensure tests are independent and can run in any order

## Test Performance

- **Fast execution**: All tests use mocks and run without external dependencies
- **Parallel execution**: Tests can be run in parallel using `pytest -n auto`
- **Selective testing**: Use markers and patterns to run specific test subsets
- **Memory efficient**: Moto provides lightweight S3 simulation

## Continuous Integration

These tests are designed to run in CI/CD environments without external dependencies:
- No real AWS credentials required
- No actual API calls made
- No MLflow server required
- No Prefect server required
- Fast execution time (typically under 30 seconds for full suite)
- Deterministic results with fixed random seeds
- Cross-platform compatibility (Windows, Linux, macOS)