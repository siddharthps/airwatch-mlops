# AirWatch MLOps

A comprehensive MLOps pipeline for predicting PM2.5 air quality levels using EPA AQS data, built with modern Python tools and containerized deployment.

## 🌟 Overview

This project implements an end-to-end machine learning operations (MLOps) pipeline that:

- **Ingests** real-time air quality data from EPA's Air Quality System (AQS) API
- **Transforms** and processes data for machine learning
- **Trains** multiple regression models (Linear Regression, Random Forest, XGBoost)
- **Selects** the best performing non-overfit model
- **Generates** predictions for future air quality levels
- **Monitors** model performance and data drift using Evidently AI
- **Orchestrates** workflows with Prefect
- **Tracks** experiments with MLflow
- **Deploys** via Docker containers for production environments

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   EPA AQS API   │───▶│  Data Ingestion  │───▶│ Data Transform  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Model Monitor   │◀───│ Model Inference  │◀───│ Model Training  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Evidently     │    │       S3         │    │     MLflow      │
│   Reports       │    │   Predictions    │    │   Experiments   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Features

### **Data Pipeline**

- ✅ **Automated data ingestion** from EPA AQS API
- ✅ **Data validation and cleaning** with pandas
- ✅ **Feature engineering** (temporal features, weekend indicators)
- ✅ **S3 storage** for raw and processed data

### **Model Training**

- ✅ **Multiple algorithms** (Linear Regression, Random Forest, XGBoost)
- ✅ **Automated hyperparameter tuning**
- ✅ **Cross-validation** and proper train/val/test splits
- ✅ **MLflow experiment tracking** with metrics and artifacts

### **Model Selection**

- ✅ **Overfitting detection** (validation RMSE threshold)
- ✅ **Automated best model selection**
- ✅ **Model artifact management** and S3 deployment

### **Inference Pipeline**

- ✅ **Batch prediction** workflows
- ✅ **Real-time data preparation** for inference
- ✅ **Prediction storage** and versioning

### **Monitoring & Observability**

- ✅ **Data drift detection** with Evidently AI
- ✅ **Model performance monitoring**
- ✅ **Automated report generation** (HTML dashboards)
- ✅ **Alert system** for drift detection

### **DevOps & Testing**

- ✅ **Comprehensive test suite** (182 tests, 90%+ coverage)
- ✅ **Modern Python tooling** (UV, Ruff, pytest)
- ✅ **Moto-based AWS mocking** for reliable testing
- ✅ **Type hints** and code quality enforcement

### **Containerization & Deployment**

- ✅ **Docker containerization** for consistent deployments
- ✅ **Docker Compose** for simplified orchestration
- ✅ **Production-ready Dockerfile** with optimized layers
- ✅ **Environment-based configuration** via .env files
- ✅ **Makefile automation** for common tasks

## 📦 Installation

### Prerequisites

- Python 3.10+
- UV package manager
- Docker and Docker Compose
- AWS credentials configured
- MLflow tracking server (optional, defaults to local)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/airwatch-mlops.git
   cd airwatch-mlops
   ```

2. **Install dependencies with UV**

   ```bash
   uv sync
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Set up AWS credentials**

   ```bash
   aws configure
   # Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
   ```

5. **Start MLflow server (optional)**

   ```bash
   uv run mlflow server --host 0.0.0.0 --port 5000
   ```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**Run the complete inference pipeline:**

```bash
# Using Docker Compose (simplest)
make docker-compose-up

# Or using Docker directly
make docker-build
make docker-run
```

**Run in background:**

```bash
make docker-compose-up-d
```

**View logs:**

```bash
make docker-logs
```

### Option 2: Local Development

**Run the complete pipeline:**

```bash
# Using Makefile
make run-pipeline

# Or manually
make run-data-prep
make run-inference
```

**Run individual components:**

```bash
# Data preparation only
make run-data-prep

# Model inference only  
make run-inference

# Generate predictions for specific year
uv run python -c "from flows.model_inference import model_batch_prediction_flow; model_batch_prediction_flow(2025)"
```

### Option 3: Training Pipeline (Local Only)

```bash
# 1. Ingest and transform data
uv run python flows/data_ingestion.py
uv run python flows/data_transformation.py

# 2. Train models
uv run python flows/model_training.py

# 3. Select best model
uv run python flows/model_selector.py

# 4. Monitor model performance
uv run python flows/model_monitoring.py
```

## 🧪 Testing

### Run All Tests

```bash
# Using Makefile
make test

# Or directly with UV
uv run pytest tests/ -v
```

### Run with Coverage

```bash
# Using Makefile
make test-cov

# Or directly with UV
uv run pytest tests/ --cov=flows --cov-report=html
```
## 📊 Monitoring & Observability

### MLflow Dashboard

Access experiment tracking at: `http://localhost:5000`

### Evidently Reports

- Data drift reports: `s3://your-bucket/monitoring_reports/data_drift_report_*.html`
- Performance reports: `s3://your-bucket/monitoring_reports/regression_performance_report_*.html`

### Key Metrics Tracked

- **Model Performance**: RMSE, MAE on validation and test sets
- **Data Drift**: Statistical tests for feature distribution changes
- **Prediction Quality**: Actual vs predicted comparisons
- **Pipeline Health**: Success rates, execution times, error rates


## 🛠️ Development Tools

### Makefile Commands

The project includes a comprehensive Makefile for common tasks:

```bash
# Environment setup
make create_environment    # Create UV virtual environment
make requirements          # Install dependencies

# Code quality
make lint                  # Run ruff linting
make format               # Format code with ruff
make test                 # Run tests
make test-cov             # Run tests with coverage

# Docker operations
make docker-build         # Build Docker image
make docker-run           # Run container
make docker-compose-up    # Build and run with compose
make docker-compose-up-d  # Run in background
make docker-compose-down  # Stop services
make docker-logs          # View logs

# MLOps workflows
make run-data-prep        # Run data preparation
make run-inference        # Run model inference
make run-pipeline         # Run complete pipeline

# Data management
make sync-data-down       # Download data from S3
make sync-data-up         # Upload data to S3
make sync-models-down     # Download models from S3

# Cleanup
make clean                # Clean Python cache
make docker-clean         # Clean Docker resources
```

### Docker Usage

**Production deployment:**

```bash
# Build and run inference pipeline
docker-compose up --build

# Run specific flows
docker run --env-file .env airwatch-inference python flows/inference_data_preparation.py
```

**Development with volume mounts:**

```bash
# Interactive development
docker run --env-file .env -v $(pwd)/flows:/app/flows -it airwatch-inference bash
```

## 📈 Performance

### Current Metrics

- **Test Coverage**: 90%+ across all modules
- **Model Accuracy**: RMSE < 5.0 μg/m³ on test data
- **Pipeline Success Rate**: 99%+ in testing
- **Data Processing**: ~1000 records/second
- **Docker Build Time**: ~2-3 minutes
- **Container Startup**: ~10 seconds

### Benchmarks

- **Data Ingestion**: ~2 minutes for 1 year of data
- **Model Training**: ~5 minutes for all models
- **Prediction Generation**: ~30 seconds for 1000 predictions
- **Monitoring Reports**: ~1 minute for drift analysis
- **Docker Image Size**: ~1.2GB (optimized)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EPA Air Quality System (AQS)** for providing comprehensive air quality data
- **Evidently AI** for excellent model monitoring capabilities
- **Prefect** for robust workflow orchestration
- **MLflow** for experiment tracking and model management
- **Data Talks Club** for the MLOps Zoomcamp 
- **Alexey Grigorev** 
