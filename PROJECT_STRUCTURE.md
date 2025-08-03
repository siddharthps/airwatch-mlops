# Project Structure

This document describes the organization of the AirWatch MLOps project.

## Directory Structure

```
airwatch-mlops/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT license
├── README.md                 # Main project documentation
├── pyproject.toml           # Python project configuration (UV)
├── uv.lock                  # Dependency lock file
├── Makefile                 # Build and development commands
├── RUFF_GUIDE.md           # Code formatting guidelines
├── UV_SETUP.md             # UV package manager setup
├── PROJECT_STRUCTURE.md    # This file
│
├── data/                    # Data storage (gitignored except .gitkeep)
│   ├── raw/                # Raw data from EPA AQS API
│   ├── interim/            # Intermediate processing results
│   ├── processed/          # Final processed data for ML
│   └── external/           # External reference data
│
├── flows/                   # Prefect workflow definitions
│   ├── data_ingestion.py           # EPA AQS data fetching
│   ├── data_transformation.py     # Data cleaning and feature engineering
│   ├── inference_data_preparation.py  # Prepare data for inference
│   ├── model_training.py          # Train ML models
│   ├── model_selector.py          # Select best model
│   ├── model_inference.py         # Generate predictions
│   ├── model_monitoring.py        # Monitor model performance
│   ├── register_blocks.py         # Prefect block registration
│   └── *.md                       # Flow documentation
│
├── tests/                   # Test suite (182 tests, 90%+ coverage)
│   ├── test_*.py           # Unit tests for each flow
│   ├── conftest.py         # Pytest configuration and fixtures
│   └── README.md           # Testing documentation
│
├── scripts/                 # Utility scripts
│   ├── deploy_monitoring.py       # Prefect deployment setup
│   ├── inspect_s3_parquet.py     # S3 data inspection
│   ├── test_mlflow_setup.py      # MLflow connectivity test
│   ├── prefect_email_notification.txt  # Email setup guide
│   └── README.md                  # Scripts documentation
│
├── models/                  # Model artifacts (gitignored)
│   └── .gitkeep            # Keep directory structure
│
├── reports/                 # Generated reports
│   ├── figures/            # Plots and visualizations
│   └── *.html             # Evidently monitoring reports
│
├── docs/                    # Documentation
│   ├── mkdocs.yml          # MkDocs configuration
│   └── docs/               # Documentation source
│
├── mlruns/                  # MLflow experiment tracking (gitignored)
├── mlartifacts/            # MLflow artifacts (gitignored)
└── htmlcov/                # Test coverage reports (gitignored)
```

## Key Files

### Configuration
- **pyproject.toml**: Python project metadata, dependencies, and tool configuration
- **.env.example**: Template for environment variables (copy to .env)
- **Makefile**: Common development tasks and commands

### Core Workflows
- **flows/**: All Prefect workflows for the MLOps pipeline
- **tests/**: Comprehensive test suite with mocking for AWS services

### Data Management
- **data/**: Structured data storage following data science conventions
- **models/**: Trained model artifacts and metadata
- **reports/**: Generated monitoring and analysis reports

## Development Workflow

1. **Setup**: Copy `.env.example` to `.env` and configure
2. **Install**: Run `uv sync` to install dependencies
3. **Test**: Run `uv run pytest` to execute test suite
4. **Develop**: Modify flows and add corresponding tests
5. **Format**: Code is auto-formatted with Ruff
6. **Deploy**: Use scripts in `scripts/` for deployment

## Data Flow

```
EPA AQS API → data/raw → data/processed → MLflow → S3 → Monitoring Reports
     ↑              ↑           ↑          ↑       ↑         ↑
data_ingestion  data_transform  training  selector inference monitoring
```

## Testing Strategy

- **Unit Tests**: Mock external services (AWS, EPA API)
- **Integration Tests**: End-to-end workflow validation (planned)
- **Coverage**: 90%+ code coverage across all modules
- **CI/CD**: Automated testing on code changes (planned)

## Deployment

- **Local**: Run individual flows with `uv run python flows/flow_name.py`
- **Scheduled**: Use Prefect deployments for automation
- **Cloud**: Deploy to AWS/GCP/Azure with containerization (planned)

## Monitoring

- **MLflow**: Experiment tracking and model registry
- **Evidently**: Data drift and model performance monitoring
- **Prefect**: Workflow orchestration and monitoring
- **Coverage**: Test coverage reporting with pytest-cov