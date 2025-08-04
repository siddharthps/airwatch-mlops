# UV Environment Setup Guide

This project is configured to work with `uv`, a fast Python package installer and
resolver.

## Prerequisites

Make sure you have `uv` installed. If not, install it:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

## Project Setup

### 1. Create and activate virtual environment

```bash
# Create virtual environment with Python 3.10+
uv venv --python 3.10

# Activate the environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
# Install all dependencies (production + dev)
uv pip install -e ".[all]"

# Or install specific dependency groups:
uv pip install -e ".[dev]"     # Development dependencies
uv pip install -e ".[test]"    # Testing dependencies only
uv pip install -e .            # Production dependencies only
```

### 3. Verify installation

```bash
# Run tests to verify everything is working
python run_tests.py

# Or use pytest directly
pytest tests/ -v
```

## Available Dependency Groups

- **Production** (`uv pip install -e .`): Core dependencies needed to run the
  application
- **Development** (`uv pip install -e ".[dev]"`): Ruff (formatting & linting) and
  testing tools
- **Testing** (`uv pip install -e ".[test]"`): Testing framework and mocking libraries
- **Documentation** (`uv pip install -e ".[docs]"`): Documentation generation tools
- **All** (`uv pip install -e ".[all]"`): All dependency groups combined

## Common Commands

```bash
# Install/update dependencies
uv pip install -e ".[all]"

# Run tests
python run_tests.py
pytest tests/ -v

# Run tests with coverage
python run_tests.py --coverage
pytest tests/ --cov=flows --cov-report=html

# Run specific test file
python run_tests.py -f test_model_training.py
pytest tests/test_model_training.py -v

# Format code
ruff format flows/ tests/

# Lint code
ruff check flows/ tests/

# Lint and fix automatically
ruff check --fix flows/ tests/

# Generate documentation
mkdocs serve
```

## Benefits of UV

- **Fast**: Much faster than pip for dependency resolution and installation
- **Reliable**: Better dependency resolution with conflict detection
- **Compatible**: Works with existing pip and requirements.txt workflows
- **Modern**: Built with Rust for performance

## Migration from requirements.txt

The old `requirements.txt` file is still present for compatibility, but the project now
uses `pyproject.toml` as the primary dependency specification. You can still use:

```bash
# Legacy pip installation (slower)
pip install -r requirements.txt

# Modern uv installation (faster)
uv pip install -e ".[all]"
```

## Troubleshooting

### Common Issues

1. **Python version mismatch**: Ensure you're using Python 3.10+

   ```bash
   uv venv --python 3.10
   ```

1. **Permission errors**: Make sure virtual environment is activated

```bash
   source .venv/bin/activate  # Unix/macOS
   .venv\Scripts\activate     # Windows
```

3. **Dependency conflicts**: Use uv's resolver to check conflicts

   ```bash
   ```

uv pip check

````

### Environment Variables

Make sure to set up your `.env` file with required environment variables:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your actual values
# EPA_AQS_EMAIL=your_email@example.com
# EPA_AQS_API_KEY=your_api_key
# S3_DATA_BUCKET_NAME=your_bucket_name
# AWS_REGION=us-east-1
````

## Performance Comparison

| Operation             | pip  | uv  | Improvement |
| --------------------- | ---- | --- | ----------- |
| Fresh install         | ~45s | ~8s | 5.6x faster |
| Cached install        | ~12s | ~2s | 6x faster   |
| Dependency resolution | ~15s | ~1s | 15x faster  |

*Times are approximate and may vary based on system and network conditions.*
