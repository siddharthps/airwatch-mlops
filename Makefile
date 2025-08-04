#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = airwatch-mlops
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
DOCKER_IMAGE = airwatch-inference

#################################################################################
# ENVIRONMENT SETUP                                                             #
#################################################################################

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt

#################################################################################
# CODE QUALITY                                                                  #
#################################################################################

## Lint using ruff
.PHONY: lint
lint:
	ruff check flows/ tests/

## Format source code with ruff
.PHONY: format
format:
	ruff format flows/ tests/
	ruff check --fix flows/ tests/

## Run tests
.PHONY: test
test:
	python -m pytest tests/ -v

## Run tests with coverage
.PHONY: test-cov
test-cov:
	python -m pytest tests/ --cov=flows --cov-report=html --cov-report=term

#################################################################################
# DOCKER COMMANDS                                                               #
#################################################################################

## Build Docker image
.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE) .

## Run Docker container
.PHONY: docker-run
docker-run:
	docker run --env-file .env $(DOCKER_IMAGE)

## Build and run with docker-compose
.PHONY: docker-compose-up
docker-compose-up:
	docker-compose up --build

## Run docker-compose in background
.PHONY: docker-compose-up-d
docker-compose-up-d:
	docker-compose up --build -d

## Stop docker-compose services
.PHONY: docker-compose-down
docker-compose-down:
	docker-compose down

## View docker-compose logs
.PHONY: docker-logs
docker-logs:
	docker-compose logs -f

#################################################################################
# MLOps WORKFLOWS                                                               #
#################################################################################

## Run data preparation flow locally
.PHONY: run-data-prep
run-data-prep:
	$(PYTHON_INTERPRETER) flows/inference_data_preparation.py

## Run model inference flow locally
.PHONY: run-inference
run-inference:
	$(PYTHON_INTERPRETER) flows/model_inference.py

## Run both flows locally
.PHONY: run-pipeline
run-pipeline: run-data-prep run-inference

#################################################################################
# DATA MANAGEMENT                                                               #
#################################################################################

## Download data from S3
.PHONY: sync-data-down
sync-data-down:
	aws s3 sync s3://air-quality-mlops-data-chicago-2025/data/ data/

## Upload data to S3
.PHONY: sync-data-up
sync-data-up:
	aws s3 sync data/ s3://air-quality-mlops-data-chicago-2025/data/

## Download models from S3
.PHONY: sync-models-down
sync-models-down:
	aws s3 sync s3://mlflow-artifacts-chicago-2025/artifacts/models/ models/

#################################################################################
# CLEANUP                                                                       #
#################################################################################

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

## Clean Docker images and containers
.PHONY: docker-clean
docker-clean:
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
