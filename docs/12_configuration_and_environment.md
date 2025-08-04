Configuration and Environment Management Overview

This document outlines the environment setup and configuration practices used in the
AirWatch MLOps project. It ensures reproducibility, security, and ease of collaboration
across development and production environments. Environment Management

1. Dependency Management

   Project uses UV for deterministic Python dependency resolution and locking.

   All packages are defined in pyproject.toml and resolved via uv pip compile.

   This ensures consistency across all development and deployment environments.

1. Python Versioning

   Python version is standardized across the team using the requires-python setting in
   pyproject.toml.

   Local environments and CI pipelines respect the defined version for compatibility.

Configuration with Environment Variables

```
All secrets and sensitive configuration (e.g., AWS credentials, API keys) are externalized using environment variables.

.env file provides local development values but is excluded from version control for security.

Examples of configuration controlled via env variables:

    AWS S3 bucket names and regions

    AQS API credentials

    MLflow tracking server URI

    Model and data version identifiers
```

Local Development Setup

```
Clone the repository and navigate to the project root.

Install dependencies using UV or another preferred tool.

Set up a .env file based on .env.example.

Activate environment and run flows or tests as needed.
```

Remote/Production Setup

```
Environments such as CI/CD runners or cloud VMs read from securely injected env variables (e.g., GitHub secrets, AWS SSM, or Docker secrets).

Prefect agents and MLflow servers run with appropriate environment contexts.
```

Best Practices

```
Keep configuration and credentials out of code.

Use .env.example to document all expected environment variables.

Rotate and manage secrets using tools like AWS Secrets Manager if scaling up.

Validate critical configuration keys at startup to avoid silent failures.
```
