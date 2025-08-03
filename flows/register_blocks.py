"""
Prefect S3 Block Registration

This module registers S3 bucket blocks in Prefect for use in workflows.
It loads AWS credentials from environment variables and creates reusable
S3 bucket blocks that can be referenced in Prefect flows.
"""

import os

from dotenv import load_dotenv
from prefect_aws.s3 import S3Bucket


def load_aws_credentials() -> tuple[str, str, str, str]:
    """
    Load AWS credentials and configuration from environment variables.

    Returns:
        Tuple of (bucket_name, aws_access_key_id, aws_secret_access_key, region)

    Raises:
        ValueError: If any required environment variables are missing
    """
    # Load from .env (parent directory)
    load_dotenv(dotenv_path="../.env")

    bucket_name = os.getenv("S3_DATA_BUCKET_NAME")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION")

    if not all([bucket_name, aws_key, aws_secret, region]):
        missing_vars = [
            var
            for var, val in [
                ("S3_DATA_BUCKET_NAME", bucket_name),
                ("AWS_ACCESS_KEY_ID", aws_key),
                ("AWS_SECRET_ACCESS_KEY", aws_secret),
                ("AWS_REGION", region),
            ]
            if not val
        ]
        raise ValueError(
            f"Missing required AWS environment variables: {', '.join(missing_vars)}"
        )

    return bucket_name, aws_key, aws_secret, region


def register_s3_block(
    bucket_name: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region_name: str,
    block_name: str | None = None,
    overwrite: bool = True,
) -> None:
    """
    Register an S3 bucket block in Prefect.

    Args:
        bucket_name: Name of the S3 bucket
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        region_name: AWS region name
        block_name: Name for the Prefect block (defaults to bucket_name)
        overwrite: Whether to overwrite existing block

    Raises:
        Exception: If block registration fails
    """
    if block_name is None:
        block_name = bucket_name

    try:
        # Create the S3 block in Prefect
        s3_block = S3Bucket(
            bucket_name=bucket_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

        s3_block.save(name=block_name, overwrite=overwrite)
        print(f"✅ Prefect S3 block '{block_name}' registered successfully.")

    except Exception as exc:
        print(f"❌ Failed to register S3 block '{block_name}': {exc}")
        raise


def main() -> None:
    """
    Main function to register S3 blocks.
    """
    try:
        bucket_name, aws_key, aws_secret, region = load_aws_credentials()
        register_s3_block(bucket_name, aws_key, aws_secret, region)

    except ValueError as ve:
        print(f"❌ Configuration error: {ve}")
        raise
    except Exception as exc:
        print(f"❌ Unexpected error: {exc}")
        raise


if __name__ == "__main__":
    main()
