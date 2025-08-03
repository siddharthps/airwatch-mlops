"""
Simple S3 test to verify moto works with the new import.
"""

import boto3
import pytest
from moto import mock_aws


class TestS3Simple:
    """Simple S3 tests using moto."""

    @mock_aws
    def test_s3_bucket_creation(self):
        """Test that we can create an S3 bucket with moto."""
        # Create S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Create bucket
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        
        # List buckets
        response = s3_client.list_buckets()
        bucket_names = [bucket['Name'] for bucket in response['Buckets']]
        
        assert bucket_name in bucket_names

    @mock_aws
    def test_s3_put_get_object(self):
        """Test that we can put and get objects from S3."""
        # Create S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Create bucket
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        
        # Put object
        test_content = b'Hello, World!'
        s3_client.put_object(
            Bucket=bucket_name,
            Key='test-file.txt',
            Body=test_content
        )
        
        # Get object
        response = s3_client.get_object(Bucket=bucket_name, Key='test-file.txt')
        retrieved_content = response['Body'].read()
        
        assert retrieved_content == test_content