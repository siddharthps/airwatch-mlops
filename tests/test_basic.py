"""
Basic test to verify the test setup is working.
"""
import boto3
from moto import mock_aws
import pandas as pd
import pytest
import requests_mock


def test_basic_functionality():
    """Test that basic imports and functionality work."""
    # Test pandas
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ["a", "b"]


def test_pytest_fixtures():
    """Test that pytest fixtures work."""
    assert True


@pytest.fixture
def test_sample_data():
    """Sample fixture for testing."""
    return {"test": "data"}


def test_fixture_usage(test_sample_data):
    """Test that fixtures can be used."""
    assert test_sample_data["test"] == "data"


def test_moto_import():
    """Test that moto can be imported and works."""
    assert mock_aws is not None


def test_boto3_import():
    """Test that boto3 can be imported and works."""
    assert boto3 is not None


def test_requests_mock_import():
    """Test that requests-mock can be imported and works."""
    assert requests_mock is not None


@mock_aws
def test_moto_s3_functionality():
    """Test that moto S3 mocking works."""
    # Create a mock S3 client
    s3_client = boto3.client("s3", region_name="us-east-1")

    # Create a test bucket
    bucket_name = "test-bucket"
    s3_client.create_bucket(Bucket=bucket_name)

    # List buckets to verify it was created
    response = s3_client.list_buckets()
    bucket_names = [bucket["Name"] for bucket in response["Buckets"]]

    assert bucket_name in bucket_names


def test_requests_mock_functionality():
    """Test that requests-mock works."""
    import requests

    with requests_mock.Mocker() as m:
        # Mock a GET request
        m.get("http://test.com/api", json={"key": "value"})

        # Make the request
        response = requests.get("http://test.com/api")

        # Verify the response
        assert response.json() == {"key": "value"}
        assert response.status_code == 200


def test_environment_setup():
    """Test that the testing environment is properly configured."""
    # Test that we can create DataFrames
    df = pd.DataFrame({"col1": [1, 2, 3]})
    assert not df.empty

    # Test that pytest is working
    assert pytest is not None

    # Test that all required testing packages are available
    assert mock_aws is not None
    assert boto3 is not None
    assert requests_mock is not None
