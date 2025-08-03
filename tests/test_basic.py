"""
Basic test to verify the test setup is working.
"""

import pandas as pd
import pytest


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
def sample_data():
    """Sample fixture for testing."""
    return {"test": "data"}


def test_fixture_usage(sample_data):
    """Test that fixtures can be used."""
    assert sample_data["test"] == "data"


def test_moto_import():
    """Test that moto can be imported."""
    try:
        from moto import mock_aws

        assert mock_aws is not None
    except ImportError as e:
        pytest.fail(f"Failed to import moto: {e}")


def test_boto3_import():
    """Test that boto3 can be imported."""
    try:
        import boto3

        assert boto3 is not None
    except ImportError as e:
        pytest.fail(f"Failed to import boto3: {e}")


def test_requests_mock_import():
    """Test that requests-mock can be imported."""
    try:
        import requests_mock

        assert requests_mock is not None
    except ImportError as e:
        pytest.fail(f"Failed to import requests_mock: {e}")
