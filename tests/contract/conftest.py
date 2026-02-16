"""Pytest configuration for contract tests."""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture(scope="session")
def test_client():
    """Create a test client for the FastAPI application."""
    with TestClient(app) as client:
        yield client
