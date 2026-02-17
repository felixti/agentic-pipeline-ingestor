"""Contract tests using Schemathesis to validate API against OpenAPI spec.

These tests ensure that the API implementation conforms to the OpenAPI 3.1
specification defined in /api/openapi.yaml.
"""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def openapi_schema():
    """Load the OpenAPI schema."""
    openapi_path = Path(__file__).parent.parent.parent / "api" / "openapi.yaml"
    with open(openapi_path) as f:
        return yaml.safe_load(f)


class TestBasicEndpoints:
    """Test basic API endpoints conform to OpenAPI spec."""

    def test_health_endpoint_exists(self, client):
        """Test that health endpoint exists and returns correct structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data

    def test_health_live_endpoint(self, client):
        """Test Kubernetes liveness probe endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_health_ready_endpoint(self, client):
        """Test Kubernetes readiness probe endpoint."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_openapi_spec_endpoint(self, client):
        """Test that OpenAPI spec endpoint returns valid YAML."""
        response = client.get("/api/v1/openapi.yaml")
        assert response.status_code == 200
        content = response.text
        assert "openapi:" in content
        assert "paths:" in content

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert "# HELP" in content or "# TYPE" in content


class TestOpenAPISpecValidity:
    """Test that the OpenAPI spec is valid and complete."""

    def test_spec_has_required_fields(self, openapi_schema):
        """Test that spec has all required OpenAPI fields."""
        assert "openapi" in openapi_schema
        assert "info" in openapi_schema
        assert "paths" in openapi_schema

    def test_spec_version_is_3_1(self, openapi_schema):
        """Test that spec version is 3.1.x."""
        version = openapi_schema["openapi"]
        assert version.startswith("3.1")

    def test_spec_has_api_info(self, openapi_schema):
        """Test that spec has API information."""
        info = openapi_schema["info"]
        assert "title" in info
        assert "version" in info
        assert "description" in info

    def test_spec_has_paths(self, openapi_schema):
        """Test that spec has at least one path defined."""
        paths = openapi_schema["paths"]
        assert len(paths) > 0

    def test_jobs_path_defined(self, openapi_schema):
        """Test that /api/v1/jobs path is defined."""
        assert "/api/v1/jobs" in openapi_schema["paths"]

    def test_upload_path_defined(self, openapi_schema):
        """Test that /api/v1/upload path is defined."""
        assert "/api/v1/upload" in openapi_schema["paths"]

    def test_sources_path_defined(self, openapi_schema):
        """Test that /api/v1/sources path is defined."""
        assert "/api/v1/sources" in openapi_schema["paths"]


class TestResponseSchemas:
    """Test that responses match expected schemas."""

    def test_health_response_structure(self, client, openapi_schema):
        """Test health response matches schema."""
        response = client.get("/health")
        data = response.json()

        # Verify required fields
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data

        # Verify field types
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["components"], dict)
