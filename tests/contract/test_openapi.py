"""Contract tests using Schemathesis to validate API against OpenAPI spec.

These tests ensure that the API implementation conforms to the OpenAPI 3.1
specification defined in /api/openapi.yaml.
"""

from pathlib import Path

import pytest
import schemathesis
from fastapi.testclient import TestClient

from src.main import app

# Load OpenAPI schema
openapi_path = Path(__file__).parent.parent.parent / "api" / "openapi.yaml"

# Configure Schemathesis
schema = schemathesis.from_path(
    str(openapi_path),
    app=app,
    base_url="http://localhost:8000",
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ============================================================================
# Health Endpoint Tests
# ============================================================================

@schema.parametrize(endpoint="/health")
def test_health_endpoint(case, client):
    """Test health endpoint conforms to OpenAPI spec."""
    response = case.call_and_validate(session=client)
    assert response.status_code == 200
    assert "status" in response.json()


@schema.parametrize(endpoint="/health/ready")
def test_readiness_endpoint(case, client):
    """Test readiness endpoint conforms to OpenAPI spec."""
    response = case.call_and_validate(session=client)
    assert response.status_code == 200


@schema.parametrize(endpoint="/health/live")
def test_liveness_endpoint(case, client):
    """Test liveness endpoint conforms to OpenAPI spec."""
    response = case.call_and_validate(session=client)
    assert response.status_code == 200


# ============================================================================
# API Schema Validation
# ============================================================================

def test_openapi_schema_is_valid():
    """Verify the OpenAPI schema is valid and can be loaded."""
    assert schema.raw_schema is not None
    assert "paths" in schema.raw_schema
    assert "/api/v1/jobs" in schema.raw_schema["paths"]


def test_all_endpoints_have_operations():
    """Verify all endpoints have at least one operation defined."""
    paths = schema.raw_schema.get("paths", {})
    for path, operations in paths.items():
        # Skip paths that are just parameter definitions
        if not any(op in operations for op in ["get", "post", "put", "delete", "patch"]):
            continue
        
        # Each operation should have a summary or description
        for method in ["get", "post", "put", "delete", "patch"]:
            if method in operations:
                operation = operations[method]
                assert "operationId" in operation, f"{method.upper()} {path} missing operationId"


# ============================================================================
# Response Schema Validation
# ============================================================================

def test_health_response_schema():
    """Validate health endpoint response schema."""
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data
    assert "components" in data
    
    # Check status values
    assert data["status"] in ["healthy", "degraded", "unhealthy"]


def test_error_response_schema():
    """Validate error response follows standard format."""
    client = TestClient(app)
    
    # Trigger a 404 error
    response = client.get("/api/v1/jobs/invalid-uuid")
    
    # Should return standardized error format
    assert response.status_code in [404, 422]
    
    # Error responses should follow standard format
    data = response.json()
    if "error" in data:
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert "meta" in data
        assert "request_id" in data["meta"]


# ============================================================================
# Content-Type Validation
# ============================================================================

def test_api_returns_json():
    """Verify API returns JSON for API endpoints."""
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.headers["content-type"] == "application/json"


def test_metrics_returns_prometheus_format():
    """Verify metrics endpoint returns Prometheus format."""
    client = TestClient(app)
    
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


# ============================================================================
# Security Header Validation
# ============================================================================

def test_security_headers_present():
    """Verify security headers are present in responses."""
    client = TestClient(app)
    
    response = client.get("/health")
    
    # Check for request ID header
    assert "X-Request-ID" in response.headers
    assert "X-API-Version" in response.headers


# ============================================================================
# API Version Validation
# ============================================================================

def test_api_version_header():
    """Verify API version header is correct."""
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.headers["X-API-Version"] == "v1"
