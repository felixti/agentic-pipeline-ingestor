"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_endpoint(self, client):
        """Test comprehensive health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data

    def test_readiness_endpoint(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_liveness_endpoint(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/health/live")

        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_openapi_yaml_endpoint(self, client):
        """Test OpenAPI YAML endpoint."""
        response = client.get("/api/v1/openapi.yaml")

        # Should return 200 if file exists, 404 otherwise
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            assert "text/yaml" in response.headers["content-type"]


class TestJobEndpoints:
    """Tests for job management endpoints."""

    def test_create_job(self, client):
        """Test job creation endpoint."""
        response = client.post(
            "/api/v1/jobs",
            json={
                "source_type": "upload",
                "source_uri": "/test/file.pdf",
            },
        )

        assert response.status_code == 202
        data = response.json()

        assert "data" in data
        assert "meta" in data
        assert "request_id" in data["meta"]

    def test_list_jobs(self, client):
        """Test job listing endpoint."""
        response = client.get("/api/v1/jobs")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert "meta" in data

    def test_get_job(self, client):
        """Test job retrieval endpoint."""
        response = client.get("/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data

    def test_cancel_job(self, client):
        """Test job cancellation endpoint."""
        response = client.delete("/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000")

        assert response.status_code in [204, 404]

    def test_retry_job(self, client):
        """Test job retry endpoint."""
        response = client.post(
            "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/retry",
            json={},
        )

        assert response.status_code == 202


class TestUploadEndpoints:
    """Tests for file upload endpoints."""

    def test_upload_files(self, client):
        """Test file upload endpoint."""
        response = client.post(
            "/api/v1/upload",
            data={"priority": 5},
        )

        assert response.status_code == 202
        data = response.json()

        assert "data" in data

    def test_ingest_from_url(self, client):
        """Test URL ingestion endpoint."""
        response = client.post(
            "/api/v1/upload/url",
            json={"url": "https://example.com/document.pdf"},
        )

        assert response.status_code == 202


class TestPipelineEndpoints:
    """Tests for pipeline configuration endpoints."""

    def test_list_pipelines(self, client):
        """Test pipeline listing endpoint."""
        response = client.get("/api/v1/pipelines")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data

    def test_create_pipeline(self, client):
        """Test pipeline creation endpoint."""
        response = client.post(
            "/api/v1/pipelines",
            json={
                "name": "test-pipeline",
                "description": "Test pipeline",
            },
        )

        assert response.status_code == 201

    def test_get_pipeline(self, client):
        """Test pipeline retrieval endpoint."""
        response = client.get(
            "/api/v1/pipelines/550e8400-e29b-41d4-a716-446655440000"
        )

        assert response.status_code == 200


class TestPluginEndpoints:
    """Tests for plugin management endpoints."""

    def test_list_sources(self, client):
        """Test source plugin listing endpoint."""
        response = client.get("/api/v1/sources")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert "plugins" in data["data"] or "configurations" in data["data"]

    def test_list_destinations(self, client):
        """Test destination plugin listing endpoint."""
        response = client.get("/api/v1/destinations")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data


class TestAuditEndpoints:
    """Tests for audit endpoints."""

    def test_query_audit_logs(self, client):
        """Test audit log query endpoint."""
        response = client.get("/api/v1/audit/logs")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data


class TestResponseHeaders:
    """Tests for response headers."""

    def test_request_id_header(self, client):
        """Test that X-Request-ID header is present."""
        response = client.get("/health")

        assert "X-Request-ID" in response.headers
        assert "X-API-Version" in response.headers
        assert response.headers["X-API-Version"] == "v1"

    def test_content_type_header(self, client):
        """Test that Content-Type is correct."""
        response = client.get("/health")

        assert "application/json" in response.headers["content-type"]


class TestErrorResponses:
    """Tests for error response formats."""

    def test_not_found_response(self, client):
        """Test 404 error response format."""
        response = client.get("/nonexistent-endpoint")

        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 error response."""
        response = client.post("/health")

        assert response.status_code == 405


class TestCorsHeaders:
    """Tests for CORS headers."""

    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/api/v1/jobs",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
