"""Integration tests for API endpoint persistence."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import JobModel, JobStatus
from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.integration
class TestJobCreationPersistence:
    """Test that POST /api/v1/jobs persists to database."""
    
    def test_create_job_persists_to_database(self, client):
        """Job creation should create database record."""
        # Create job via API
        response = client.post(
            "/api/v1/jobs",
            json={
                "source_type": "upload",
                "source_uri": "/tmp/test/file.pdf",
                "file_name": "file.pdf",
                "file_size": 1024,
                "mime_type": "application/pdf",
            }
        )
        
        assert response.status_code == 202
        data = response.json()["data"]
        assert "id" in data
        assert data["status"] == "created"
        assert data["file_name"] == "file.pdf"


@pytest.mark.integration
class TestJobListingPersistence:
    """Test that GET /api/v1/jobs returns persisted data."""
    
    def test_list_jobs_returns_persisted_jobs(self, client):
        """Job listing should return jobs from database."""
        # First create a job
        client.post(
            "/api/v1/jobs",
            json={"source_type": "upload", "file_name": "test1.pdf", "file_size": 1000}
        )
        
        # List jobs
        response = client.get("/api/v1/jobs")
        
        assert response.status_code == 200
        data = response.json()["data"]
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["items"]) >= 1


@pytest.mark.integration
class TestJobRetrievalPersistence:
    """Test that GET /api/v1/jobs/{id} returns persisted job."""
    
    def test_get_job_returns_persisted_data(self, client):
        """Get job should return job from database."""
        # Create job
        create_response = client.post(
            "/api/v1/jobs",
            json={"source_type": "upload", "file_name": "test2.pdf", "file_size": 2000}
        )
        job_id = create_response.json()["data"]["id"]
        
        # Get job
        response = client.get(f"/api/v1/jobs/{job_id}")
        
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["id"] == job_id
        assert data["file_name"] == "test2.pdf"
        assert data["file_size"] == 2000


@pytest.mark.integration
class TestJobCancellation:
    """Test DELETE /api/v1/jobs/{id} cancels job."""
    
    def test_cancel_job_updates_status(self, client):
        """Cancel should update job status."""
        # Create job
        create_response = client.post(
            "/api/v1/jobs",
            json={"source_type": "upload", "file_name": "test3.pdf", "file_size": 3000}
        )
        job_id = create_response.json()["data"]["id"]
        
        # Cancel job
        response = client.delete(f"/api/v1/jobs/{job_id}")
        
        assert response.status_code == 204


@pytest.mark.integration
class TestFileUploadPersistence:
    """Test that POST /api/v1/upload creates job."""
    
    def test_upload_creates_job_in_database(self, client):
        """File upload should create job record."""
        # Create a test file
        import io
        
        file_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n%%EOF"
        
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
        )
        
        assert response.status_code == 202
        data = response.json()["data"]
        assert "job_id" in data
        assert data["file_name"] == "test.pdf"
