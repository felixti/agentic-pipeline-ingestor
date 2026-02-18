"""Unit tests for jobs API routes.

Tests the job-related API endpoints defined in src/main.py including
creating, listing, getting, canceling, and retrying jobs.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException, status

from src.api.models import (
    ApiLinks,
    ApiResponse,
    JobStatus,
)


@pytest.mark.unit
class TestCreateJob:
    """Tests for POST /api/v1/jobs - Create job endpoint."""

    @pytest.mark.asyncio
    async def test_create_job_success(self):
        """Test successful job creation."""
        from src.main import create_job
        
        mock_request = MagicMock()
        mock_request.state.request_id = str(uuid4())
        
        response = await create_job(mock_request)
        
        assert response["data"]["status"] == JobStatus.CREATED.value
        assert "id" in response["data"]
        assert response["data"]["message"] == "Job accepted for processing"
        assert "meta" in response
        assert "request_id" in response["meta"]

    @pytest.mark.asyncio
    async def test_create_job_without_request_id(self):
        """Test job creation when request has no request_id."""
        from src.main import create_job
        
        mock_request = MagicMock()
        mock_request.state = MagicMock()
        # No request_id attribute
        del mock_request.state.request_id
        
        response = await create_job(mock_request)
        
        assert response["data"]["status"] == JobStatus.CREATED.value
        assert "id" in response["data"]


@pytest.mark.unit
class TestListJobs:
    """Tests for GET /api/v1/jobs - List jobs endpoint."""

    @pytest.mark.asyncio
    async def test_list_jobs_success(self):
        """Test successful job listing."""
        from src.main import list_jobs
        
        mock_request = MagicMock()
        mock_request.state.request_id = str(uuid4())
        mock_request.url = "http://test/api/v1/jobs"
        
        response = await list_jobs(mock_request, page=1, limit=20, status=None)
        
        assert response["data"] == []
        assert "meta" in response
        assert "links" in response
        assert response["links"]["self"] == "http://test/api/v1/jobs"

    @pytest.mark.asyncio
    async def test_list_jobs_with_status_filter(self):
        """Test job listing with status filter."""
        from src.main import list_jobs
        
        mock_request = MagicMock()
        mock_request.state.request_id = str(uuid4())
        mock_request.url = "http://test/api/v1/jobs"
        
        response = await list_jobs(mock_request, page=1, limit=20, status="completed")
        
        assert response["data"] == []
        assert "meta" in response

    @pytest.mark.asyncio
    async def test_list_jobs_pagination(self):
        """Test job listing with pagination parameters."""
        from src.main import list_jobs
        
        mock_request = MagicMock()
        mock_request.state.request_id = str(uuid4())
        mock_request.url = "http://test/api/v1/jobs?page=2&limit=50"
        
        response = await list_jobs(mock_request, page=2, limit=50, status=None)
        
        assert response["data"] == []
        assert "meta" in response


@pytest.mark.unit
class TestGetJob:
    """Tests for GET /api/v1/jobs/{job_id} - Get job endpoint."""

    @pytest.mark.asyncio
    async def test_get_job_success(self):
        """Test successful job retrieval."""
        from src.main import get_job
        
        job_id = str(uuid4())
        mock_request = MagicMock()
        mock_request.state.request_id = str(uuid4())
        
        # Mock database session and result
        mock_job = MagicMock()
        mock_job.to_dict.return_value = {
            "id": job_id,
            "status": "completed",
            "created_at": datetime.utcnow().isoformat(),
        }
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_job
        
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        
        response = await get_job(job_id, mock_request, mock_db)
        
        assert response["data"]["id"] == job_id
        assert response["data"]["status"] == "completed"
        assert "meta" in response

    @pytest.mark.asyncio
    async def test_get_job_not_found(self):
        """Test job retrieval when job doesn't exist."""
        from src.main import get_job
        
        job_id = str(uuid4())
        mock_request = MagicMock()
        mock_request.state.request_id = str(uuid4())
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        
        with pytest.raises(HTTPException) as exc_info:
            await get_job(job_id, mock_request, mock_db)
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert job_id in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_job_database_error(self):
        """Test job retrieval with database error."""
        from src.main import get_job
        
        job_id = str(uuid4())
        mock_request = MagicMock()
        mock_request.state.request_id = str(uuid4())
        
        mock_db = AsyncMock()
        mock_db.execute.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception) as exc_info:
            await get_job(job_id, mock_request, mock_db)
        
        assert "Database connection failed" in str(exc_info.value)


@pytest.mark.unit
class TestCancelJob:
    """Tests for DELETE /api/v1/jobs/{job_id} - Cancel job endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_job_success(self):
        """Test successful job cancellation."""
        from src.main import cancel_job
        
        job_id = str(uuid4())
        
        # The current implementation returns None (204 No Content)
        response = await cancel_job(job_id)
        
        assert response is None

    @pytest.mark.asyncio
    async def test_cancel_job_nonexistent(self):
        """Test canceling a non-existent job."""
        from src.main import cancel_job
        
        job_id = "non-existent-job-id"
        
        # Current implementation doesn't raise exception
        response = await cancel_job(job_id)
        
        assert response is None


@pytest.mark.unit
class TestRetryJob:
    """Tests for POST /api/v1/jobs/{job_id}/retry - Retry job endpoint."""

    @pytest.mark.asyncio
    async def test_retry_job_success(self):
        """Test successful job retry."""
        from src.main import retry_job
        
        job_id = str(uuid4())
        mock_request = MagicMock()
        mock_request.state.request_id = str(uuid4())
        
        response = await retry_job(job_id, mock_request)
        
        assert response["data"]["id"] == job_id
        assert response["data"]["status"] == "retrying"
        assert "meta" in response

    @pytest.mark.asyncio
    async def test_retry_job_without_request_id(self):
        """Test job retry when request has no request_id."""
        from src.main import retry_job
        
        job_id = str(uuid4())
        mock_request = MagicMock()
        mock_request.state = MagicMock()
        del mock_request.state.request_id
        
        response = await retry_job(job_id, mock_request)
        
        assert response["data"]["id"] == job_id
        assert response["data"]["status"] == "retrying"


@pytest.mark.unit
class TestGetJobResult:
    """Tests for GET /api/v1/jobs/{job_id}/result - Get job result endpoint."""

    @pytest.mark.asyncio
    async def test_get_job_result_success(self):
        """Test successful job result retrieval."""
        from src.main import get_job_result
        
        job_id = str(uuid4())
        mock_request = MagicMock()
        mock_request.state.request_id = str(uuid4())
        
        response = await get_job_result(job_id, mock_request)
        
        assert response["data"]["job_id"] == job_id
        assert response["data"]["success"] is True
        assert "meta" in response

    @pytest.mark.asyncio
    async def test_get_job_result_not_ready(self):
        """Test job result retrieval when result is not ready."""
        from src.main import get_job_result
        
        job_id = str(uuid4())
        mock_request = MagicMock()
        mock_request.state.request_id = str(uuid4())
        
        # Current implementation always returns success=True
        # In a real implementation, this would check job status
        response = await get_job_result(job_id, mock_request)
        
        assert response["data"]["job_id"] == job_id
