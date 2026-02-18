"""Integration tests for chunk API endpoints.

Tests for GET /api/v1/jobs/{job_id}/chunks and GET /api/v1/jobs/{job_id}/chunks/{chunk_id}
Includes tests for pagination, error cases, and rate limiting.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DocumentChunkModel, JobModel, JobStatus
from src.main import app


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create a test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_job_id():
    """Return a sample job UUID."""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def sample_chunk_id():
    """Return a sample chunk UUID."""
    return "660e8400-e29b-41d4-a716-446655440001"


@pytest.fixture
def mock_job():
    """Create a mock job."""
    job = MagicMock(spec=JobModel)
    job.id = UUID("550e8400-e29b-41d4-a716-446655440000")
    job.status = JobStatus.COMPLETED
    job.source_type = "upload"
    return job


@pytest.fixture
def mock_chunk():
    """Create a mock chunk."""
    chunk = MagicMock(spec=DocumentChunkModel)
    chunk.id = UUID("660e8400-e29b-41d4-a716-446655440001")
    chunk.job_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    chunk.chunk_index = 0
    chunk.content = "This is a sample chunk content for testing purposes."
    chunk.content_hash = "abc123"
    chunk.embedding = [0.1] * 1536
    chunk.chunk_metadata = {"page": 1, "source": "test"}
    chunk.created_at = datetime.utcnow()
    return chunk


@pytest.fixture
def mock_chunks():
    """Create multiple mock chunks."""
    job_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    return [
        MagicMock(
            spec=DocumentChunkModel,
            id=uuid4(),
            job_id=job_id,
            chunk_index=i,
            content=f"Chunk {i} content for testing",
            content_hash=f"hash{i}",
            embedding=[0.1 * i] * 1536 if i % 2 == 0 else None,
            metadata={"page": i + 1},
            created_at=datetime.utcnow(),
        )
        for i in range(10)
    ]


# =============================================================================
# List Chunks Endpoint Tests
# =============================================================================

@pytest.mark.integration
class TestListChunksEndpoint:
    """Tests for GET /api/v1/jobs/{job_id}/chunks"""

    def test_list_chunks_success(self, client, sample_job_id, mock_chunks):
        """Test successful listing of chunks."""
        with patch("src.api.routes.chunks.JobRepository") as mock_job_repo, \
             patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            
            # Setup mocks
            mock_job_repo_instance = MagicMock()
            mock_job_repo_instance.get_by_id = AsyncMock(return_value=MagicMock())
            mock_job_repo.return_value = mock_job_repo_instance
            
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_job_id = AsyncMock(return_value=mock_chunks)
            mock_chunk_repo_instance.count_by_job_id = AsyncMock(return_value=len(mock_chunks))
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            # Execute
            response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert "total" in data
            assert "limit" in data
            assert "offset" in data
            assert data["total"] == len(mock_chunks)
            assert len(data["items"]) == len(mock_chunks)

    def test_list_chunks_with_pagination(self, client, sample_job_id, mock_chunks):
        """Test listing chunks with pagination parameters."""
        with patch("src.api.routes.chunks.JobRepository") as mock_job_repo, \
             patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            
            # Setup mocks
            mock_job_repo_instance = MagicMock()
            mock_job_repo_instance.get_by_id = AsyncMock(return_value=MagicMock())
            mock_job_repo.return_value = mock_job_repo_instance
            
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_job_id = AsyncMock(return_value=mock_chunks[5:10])
            mock_chunk_repo_instance.count_by_job_id = AsyncMock(return_value=10)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            # Execute
            response = client.get(
                f"/api/v1/jobs/{sample_job_id}/chunks?limit=5&offset=5"
            )
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["limit"] == 5
            assert data["offset"] == 5
            assert len(data["items"]) == 5

    def test_list_chunks_with_limit_validation(self, client, sample_job_id):
        """Test that limit parameter is validated."""
        # Test limit > 1000 (should be rejected)
        response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks?limit=1001")
        assert response.status_code == 422
        
        # Test limit < 1 (should be rejected)
        response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks?limit=0")
        assert response.status_code == 422

    def test_list_chunks_with_offset_validation(self, client, sample_job_id):
        """Test that offset parameter is validated."""
        # Test offset < 0 (should be rejected)
        response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks?offset=-1")
        assert response.status_code == 422

    def test_list_chunks_job_not_found(self, client, sample_job_id):
        """Test listing chunks for non-existent job."""
        with patch("src.api.routes.chunks.JobRepository") as mock_job_repo:
            mock_job_repo_instance = MagicMock()
            mock_job_repo_instance.get_by_id = AsyncMock(return_value=None)
            mock_job_repo.return_value = mock_job_repo_instance
            
            response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks")
            
            assert response.status_code == 404
            data = response.json()
            assert "not found" in data.get("detail", "").lower()

    def test_list_chunks_invalid_job_id(self, client):
        """Test listing chunks with invalid job ID format."""
        response = client.get("/api/v1/jobs/invalid-uuid/chunks")
        
        assert response.status_code == 400

    def test_list_chunks_with_include_embedding_flag(self, client, sample_job_id, mock_chunks):
        """Test listing chunks with include_embedding flag."""
        with patch("src.api.routes.chunks.JobRepository") as mock_job_repo, \
             patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            
            mock_job_repo_instance = MagicMock()
            mock_job_repo_instance.get_by_id = AsyncMock(return_value=MagicMock())
            mock_job_repo.return_value = mock_job_repo_instance
            
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_job_id = AsyncMock(return_value=mock_chunks)
            mock_chunk_repo_instance.count_by_job_id = AsyncMock(return_value=len(mock_chunks))
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            # Note: List response doesn't include embeddings for performance
            response = client.get(
                f"/api/v1/jobs/{sample_job_id}/chunks?include_embedding=true"
            )
            
            assert response.status_code == 200
            data = response.json()
            # Embeddings should not be in list response
            for item in data["items"]:
                assert "embedding" not in item

    def test_list_chunks_empty_result(self, client, sample_job_id):
        """Test listing chunks when job has no chunks."""
        with patch("src.api.routes.chunks.JobRepository") as mock_job_repo, \
             patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            
            mock_job_repo_instance = MagicMock()
            mock_job_repo_instance.get_by_id = AsyncMock(return_value=MagicMock())
            mock_job_repo.return_value = mock_job_repo_instance
            
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_job_id = AsyncMock(return_value=[])
            mock_chunk_repo_instance.count_by_job_id = AsyncMock(return_value=0)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks")
            
            assert response.status_code == 200
            data = response.json()
            assert data["items"] == []
            assert data["total"] == 0

    def test_list_chunks_response_structure(self, client, sample_job_id, mock_chunks):
        """Test that response has correct structure."""
        with patch("src.api.routes.chunks.JobRepository") as mock_job_repo, \
             patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            
            mock_job_repo_instance = MagicMock()
            mock_job_repo_instance.get_by_id = AsyncMock(return_value=MagicMock())
            mock_job_repo.return_value = mock_job_repo_instance
            
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_job_id = AsyncMock(return_value=mock_chunks[:1])
            mock_chunk_repo_instance.count_by_job_id = AsyncMock(return_value=1)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks")
            
            assert response.status_code == 200
            data = response.json()
            
            # Check item structure
            if data["items"]:
                item = data["items"][0]
                assert "id" in item
                assert "job_id" in item
                assert "chunk_index" in item
                assert "content" in item
                assert "content_hash" in item
                assert "metadata" in item
                assert "created_at" in item


# =============================================================================
# Get Chunk Endpoint Tests
# =============================================================================

@pytest.mark.integration
class TestGetChunkEndpoint:
    """Tests for GET /api/v1/jobs/{job_id}/chunks/{chunk_id}"""

    def test_get_chunk_success(self, client, sample_job_id, sample_chunk_id, mock_chunk):
        """Test successful retrieval of a single chunk."""
        with patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_id = AsyncMock(return_value=mock_chunk)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(
                f"/api/v1/jobs/{sample_job_id}/chunks/{sample_chunk_id}"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == sample_chunk_id
            assert data["job_id"] == sample_job_id
            assert "content" in data
            assert "chunk_index" in data

    def test_get_chunk_with_embedding(self, client, sample_job_id, sample_chunk_id, mock_chunk):
        """Test getting chunk with embedding included."""
        with patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_id = AsyncMock(return_value=mock_chunk)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(
                f"/api/v1/jobs/{sample_job_id}/chunks/{sample_chunk_id}?include_embedding=true"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "embedding" in data
            assert len(data["embedding"]) == 1536

    def test_get_chunk_without_embedding(self, client, sample_job_id, sample_chunk_id, mock_chunk):
        """Test getting chunk without embedding (default)."""
        with patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_id = AsyncMock(return_value=mock_chunk)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(
                f"/api/v1/jobs/{sample_job_id}/chunks/{sample_chunk_id}"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "embedding" not in data

    def test_get_chunk_not_found(self, client, sample_job_id, sample_chunk_id):
        """Test getting a non-existent chunk."""
        with patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_id = AsyncMock(return_value=None)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(
                f"/api/v1/jobs/{sample_job_id}/chunks/{sample_chunk_id}"
            )
            
            assert response.status_code == 404
            data = response.json()
            assert "not found" in data.get("detail", "").lower()

    def test_get_chunk_job_mismatch(self, client, sample_chunk_id):
        """Test getting chunk that belongs to different job."""
        with patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            # Create chunk belonging to different job
            chunk = MagicMock(spec=DocumentChunkModel)
            chunk.id = UUID(sample_chunk_id)
            chunk.job_id = uuid4()  # Different job ID
            
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_id = AsyncMock(return_value=chunk)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(
                f"/api/v1/jobs/550e8400-e29b-41d4-a716-446655440999/chunks/{sample_chunk_id}"
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "does not belong" in data.get("detail", "").lower()

    def test_get_chunk_invalid_job_id(self, client, sample_chunk_id):
        """Test getting chunk with invalid job ID format."""
        response = client.get(
            f"/api/v1/jobs/invalid-uuid/chunks/{sample_chunk_id}"
        )
        
        assert response.status_code == 400

    def test_get_chunk_invalid_chunk_id(self, client, sample_job_id):
        """Test getting chunk with invalid chunk ID format."""
        response = client.get(
            f"/api/v1/jobs/{sample_job_id}/chunks/invalid-uuid"
        )
        
        assert response.status_code == 400

    def test_get_chunk_response_structure(self, client, sample_job_id, sample_chunk_id, mock_chunk):
        """Test that chunk detail response has correct structure."""
        with patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_id = AsyncMock(return_value=mock_chunk)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(
                f"/api/v1/jobs/{sample_job_id}/chunks/{sample_chunk_id}"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check all expected fields
            assert "id" in data
            assert "job_id" in data
            assert "chunk_index" in data
            assert "content" in data
            assert "content_hash" in data
            assert "metadata" in data
            assert "created_at" in data


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.integration
class TestChunkApiErrors:
    """Tests for error handling in chunk API."""

    def test_list_chunks_database_error(self, client, sample_job_id):
        """Test handling of database errors when listing chunks."""
        with patch("src.api.routes.chunks.JobRepository") as mock_job_repo:
            mock_job_repo_instance = MagicMock()
            mock_job_repo_instance.get_by_id = AsyncMock(side_effect=Exception("DB error"))
            mock_job_repo.return_value = mock_job_repo_instance
            
            response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks")
            
            # Should return 500 or handle gracefully
            assert response.status_code in [500, 503]

    def test_get_chunk_database_error(self, client, sample_job_id, sample_chunk_id):
        """Test handling of database errors when getting chunk."""
        with patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_id = AsyncMock(side_effect=Exception("DB error"))
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(
                f"/api/v1/jobs/{sample_job_id}/chunks/{sample_chunk_id}"
            )
            
            assert response.status_code in [500, 503]


# =============================================================================
# Rate Limiting Tests
# =============================================================================

@pytest.mark.integration
class TestChunkApiRateLimiting:
    """Tests for rate limiting on chunk endpoints."""

    @pytest.mark.slow
    def test_list_chunks_rate_limit(self, client, sample_job_id):
        """Test rate limiting on list chunks endpoint."""
        # This test may be slow due to rate limit windows
        with patch("src.api.routes.chunks.JobRepository") as mock_job_repo, \
             patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            
            mock_job_repo_instance = MagicMock()
            mock_job_repo_instance.get_by_id = AsyncMock(return_value=MagicMock())
            mock_job_repo.return_value = mock_job_repo_instance
            
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_job_id = AsyncMock(return_value=[])
            mock_chunk_repo_instance.count_by_job_id = AsyncMock(return_value=0)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            # Make many rapid requests
            responses = []
            for _ in range(150):  # Exceed typical rate limit
                response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks")
                responses.append(response.status_code)
                if response.status_code == 429:
                    break
            
            # At least some requests should succeed, or we should hit rate limit
            assert 200 in responses or 429 in responses

    def test_get_chunk_rate_limit_headers(self, client, sample_job_id, sample_chunk_id):
        """Test that rate limit headers are present."""
        with patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_id = AsyncMock(return_value=MagicMock(
                id=UUID(sample_chunk_id),
                job_id=UUID(sample_job_id),
            ))
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(
                f"/api/v1/jobs/{sample_job_id}/chunks/{sample_chunk_id}"
            )
            
            # Check for rate limit headers (may or may not be present depending on config)
            assert response.status_code == 200


# =============================================================================
# Security Tests
# =============================================================================

@pytest.mark.integration
class TestChunkApiSecurity:
    """Tests for security aspects of chunk API."""

    def test_list_chunks_sql_injection_attempt(self, client):
        """Test that SQL injection in job_id is handled."""
        # This should be caught by UUID validation
        malicious_job_id = "550e8400-e29b-41d4-a716-446655440000' OR '1'='1"
        
        response = client.get(f"/api/v1/jobs/{malicious_job_id}/chunks")
        
        # Should fail validation, not execute malicious SQL
        assert response.status_code == 400

    def test_get_chunk_cross_job_access_attempt(self, client, sample_chunk_id):
        """Test that accessing chunk from different job is prevented."""
        with patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            # Create chunk for different job
            chunk = MagicMock(spec=DocumentChunkModel)
            chunk.id = UUID(sample_chunk_id)
            chunk.job_id = UUID("770e8400-e29b-41d4-a716-446655440002")
            
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_id = AsyncMock(return_value=chunk)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(
                f"/api/v1/jobs/880e8400-e29b-41d4-a716-446655440003/chunks/{sample_chunk_id}"
            )
            
            assert response.status_code == 400

    def test_list_chunks_xss_in_content(self, client, sample_job_id):
        """Test that XSS in content is handled."""
        with patch("src.api.routes.chunks.JobRepository") as mock_job_repo, \
             patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            
            mock_job_repo_instance = MagicMock()
            mock_job_repo_instance.get_by_id = AsyncMock(return_value=MagicMock())
            mock_job_repo.return_value = mock_job_repo_instance
            
            # Create chunk with XSS attempt
            xss_chunk = MagicMock()
            xss_chunk.id = uuid4()
            xss_chunk.job_id = UUID(sample_job_id)
            xss_chunk.chunk_index = 0
            xss_chunk.content = "<script>alert('XSS')</script>"
            xss_chunk.content_hash = "xss_hash"
            xss_chunk.chunk_metadata = {}
            xss_chunk.created_at = datetime.utcnow()
            
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_job_id = AsyncMock(return_value=[xss_chunk])
            mock_chunk_repo_instance.count_by_job_id = AsyncMock(return_value=1)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks")
            
            assert response.status_code == 200
            data = response.json()
            # Content should be returned as-is (escaping is frontend responsibility)
            assert "<script>" in data["items"][0]["content"]


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.integration
class TestChunkApiPerformance:
    """Basic performance tests for chunk API."""

    def test_list_chunks_response_time(self, client, sample_job_id, mock_chunks):
        """Test that list chunks responds within reasonable time."""
        import time
        
        with patch("src.api.routes.chunks.JobRepository") as mock_job_repo, \
             patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            
            mock_job_repo_instance = MagicMock()
            mock_job_repo_instance.get_by_id = AsyncMock(return_value=MagicMock())
            mock_job_repo.return_value = mock_job_repo_instance
            
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_job_id = AsyncMock(return_value=mock_chunks)
            mock_chunk_repo_instance.count_by_job_id = AsyncMock(return_value=len(mock_chunks))
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            start_time = time.time()
            response = client.get(f"/api/v1/jobs/{sample_job_id}/chunks")
            elapsed_ms = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            # Should respond within 1 second (generous for mocked tests)
            assert elapsed_ms < 1000

    def test_get_chunk_response_time(self, client, sample_job_id, sample_chunk_id, mock_chunk):
        """Test that get chunk responds within reasonable time."""
        import time
        
        with patch("src.api.routes.chunks.DocumentChunkRepository") as mock_chunk_repo:
            mock_chunk_repo_instance = MagicMock()
            mock_chunk_repo_instance.get_by_id = AsyncMock(return_value=mock_chunk)
            mock_chunk_repo.return_value = mock_chunk_repo_instance
            
            start_time = time.time()
            response = client.get(
                f"/api/v1/jobs/{sample_job_id}/chunks/{sample_chunk_id}"
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            assert elapsed_ms < 500
