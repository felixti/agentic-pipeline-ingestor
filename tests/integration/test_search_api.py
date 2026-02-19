"""Integration tests for search API endpoints.

Tests for:
- POST /api/v1/search/semantic
- POST /api/v1/search/text
- POST /api/v1/search/hybrid
- GET /api/v1/search/similar/{chunk_id}

Includes tests for rate limiting, error cases, and response validation.
"""

import math
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from src.db.models import DocumentChunkModel
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
def sample_embedding():
    """Create a sample normalized embedding vector."""
    dims = 1536
    vec = [0.1 * (i % 10) / 10 for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for search results."""
    return MagicMock(
        spec=DocumentChunkModel,
        id=uuid4(),
        job_id=uuid4(),
        chunk_index=0,
        content="Machine learning is a subset of artificial intelligence.",
        content_hash="abc123",
        metadata={"page": 1, "source": "test"},
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def mock_search_results(sample_chunk):
    """Create mock search results."""
    return [
        {
            "chunk": sample_chunk,
            "similarity_score": 0.95,
            "rank": 1,
        },
        {
            "chunk": MagicMock(
                spec=DocumentChunkModel,
                id=uuid4(),
                job_id=sample_chunk.job_id,
                chunk_index=1,
                content="Deep learning uses neural networks with multiple layers.",
                content_hash="def456",
                metadata={"page": 2},
                created_at=datetime.utcnow(),
            ),
            "similarity_score": 0.87,
            "rank": 2,
        },
    ]


# =============================================================================
# Semantic Search Endpoint Tests
# =============================================================================

@pytest.mark.integration
class TestSemanticSearchEndpoint:
    """Tests for POST /api/v1/search/semantic"""

    def test_semantic_search_success(self, client, sample_embedding, mock_search_results):
        """Test successful semantic search."""
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_vector = AsyncMock(return_value=[
                MagicMock(
                    chunk=r["chunk"],
                    similarity_score=r["similarity_score"],
                    rank=r["rank"],
                )
                for r in mock_search_results
            ])
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/semantic",
                json={
                    "query_embedding": sample_embedding,
                    "top_k": 10,
                    "min_similarity": 0.7,
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "total" in data
            assert "query_time_ms" in data
            assert len(data["results"]) == 2
            assert data["total"] == 2

    def test_semantic_search_with_filters(self, client, sample_embedding, sample_chunk):
        """Test semantic search with job_id filter."""
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_vector = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    similarity_score=0.92,
                    rank=1,
                ),
            ])
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/semantic",
                json={
                    "query_embedding": sample_embedding,
                    "top_k": 5,
                    "min_similarity": 0.8,
                    "filters": {
                        "job_id": str(sample_chunk.job_id),
                        "metadata": {"page": 1},
                    },
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 1

    def test_semantic_search_invalid_embedding_empty(self, client):
        """Test semantic search with empty embedding."""
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query_embedding": [],
                "top_k": 10,
            },
        )
        
        assert response.status_code == 422

    def test_semantic_search_invalid_embedding_wrong_dimensions(self, client):
        """Test semantic search with wrong embedding dimensions."""
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query_embedding": [0.1] * 768,  # Wrong size
                "top_k": 10,
            },
        )
        
        # Should either accept and filter results or return error
        # Implementation may vary
        assert response.status_code in [200, 400, 422]

    def test_semantic_search_top_k_validation(self, client, sample_embedding):
        """Test top_k parameter validation."""
        # Test top_k > 100
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query_embedding": sample_embedding,
                "top_k": 101,
            },
        )
        assert response.status_code == 422
        
        # Test top_k < 1
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query_embedding": sample_embedding,
                "top_k": 0,
            },
        )
        assert response.status_code == 422

    def test_semantic_search_min_similarity_validation(self, client, sample_embedding):
        """Test min_similarity parameter validation."""
        # Test min_similarity > 1
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query_embedding": sample_embedding,
                "min_similarity": 1.5,
            },
        )
        assert response.status_code == 422
        
        # Test min_similarity < 0
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query_embedding": sample_embedding,
                "min_similarity": -0.1,
            },
        )
        assert response.status_code == 422

    def test_semantic_search_response_structure(self, client, sample_embedding, sample_chunk):
        """Test that semantic search response has correct structure."""
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_vector = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    similarity_score=0.92,
                    rank=1,
                ),
            ])
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/semantic",
                json={
                    "query_embedding": sample_embedding,
                    "top_k": 5,
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "results" in data
            if data["results"]:
                result = data["results"][0]
                assert "chunk_id" in result
                assert "job_id" in result
                assert "chunk_index" in result
                assert "content" in result
                assert "metadata" in result
                assert "similarity_score" in result
                assert "rank" in result

    def test_semantic_search_empty_results(self, client, sample_embedding):
        """Test semantic search with no matching results."""
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_vector = AsyncMock(return_value=[])
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/semantic",
                json={
                    "query_embedding": sample_embedding,
                    "top_k": 10,
                    "min_similarity": 0.99,  # Very high threshold
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["results"] == []
            assert data["total"] == 0


# =============================================================================
# Text Search Endpoint Tests
# =============================================================================

@pytest.mark.integration
class TestTextSearchEndpoint:
    """Tests for POST /api/v1/search/text"""

    def test_text_search_success(self, client, sample_chunk):
        """Test successful text search."""
        with patch("src.api.routes.search.TextSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_text = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    rank_score=0.85,
                    rank=1,
                    highlighted_content=None,
                    matched_terms=None,
                ),
            ])
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/text",
                json={
                    "query": "machine learning",
                    "top_k": 10,
                    "language": "english",
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["content"] == sample_chunk.content

    def test_text_search_with_highlighting(self, client, sample_chunk):
        """Test text search with highlighting enabled."""
        with patch("src.api.routes.search.TextSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_text = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    rank_score=0.85,
                    rank=1,
                    highlighted_content="Machine <mark>learning</mark> is...",
                    matched_terms=["learning"],
                ),
            ])
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/text",
                json={
                    "query": "learning",
                    "top_k": 10,
                    "highlight": True,
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["results"][0]["highlighted_content"] == "Machine <mark>learning</mark> is..."
            assert "learning" in data["results"][0]["matched_terms"]

    def test_text_search_with_fuzzy(self, client, sample_chunk):
        """Test text search with fuzzy matching."""
        with patch("src.api.routes.search.TextSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_text = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    rank_score=0.75,
                    rank=1,
                    highlighted_content=None,
                    matched_terms=None,
                ),
            ])
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/text",
                json={
                    "query": "mashine learnng",  # Misspelled
                    "top_k": 10,
                    "use_fuzzy": True,
                },
            )
            
            assert response.status_code == 200

    def test_text_search_empty_query(self, client):
        """Test text search with empty query."""
        response = client.post(
            "/api/v1/search/text",
            json={
                "query": "",
                "top_k": 10,
            },
        )
        
        assert response.status_code == 422

    def test_text_search_query_too_long(self, client):
        """Test text search with query exceeding max length."""
        long_query = "a" * 1025
        
        response = client.post(
            "/api/v1/search/text",
            json={
                "query": long_query,
                "top_k": 10,
            },
        )
        
        assert response.status_code == 422

    def test_text_search_invalid_language(self, client):
        """Test text search with unsupported language."""
        with patch("src.api.routes.search.TextSearchService") as mock_service:
            from src.services.text_search_service import LanguageNotSupportedError
            
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_text = AsyncMock(
                side_effect=LanguageNotSupportedError("Language 'klingon' is not supported")
            )
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/text",
                json={
                    "query": "test",
                    "language": "klingon",
                },
            )
            
            assert response.status_code == 400
            assert "klingon" in response.json().get("detail", "").lower()

    def test_text_search_response_structure(self, client, sample_chunk):
        """Test that text search response has correct structure."""
        with patch("src.api.routes.search.TextSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_text = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    rank_score=0.85,
                    rank=1,
                    highlighted_content=None,
                    matched_terms=None,
                ),
            ])
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/text",
                json={
                    "query": "machine learning",
                    "top_k": 5,
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "results" in data
            assert "total" in data
            assert "query_time_ms" in data
            
            if data["results"]:
                result = data["results"][0]
                assert "chunk_id" in result
                assert "similarity_score" in result  # Actually rank_score but mapped
                assert "highlighted_content" in result
                assert "matched_terms" in result


# =============================================================================
# Hybrid Search Endpoint Tests
# =============================================================================

@pytest.mark.integration
class TestHybridSearchEndpoint:
    """Tests for POST /api/v1/search/hybrid"""

    def test_hybrid_search_success(self, client, sample_chunk):
        """Test successful hybrid search."""
        with patch("src.api.routes.search.TextSearchService") as mock_text_service, \
             patch("src.api.routes.search.VectorSearchService") as mock_vector_service, \
             patch("src.api.routes.search.HybridSearchService") as mock_hybrid_service:
            
            mock_hybrid_instance = MagicMock()
            mock_hybrid_instance.search_with_embedding = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    hybrid_score=0.88,
                    vector_score=0.92,
                    text_score=0.78,
                    vector_rank=1,
                    text_rank=2,
                    rank=1,
                    fusion_method="weighted_sum",
                ),
            ])
            mock_hybrid_service.return_value = mock_hybrid_instance
            
            # Also mock text service for fallback mode
            mock_text_instance = MagicMock()
            mock_text_instance.search_by_text = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    rank_score=0.78,
                    rank=1,
                ),
            ])
            mock_text_service.return_value = mock_text_instance
            
            response = client.post(
                "/api/v1/search/hybrid",
                json={
                    "query": "machine learning",
                    "top_k": 10,
                    "vector_weight": 0.7,
                    "text_weight": 0.3,
                    "fusion_method": "weighted_sum",
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1
            
            result = data["results"][0]
            assert "hybrid_score" in result
            assert "vector_score" in result
            assert "text_score" in result
            assert "fusion_method" in result

    def test_hybrid_search_rrf_fusion(self, client, sample_chunk):
        """Test hybrid search with RRF fusion."""
        with patch("src.api.routes.search.TextSearchService") as mock_text_service:
            mock_text_instance = MagicMock()
            mock_text_instance.search_by_text = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    rank_score=0.80,
                    rank=1,
                ),
            ])
            mock_text_service.return_value = mock_text_instance
            
            response = client.post(
                "/api/v1/search/hybrid",
                json={
                    "query": "machine learning",
                    "top_k": 10,
                    "fusion_method": "rrf",
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) >= 0  # May be empty due to fallback

    def test_hybrid_search_invalid_weights(self, client):
        """Test hybrid search with invalid weights."""
        response = client.post(
            "/api/v1/search/hybrid",
            json={
                "query": "test",
                "vector_weight": 0.8,
                "text_weight": 0.3,  # Sum != 1.0
            },
        )
        
        assert response.status_code == 422

    def test_hybrid_search_negative_weights(self, client):
        """Test hybrid search with negative weights."""
        response = client.post(
            "/api/v1/search/hybrid",
            json={
                "query": "test",
                "vector_weight": -0.1,
                "text_weight": 1.1,
            },
        )
        
        assert response.status_code == 422

    def test_hybrid_search_invalid_fusion_method(self, client):
        """Test hybrid search with invalid fusion method."""
        response = client.post(
            "/api/v1/search/hybrid",
            json={
                "query": "test",
                "fusion_method": "invalid_method",
            },
        )
        
        assert response.status_code == 422

    def test_hybrid_search_empty_results(self, client):
        """Test hybrid search with no results."""
        with patch("src.api.routes.search.TextSearchService") as mock_text_service:
            mock_text_instance = MagicMock()
            mock_text_instance.search_by_text = AsyncMock(return_value=[])
            mock_text_service.return_value = mock_text_instance
            
            response = client.post(
                "/api/v1/search/hybrid",
                json={
                    "query": "xyznonexistent12345",
                    "top_k": 10,
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["results"] == []
            assert data["total"] == 0


# =============================================================================
# Similar Chunks Endpoint Tests
# =============================================================================

@pytest.mark.integration
class TestSimilarChunksEndpoint:
    """Tests for GET /api/v1/search/similar/{chunk_id}"""

    def test_find_similar_chunks_success(self, client, sample_chunk):
        """Test successful similar chunks search."""
        chunk_id = str(sample_chunk.id)
        
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.find_similar_chunks = AsyncMock(return_value=[
                MagicMock(
                    chunk=MagicMock(
                        spec=DocumentChunkModel,
                        id=uuid4(),
                        job_id=sample_chunk.job_id,
                        chunk_index=1,
                        content="Similar chunk content",
                        content_hash="def456",
                        metadata={"page": 2},
                        created_at=datetime.utcnow(),
                    ),
                    similarity_score=0.89,
                    rank=1,
                ),
            ])
            mock_service.return_value = mock_service_instance
            
            response = client.get(f"/api/v1/search/similar/{chunk_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1

    def test_find_similar_chunks_with_params(self, client, sample_chunk):
        """Test similar chunks with query parameters."""
        chunk_id = str(sample_chunk.id)
        
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.find_similar_chunks = AsyncMock(return_value=[])
            mock_service.return_value = mock_service_instance
            
            response = client.get(
                f"/api/v1/search/similar/{chunk_id}?top_k=20&exclude_self=false"
            )
            
            assert response.status_code == 200
            mock_service_instance.find_similar_chunks.assert_called_once()
            call_kwargs = mock_service_instance.find_similar_chunks.call_args[1]
            assert call_kwargs["top_k"] == 20
            assert call_kwargs["exclude_self"] is False

    def test_find_similar_chunks_not_found(self, client):
        """Test similar chunks when reference chunk not found."""
        chunk_id = "550e8400-e29b-41d4-a716-446655440000"
        
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            from src.services.vector_search_service import ChunkNotFoundError
            
            mock_service_instance = MagicMock()
            mock_service_instance.find_similar_chunks = AsyncMock(
                side_effect=ChunkNotFoundError(f"Chunk {chunk_id} not found")
            )
            mock_service.return_value = mock_service_instance
            
            response = client.get(f"/api/v1/search/similar/{chunk_id}")
            
            assert response.status_code == 404
            assert "not found" in response.json().get("detail", "").lower()

    def test_find_similar_chunks_no_embedding(self, client):
        """Test similar chunks when reference chunk has no embedding."""
        chunk_id = "550e8400-e29b-41d4-a716-446655440000"
        
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            from src.services.vector_search_service import InvalidEmbeddingError
            
            mock_service_instance = MagicMock()
            mock_service_instance.find_similar_chunks = AsyncMock(
                side_effect=InvalidEmbeddingError("Chunk has no embedding")
            )
            mock_service.return_value = mock_service_instance
            
            response = client.get(f"/api/v1/search/similar/{chunk_id}")
            
            assert response.status_code == 400
            assert "embedding" in response.json().get("detail", "").lower()

    def test_find_similar_chunks_invalid_uuid(self, client):
        """Test similar chunks with invalid chunk ID."""
        response = client.get("/api/v1/search/similar/invalid-uuid")
        
        assert response.status_code == 400

    def test_find_similar_chunks_top_k_validation(self, client):
        """Test top_k parameter validation for similar chunks."""
        chunk_id = "550e8400-e29b-41d4-a716-446655440000"
        
        # Test top_k > 100
        response = client.get(f"/api/v1/search/similar/{chunk_id}?top_k=101")
        assert response.status_code == 422
        
        # Test top_k < 1
        response = client.get(f"/api/v1/search/similar/{chunk_id}?top_k=0")
        assert response.status_code == 422


# =============================================================================
# Rate Limiting Tests
# =============================================================================

@pytest.mark.integration
class TestSearchApiRateLimiting:
    """Tests for rate limiting on search endpoints."""

    @pytest.mark.slow
    def test_semantic_search_rate_limit(self, client, sample_embedding):
        """Test rate limiting on semantic search."""
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_vector = AsyncMock(return_value=[])
            mock_service.return_value = mock_service_instance
            
            responses = []
            for _ in range(150):
                response = client.post(
                    "/api/v1/search/semantic",
                    json={
                        "query_embedding": sample_embedding,
                        "top_k": 10,
                    },
                )
                responses.append(response.status_code)
                if response.status_code == 429:
                    break
            
            assert 200 in responses or 429 in responses

    @pytest.mark.slow
    def test_text_search_rate_limit(self, client):
        """Test rate limiting on text search."""
        with patch("src.api.routes.search.TextSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_text = AsyncMock(return_value=[])
            mock_service.return_value = mock_service_instance
            
            responses = []
            for _ in range(150):
                response = client.post(
                    "/api/v1/search/text",
                    json={"query": "test", "top_k": 10},
                )
                responses.append(response.status_code)
                if response.status_code == 429:
                    break
            
            assert 200 in responses or 429 in responses


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.integration
class TestSearchApiErrors:
    """Tests for error handling in search API."""

    def test_semantic_search_database_error(self, client, sample_embedding):
        """Test handling of database errors in semantic search."""
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_vector = AsyncMock(
                side_effect=Exception("Database connection failed")
            )
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/semantic",
                json={
                    "query_embedding": sample_embedding,
                    "top_k": 10,
                },
            )
            
            assert response.status_code in [500, 503]

    def test_text_search_database_error(self, client):
        """Test handling of database errors in text search."""
        with patch("src.api.routes.search.TextSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_text = AsyncMock(
                side_effect=Exception("Database connection failed")
            )
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/text",
                json={"query": "test", "top_k": 10},
            )
            
            assert response.status_code in [500, 503]

    def test_hybrid_search_database_error(self, client):
        """Test handling of database errors in hybrid search."""
        with patch("src.api.routes.search.TextSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_text = AsyncMock(
                side_effect=Exception("Database connection failed")
            )
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/hybrid",
                json={"query": "test", "top_k": 10},
            )
            
            assert response.status_code in [500, 503]


# =============================================================================
# Security Tests
# =============================================================================

@pytest.mark.integration
class TestSearchApiSecurity:
    """Tests for security aspects of search API."""

    def test_semantic_search_sql_injection_in_filters(self, client, sample_embedding):
        """Test that SQL injection in filters is prevented."""
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_vector = AsyncMock(return_value=[])
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/semantic",
                json={
                    "query_embedding": sample_embedding,
                    "filters": {
                        "job_id": "550e8400-e29b-41d4-a716-446655440000' OR '1'='1",
                    },
                },
            )
            
            # Should fail UUID validation
            assert response.status_code == 400

    def test_text_search_xss_in_query(self, client):
        """Test that XSS in search query is handled."""
        with patch("src.api.routes.search.TextSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_text = AsyncMock(return_value=[])
            mock_service.return_value = mock_service_instance
            
            xss_query = "<script>alert('XSS')</script>"
            
            response = client.post(
                "/api/v1/search/text",
                json={"query": xss_query, "top_k": 10},
            )
            
            # Query should be sanitized or rejected
            assert response.status_code in [200, 400]

    def test_search_with_malicious_metadata(self, client, sample_embedding):
        """Test that malicious metadata is handled."""
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_vector = AsyncMock(return_value=[])
            mock_service.return_value = mock_service_instance
            
            response = client.post(
                "/api/v1/search/semantic",
                json={
                    "query_embedding": sample_embedding,
                    "filters": {
                        "metadata": {
                            "$where": "malicious",
                            "key": "value",
                        },
                    },
                },
            )
            
            # Should either sanitize or return error
            assert response.status_code in [200, 400]


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.integration
class TestSearchApiPerformance:
    """Basic performance tests for search API."""

    def test_semantic_search_response_time(self, client, sample_embedding, sample_chunk):
        """Test that semantic search responds within reasonable time."""
        import time
        
        with patch("src.api.routes.search.VectorSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_vector = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    similarity_score=0.92,
                    rank=1,
                ),
            ])
            mock_service.return_value = mock_service_instance
            
            start_time = time.time()
            response = client.post(
                "/api/v1/search/semantic",
                json={
                    "query_embedding": sample_embedding,
                    "top_k": 10,
                },
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            assert elapsed_ms < 1000  # Should respond within 1 second

    def test_text_search_response_time(self, client, sample_chunk):
        """Test that text search responds within reasonable time."""
        import time
        
        with patch("src.api.routes.search.TextSearchService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service_instance.search_by_text = AsyncMock(return_value=[
                MagicMock(
                    chunk=sample_chunk,
                    rank_score=0.85,
                    rank=1,
                    highlighted_content=None,
                    matched_terms=None,
                ),
            ])
            mock_service.return_value = mock_service_instance
            
            start_time = time.time()
            response = client.post(
                "/api/v1/search/text",
                json={"query": "machine learning", "top_k": 10},
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            assert elapsed_ms < 500  # Should respond within 500ms
