"""Unit tests for HippoRAG Multi-Hop RAG API routes.

Tests the HippoRAG endpoints including retrieve, QA, and triple extraction.
All tests mock the HippoRAGDestination to avoid external dependencies.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.models.hipporag import (
    HippoRAGExtractTriplesResponse,
    HippoRAGQAResponse,
    HippoRAGQAResult,
    HippoRAGRetrievalResult,
    HippoRAGRetrieveResponse,
    HippoRAGTriple,
)
from src.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_hipporag_destination():
    """Create a mock HippoRAGDestination."""
    mock = MagicMock()
    mock._is_initialized = True
    mock._llm_provider = MagicMock()
    return mock


@pytest.mark.unit
class TestHippoRAGRetrieveEndpoint:
    """Tests for the HippoRAG retrieve endpoint."""

    def test_retrieve_endpoint_exists(self, client):
        """Test that the retrieve endpoint exists and returns 200."""
        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.retrieve = AsyncMock(return_value=[])
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/retrieve",
                json={
                    "queries": ["What is machine learning?"],
                    "num_to_retrieve": 10,
                },
            )

        assert response.status_code == status.HTTP_200_OK

    def test_retrieve_endpoint_returns_valid_response(self, client):
        """Test that retrieve returns a properly structured response."""
        from src.plugins.destinations.hipporag import RetrievalResult

        mock_result = RetrievalResult(
            query="What is machine learning?",
            passages=["Machine learning is a subset of AI."],
            scores=[0.95],
            source_documents=["doc1.pdf"],
            entities=["machine learning", "AI"],
        )

        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.retrieve = AsyncMock(return_value=[mock_result])
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/retrieve",
                json={
                    "queries": ["What is machine learning?"],
                    "num_to_retrieve": 10,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert "query_time_ms" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["query"] == "What is machine learning?"
        assert len(data["results"][0]["passages"]) == 1
        assert data["results"][0]["passages"][0] == "Machine learning is a subset of AI."
        assert data["results"][0]["scores"][0] == 0.95

    def test_retrieve_endpoint_with_multiple_queries(self, client):
        """Test retrieve with multiple queries."""
        from src.plugins.destinations.hipporag import RetrievalResult

        mock_results = [
            RetrievalResult(
                query="What is machine learning?",
                passages=["ML is a subset of AI."],
                scores=[0.9],
                source_documents=["doc1.pdf"],
                entities=["ML"],
            ),
            RetrievalResult(
                query="How does neural networks work?",
                passages=["Neural networks are inspired by the brain."],
                scores=[0.85],
                source_documents=["doc2.pdf"],
                entities=["neural networks"],
            ),
        ]

        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.retrieve = AsyncMock(return_value=mock_results)
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/retrieve",
                json={
                    "queries": [
                        "What is machine learning?",
                        "How does neural networks work?",
                    ],
                    "num_to_retrieve": 5,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["query"] == "What is machine learning?"
        assert data["results"][1]["query"] == "How does neural networks work?"

    def test_retrieve_endpoint_validation_error_missing_queries(self, client):
        """Test that retrieve returns 422 when queries is missing."""
        response = client.post(
            "/api/v1/hipporag/retrieve",
            json={
                "num_to_retrieve": 10,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_retrieve_endpoint_validation_error_empty_queries(self, client):
        """Test that retrieve returns 422 when queries is empty."""
        response = client.post(
            "/api/v1/hipporag/retrieve",
            json={
                "queries": [],
                "num_to_retrieve": 10,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_retrieve_endpoint_validation_error_num_to_retrieve_out_of_range(self, client):
        """Test that retrieve returns 422 for num_to_retrieve out of range."""
        response = client.post(
            "/api/v1/hipporag/retrieve",
            json={
                "queries": ["test query"],
                "num_to_retrieve": 100,  # Max is 50
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_retrieve_endpoint_service_unavailable(self, client):
        """Test that retrieve returns 503 when HippoRAG is unavailable."""
        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            from fastapi import HTTPException

            mock_get_dest.side_effect = HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "HIPPO_RAG_UNAVAILABLE",
                        "message": "HippoRAG service is not available",
                    }
                },
            )

            response = client.post(
                "/api/v1/hipporag/retrieve",
                json={
                    "queries": ["test query"],
                    "num_to_retrieve": 10,
                },
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "error" in data["detail"]
        assert data["detail"]["error"]["code"] == "HIPPO_RAG_UNAVAILABLE"


@pytest.mark.unit
class TestHippoRAGQAEndpoint:
    """Tests for the HippoRAG QA endpoint."""

    def test_qa_endpoint_exists(self, client):
        """Test that the QA endpoint exists and returns 200."""
        from src.plugins.destinations.hipporag import QAResult, RetrievalResult

        mock_result = QAResult(
            query="What is machine learning?",
            answer="Machine learning is a subset of AI.",
            sources=["doc1.pdf"],
            retrieval_results=RetrievalResult(query="What is machine learning?"),
            confidence=0.95,
        )

        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.rag_qa = AsyncMock(return_value=[mock_result])
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/qa",
                json={
                    "queries": ["What is machine learning?"],
                    "num_to_retrieve": 10,
                },
            )

        assert response.status_code == status.HTTP_200_OK

    def test_qa_endpoint_returns_valid_response(self, client):
        """Test that QA returns a properly structured response."""
        from src.plugins.destinations.hipporag import QAResult, RetrievalResult

        mock_result = QAResult(
            query="What is machine learning?",
            answer="Machine learning is a subset of AI that enables computers to learn from data.",
            sources=["doc1.pdf", "doc2.pdf"],
            retrieval_results=RetrievalResult(
                query="What is machine learning?",
                passages=["ML is a subset of AI."],
                scores=[0.92],
                source_documents=["doc1.pdf"],
                entities=["ML", "AI"],
            ),
            confidence=0.92,
        )

        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.rag_qa = AsyncMock(return_value=[mock_result])
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/qa",
                json={
                    "queries": ["What is machine learning?"],
                    "num_to_retrieve": 10,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert "total_tokens" in data
        assert "query_time_ms" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["query"] == "What is machine learning?"
        assert "Machine learning is a subset of AI" in data["results"][0]["answer"]
        assert data["results"][0]["confidence"] == 0.92
        assert data["results"][0]["sources"] == ["doc1.pdf", "doc2.pdf"]

    def test_qa_endpoint_with_multiple_questions(self, client):
        """Test QA with multiple questions."""
        from src.plugins.destinations.hipporag import QAResult, RetrievalResult

        mock_results = [
            QAResult(
                query="What is the capital of France?",
                answer="Paris",
                sources=["doc1.pdf"],
                retrieval_results=RetrievalResult(query="What is the capital of France?"),
                confidence=0.98,
            ),
            QAResult(
                query="Who invented the telephone?",
                answer="Alexander Graham Bell",
                sources=["doc2.pdf"],
                retrieval_results=RetrievalResult(query="Who invented the telephone?"),
                confidence=0.95,
            ),
        ]

        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.rag_qa = AsyncMock(return_value=mock_results)
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/qa",
                json={
                    "queries": [
                        "What is the capital of France?",
                        "Who invented the telephone?",
                    ],
                    "num_to_retrieve": 5,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["answer"] == "Paris"
        assert data["results"][1]["answer"] == "Alexander Graham Bell"

    def test_qa_endpoint_validation_error_missing_queries(self, client):
        """Test that QA returns 422 when queries is missing."""
        response = client.post(
            "/api/v1/hipporag/qa",
            json={
                "num_to_retrieve": 10,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_qa_endpoint_validation_error_empty_queries(self, client):
        """Test that QA returns 422 when queries is empty."""
        response = client.post(
            "/api/v1/hipporag/qa",
            json={
                "queries": [],
                "num_to_retrieve": 10,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_qa_endpoint_service_unavailable(self, client):
        """Test that QA returns 503 when HippoRAG is unavailable."""
        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            from fastapi import HTTPException

            mock_get_dest.side_effect = HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "HIPPO_RAG_UNAVAILABLE",
                        "message": "HippoRAG service is not available",
                    }
                },
            )

            response = client.post(
                "/api/v1/hipporag/qa",
                json={
                    "queries": ["test question"],
                    "num_to_retrieve": 10,
                },
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.unit
class TestHippoRAGExtractTriplesEndpoint:
    """Tests for the HippoRAG extract-triples endpoint."""

    def test_extract_triples_endpoint_exists(self, client):
        """Test that the extract-triples endpoint exists and returns 200."""
        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination._llm_provider = MagicMock()
            mock_destination._llm_provider.extract_triples = AsyncMock(
                return_value=[
                    ("Steve Jobs", "founded", "Apple Inc."),
                ]
            )
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/extract-triples",
                json={
                    "text": "Steve Jobs founded Apple Inc.",
                },
            )

        assert response.status_code == status.HTTP_200_OK

    def test_extract_triples_returns_valid_response(self, client):
        """Test that extract-triples returns properly structured response."""
        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination._llm_provider = MagicMock()
            mock_destination._llm_provider.extract_triples = AsyncMock(
                return_value=[
                    ("Steve Jobs", "founded", "Apple Inc."),
                    ("Steve Jobs", "served as", "CEO"),
                    ("Apple Inc.", "headquartered in", "Cupertino"),
                ]
            )
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/extract-triples",
                json={
                    "text": "Steve Jobs founded Apple Inc. He served as CEO. Apple Inc. is headquartered in Cupertino.",
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "triples" in data
        assert len(data["triples"]) == 3
        assert data["triples"][0]["subject"] == "Steve Jobs"
        assert data["triples"][0]["predicate"] == "founded"
        assert data["triples"][0]["object"] == "Apple Inc."

    def test_extract_triples_validation_error_missing_text(self, client):
        """Test that extract-triples returns 422 when text is missing."""
        response = client.post(
            "/api/v1/hipporag/extract-triples",
            json={},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_extract_triples_validation_error_empty_text(self, client):
        """Test that extract-triples returns 422 when text is empty."""
        response = client.post(
            "/api/v1/hipporag/extract-triples",
            json={
                "text": "",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_extract_triples_service_unavailable(self, client):
        """Test that extract-triples returns 503 when HippoRAG is unavailable."""
        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            from fastapi import HTTPException

            mock_get_dest.side_effect = HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "HIPPO_RAG_UNAVAILABLE",
                        "message": "HippoRAG service is not available",
                    }
                },
            )

            response = client.post(
                "/api/v1/hipporag/extract-triples",
                json={
                    "text": "Steve Jobs founded Apple Inc.",
                },
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_extract_triples_llm_unavailable(self, client):
        """Test that extract-triples returns 503 when LLM provider is unavailable."""
        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination._llm_provider = None  # LLM not available
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/extract-triples",
                json={
                    "text": "Steve Jobs founded Apple Inc.",
                },
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["detail"]["error"]["code"] == "LLM_UNAVAILABLE"


@pytest.mark.unit
class TestHippoRAGResponseModels:
    """Tests for HippoRAG response model validation."""

    def test_hipporag_retrieval_result_model(self):
        """Test HippoRAGRetrievalResult model creation."""
        result = HippoRAGRetrievalResult(
            query="What is AI?",
            passages=["AI is artificial intelligence."],
            scores=[0.95],
            source_documents=["doc1.pdf"],
            entities=["AI"],
        )

        assert result.query == "What is AI?"
        assert len(result.passages) == 1
        assert result.scores[0] == 0.95

    def test_hipporag_qa_result_model(self):
        """Test HippoRAGQAResult model creation."""
        retrieval = HippoRAGRetrievalResult(
            query="What is AI?",
            passages=["AI is artificial intelligence."],
            scores=[0.95],
            source_documents=["doc1.pdf"],
            entities=["AI"],
        )
        result = HippoRAGQAResult(
            query="What is AI?",
            answer="AI is artificial intelligence.",
            sources=["doc1.pdf"],
            confidence=0.95,
            retrieval_results=retrieval,
        )

        assert result.query == "What is AI?"
        assert result.answer == "AI is artificial intelligence."
        assert result.confidence == 0.95
        assert result.retrieval_results.query == "What is AI?"

    def test_hipporag_extract_triples_response_model(self):
        """Test HippoRAGExtractTriplesResponse model creation."""
        triples = [
            HippoRAGTriple(subject="A", predicate="is", object="B"),
            HippoRAGTriple(subject="C", predicate="has", object="D"),
        ]
        response = HippoRAGExtractTriplesResponse(triples=triples)

        assert len(response.triples) == 2
        assert response.triples[0].subject == "A"
        assert response.triples[0].predicate == "is"
        assert response.triples[0].object == "B"


@pytest.mark.unit
class TestHippoRAGEdgeCases:
    """Tests for edge cases and error handling."""

    def test_retrieve_with_no_results(self, client):
        """Test retrieve when no results are found."""
        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.retrieve = AsyncMock(return_value=[])
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/retrieve",
                json={
                    "queries": ["xyznonexistentquery123"],
                    "num_to_retrieve": 10,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["results"] == []

    def test_qa_with_empty_retrieval(self, client):
        """Test QA when retrieval returns no results."""
        from src.plugins.destinations.hipporag import QAResult, RetrievalResult

        mock_result = QAResult(
            query="Obscure question?",
            answer="I couldn't find relevant information to answer this question.",
            sources=[],
            retrieval_results=RetrievalResult(query="Obscure question?"),
            confidence=0.0,
        )

        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.rag_qa = AsyncMock(return_value=[mock_result])
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/qa",
                json={
                    "queries": ["Obscure question?"],
                    "num_to_retrieve": 10,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "couldn't find relevant information" in data["results"][0]["answer"]

    def test_extract_triples_with_no_triples(self, client):
        """Test extract-triples when no triples are found."""
        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination._llm_provider = MagicMock()
            mock_destination._llm_provider.extract_triples = AsyncMock(return_value=[])
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/extract-triples",
                json={
                    "text": "The weather is nice today.",
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["triples"] == []

    def test_retrieve_with_long_query(self, client):
        """Test retrieve with a very long query."""
        from src.plugins.destinations.hipporag import RetrievalResult

        long_query = "machine learning " * 100

        mock_result = RetrievalResult(
            query=long_query,
            passages=["Result passage."],
            scores=[0.9],
            source_documents=["doc1.pdf"],
            entities=["entity"],
        )

        with patch(
            "src.api.routes.hipporag._get_hipporag_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.retrieve = AsyncMock(return_value=[mock_result])
            mock_get_dest.return_value = mock_destination

            response = client.post(
                "/api/v1/hipporag/retrieve",
                json={
                    "queries": [long_query],
                    "num_to_retrieve": 10,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) == 1
