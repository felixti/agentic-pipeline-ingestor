"""Unit tests for Cognee GraphRAG API routes.

Tests the Cognee endpoints including search, entity extraction, and statistics.
All tests mock the CogneeLocalDestination to avoid external dependencies.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.models.cognee import (
    CogneeEntity,
    CogneeExtractEntitiesResponse,
    CogneeRelationship,
    CogneeSearchResponse,
    CogneeSearchResult,
    CogneeStatsResponse,
)
from src.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_cognee_destination():
    """Create a mock CogneeLocalDestination."""
    mock = MagicMock()
    mock._is_initialized = True
    return mock


@pytest.mark.unit
class TestCogneeSearchEndpoint:
    """Tests for the Cognee search endpoint."""

    def test_search_endpoint_exists(self, client):
        """Test that the search endpoint exists and returns 200."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.search = AsyncMock(return_value=[])
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.post(
                    "/api/v1/cognee/search",
                    json={
                        "query": "What is machine learning?",
                        "search_type": "hybrid",
                        "top_k": 10,
                        "dataset_id": "default",
                    },
                )

        assert response.status_code == status.HTTP_200_OK

    def test_search_endpoint_returns_valid_response(self, client):
        """Test that search returns a properly structured response."""
        mock_result = MagicMock()
        mock_result.chunk_id = "chunk_001"
        mock_result.content = "Machine learning is a subset of AI."
        mock_result.score = 0.95
        mock_result.source_document = "ai_overview.pdf"
        mock_result.entities = ["machine learning", "AI"]

        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.search = AsyncMock(return_value=[mock_result])
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.post(
                    "/api/v1/cognee/search",
                    json={
                        "query": "What is machine learning?",
                        "search_type": "hybrid",
                        "top_k": 10,
                        "dataset_id": "default",
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert "search_type" in data
        assert "dataset_id" in data
        assert "query_time_ms" in data
        assert data["search_type"] == "hybrid"
        assert data["dataset_id"] == "default"
        assert len(data["results"]) == 1
        assert data["results"][0]["chunk_id"] == "chunk_001"
        assert data["results"][0]["score"] == 0.95

    def test_search_endpoint_with_vector_type(self, client):
        """Test search with vector search type."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.search = AsyncMock(return_value=[])
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.post(
                    "/api/v1/cognee/search",
                    json={
                        "query": "neural networks",
                        "search_type": "vector",
                        "top_k": 5,
                        "dataset_id": "test_dataset",
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["search_type"] == "vector"

    def test_search_endpoint_with_graph_type(self, client):
        """Test search with graph search type."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.search = AsyncMock(return_value=[])
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.post(
                    "/api/v1/cognee/search",
                    json={
                        "query": "entity relationships",
                        "search_type": "graph",
                        "top_k": 20,
                        "dataset_id": "graph_dataset",
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["search_type"] == "graph"

    def test_search_endpoint_validation_error_missing_query(self, client):
        """Test that search returns 422 when query is missing."""
        response = client.post(
            "/api/v1/cognee/search",
            json={
                "search_type": "hybrid",
                "top_k": 10,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_endpoint_validation_error_empty_query(self, client):
        """Test that search returns 422 when query is empty."""
        response = client.post(
            "/api/v1/cognee/search",
            json={
                "query": "",
                "search_type": "hybrid",
                "top_k": 10,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_endpoint_validation_error_invalid_search_type(self, client):
        """Test that search returns 422 for invalid search_type."""
        response = client.post(
            "/api/v1/cognee/search",
            json={
                "query": "test query",
                "search_type": "invalid_type",
                "top_k": 10,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_endpoint_validation_error_top_k_out_of_range(self, client):
        """Test that search returns 422 for top_k out of range."""
        response = client.post(
            "/api/v1/cognee/search",
            json={
                "query": "test query",
                "search_type": "hybrid",
                "top_k": 200,  # Max is 100
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_endpoint_service_unavailable(self, client):
        """Test that search returns 503 when Cognee is unavailable."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            from fastapi import HTTPException

            mock_get_dest.side_effect = HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "COGNEE_UNAVAILABLE",
                        "message": "Cognee service is not available",
                    }
                },
            )

            response = client.post(
                "/api/v1/cognee/search",
                json={
                    "query": "test query",
                    "search_type": "hybrid",
                    "top_k": 10,
                },
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "error" in data["detail"]
        assert data["detail"]["error"]["code"] == "COGNEE_UNAVAILABLE"


@pytest.mark.unit
class TestCogneeExtractEntitiesEndpoint:
    """Tests for the Cognee extract-entities endpoint."""

    def test_extract_entities_endpoint_exists(self, client):
        """Test that the extract-entities endpoint exists and returns 200."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.extract_entities = AsyncMock(
                return_value={
                    "entities": [],
                    "relationships": [],
                }
            )
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.post(
                    "/api/v1/cognee/extract-entities",
                    json={
                        "text": "Apple Inc. was founded by Steve Jobs.",
                        "dataset_id": "default",
                    },
                )

        assert response.status_code == status.HTTP_200_OK

    def test_extract_entities_returns_valid_response(self, client):
        """Test that extract-entities returns properly structured response."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.extract_entities = AsyncMock(
                return_value={
                    "entities": [
                        {"name": "Steve Jobs", "type": "PERSON", "description": "Co-founder"},
                        {"name": "Apple Inc.", "type": "ORGANIZATION", "description": "Tech company"},
                    ],
                    "relationships": [
                        {"source": "Steve Jobs", "target": "Apple Inc.", "type": "FOUNDED"},
                    ],
                }
            )
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.post(
                    "/api/v1/cognee/extract-entities",
                    json={
                        "text": "Apple Inc. was founded by Steve Jobs.",
                        "dataset_id": "default",
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "entities" in data
        assert "relationships" in data
        assert len(data["entities"]) == 2
        assert len(data["relationships"]) == 1
        assert data["entities"][0]["name"] == "Steve Jobs"
        assert data["entities"][0]["type"] == "PERSON"
        assert data["relationships"][0]["source"] == "Steve Jobs"
        assert data["relationships"][0]["type"] == "FOUNDED"

    def test_extract_entities_validation_error_missing_text(self, client):
        """Test that extract-entities returns 422 when text is missing."""
        response = client.post(
            "/api/v1/cognee/extract-entities",
            json={
                "dataset_id": "default",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_extract_entities_validation_error_empty_text(self, client):
        """Test that extract-entities returns 422 when text is empty."""
        response = client.post(
            "/api/v1/cognee/extract-entities",
            json={
                "text": "",
                "dataset_id": "default",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_extract_entities_service_unavailable(self, client):
        """Test that extract-entities returns 503 when Cognee is unavailable."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            from fastapi import HTTPException

            mock_get_dest.side_effect = HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "COGNEE_UNAVAILABLE",
                        "message": "Cognee service is not available",
                    }
                },
            )

            response = client.post(
                "/api/v1/cognee/extract-entities",
                json={
                    "text": "Apple Inc. was founded by Steve Jobs.",
                    "dataset_id": "default",
                },
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.unit
class TestCogneeStatsEndpoint:
    """Tests for the Cognee stats endpoint."""

    def test_stats_endpoint_exists(self, client):
        """Test that the stats endpoint exists and returns 200."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.get_stats = AsyncMock(
                return_value={
                    "dataset_id": "default",
                    "document_count": 100,
                    "chunk_count": 500,
                    "entity_count": 250,
                    "relationship_count": 750,
                    "graph_density": 0.35,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
            )
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.get("/api/v1/cognee/stats?dataset_id=default")

        assert response.status_code == status.HTTP_200_OK

    def test_stats_returns_valid_response(self, client):
        """Test that stats returns properly structured response."""
        last_updated = datetime.now(timezone.utc).isoformat()

        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.get_stats = AsyncMock(
                return_value={
                    "dataset_id": "test_dataset",
                    "document_count": 100,
                    "chunk_count": 500,
                    "entity_count": 250,
                    "relationship_count": 750,
                    "graph_density": 0.35,
                    "last_updated": last_updated,
                }
            )
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.get("/api/v1/cognee/stats?dataset_id=test_dataset")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "dataset_id" in data
        assert "document_count" in data
        assert "chunk_count" in data
        assert "entity_count" in data
        assert "relationship_count" in data
        assert "graph_density" in data
        assert "last_updated" in data
        assert data["dataset_id"] == "test_dataset"
        assert data["document_count"] == 100
        assert data["chunk_count"] == 500
        assert data["entity_count"] == 250
        assert data["relationship_count"] == 750
        assert data["graph_density"] == 0.35

    def test_stats_endpoint_default_dataset(self, client):
        """Test that stats uses default dataset when not specified."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.get_stats = AsyncMock(
                return_value={
                    "dataset_id": "default",
                    "document_count": 10,
                    "chunk_count": 50,
                    "entity_count": 25,
                    "relationship_count": 75,
                    "graph_density": 0.25,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
            )
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.get("/api/v1/cognee/stats")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["dataset_id"] == "default"

    def test_stats_service_unavailable(self, client):
        """Test that stats returns 503 when Cognee is unavailable."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            from fastapi import HTTPException

            mock_get_dest.side_effect = HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "COGNEE_UNAVAILABLE",
                        "message": "Cognee service is not available",
                    }
                },
            )

            response = client.get("/api/v1/cognee/stats?dataset_id=default")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.unit
class TestCogneeResponseModels:
    """Tests for Cognee response model validation."""

    def test_cognee_search_response_model(self):
        """Test CogneeSearchResponse model creation."""
        result = CogneeSearchResult(
            chunk_id="chunk_001",
            content="Test content",
            score=0.95,
            source_document="test.pdf",
            entities=["entity1", "entity2"],
        )
        response = CogneeSearchResponse(
            results=[result],
            search_type="hybrid",
            dataset_id="default",
            query_time_ms=150.5,
        )

        assert len(response.results) == 1
        assert response.results[0].chunk_id == "chunk_001"
        assert response.results[0].score == 0.95
        assert response.search_type == "hybrid"
        assert response.query_time_ms == 150.5

    def test_cognee_extract_entities_response_model(self):
        """Test CogneeExtractEntitiesResponse model creation."""
        entities = [
            CogneeEntity(name="Entity1", type="PERSON", description="A person"),
            CogneeEntity(name="Entity2", type="ORG", description="An org"),
        ]
        relationships = [
            CogneeRelationship(source="Entity1", target="Entity2", type="WORKS_AT"),
        ]
        response = CogneeExtractEntitiesResponse(
            entities=entities,
            relationships=relationships,
        )

        assert len(response.entities) == 2
        assert len(response.relationships) == 1
        assert response.entities[0].name == "Entity1"
        assert response.relationships[0].source == "Entity1"

    def test_cognee_stats_response_model(self):
        """Test CogneeStatsResponse model creation."""
        last_updated = datetime.now(timezone.utc)
        response = CogneeStatsResponse(
            dataset_id="test",
            document_count=100,
            chunk_count=500,
            entity_count=250,
            relationship_count=750,
            graph_density=0.35,
            last_updated=last_updated,
        )

        assert response.dataset_id == "test"
        assert response.document_count == 100
        assert response.chunk_count == 500
        assert response.entity_count == 250
        assert response.relationship_count == 750
        assert response.graph_density == 0.35


@pytest.mark.unit
class TestCogneeEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_with_no_results(self, client):
        """Test search when no results are found."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.search = AsyncMock(return_value=[])
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.post(
                    "/api/v1/cognee/search",
                    json={
                        "query": "xyznonexistentquery123",
                        "search_type": "hybrid",
                        "top_k": 10,
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["results"] == []

    def test_extract_entities_with_long_text(self, client):
        """Test extract-entities with a long text input."""
        long_text = "Apple Inc. " * 1000  # Long text

        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.extract_entities = AsyncMock(
                return_value={
                    "entities": [{"name": "Apple Inc.", "type": "ORG", "description": "Company"}],
                    "relationships": [],
                }
            )
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.post(
                    "/api/v1/cognee/extract-entities",
                    json={
                        "text": long_text,
                        "dataset_id": "default",
                    },
                )

        assert response.status_code == status.HTTP_200_OK

    def test_stats_with_empty_dataset(self, client):
        """Test stats for an empty dataset."""
        with patch(
            "src.api.routes.cognee._get_cognee_destination_async"
        ) as mock_get_dest:
            mock_destination = MagicMock()
            mock_destination.get_stats = AsyncMock(
                return_value={
                    "dataset_id": "empty_dataset",
                    "document_count": 0,
                    "chunk_count": 0,
                    "entity_count": 0,
                    "relationship_count": 0,
                    "graph_density": 0.0,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
            )
            mock_get_dest.return_value = mock_destination

            with patch(
                "src.api.routes.cognee._get_connection"
            ) as mock_get_conn:
                mock_get_conn.return_value = MagicMock()

                response = client.get("/api/v1/cognee/stats?dataset_id=empty_dataset")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["document_count"] == 0
        assert data["entity_count"] == 0
        assert data["graph_density"] == 0.0
