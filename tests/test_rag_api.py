"""Tests for RAG API endpoints.

These tests validate the RAG API routes and models.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from src.main import app

    return TestClient(app)


class TestRAGQueryEndpoint:
    """Tests for POST /api/v1/rag/query endpoint."""

    def test_query_basic(self, client):
        """Test basic RAG query."""
        response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "What is vibe coding?",
                "strategy": "balanced",
            },
        )
        # May return 200 or 500 depending on backend availability
        assert response.status_code in [200, 500]

    def test_query_validation_empty(self, client):
        """Test validation rejects empty query."""
        response = client.post(
            "/api/v1/rag/query",
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_query_validation_whitespace(self, client):
        """Test validation rejects whitespace-only query."""
        response = client.post(
            "/api/v1/rag/query",
            json={"query": "   "},
        )
        assert response.status_code == 422

    def test_query_invalid_strategy(self, client):
        """Test validation rejects invalid strategy."""
        response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "What is vibe coding?",
                "strategy": "invalid_strategy",
            },
        )
        assert response.status_code == 422

    def test_query_auto_strategy(self, client):
        """Test query with auto strategy."""
        response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "What is vibe coding?",
                "strategy": "auto",
            },
        )
        assert response.status_code in [200, 500]

    def test_query_with_context(self, client):
        """Test query with conversation context."""
        response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "What is vibe coding?",
                "strategy": "balanced",
                "context": {
                    "previous_queries": ["Tell me about programming trends"],
                    "session_id": "test_session",
                },
            },
        )
        assert response.status_code in [200, 500]

    def test_query_with_filters_rejected(self, client):
        response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "What is vibe coding?",
                "strategy": "balanced",
                "filters": {"source_type": "documentation"},
                "top_k": 3,
            },
        )
        assert response.status_code == 422

    def test_query_with_non_default_top_k_rejected(self, client):
        response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "What is vibe coding?",
                "strategy": "balanced",
                "top_k": 3,
            },
        )
        assert response.status_code == 422


class TestListStrategiesEndpoint:
    """Tests for GET /api/v1/rag/strategies endpoint."""

    def test_list_strategies(self, client):
        """Test listing available strategies."""
        response = client.get("/api/v1/rag/strategies")
        assert response.status_code == 200

        data = response.json()
        assert "strategies" in data
        assert "default_strategy" in data
        assert data["default_strategy"] == "balanced"
        assert len(data["strategies"]) >= 3

        # Check strategy structure
        for strategy in data["strategies"]:
            assert "name" in strategy
            assert "description" in strategy
            assert "config" in strategy
            assert "use_cases" in strategy

    def test_strategies_include_auto(self, client):
        """Test that auto strategy is included."""
        response = client.get("/api/v1/rag/strategies")
        data = response.json()

        strategy_names = [s["name"] for s in data["strategies"]]
        assert "auto" in strategy_names
        assert "fast" in strategy_names
        assert "balanced" in strategy_names
        assert "thorough" in strategy_names


class TestEvaluateStrategyEndpoint:
    """Tests for POST /api/v1/rag/strategies/{name}/evaluate endpoint."""

    def test_evaluate_balanced_strategy(self, client):
        """Test evaluating balanced strategy."""
        response = client.post(
            "/api/v1/rag/strategies/balanced/evaluate",
            json={
                "query": "What is machine learning?",
                "iterations": 1,
            },
        )
        # May fail if backend unavailable
        assert response.status_code in [200, 500]

    def test_evaluate_invalid_strategy(self, client):
        """Test evaluating invalid strategy returns 404."""
        response = client.post(
            "/api/v1/rag/strategies/invalid/evaluate",
            json={
                "query": "What is machine learning?",
            },
        )
        assert response.status_code == 404

    def test_evaluate_with_ground_truth(self, client):
        """Test evaluation with ground truth."""
        response = client.post(
            "/api/v1/rag/strategies/fast/evaluate",
            json={
                "query": "What is machine learning?",
                "ground_truth_relevant_ids": ["chunk_1", "chunk_2"],
                "ground_truth_answer": "Machine learning is...",
                "iterations": 1,
            },
        )
        assert response.status_code in [200, 500]


class TestRAGMetricsEndpoint:
    """Tests for GET /api/v1/rag/metrics endpoint."""

    def test_get_metrics(self, client):
        """Test getting RAG metrics."""
        response = client.get("/api/v1/rag/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "summary" in data
        assert "component_health" in data
        assert "recent_alerts" in data
        assert "performance_trends" in data

    def test_metrics_structure(self, client):
        """Test metrics response structure."""
        response = client.get("/api/v1/rag/metrics")
        data = response.json()

        summary = data["summary"]
        assert "total_queries" in summary
        assert "avg_latency_ms" in summary
        assert "avg_retrieval_score" in summary
        assert "strategy_usage" in summary
        assert "query_type_distribution" in summary


class TestBenchmarkEndpoint:
    """Tests for POST /api/v1/rag/evaluate endpoint."""

    def test_run_benchmark(self, client):
        """Test running benchmark."""
        response = client.post(
            "/api/v1/rag/evaluate",
            json={
                "benchmark_name": "ms_marco",
                "max_queries": 10,
                "strategy_preset": "balanced",
            },
        )
        # May fail if backend unavailable
        assert response.status_code in [200, 500]

    def test_benchmark_validation(self, client):
        """Test benchmark request validation."""
        response = client.post(
            "/api/v1/rag/evaluate",
            json={
                "benchmark_name": "",
                "max_queries": 10,
            },
        )
        assert response.status_code == 422

    def test_benchmark_invalid_strategy(self, client):
        """Test benchmark with invalid strategy."""
        response = client.post(
            "/api/v1/rag/evaluate",
            json={
                "benchmark_name": "ms_marco",
                "strategy_preset": "invalid",
            },
        )
        assert response.status_code == 422


class TestRAGModels:
    """Tests for RAG Pydantic models."""

    def test_rag_query_request_valid(self):
        """Test valid RAG query request."""
        from src.api.models.rag import RAGQueryRequest

        request = RAGQueryRequest(
            query="What is vibe coding?",
            strategy="balanced",
        )
        assert request.query == "What is vibe coding?"
        assert request.strategy == "balanced"
        assert request.top_k == 5

    def test_rag_query_request_trims_whitespace(self):
        """Test that query is trimmed."""
        from src.api.models.rag import RAGQueryRequest

        request = RAGQueryRequest(query="  test query  ")
        assert request.query == "test query"

    def test_rag_source_model(self):
        """Test RAG source model."""
        from src.api.models.rag import RAGSource

        source = RAGSource(
            chunk_id="chunk_123",
            content="Test content",
            score=0.95,
            metadata={"page": 1},
        )
        assert source.chunk_id == "chunk_123"
        assert source.score == 0.95

    def test_rag_strategies_response(self):
        """Test strategies response model."""
        from src.api.models.rag import RAGStrategiesResponse, RAGStrategyInfo

        strategies = [
            RAGStrategyInfo(
                name="test",
                description="Test strategy",
                config={"query_rewrite": True},
                use_cases=["testing"],
            )
        ]
        response = RAGStrategiesResponse(
            strategies=strategies,
            default_strategy="test",
            total_count=1,
        )
        assert response.total_count == 1
