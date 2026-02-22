"""Unit tests for Agentic RAG Router.

This module tests the AgenticRAG class and related components including:
- Strategy selection based on query type
- Self-correction logic
- Metrics tracking
- Integration with query rewriter, classifier, and re-ranker
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.classification import STRATEGY_MATRIX, QueryClassifier
from src.rag.models import (
    QueryClassification,
    QueryRewriteResult,
    QueryType,
    RAGConfig,
    RAGMetrics,
    RAGResult,
    RankedChunk,
    Source,
)
from src.rag.router import (
    STRATEGY_PRESETS,
    AgenticRAG,
    AgenticRAGError,
    get_strategy_presets,
)
from src.rag.strategies.hyde import HyDERewriter
from src.rag.strategies.query_rewriting import QueryRewriter
from src.rag.strategies.reranking import ReRanker

# ============================================================================
# Strategy Presets Tests
# ============================================================================


class TestStrategyPresets:
    """Tests for strategy preset configurations."""

    def test_fast_preset(self):
        """Test fast strategy preset configuration."""
        preset = STRATEGY_PRESETS["fast"]
        assert preset["query_rewrite"] is True
        assert preset["hyde"] is False
        assert preset["reranking"] is False
        assert preset["hybrid_search"] is True

    def test_balanced_preset(self):
        """Test balanced strategy preset configuration."""
        preset = STRATEGY_PRESETS["balanced"]
        assert preset["query_rewrite"] is True
        assert preset["hyde"] is False
        assert preset["reranking"] is True
        assert preset["hybrid_search"] is True

    def test_thorough_preset(self):
        """Test thorough strategy preset configuration."""
        preset = STRATEGY_PRESETS["thorough"]
        assert preset["query_rewrite"] is True
        assert preset["hyde"] is True
        assert preset["reranking"] is True
        assert preset["hybrid_search"] is True

    def test_get_strategy_presets(self):
        """Test get_strategy_presets utility function."""
        presets = get_strategy_presets()
        assert "fast" in presets
        assert "balanced" in presets
        assert "thorough" in presets
        # Ensure it's a copy
        presets["fast"]["hyde"] = True
        assert STRATEGY_PRESETS["fast"]["hyde"] is False


# ============================================================================
# RAGConfig Model Tests
# ============================================================================


class TestRAGConfig:
    """Tests for RAGConfig Pydantic model."""

    def test_default_creation(self):
        """Test creating config with default values."""
        config = RAGConfig()

        assert config.query_rewrite is True
        assert config.hyde is False
        assert config.reranking is True
        assert config.hybrid_search is True
        assert config.strategy_preset == "balanced"

    def test_custom_creation(self):
        """Test creating config with custom values."""
        config = RAGConfig(
            query_rewrite=False,
            hyde=True,
            reranking=False,
            hybrid_search=False,
            strategy_preset="custom",
        )

        assert config.query_rewrite is False
        assert config.hyde is True
        assert config.reranking is False
        assert config.hybrid_search is False
        assert config.strategy_preset == "custom"

    def test_serialization(self):
        """Test JSON serialization."""
        config = RAGConfig(strategy_preset="fast")
        json_str = config.model_dump_json()
        assert "fast" in json_str
        assert "query_rewrite" in json_str


# ============================================================================
# Source Model Tests
# ============================================================================


class TestSource:
    """Tests for Source Pydantic model."""

    def test_creation(self):
        """Test creating a source."""
        source = Source(
            chunk_id="chunk-123",
            content="Test content",
            score=0.95,
            metadata={"page": 1},
        )

        assert source.chunk_id == "chunk-123"
        assert source.content == "Test content"
        assert source.score == 0.95
        assert source.metadata == {"page": 1}

    def test_score_bounds(self):
        """Test that score must be between 0 and 1."""
        # Valid values
        Source(chunk_id="1", content="test", score=0.0)
        Source(chunk_id="1", content="test", score=1.0)
        Source(chunk_id="1", content="test", score=0.5)

        # Invalid values
        with pytest.raises(ValueError):
            Source(chunk_id="1", content="test", score=-0.1)

        with pytest.raises(ValueError):
            Source(chunk_id="1", content="test", score=1.1)


# ============================================================================
# RAGMetrics Model Tests
# ============================================================================


class TestRAGMetrics:
    """Tests for RAGMetrics Pydantic model."""

    def test_creation(self):
        """Test creating metrics."""
        metrics = RAGMetrics(
            latency_ms=100.0,
            retrieval_score=0.85,
            classification_confidence=0.95,
        )

        assert metrics.latency_ms == 100.0
        assert metrics.retrieval_score == 0.85
        assert metrics.classification_confidence == 0.95
        assert metrics.tokens_used == 0  # default

    def test_full_creation(self):
        """Test creating metrics with all fields."""
        metrics = RAGMetrics(
            latency_ms=250.5,
            tokens_used=1000,
            retrieval_score=0.82,
            classification_confidence=0.95,
            rewrite_time_ms=50.0,
            retrieval_time_ms=120.0,
            reranking_time_ms=30.0,
            generation_time_ms=50.5,
            chunks_retrieved=20,
            chunks_used=5,
            self_correction_iterations=1,
        )

        assert metrics.latency_ms == 250.5
        assert metrics.tokens_used == 1000
        assert metrics.chunks_retrieved == 20
        assert metrics.self_correction_iterations == 1

    def test_score_bounds(self):
        """Test that scores must be between 0 and 1."""
        # Valid values
        RAGMetrics(
            latency_ms=100.0,
            retrieval_score=0.0,
            classification_confidence=1.0,
        )

        # Invalid retrieval_score
        with pytest.raises(ValueError):
            RAGMetrics(
                latency_ms=100.0,
                retrieval_score=1.5,
                classification_confidence=0.5,
            )

        # Invalid classification_confidence
        with pytest.raises(ValueError):
            RAGMetrics(
                latency_ms=100.0,
                retrieval_score=0.5,
                classification_confidence=-0.5,
            )


# ============================================================================
# RAGResult Model Tests
# ============================================================================


class TestRAGResult:
    """Tests for RAGResult Pydantic model."""

    def test_creation(self):
        """Test creating a result."""
        result = RAGResult(
            answer="Test answer",
            sources=[
                Source(chunk_id="1", content="Source 1", score=0.9),
            ],
            metrics=RAGMetrics(
                latency_ms=100.0,
                retrieval_score=0.85,
                classification_confidence=0.95,
            ),
        )

        assert result.answer == "Test answer"
        assert len(result.sources) == 1
        assert result.strategy_used == "balanced"  # default
        assert result.query_type == QueryType.FACTUAL  # default

    def test_with_sources(self):
        """Test result with multiple sources."""
        result = RAGResult(
            answer="Answer with sources",
            sources=[
                Source(chunk_id="1", content="Source 1", score=0.9),
                Source(chunk_id="2", content="Source 2", score=0.85),
            ],
            metrics=RAGMetrics(
                latency_ms=200.0,
                retrieval_score=0.88,
                classification_confidence=0.92,
            ),
            strategy_used="thorough",
            query_type=QueryType.ANALYTICAL,
        )

        assert len(result.sources) == 2
        assert result.strategy_used == "thorough"
        assert result.query_type == QueryType.ANALYTICAL


# ============================================================================
# AgenticRAG Initialization Tests
# ============================================================================


@pytest.mark.asyncio
class TestAgenticRAGInitialization:
    """Tests for AgenticRAG initialization."""

    async def test_init_with_required_services(self):
        """Test initialization with required services."""
        mock_rewriter = MagicMock(spec=QueryRewriter)
        mock_classifier = MagicMock(spec=QueryClassifier)

        router = AgenticRAG(
            query_rewriter=mock_rewriter,
            classifier=mock_classifier,
        )

        assert router.query_rewriter == mock_rewriter
        assert router.classifier == mock_classifier
        assert router.hyde_rewriter is None
        assert router.reranker is None
        assert router.quality_threshold == 0.7
        assert router.max_iterations == 3

    async def test_init_with_all_services(self):
        """Test initialization with all optional services."""
        mock_rewriter = MagicMock(spec=QueryRewriter)
        mock_classifier = MagicMock(spec=QueryClassifier)
        mock_hyde = MagicMock(spec=HyDERewriter)
        mock_reranker = MagicMock(spec=ReRanker)

        router = AgenticRAG(
            query_rewriter=mock_rewriter,
            classifier=mock_classifier,
            hyde_rewriter=mock_hyde,
            reranker=mock_reranker,
            quality_threshold=0.8,
            max_iterations=2,
        )

        assert router.hyde_rewriter == mock_hyde
        assert router.reranker == mock_reranker
        assert router.quality_threshold == 0.8
        assert router.max_iterations == 2


# ============================================================================
# AgenticRAG Strategy Selection Tests
# ============================================================================


@pytest.mark.asyncio
class TestAgenticRAGStrategySelection:
    """Tests for strategy selection logic."""

    @pytest.fixture
    def router(self):
        """Create AgenticRAG with mocked services."""
        mock_rewriter = MagicMock(spec=QueryRewriter)
        mock_classifier = MagicMock(spec=QueryClassifier)
        return AgenticRAG(
            query_rewriter=mock_rewriter,
            classifier=mock_classifier,
        )

    def test_select_fast_preset(self, router):
        """Test selecting fast preset."""
        classification = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="test",
        )

        config = router._select_strategy_config(classification, "fast")

        assert config.query_rewrite is True
        assert config.hyde is False
        assert config.reranking is False
        assert config.hybrid_search is True
        assert config.strategy_preset == "fast"

    def test_select_balanced_preset(self, router):
        """Test selecting balanced preset."""
        classification = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="test",
        )

        config = router._select_strategy_config(classification, "balanced")

        assert config.query_rewrite is True
        assert config.hyde is False
        assert config.reranking is True
        assert config.hybrid_search is True
        assert config.strategy_preset == "balanced"

    def test_select_thorough_preset(self, router):
        """Test selecting thorough preset."""
        classification = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="test",
        )

        config = router._select_strategy_config(classification, "thorough")

        assert config.query_rewrite is True
        assert config.hyde is True
        assert config.reranking is True
        assert config.hybrid_search is True
        assert config.strategy_preset == "thorough"

    def test_select_auto_factual(self, router):
        """Test auto selection for factual query."""
        classification = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="test",
        )

        config = router._select_strategy_config(classification, "auto")

        # Should use strategy matrix for factual
        expected = STRATEGY_MATRIX[QueryType.FACTUAL]
        assert config.hyde == expected["hyde"]
        assert config.reranking == expected["reranking"]
        assert config.strategy_preset == "auto_factual"

    def test_select_auto_analytical(self, router):
        """Test auto selection for analytical query."""
        classification = QueryClassification(
            query_type=QueryType.ANALYTICAL,
            confidence=0.9,
            reasoning="test",
        )

        config = router._select_strategy_config(classification, "auto")

        # Should use strategy matrix for analytical
        expected = STRATEGY_MATRIX[QueryType.ANALYTICAL]
        assert config.hyde == expected["hyde"]
        assert config.strategy_preset == "auto_analytical"


# ============================================================================
# AgenticRAG Quality Evaluation Tests
# ============================================================================


@pytest.mark.asyncio
class TestAgenticRAGQualityEvaluation:
    """Tests for retrieval quality evaluation."""

    @pytest.fixture
    def router(self):
        """Create AgenticRAG with mocked services."""
        mock_rewriter = MagicMock(spec=QueryRewriter)
        mock_classifier = MagicMock(spec=QueryClassifier)
        return AgenticRAG(
            query_rewriter=mock_rewriter,
            classifier=mock_classifier,
        )

    def test_evaluate_quality_empty_chunks(self, router):
        """Test quality evaluation with empty chunks."""
        quality = router._evaluate_quality([])
        assert quality == 0.0

    def test_evaluate_quality_single_chunk(self, router):
        """Test quality evaluation with single chunk."""
        chunks = [
            RankedChunk(chunk_id="1", content="Test", score=0.8, rank=1),
        ]
        quality = router._evaluate_quality(chunks)
        assert quality == 0.8

    def test_evaluate_quality_multiple_chunks(self, router):
        """Test quality evaluation averages top chunks with boost."""
        chunks = [
            RankedChunk(chunk_id="1", content="Test 1", score=0.9, rank=1),
            RankedChunk(chunk_id="2", content="Test 2", score=0.8, rank=2),
            RankedChunk(chunk_id="3", content="Test 3", score=0.7, rank=3),
        ]
        quality = router._evaluate_quality(chunks)
        # Average is (0.9 + 0.8 + 0.7) / 3 = 0.8, but boosted by 10% since all > 0.5
        expected_avg = (0.9 + 0.8 + 0.7) / 3
        expected_boosted = min(1.0, expected_avg * 1.1)
        assert quality == pytest.approx(expected_boosted, abs=1e-10)

    def test_evaluate_quality_ignores_extra_chunks(self, router):
        """Test that only top 5 chunks are considered."""
        chunks = [
            RankedChunk(chunk_id="1", content="Test", score=1.0, rank=1),
            RankedChunk(chunk_id="2", content="Test", score=1.0, rank=2),
            RankedChunk(chunk_id="3", content="Test", score=1.0, rank=3),
            RankedChunk(chunk_id="4", content="Test", score=1.0, rank=4),
            RankedChunk(chunk_id="5", content="Test", score=1.0, rank=5),
            RankedChunk(chunk_id="6", content="Test", score=0.1, rank=6),  # Should be ignored
        ]
        quality = router._evaluate_quality(chunks)
        assert quality == 1.0  # Average of top 5


# ============================================================================
# AgenticRAG Query Rewriting Tests
# ============================================================================


@pytest.mark.asyncio
class TestAgenticRAGQueryRewriting:
    """Tests for query rewriting in the router."""

    @pytest.fixture
    def mock_rewriter(self):
        """Create mock query rewriter."""
        rewriter = MagicMock(spec=QueryRewriter)
        rewriter.rewrite = AsyncMock()
        return rewriter

    @pytest.fixture
    def mock_classifier(self):
        """Create mock classifier."""
        return MagicMock(spec=QueryClassifier)

    @pytest.fixture
    def router(self, mock_rewriter, mock_classifier):
        """Create AgenticRAG with mocked services."""
        return AgenticRAG(
            query_rewriter=mock_rewriter,
            classifier=mock_classifier,
        )

    async def test_rewrite_with_standard_rewriter(self, router, mock_rewriter):
        """Test standard query rewriting."""
        config = RAGConfig(hyde=False)
        expected_result = QueryRewriteResult(
            search_rag=True,
            embedding_source_text="test keywords",
            llm_query="test query",
        )
        mock_rewriter.rewrite.return_value = expected_result

        result, elapsed = await router._rewrite_query("test query", config)

        assert result == expected_result
        assert elapsed >= 0
        mock_rewriter.rewrite.assert_called_once()

    async def test_rewrite_with_hyde(self, mock_rewriter, mock_classifier):
        """Test query rewriting with HyDE enabled."""
        mock_hyde = MagicMock(spec=HyDERewriter)
        mock_hyde.rewrite = AsyncMock()

        router = AgenticRAG(
            query_rewriter=mock_rewriter,
            classifier=mock_classifier,
            hyde_rewriter=mock_hyde,
        )

        config = RAGConfig(hyde=True)
        expected_result = QueryRewriteResult(
            search_rag=True,
            embedding_source_text="hypothetical document",
            llm_query="test query",
        )
        mock_hyde.rewrite.return_value = expected_result

        result, elapsed = await router._rewrite_query("test query", config)

        assert result == expected_result
        mock_hyde.rewrite.assert_called_once()

    async def test_rewrite_fallback_on_error(self, router, mock_rewriter):
        """Test fallback when rewriting fails."""
        config = RAGConfig(hyde=False)
        mock_rewriter.rewrite.side_effect = Exception("Rewrite failed")

        result, elapsed = await router._rewrite_query("@knowledgebase test", config)

        assert result.search_rag is True
        assert result.embedding_source_text == "@knowledgebase test"
        assert "Based on the provided context" in result.llm_query
        assert elapsed >= 0


# ============================================================================
# AgenticRAG Process Tests
# ============================================================================


@pytest.mark.asyncio
class TestAgenticRAGProcess:
    """Tests for the main process method."""

    @pytest.fixture
    def mock_services(self):
        """Create all mocked services."""
        mock_rewriter = MagicMock(spec=QueryRewriter)
        mock_classifier = MagicMock(spec=QueryClassifier)
        mock_hyde = MagicMock(spec=HyDERewriter)
        mock_reranker = MagicMock(spec=ReRanker)

        # Setup default returns
        mock_classifier.classify = AsyncMock(return_value=QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="Factual query",
            suggested_strategies=["query_rewrite", "reranking"],
        ))

        mock_rewriter.rewrite = AsyncMock(return_value=QueryRewriteResult(
            search_rag=True,
            embedding_source_text="test keywords",
            llm_query="Based on context, answer: test",
        ))

        mock_rewriter.health_check = AsyncMock(return_value={"healthy": True})
        mock_classifier.health_check = AsyncMock(return_value={"healthy": True})

        return {
            "rewriter": mock_rewriter,
            "classifier": mock_classifier,
            "hyde": mock_hyde,
            "reranker": mock_reranker,
        }

    @pytest.fixture
    def router(self, mock_services):
        """Create AgenticRAG with mocked services."""
        return AgenticRAG(
            query_rewriter=mock_services["rewriter"],
            classifier=mock_services["classifier"],
        )

    async def test_process_basic(self, router, mock_services):
        """Test basic process execution."""
        result = await router.process("What is AI?")

        assert isinstance(result, RAGResult)
        assert result.answer is not None
        assert isinstance(result.metrics, RAGMetrics)
        assert result.metrics.latency_ms >= 0

        # Verify classifier was called
        mock_services["classifier"].classify.assert_called_once()

    async def test_process_with_context(self, router, mock_services):
        """Test process with conversation context."""
        context = {
            "previous_queries": ["What is machine learning?"],
            "session_id": "sess_123",
        }

        result = await router.process("How does it work?", context=context)

        assert isinstance(result, RAGResult)
        # Verify context was passed to classifier
        mock_services["classifier"].classify.assert_called_once()
        call_args = mock_services["classifier"].classify.call_args
        # Check that context was passed (either as positional or keyword argument)
        # call_args.args = (query, context) when context is passed
        if len(call_args.args) >= 2:
            assert call_args.args[1] == context
        else:
            assert "context" in call_args.kwargs
            assert call_args.kwargs["context"] == context

    async def test_process_with_preset(self, router, mock_services):
        """Test process with specific strategy preset."""
        result = await router.process("What is AI?", strategy_preset="fast")

        assert isinstance(result, RAGResult)
        assert result.strategy_used == "fast"

    async def test_process_tracks_metrics(self, router, mock_services):
        """Test that process tracks all metrics."""
        result = await router.process("What is AI?")

        metrics = result.metrics
        assert metrics.latency_ms >= 0
        assert metrics.rewrite_time_ms >= 0
        assert metrics.retrieval_time_ms >= 0
        assert metrics.generation_time_ms >= 0
        assert metrics.classification_confidence == 0.95

    async def test_process_empty_query(self, router, mock_services):
        """Test process with empty query."""
        mock_services["classifier"].classify.return_value = QueryClassification(
            query_type=QueryType.VAGUE,
            confidence=0.5,
            reasoning="Empty query",
            suggested_strategies=["hyde", "reranking"],
        )

        result = await router.process("")

        assert isinstance(result, RAGResult)


# ============================================================================
# AgenticRAG Multi-hop Tests
# ============================================================================


@pytest.mark.asyncio
class TestAgenticRAGMultiHop:
    """Tests for multi-hop query processing."""

    @pytest.fixture
    def mock_services(self):
        """Create all mocked services."""
        mock_rewriter = MagicMock(spec=QueryRewriter)
        mock_classifier = MagicMock(spec=QueryClassifier)

        mock_classifier.classify = AsyncMock(return_value=QueryClassification(
            query_type=QueryType.MULTI_HOP,
            confidence=0.9,
            reasoning="Multi-hop query",
        ))

        mock_rewriter.rewrite = AsyncMock(return_value=QueryRewriteResult(
            search_rag=True,
            embedding_source_text="test",
            llm_query="test",
        ))

        return {
            "rewriter": mock_rewriter,
            "classifier": mock_classifier,
        }

    @pytest.fixture
    def router(self, mock_services):
        """Create AgenticRAG with mocked services."""
        return AgenticRAG(
            query_rewriter=mock_services["rewriter"],
            classifier=mock_services["classifier"],
        )

    async def test_process_multi_hop(self, router, mock_services):
        """Test multi-hop query processing."""
        result = await router.process_multi_hop(
            "What did the author of X say about Y?"
        )

        assert isinstance(result, RAGResult)
        assert result.query_type == QueryType.MULTI_HOP


# ============================================================================
# AgenticRAG Health Check Tests
# ============================================================================


@pytest.mark.asyncio
class TestAgenticRAGHealthCheck:
    """Tests for health check functionality."""

    @pytest.fixture
    def mock_services(self):
        """Create all mocked services."""
        mock_rewriter = MagicMock(spec=QueryRewriter)
        mock_classifier = MagicMock(spec=QueryClassifier)
        mock_hyde = MagicMock(spec=HyDERewriter)
        mock_reranker = MagicMock(spec=ReRanker)

        mock_rewriter.health_check = AsyncMock(return_value={"healthy": True})
        mock_classifier.health_check = AsyncMock(return_value={"healthy": True})
        mock_hyde.health_check = AsyncMock(return_value={"healthy": True})
        mock_reranker.health_check = AsyncMock(return_value={"healthy": True})

        return {
            "rewriter": mock_rewriter,
            "classifier": mock_classifier,
            "hyde": mock_hyde,
            "reranker": mock_reranker,
        }

    @pytest.fixture
    def router(self, mock_services):
        """Create AgenticRAG with all services."""
        return AgenticRAG(
            query_rewriter=mock_services["rewriter"],
            classifier=mock_services["classifier"],
            hyde_rewriter=mock_services["hyde"],
            reranker=mock_services["reranker"],
        )

    async def test_health_check_all_healthy(self, router, mock_services):
        """Test health check when all components are healthy."""
        health = await router.health_check()

        assert health["healthy"] is True
        assert health["components"]["query_rewriter"]["healthy"] is True
        assert health["components"]["classifier"]["healthy"] is True
        assert health["components"]["hyde_rewriter"]["healthy"] is True
        assert health["components"]["reranker"]["healthy"] is True

    async def test_health_check_rewriter_unhealthy(self, mock_services):
        """Test health check when rewriter is unhealthy."""
        mock_services["rewriter"].health_check = AsyncMock(
            return_value={"healthy": False, "error": "Service down"}
        )

        router = AgenticRAG(
            query_rewriter=mock_services["rewriter"],
            classifier=mock_services["classifier"],
        )

        health = await router.health_check()

        assert health["healthy"] is False
        assert health["components"]["query_rewriter"]["healthy"] is False

    async def test_health_check_without_optional_services(self, mock_services):
        """Test health check without optional services."""
        router = AgenticRAG(
            query_rewriter=mock_services["rewriter"],
            classifier=mock_services["classifier"],
        )

        health = await router.health_check()

        assert health["healthy"] is True
        assert health["components"]["hyde_rewriter"]["enabled"] is False
        assert health["components"]["reranker"]["enabled"] is False


# ============================================================================
# AgenticRAG Chunks to Sources Tests
# ============================================================================


@pytest.mark.asyncio
class TestAgenticRAGChunksToSources:
    """Tests for chunk to source conversion."""

    @pytest.fixture
    def router(self):
        """Create AgenticRAG with mocked services."""
        mock_rewriter = MagicMock(spec=QueryRewriter)
        mock_classifier = MagicMock(spec=QueryClassifier)
        return AgenticRAG(
            query_rewriter=mock_rewriter,
            classifier=mock_classifier,
        )

    def test_chunks_to_sources_empty(self, router):
        """Test conversion with empty chunks."""
        sources = router._chunks_to_sources([])
        assert sources == []

    def test_chunks_to_sources_single(self, router):
        """Test conversion with single chunk."""
        chunks = [
            RankedChunk(
                chunk_id="1",
                content="Test content",
                score=0.9,
                rank=1,
                metadata={"page": 1},
            ),
        ]
        sources = router._chunks_to_sources(chunks)

        assert len(sources) == 1
        assert sources[0].chunk_id == "1"
        assert sources[0].content == "Test content"
        assert sources[0].score == 0.9
        assert sources[0].metadata == {"page": 1}

    def test_chunks_to_sources_multiple(self, router):
        """Test conversion with multiple chunks."""
        chunks = [
            RankedChunk(chunk_id="1", content="Content 1", score=0.9, rank=1),
            RankedChunk(chunk_id="2", content="Content 2", score=0.8, rank=2),
        ]
        sources = router._chunks_to_sources(chunks)

        assert len(sources) == 2
        assert sources[0].chunk_id == "1"
        assert sources[1].chunk_id == "2"


# ============================================================================
# AgenticRAG Self-Correction Tests
# ============================================================================


@pytest.mark.asyncio
class TestAgenticRAGSelfCorrection:
    """Tests for self-correction logic."""

    @pytest.fixture
    def router(self):
        """Create AgenticRAG with mocked services."""
        mock_rewriter = MagicMock(spec=QueryRewriter)
        mock_classifier = MagicMock(spec=QueryClassifier)
        mock_hyde = MagicMock(spec=HyDERewriter)

        # Setup HyDE rewriter mock
        mock_hyde.rewrite = AsyncMock(return_value=QueryRewriteResult(
            search_rag=True,
            embedding_source_text="hypothetical document",
            llm_query="test",
        ))

        return AgenticRAG(
            query_rewriter=mock_rewriter,
            classifier=mock_classifier,
            hyde_rewriter=mock_hyde,
        )

    async def test_self_correction_enables_hyde(self, router):
        """Test that self-correction enables HyDE."""
        config = RAGConfig(hyde=False)
        chunks = []

        _, new_config, was_corrected = await router._self_correct(
            query="test",
            chunks=chunks,
            config=config,
            iteration=0,
        )

        assert was_corrected is True
        assert new_config.hyde is True

    async def test_self_correction_max_iterations(self, router):
        """Test that self-correction respects max iterations."""
        config = RAGConfig(hyde=True)  # HyDE already enabled
        chunks = []

        _, new_config, was_corrected = await router._self_correct(
            query="test",
            chunks=chunks,
            config=config,
            iteration=router.max_iterations,  # At max iterations
        )

        assert was_corrected is False

    async def test_self_correction_no_hyde_rewriter(self):
        """Test self-correction when HyDE rewriter is not available."""
        mock_rewriter = MagicMock(spec=QueryRewriter)
        mock_classifier = MagicMock(spec=QueryClassifier)

        router = AgenticRAG(
            query_rewriter=mock_rewriter,
            classifier=mock_classifier,
            hyde_rewriter=None,  # No HyDE
        )

        config = RAGConfig(hyde=False)
        chunks = []

        _, new_config, was_corrected = await router._self_correct(
            query="test",
            chunks=chunks,
            config=config,
            iteration=0,
        )

        # Should try other strategies since HyDE is not available
        assert was_corrected is True  # Increased retrieval is attempted
