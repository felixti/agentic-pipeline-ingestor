"""Unit tests for Re-ranking Service.

This module tests the ReRanker class and related components including
model loading, scoring, batching, and error handling.
"""

import asyncio
import sys
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from src.rag.models import RankedChunk
from src.rag.strategies.reranking import (
    ALTERNATIVE_MODEL,
    DEFAULT_MODEL,
    FAST_MODEL,
    HIGH_PRECISION_MODEL,
    SUPPORTED_MODELS,
    Chunk,
    ReRanker,
    ReRankingError,
    ReRankingModelError,
    ReRankingTimeoutError,
)

# Create a mock module for sentence_transformers
mock_sentence_transformers = MagicMock()
mock_cross_encoder = MagicMock()
mock_sentence_transformers.CrossEncoder = mock_cross_encoder

# Patch the module before importing
try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    sys.modules["sentence_transformers"] = mock_sentence_transformers


# ============================================================================
# Chunk Model Tests
# ============================================================================

class TestChunk:
    """Tests for Chunk input model."""

    def test_default_creation(self):
        """Test creating chunk with default values."""
        chunk = Chunk(chunk_id="123", content="Test content")

        assert chunk.chunk_id == "123"
        assert chunk.content == "Test content"
        assert chunk.metadata == {}

    def test_full_creation(self):
        """Test creating chunk with all values."""
        chunk = Chunk(
            chunk_id="456",
            content="Full test content",
            metadata={"source": "test.pdf", "page": 1},
        )

        assert chunk.chunk_id == "456"
        assert chunk.content == "Full test content"
        assert chunk.metadata == {"source": "test.pdf", "page": 1}

    def test_with_uuid(self):
        """Test creating chunk with UUID as chunk_id."""
        chunk_id = uuid4()
        chunk = Chunk(chunk_id=str(chunk_id), content="Content with UUID")

        assert chunk.chunk_id == str(chunk_id)


# ============================================================================
# RankedChunk Model Tests
# ============================================================================

class TestRankedChunk:
    """Tests for RankedChunk result model."""

    def test_default_creation(self):
        """Test creating ranked chunk with valid values."""
        chunk_id = str(uuid4())
        ranked = RankedChunk(
            chunk_id=chunk_id,
            content="Test content",
            score=0.85,
            rank=1,
        )

        assert ranked.chunk_id == chunk_id
        assert ranked.content == "Test content"
        assert ranked.score == 0.85
        assert ranked.rank == 1
        assert ranked.metadata == {}

    def test_score_range_validation(self):
        """Test that scores must be in [0, 1] range."""
        chunk_id = str(uuid4())
        
        # Valid scores
        RankedChunk(chunk_id=chunk_id, content="test", score=0.0, rank=1)
        RankedChunk(chunk_id=chunk_id, content="test", score=1.0, rank=1)
        RankedChunk(chunk_id=chunk_id, content="test", score=0.5, rank=1)

        # Invalid scores should raise ValueError
        with pytest.raises(ValueError):
            RankedChunk(chunk_id=chunk_id, content="test", score=-0.1, rank=1)
        
        with pytest.raises(ValueError):
            RankedChunk(chunk_id=chunk_id, content="test", score=1.1, rank=1)

    def test_rank_validation(self):
        """Test that rank must be >= 1."""
        chunk_id = str(uuid4())
        
        # Valid rank
        RankedChunk(chunk_id=chunk_id, content="test", score=0.5, rank=1)
        RankedChunk(chunk_id=chunk_id, content="test", score=0.5, rank=100)

        # Invalid rank
        with pytest.raises(ValueError):
            RankedChunk(chunk_id=chunk_id, content="test", score=0.5, rank=0)
        
        with pytest.raises(ValueError):
            RankedChunk(chunk_id=chunk_id, content="test", score=0.5, rank=-1)

    def test_with_metadata(self):
        """Test creating ranked chunk with metadata."""
        chunk_id = str(uuid4())
        ranked = RankedChunk(
            chunk_id=chunk_id,
            content="Test",
            score=0.9,
            rank=1,
            metadata={"source": "doc.pdf", "page": 3, "confidence": "high"},
        )

        assert ranked.metadata["source"] == "doc.pdf"
        assert ranked.metadata["page"] == 3


# ============================================================================
# ReRanker Initialization Tests
# ============================================================================

class TestReRankerInitialization:
    """Tests for ReRanker initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        reranker = ReRanker()

        assert reranker.model_name == DEFAULT_MODEL
        assert reranker.batch_size == 8
        assert reranker.timeout_ms == 500
        assert reranker.device is None
        assert reranker._model is None  # Model loaded lazily

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        reranker = ReRanker(
            model_name=HIGH_PRECISION_MODEL,
            batch_size=16,
            timeout_ms=1000,
            device="cuda",
        )

        assert reranker.model_name == HIGH_PRECISION_MODEL
        assert reranker.batch_size == 16
        assert reranker.timeout_ms == 1000
        assert reranker.device == "cuda"

    def test_init_with_preset_key(self):
        """Test initialization with preset model key."""
        reranker = ReRanker(model_name="fast")
        assert reranker.model_name == FAST_MODEL

        reranker2 = ReRanker(model_name="high_precision")
        assert reranker2.model_name == HIGH_PRECISION_MODEL

        reranker3 = ReRanker(model_name="alternative")
        assert reranker3.model_name == ALTERNATIVE_MODEL

    def test_supported_models_constant(self):
        """Test that SUPPORTED_MODELS contains expected presets."""
        assert "default" in SUPPORTED_MODELS
        assert "high_precision" in SUPPORTED_MODELS
        assert "fast" in SUPPORTED_MODELS
        assert "alternative" in SUPPORTED_MODELS


# ============================================================================
# ReRanker from_settings Tests
# ============================================================================

class TestReRankerFromSettings:
    """Tests for ReRanker.from_settings factory method."""

    def test_from_settings_with_defaults(self):
        """Test creating from default settings."""
        mock_settings = MagicMock()
        mock_settings.reranking.models = {
            "default": "cross-encoder/test-model",
            "fast": "cross-encoder/fast",
        }
        mock_settings.reranking.batch_size = 16
        mock_settings.reranking.timeout_ms = 750

        reranker = ReRanker.from_settings(mock_settings)

        assert reranker.model_name == "cross-encoder/test-model"
        assert reranker.batch_size == 16
        assert reranker.timeout_ms == 750

    def test_from_settings_with_preset(self):
        """Test creating with specific model preset."""
        mock_settings = MagicMock()
        mock_settings.reranking.models = {
            "default": "cross-encoder/default",
            "fast": "cross-encoder/fast-model",
        }
        mock_settings.reranking.batch_size = 8
        mock_settings.reranking.timeout_ms = 500

        reranker = ReRanker.from_settings(mock_settings, model_preset="fast")

        assert reranker.model_name == "cross-encoder/fast-model"

    def test_from_settings_no_reranking_config(self):
        """Test creating when no reranking settings exist."""
        mock_settings = MagicMock()
        mock_settings.reranking = None

        reranker = ReRanker.from_settings(mock_settings)

        assert reranker.model_name == DEFAULT_MODEL
        assert reranker.batch_size == 8
        assert reranker.timeout_ms == 500


# ============================================================================
# ReRanker Model Loading Tests
# ============================================================================

class TestReRankerModelLoading:
    """Tests for model loading functionality."""

    @patch("src.rag.strategies.reranking.ReRanker._load_model")
    def test_ensure_model_loaded_when_none(self, mock_load):
        """Test that model is loaded when not already loaded."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        reranker = ReRanker()
        reranker._model = None

        result = reranker._ensure_model_loaded()

        assert result == mock_model
        mock_load.assert_called_once()

    def test_ensure_model_loaded_when_exists(self):
        """Test that existing model is returned without reloading."""
        reranker = ReRanker()
        mock_model = MagicMock()
        reranker._model = mock_model

        result = reranker._ensure_model_loaded()

        assert result == mock_model

    def test_load_model_success(self):
        """Test successful model loading."""
        mock_model = MagicMock()
        mock_cross_encoder.return_value = mock_model

        reranker = ReRanker(model_name="cross-encoder/test")
        # Clear any cached model
        reranker._model = None
        result = reranker._load_model()

        assert result == mock_model
        mock_cross_encoder.assert_called_once_with("cross-encoder/test")

    def test_load_model_with_device(self):
        """Test model loading with specific device."""
        mock_model = MagicMock()
        mock_cross_encoder.return_value = mock_model

        reranker = ReRanker(model_name="cross-encoder/test", device="cuda:0")
        # Clear any cached model
        reranker._model = None
        reranker._load_model()

        # Check the last call was with device
        assert mock_cross_encoder.call_count >= 1
        last_call = mock_cross_encoder.call_args
        assert last_call[0][0] == "cross-encoder/test"
        assert last_call[1].get("device") == "cuda:0"

    def test_load_model_import_error(self):
        """Test handling of missing sentence-transformers."""
        # Temporarily replace the module reference
        original_module = sys.modules.get("sentence_transformers")
        
        # Create a mock that raises ImportError when accessed
        class ImportRaiser:
            def __getattr__(self, name):
                raise ImportError("No module named sentence_transformers")
        
        sys.modules["sentence_transformers"] = ImportRaiser()  # type: ignore
        
        try:
            reranker = ReRanker()
            reranker._model = None
            with pytest.raises(ReRankingModelError) as exc_info:
                reranker._load_model()
            
            assert "sentence-transformers" in str(exc_info.value)
        finally:
            # Restore the original module
            if original_module:
                sys.modules["sentence_transformers"] = original_module
            else:
                sys.modules.pop("sentence_transformers", None)

    def test_load_model_error(self):
        """Test handling of model loading error."""
        mock_cross_encoder.side_effect = Exception("Download failed")

        reranker = ReRanker()
        reranker._model = None
        with pytest.raises(ReRankingModelError) as exc_info:
            reranker._load_model()
        
        assert "Failed to load model" in str(exc_info.value)
        
        # Reset the mock for other tests
        mock_cross_encoder.side_effect = None


# ============================================================================
# ReRanker Score Normalization Tests
# ============================================================================

class TestReRankerScoreNormalization:
    """Tests for score normalization."""

    def test_normalize_already_in_range(self):
        """Test scores already in [0, 1] are unchanged."""
        reranker = ReRanker()

        assert reranker._normalize_score(0.0) == 0.0
        assert reranker._normalize_score(0.5) == 0.5
        assert reranker._normalize_score(1.0) == 1.0
        assert reranker._normalize_score(0.75) == 0.75

    def test_normalize_positive_scores(self):
        """Test normalization of positive unbounded scores."""
        reranker = ReRanker()

        # High positive scores should approach 1
        assert reranker._normalize_score(5.0) > 0.99
        assert reranker._normalize_score(2.0) > 0.88

    def test_normalize_negative_scores(self):
        """Test normalization of negative unbounded scores."""
        reranker = ReRanker()

        # Negative scores should approach 0
        assert reranker._normalize_score(-5.0) < 0.01
        assert reranker._normalize_score(-2.0) < 0.12

    def test_normalize_zero(self):
        """Test normalization of zero - already in [0, 1] range so returned as-is."""
        reranker = ReRanker()

        # 0.0 is in the [0, 1] range, so returned as-is
        result = reranker._normalize_score(0.0)
        assert result == 0.0


# ============================================================================
# ReRanker rerank Tests
# ============================================================================

@pytest.mark.asyncio
class TestReRankerRerank:
    """Tests for the rerank method."""

    @pytest.fixture
    def reranker(self):
        """Create ReRanker with mocked model."""
        reranker = ReRanker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.6, 0.9]
        reranker._model = mock_model
        return reranker

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(chunk_id="1", content="Python is a programming language"),
            Chunk(chunk_id="2", content="JavaScript runs in browsers"),
            Chunk(chunk_id="3", content="Python is easy to learn"),
        ]

    async def test_rerank_success(self, reranker, sample_chunks):
        """Test successful re-ranking."""
        result = await reranker.rerank(
            query="What is Python?",
            chunks=sample_chunks,
            top_k=2,
        )

        assert len(result) == 2
        # Should be sorted by score descending
        assert result[0].score >= result[1].score
        # Ranks should be 1-indexed
        assert result[0].rank == 1
        assert result[1].rank == 2

    async def test_rerank_returns_top_k(self, reranker, sample_chunks):
        """Test that only top_k results are returned."""
        result = await reranker.rerank(
            query="test",
            chunks=sample_chunks,
            top_k=2,
        )

        assert len(result) == 2

    async def test_rerank_with_empty_query(self, reranker, sample_chunks):
        """Test handling of empty query."""
        result = await reranker.rerank(
            query="",
            chunks=sample_chunks,
            top_k=2,
        )

        assert result == []

    async def test_rerank_with_whitespace_query(self, reranker, sample_chunks):
        """Test handling of whitespace-only query."""
        result = await reranker.rerank(
            query="   ",
            chunks=sample_chunks,
            top_k=2,
        )

        assert result == []

    async def test_rerank_with_empty_chunks(self, reranker):
        """Test handling of empty chunks list."""
        result = await reranker.rerank(
            query="test",
            chunks=[],
            top_k=2,
        )

        assert result == []

    async def test_rerank_with_invalid_top_k(self, reranker, sample_chunks):
        """Test handling of invalid top_k value."""
        result = await reranker.rerank(
            query="test",
            chunks=sample_chunks,
            top_k=0,
        )

        # Should default to 5
        assert len(result) == min(5, len(sample_chunks))

    async def test_rerank_preserves_chunk_data(self, reranker, sample_chunks):
        """Test that chunk data is preserved in results."""
        result = await reranker.rerank(
            query="test",
            chunks=sample_chunks,
            top_k=3,
        )

        # Find the chunk with highest score (chunk 3 with score 0.9)
        top_result = result[0]
        assert top_result.chunk_id == "3"
        assert "Python is easy to learn" in top_result.content

    async def test_rerank_preserves_metadata(self, reranker):
        """Test that chunk metadata is preserved."""
        chunks = [
            Chunk(
                chunk_id="1",
                content="Test",
                metadata={"source": "test.pdf", "page": 5},
            ),
        ]
        reranker._model.predict.return_value = [0.9]

        result = await reranker.rerank(
            query="test",
            chunks=chunks,
            top_k=1,
        )

        assert result[0].metadata["source"] == "test.pdf"
        assert result[0].metadata["page"] == 5

    async def test_rerank_timeout(self, sample_chunks):
        """Test timeout handling."""
        reranker = ReRanker(timeout_ms=1)  # Very short timeout
        
        # Mock slow model
        mock_model = MagicMock()
        
        def slow_predict(*args, **kwargs):
            import time
            time.sleep(0.1)  # 100ms sleep, longer than timeout
            return [0.5]
        
        mock_model.predict = slow_predict
        reranker._model = mock_model

        with pytest.raises(ReRankingTimeoutError):
            await reranker.rerank(
                query="test",
                chunks=sample_chunks[:1],
                top_k=1,
            )

    async def test_rerank_sorts_by_score_descending(self, reranker, sample_chunks):
        """Test that results are sorted by score in descending order."""
        # Set scores that are not in order
        reranker._model.predict.return_value = [0.5, 0.9, 0.3]

        result = await reranker.rerank(
            query="test",
            chunks=sample_chunks,
            top_k=3,
        )

        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)
        # Verify the order: 0.9, 0.5, 0.3
        assert result[0].score == 0.9
        assert result[1].score == 0.5
        assert result[2].score == 0.3


# ============================================================================
# ReRanker Batch Processing Tests
# ============================================================================

@pytest.mark.asyncio
class TestReRankerBatch:
    """Tests for batch re-ranking."""

    @pytest.fixture
    def reranker(self):
        """Create ReRanker with mocked model."""
        reranker = ReRanker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        reranker._model = mock_model
        return reranker

    async def test_rerank_batch_success(self, reranker):
        """Test successful batch re-ranking."""
        queries = ["What is Python?", "What is JavaScript?"]
        chunks_per_query = [
            [Chunk(chunk_id="1", content="Python content")],
            [Chunk(chunk_id="2", content="JavaScript content")],
        ]

        results = await reranker.rerank_batch(
            queries=queries,
            chunks_per_query=chunks_per_query,
            top_k=1,
        )

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1

    async def test_rerank_batch_mismatched_lengths(self, reranker):
        """Test error on mismatched query and chunk list lengths."""
        queries = ["Query 1", "Query 2"]
        chunks_per_query = [[Chunk(chunk_id="1", content="test")]]  # Only one list

        with pytest.raises(ValueError) as exc_info:
            await reranker.rerank_batch(
                queries=queries,
                chunks_per_query=chunks_per_query,
                top_k=1,
            )
        
        assert "must match" in str(exc_info.value)

    async def test_rerank_batch_handles_errors(self, reranker):
        """Test that batch processing continues on individual errors."""
        queries = ["Query 1", "Query 2", "Query 3"]
        chunks_per_query = [
            [Chunk(chunk_id="1", content="test")],
            [Chunk(chunk_id="2", content="test")],
            [Chunk(chunk_id="3", content="test")],
        ]

        # Make second query fail
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Model error")
            return [0.8]
        
        reranker._model.predict = side_effect

        results = await reranker.rerank_batch(
            queries=queries,
            chunks_per_query=chunks_per_query,
            top_k=1,
        )

        assert len(results) == 3
        # First and third should succeed, second should be empty
        assert len(results[0]) == 1
        assert len(results[1]) == 0  # Failed
        assert len(results[2]) == 1


# ============================================================================
# ReRanker Batch Size Tests
# ============================================================================

class TestReRankerBatchSize:
    """Tests for batch size processing."""

    def test_score_pairs_respects_batch_size(self):
        """Test that scoring respects batch size."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        reranker = ReRanker(batch_size=2)
        reranker._model = mock_model

        chunks = [
            Chunk(chunk_id=str(i), content=f"Content {i}")
            for i in range(5)
        ]

        # Should process in batches of 2: [0,1], [2,3], [4]
        reranker._score_pairs_sync("query", chunks)

        # Verify predict was called 3 times (batches of 2, 2, 1)
        assert mock_model.predict.call_count == 3


# ============================================================================
# ReRanker Info and Health Tests
# ============================================================================

class TestReRankerInfoAndHealth:
    """Tests for get_model_info and health_check methods."""

    def test_get_model_info_not_loaded(self):
        """Test getting info when model not loaded."""
        reranker = ReRanker()
        info = reranker.get_model_info()

        assert info["model_name"] == DEFAULT_MODEL
        assert info["batch_size"] == 8
        assert info["timeout_ms"] == 500
        assert info["loaded"] is False

    def test_get_model_info_loaded(self):
        """Test getting info when model is loaded."""
        mock_model = MagicMock()
        mock_model.max_length = 512
        mock_model.num_labels = 1
        mock_cross_encoder.side_effect = None
        mock_cross_encoder.return_value = mock_model

        reranker = ReRanker()
        reranker._model = None  # Force reload
        reranker._load_model()
        info = reranker.get_model_info()

        assert info["loaded"] is True
        assert info["max_length"] == 512
        assert info["num_labels"] == 1

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when all components are healthy."""
        mock_cross_encoder.side_effect = None
        mock_cross_encoder.return_value = MagicMock()
        reranker = ReRanker()
        reranker._model = None  # Force reload
        reranker._load_model()

        health = await reranker.health_check()

        assert health["healthy"] is True
        assert health["components"]["sentence_transformers"]["healthy"] is True
        assert health["components"]["model"]["healthy"] is True

    @pytest.mark.asyncio
    async def test_health_check_missing_dependency(self):
        """Test health check when sentence-transformers is missing."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            reranker = ReRanker()
            health = await reranker.health_check()

            assert health["healthy"] is False
            assert health["components"]["sentence_transformers"]["available"] is False

    @pytest.mark.asyncio
    async def test_health_check_model_error(self):
        """Test health check when model fails to load."""
        mock_cross_encoder.side_effect = Exception("Load failed")
        reranker = ReRanker()

        health = await reranker.health_check()

        assert health["healthy"] is False
        assert "error" in health["components"]["model"]


# ============================================================================
# ReRanker Error Handling Tests
# ============================================================================

class TestReRankerErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_rerank_general_error(self):
        """Test handling of general errors during re-ranking."""
        reranker = ReRanker()
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Unexpected error")
        reranker._model = mock_model

        chunks = [Chunk(chunk_id="1", content="test")]

        with pytest.raises(ReRankingError) as exc_info:
            await reranker.rerank("query", chunks, top_k=1)
        
        assert "Re-ranking failed" in str(exc_info.value)


# ============================================================================
# Integration-style Tests
# ============================================================================

@pytest.mark.asyncio
class TestReRankerIntegration:
    """Integration-style tests with mocked model responses."""

    @pytest.fixture
    def reranker(self):
        """Create ReRanker with predictable mock responses."""
        reranker = ReRanker()
        mock_model = MagicMock()
        reranker._model = mock_model
        return reranker

    async def test_full_reranking_flow(self, reranker):
        """Test complete re-ranking flow."""
        # Simulate realistic scores for relevance
        reranker._model.predict.return_value = [0.3, 0.85, 0.6, 0.95, 0.4]

        chunks = [
            Chunk(
                chunk_id=str(i),
                content=f"Content about {topic}",
                metadata={"topic": topic}
            )
            for i, topic in enumerate([
                "Java", "Python programming", "C++", "Python basics", "Ruby"
            ])
        ]

        result = await reranker.rerank(
            query="What is Python?",
            chunks=chunks,
            top_k=3,
        )

        # Verify structure
        assert len(result) == 3
        for i, item in enumerate(result):
            assert isinstance(item, RankedChunk)
            assert item.rank == i + 1
            assert 0 <= item.score <= 1

        # Verify sorting (highest scores first)
        assert result[0].score == 0.95  # "Python basics"
        assert result[1].score == 0.85  # "Python programming"
        assert result[2].score == 0.6   # "C++"

    async def test_reranking_with_metadata_preservation(self, reranker):
        """Test that all metadata is preserved through re-ranking."""
        reranker._model.predict.return_value = [0.9]

        chunks = [
            Chunk(
                chunk_id="doc-123",
                content="Test content",
                metadata={
                    "source": "important.pdf",
                    "page": 42,
                    "section": "Introduction",
                    "author": "John Doe",
                },
            ),
        ]

        result = await reranker.rerank("query", chunks, top_k=1)

        metadata = result[0].metadata
        assert metadata["source"] == "important.pdf"
        assert metadata["page"] == 42
        assert metadata["section"] == "Introduction"
        assert metadata["author"] == "John Doe"

    async def test_performance_under_200ms_cached(self, reranker):
        """Test that cached/model-loaded re-ranking completes within 200ms."""
        import time

        reranker._model.predict.return_value = [0.5] * 20

        chunks = [
            Chunk(chunk_id=str(i), content=f"Content {i}")
            for i in range(20)
        ]

        start = time.perf_counter()
        result = await reranker.rerank("test query", chunks, top_k=5)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(result) == 5
        assert elapsed_ms < 200, f"Processing took {elapsed_ms:.2f}ms, expected <200ms"


# ============================================================================
# Exception Classes Tests
# ============================================================================

class TestReRankingExceptions:
    """Tests for custom exception classes."""

    def test_re_ranking_error_is_exception(self):
        """Test ReRankingError is an Exception."""
        err = ReRankingError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"

    def test_re_ranking_timeout_error_is_re_ranking_error(self):
        """Test ReRankingTimeoutError is a ReRankingError."""
        err = ReRankingTimeoutError("timeout")
        assert isinstance(err, ReRankingError)
        assert "timeout" in str(err)

    def test_re_ranking_model_error_is_re_ranking_error(self):
        """Test ReRankingModelError is a ReRankingError."""
        err = ReRankingModelError("model failed")
        assert isinstance(err, ReRankingError)
        assert "model failed" in str(err)

    def test_exception_chaining(self):
        """Test that exceptions can be chained."""
        original = ValueError("original")
        try:
            raise ReRankingError("wrapped") from original
        except ReRankingError as err:
            assert err.__cause__ == original
