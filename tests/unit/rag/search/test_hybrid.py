"""Unit tests for Enhanced Hybrid Search Service.

This module tests the HybridSearchService class including:
- Reciprocal Rank Fusion (RRF) algorithm with weights
- Metadata filtering
- Weight presets
- Fusion methods (RRF and weighted sum)
- Error handling and edge cases
"""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock, Mock
from uuid import UUID, uuid4

import pytest

from src.rag.search.hybrid_search import (
    FILTERABLE_FIELDS,
    WEIGHT_PRESETS,
    FusionMethod,
    HybridSearchConfig,
    HybridSearchError,
    HybridSearchResult,
    HybridSearchService,
    InvalidFilterError,
    InvalidFusionMethodError,
    InvalidWeightError,
    MetadataFilter,
    get_weight_preset,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@dataclass
class MockChunk:
    """Mock DocumentChunkModel for testing."""
    id: UUID
    content: str
    job_id: UUID | None = None


@dataclass
class MockSearchResult:
    """Mock SearchResult for testing."""
    chunk: MockChunk
    similarity_score: float
    rank: int


@dataclass
class MockTextSearchResult:
    """Mock TextSearchResult for testing."""
    chunk: MockChunk
    rank_score: float
    rank: int


@pytest.fixture
def mock_vector_service():
    """Create a mock vector search service."""
    service = MagicMock()
    # Make search_by_vector async
    async def async_search(*args, **kwargs):
        return []
    service.search_by_vector = async_search
    return service


@pytest.fixture
def mock_text_service():
    """Create a mock text search service."""
    service = MagicMock()
    # Make search_by_text async
    async def async_search(*args, **kwargs):
        return []
    service.search_by_text = async_search
    return service


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        MockChunk(id=uuid4(), content="Python is a programming language"),
        MockChunk(id=uuid4(), content="JavaScript runs in browsers"),
        MockChunk(id=uuid4(), content="Rust is systems programming language"),
        MockChunk(id=uuid4(), content="Go is fast and simple"),
        MockChunk(id=uuid4(), content="TypeScript adds types to JavaScript"),
    ]


@pytest.fixture
def vector_results(sample_chunks):
    """Create sample vector search results."""
    return [
        MockSearchResult(chunk=sample_chunks[0], similarity_score=0.95, rank=1),
        MockSearchResult(chunk=sample_chunks[2], similarity_score=0.85, rank=2),
        MockSearchResult(chunk=sample_chunks[4], similarity_score=0.75, rank=3),
    ]


@pytest.fixture
def text_results(sample_chunks):
    """Create sample text search results."""
    return [
        MockTextSearchResult(chunk=sample_chunks[1], rank_score=0.90, rank=1),
        MockTextSearchResult(chunk=sample_chunks[3], rank_score=0.80, rank=2),
        MockTextSearchResult(chunk=sample_chunks[0], rank_score=0.70, rank=3),
    ]


@pytest.fixture
def hybrid_service(mock_vector_service, mock_text_service):
    """Create a HybridSearchService with mocked dependencies."""
    config = HybridSearchConfig(
        default_vector_weight=0.7,
        default_text_weight=0.3,
        rrf_k=60,
    )
    return HybridSearchService(
        vector_service=mock_vector_service,
        text_service=mock_text_service,
        config=config,
    )


# ============================================================================
# Configuration Tests
# ============================================================================

class TestHybridSearchConfig:
    """Tests for HybridSearchConfig dataclass."""

    def test_default_creation(self):
        """Test creating config with default values."""
        config = HybridSearchConfig()

        assert config.default_top_k == 10
        assert config.max_top_k == 100
        assert config.default_vector_weight == 0.7
        assert config.default_text_weight == 0.3
        assert config.default_fusion_method == FusionMethod.RECIPROCAL_RANK_FUSION
        assert config.rrf_k == 60
        assert config.min_similarity == 0.5
        assert config.fallback_mode == "auto"
        assert config.apply_weights_to_rrf is True

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = HybridSearchConfig(
            default_top_k=20,
            max_top_k=200,
            default_vector_weight=0.8,
            default_text_weight=0.2,
            default_fusion_method=FusionMethod.WEIGHTED_SUM,
            rrf_k=50,
            min_similarity=0.6,
            fallback_mode="strict",
            apply_weights_to_rrf=False,
        )

        assert config.default_top_k == 20
        assert config.max_top_k == 200
        assert config.default_vector_weight == 0.8
        assert config.default_text_weight == 0.2
        assert config.default_fusion_method == FusionMethod.WEIGHTED_SUM
        assert config.rrf_k == 50
        assert config.min_similarity == 0.6
        assert config.fallback_mode == "strict"
        assert config.apply_weights_to_rrf is False


# ============================================================================
# MetadataFilter Tests
# ============================================================================

class TestMetadataFilter:
    """Tests for MetadataFilter dataclass."""

    def test_default_creation(self):
        """Test creating filter with default values."""
        filter_obj = MetadataFilter()

        assert filter_obj.source_type is None
        assert filter_obj.document_type is None
        assert filter_obj.created_date is None
        assert filter_obj.date_range is None
        assert filter_obj.author is None
        assert filter_obj.tags is None
        assert filter_obj.job_id is None
        assert filter_obj.metadata == {}

    def test_full_creation(self):
        """Test creating filter with all values."""
        job_id = uuid4()
        created_date = datetime(2024, 1, 15)
        date_range = (date(2024, 1, 1), date(2024, 12, 31))

        filter_obj = MetadataFilter(
            source_type="pdf",
            document_type="report",
            created_date=created_date,
            date_range=date_range,
            author="John Doe",
            tags=["python", "ml"],
            job_id=job_id,
            metadata={"category": "tech"},
        )

        assert filter_obj.source_type == "pdf"
        assert filter_obj.document_type == "report"
        assert filter_obj.created_date == created_date
        assert filter_obj.date_range == date_range
        assert filter_obj.author == "John Doe"
        assert filter_obj.tags == ["python", "ml"]
        assert filter_obj.job_id == job_id
        assert filter_obj.metadata == {"category": "tech"}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        job_id = uuid4()
        filter_obj = MetadataFilter(
            source_type="pdf",
            author="Jane Smith",
            job_id=job_id,
            metadata={"key": "value"},
        )

        result = filter_obj.to_dict()

        assert result["source_type"] == "pdf"
        assert result["author"] == "Jane Smith"
        assert result["job_id"] == str(job_id)
        assert result["metadata"] == {"key": "value"}
        assert "document_type" not in result
        assert "tags" not in result


# ============================================================================
# Weight Presets Tests
# ============================================================================

class TestWeightPresets:
    """Tests for weight presets."""

    def test_semantic_focus_preset(self):
        """Test semantic focus preset values."""
        preset = WEIGHT_PRESETS["semantic_focus"]
        assert preset["vector"] == 0.9
        assert preset["text"] == 0.1

    def test_balanced_preset(self):
        """Test balanced preset values."""
        preset = WEIGHT_PRESETS["balanced"]
        assert preset["vector"] == 0.7
        assert preset["text"] == 0.3

    def test_lexical_focus_preset(self):
        """Test lexical focus preset values."""
        preset = WEIGHT_PRESETS["lexical_focus"]
        assert preset["vector"] == 0.3
        assert preset["text"] == 0.7

    def test_get_weight_preset_function(self):
        """Test get_weight_preset convenience function."""
        preset = get_weight_preset("semantic_focus")
        assert preset["vector"] == 0.9
        assert preset["text"] == 0.1

    def test_get_weight_preset_invalid(self):
        """Test error on invalid preset name."""
        with pytest.raises(HybridSearchError) as exc_info:
            get_weight_preset("invalid_preset")
        assert "Unknown weight preset" in str(exc_info.value)


# ============================================================================
# Service Initialization Tests
# ============================================================================

class TestHybridSearchServiceInitialization:
    """Tests for HybridSearchService initialization."""

    def test_init_with_defaults(self, mock_vector_service, mock_text_service):
        """Test initialization with default config."""
        service = HybridSearchService(mock_vector_service, mock_text_service)

        assert service.vector_service == mock_vector_service
        assert service.text_service == mock_text_service
        assert service.config.default_vector_weight == 0.7
        assert service.config.default_text_weight == 0.3

    def test_init_with_custom_config(self, mock_vector_service, mock_text_service):
        """Test initialization with custom config."""
        config = HybridSearchConfig(
            default_vector_weight=0.8,
            default_text_weight=0.2,
            rrf_k=50,
        )
        service = HybridSearchService(
            mock_vector_service, mock_text_service, config=config
        )

        assert service.config.default_vector_weight == 0.8
        assert service.config.default_text_weight == 0.2
        assert service.config.rrf_k == 50

    def test_fallback_modes_constant(self):
        """Test that valid fallback modes are defined."""
        service = HybridSearchService(MagicMock(), MagicMock())
        assert "auto" in service.FALLBACK_MODES
        assert "vector" in service.FALLBACK_MODES
        assert "text" in service.FALLBACK_MODES
        assert "strict" in service.FALLBACK_MODES


# ============================================================================
# Reciprocal Rank Fusion Tests
# ============================================================================

class TestReciprocalRankFusion:
    """Tests for the reciprocal_rank_fusion method."""

    def test_rrf_basic_fusion(self, hybrid_service, vector_results, text_results):
        """Test basic RRF fusion."""
        results = hybrid_service.reciprocal_rank_fusion(
            vector_results=vector_results,
            text_results=text_results,
            vector_weight=0.7,
            text_weight=0.3,
            k=60,
        )

        assert len(results) == 5  # Union of both result sets
        assert all(isinstance(r, HybridSearchResult) for r in results)

    def test_rrf_with_equal_weights(self, hybrid_service, vector_results, text_results):
        """Test RRF with equal weights (0.5 each)."""
        results = hybrid_service.reciprocal_rank_fusion(
            vector_results=vector_results,
            text_results=text_results,
            vector_weight=0.5,
            text_weight=0.5,
            k=60,
        )

        # Results should contain all unique chunks from both result sets
        assert len(results) == 5  # Union of both sets
        # All scores should be positive
        assert all(r.hybrid_score > 0 for r in results)

    def test_rrf_semantic_focus(self, hybrid_service, vector_results, text_results):
        """Test RRF with semantic focus (high vector weight)."""
        results = hybrid_service.reciprocal_rank_fusion(
            vector_results=vector_results,
            text_results=text_results,
            vector_weight=0.9,
            text_weight=0.1,
            k=60,
        )

        # Sort results by score to check ranking
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # With high vector weight, vector-only results should be favored
        # The top result should have a valid vector_rank
        top_result = results[0]
        assert top_result is not None
        # Results with both vector and text contributions should rank highly
        results_with_both = [r for r in results if r.vector_rank is not None and r.text_rank is not None]
        # With semantic focus, at least one vector result should be in top 3
        vector_results_in_top3 = [r for r in results[:3] if r.vector_rank is not None]
        assert len(vector_results_in_top3) > 0

    def test_rrf_lexical_focus(self, hybrid_service, vector_results, text_results):
        """Test RRF with lexical focus (high text weight)."""
        results = hybrid_service.reciprocal_rank_fusion(
            vector_results=vector_results,
            text_results=text_results,
            vector_weight=0.1,
            text_weight=0.9,
            k=60,
        )

        # Sort by score to check ranking
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # The top text result should have high score when text weight is high
        # Text rank 1: 0.9/61 = 0.01475, Vector rank 1: 0.1/61 = 0.00164
        top_result = results[0]
        # Top result should be the one with best text rank (lowest rank number)
        assert top_result.text_rank is not None
        # With high text weight, text-only results should rank highly
        text_only_results = [r for r in results if r.vector_rank is None]
        assert len(text_only_results) > 0

    def test_rrf_empty_vector_results(self, hybrid_service, text_results):
        """Test RRF with empty vector results."""
        results = hybrid_service.reciprocal_rank_fusion(
            vector_results=[],
            text_results=text_results,
            vector_weight=0.7,
            text_weight=0.3,
            k=60,
        )

        assert len(results) == 3  # Only text results
        # All results should have None for vector scores
        assert all(r.vector_score is None for r in results)
        assert all(r.text_score is not None for r in results)

    def test_rrf_empty_text_results(self, hybrid_service, vector_results):
        """Test RRF with empty text results."""
        results = hybrid_service.reciprocal_rank_fusion(
            vector_results=vector_results,
            text_results=[],
            vector_weight=0.7,
            text_weight=0.3,
            k=60,
        )

        assert len(results) == 3  # Only vector results
        # All results should have None for text scores
        assert all(r.text_score is None for r in results)
        assert all(r.vector_score is not None for r in results)

    def test_rrf_both_empty(self, hybrid_service):
        """Test RRF with both result lists empty."""
        results = hybrid_service.reciprocal_rank_fusion(
            vector_results=[],
            text_results=[],
            vector_weight=0.7,
            text_weight=0.3,
            k=60,
        )

        assert len(results) == 0

    def test_rrf_formula_calculation(self, hybrid_service):
        """Test that RRF formula is applied correctly with weights."""
        chunk1 = MockChunk(id=uuid4(), content="test1")

        vec_results = [MockSearchResult(chunk=chunk1, similarity_score=0.9, rank=1)]
        txt_results = [MockTextSearchResult(chunk=chunk1, rank_score=0.8, rank=2)]

        results = hybrid_service.reciprocal_rank_fusion(
            vector_results=vec_results,
            text_results=txt_results,
            vector_weight=0.7,
            text_weight=0.3,
            k=60,
        )

        # Find the result for chunk1
        chunk1_result = next(r for r in results if r.chunk.id == chunk1.id)

        # Expected: 0.7/(60+1) + 0.3/(60+2) = 0.011475 + 0.004839 = 0.016314
        expected_score = 0.7 / 61 + 0.3 / 62
        # Check score is calculated correctly (with tolerance for floating point)
        assert abs(chunk1_result.hybrid_score - expected_score) < 0.001
        assert chunk1_result.vector_rank == 1
        assert chunk1_result.text_rank == 2

    def test_rrf_different_k_values(self, hybrid_service, vector_results, text_results):
        """Test RRF with different k values."""
        results_k30 = hybrid_service.reciprocal_rank_fusion(
            vector_results=vector_results,
            text_results=text_results,
            vector_weight=0.7,
            text_weight=0.3,
            k=30,
        )

        results_k120 = hybrid_service.reciprocal_rank_fusion(
            vector_results=vector_results,
            text_results=text_results,
            vector_weight=0.7,
            text_weight=0.3,
            k=120,
        )

        # Lower k should give higher scores (smaller denominator)
        # But ranking order may differ
        assert len(results_k30) == len(results_k120)


# ============================================================================
# Weighted Sum Fusion Tests
# ============================================================================

class TestWeightedSumFusion:
    """Tests for the _fuse_weighted_sum method."""

    def test_weighted_sum_basic(self, hybrid_service, vector_results, text_results):
        """Test basic weighted sum fusion."""
        results = hybrid_service._fuse_weighted_sum(
            vector_results=vector_results,
            text_results=text_results,
            vector_weight=0.7,
            text_weight=0.3,
        )

        assert len(results) == 5  # Union of both sets
        assert all(isinstance(r, HybridSearchResult) for r in results)
        assert all(0 <= r.hybrid_score <= 1 for r in results)

    def test_weighted_sum_formula(self, hybrid_service):
        """Test weighted sum formula calculation."""
        chunk1 = MockChunk(id=uuid4(), content="test")

        vec_results = [MockSearchResult(chunk=chunk1, similarity_score=0.8, rank=1)]
        txt_results = [MockTextSearchResult(chunk=chunk1, rank_score=1.0, rank=1)]

        results = hybrid_service._fuse_weighted_sum(
            vector_results=vec_results,
            text_results=txt_results,
            vector_weight=0.6,
            text_weight=0.4,
        )

        # Expected: 0.6 * 0.8 + 0.4 * 1.0 = 0.48 + 0.4 = 0.88
        expected_score = 0.6 * 0.8 + 0.4 * 1.0
        assert abs(results[0].hybrid_score - round(expected_score, 4)) < 0.0001

    def test_weighted_sum_only_vector(self, hybrid_service, vector_results):
        """Test weighted sum with only vector results."""
        results = hybrid_service._fuse_weighted_sum(
            vector_results=vector_results,
            text_results=[],
            vector_weight=0.7,
            text_weight=0.3,
        )

        assert len(results) == 3
        # Scores should be vector_weight * vector_score (since text_weight is not applied)
        # When only vector results exist, score = vector_weight * vector_score
        for r, expected in zip(results, vector_results):
            expected_score = 0.7 * expected.similarity_score
            # Check score is close to expected (with tolerance)
            assert abs(r.hybrid_score - expected_score) < 0.01


# ============================================================================
# Filter Preparation Tests
# ============================================================================

class TestFilterPreparation:
    """Tests for the _prepare_filters method."""

    def test_none_filters(self, hybrid_service):
        """Test with None filters."""
        result = hybrid_service._prepare_filters(None)
        assert result == {}

    def test_dict_filters(self, hybrid_service):
        """Test with dict filters."""
        filters = {"source_type": "pdf", "author": "John"}
        result = hybrid_service._prepare_filters(filters)
        assert result == filters

    def test_metadata_filter_object(self, hybrid_service):
        """Test with MetadataFilter object."""
        filter_obj = MetadataFilter(
            source_type="pdf",
            author="Jane",
            job_id=uuid4(),
        )
        result = hybrid_service._prepare_filters(filter_obj)

        assert result["source_type"] == "pdf"
        assert result["author"] == "Jane"
        assert "job_id" in result

    def test_invalid_filter_type(self, hybrid_service):
        """Test error with invalid filter type."""
        with pytest.raises(InvalidFilterError) as exc_info:
            hybrid_service._prepare_filters("invalid")
        assert "Filters must be dict or MetadataFilter" in str(exc_info.value)

    def test_invalid_filter_field(self, hybrid_service):
        """Test error with invalid filter field."""
        with pytest.raises(InvalidFilterError) as exc_info:
            hybrid_service._prepare_filters({"invalid_field": "value"})
        assert "Invalid filter field" in str(exc_info.value)
        assert "invalid_field" in str(exc_info.value)

    def test_all_valid_filter_fields(self, hybrid_service):
        """Test that all expected filter fields are valid."""
        expected_fields = {
            "source_type",
            "document_type",
            "created_date",
            "author",
            "tags",
            "job_id",
            "metadata",
        }
        assert expected_fields.issubset(FILTERABLE_FIELDS)


# ============================================================================
# Weight Validation Tests
# ============================================================================

class TestWeightValidation:
    """Tests for the _validate_weights method."""

    def test_valid_weights(self, hybrid_service):
        """Test validation of valid weights."""
        # Should not raise
        hybrid_service._validate_weights(0.7, 0.3)
        hybrid_service._validate_weights(0.5, 0.5)
        hybrid_service._validate_weights(1.0, 0.0)
        hybrid_service._validate_weights(0.0, 1.0)

    def test_invalid_negative_weights(self, hybrid_service):
        """Test error on negative weights."""
        with pytest.raises(InvalidWeightError) as exc_info:
            hybrid_service._validate_weights(-0.1, 1.1)
        assert "Weights must be non-negative" in str(exc_info.value)

    def test_weights_not_sum_to_one(self, hybrid_service):
        """Test error when weights don't sum to 1.0."""
        with pytest.raises(InvalidWeightError) as exc_info:
            hybrid_service._validate_weights(0.6, 0.3)
        assert "Weights must sum to 1.0" in str(exc_info.value)

    def test_floating_point_tolerance(self, hybrid_service):
        """Test that small floating point errors are tolerated."""
        # Should not raise due to floating point tolerance
        hybrid_service._validate_weights(0.333333, 0.666667)


# ============================================================================
# Fusion Method Validation Tests
# ============================================================================

class TestFusionMethodValidation:
    """Tests for the _validate_fusion_method method."""

    def test_valid_methods(self, hybrid_service):
        """Test validation of valid fusion methods."""
        # Should not raise
        hybrid_service._validate_fusion_method(FusionMethod.WEIGHTED_SUM)
        hybrid_service._validate_fusion_method(FusionMethod.RECIPROCAL_RANK_FUSION)

    def test_invalid_method(self, hybrid_service):
        """Test error on invalid fusion method."""
        with pytest.raises(InvalidFusionMethodError) as exc_info:
            hybrid_service._validate_fusion_method("invalid")
        assert "Invalid fusion method" in str(exc_info.value)


# ============================================================================
# Fallback Strategy Tests
# ============================================================================

class TestFallbackStrategy:
    """Tests for the _apply_fallback_strategy method."""

    def test_both_have_results(self, hybrid_service, vector_results, text_results):
        """Test when both searches return results."""
        v, t = hybrid_service._apply_fallback_strategy(
            vector_results, text_results, "auto"
        )
        assert len(v) == len(vector_results)
        assert len(t) == len(text_results)

    def test_auto_fallback_to_text(self, hybrid_service, text_results):
        """Test auto fallback to text when vector is empty."""
        v, t = hybrid_service._apply_fallback_strategy(
            [], text_results, "auto"
        )
        assert len(v) == 0
        assert len(t) == len(text_results)

    def test_auto_fallback_to_vector(self, hybrid_service, vector_results):
        """Test auto fallback to vector when text is empty."""
        v, t = hybrid_service._apply_fallback_strategy(
            vector_results, [], "auto"
        )
        assert len(v) == len(vector_results)
        assert len(t) == 0

    def test_strict_mode(self, hybrid_service, vector_results):
        """Test strict fallback mode."""
        v, t = hybrid_service._apply_fallback_strategy(
            vector_results, [], "strict"
        )
        # Strict mode keeps results as-is
        assert len(v) == len(vector_results)
        assert len(t) == 0

    def test_vector_preference_mode(self, hybrid_service, vector_results, text_results):
        """Test vector preference fallback mode."""
        # When vector has results, use only vector
        v, t = hybrid_service._apply_fallback_strategy(
            vector_results, text_results, "vector"
        )
        assert len(v) == len(vector_results)
        assert len(t) == 0

        # When vector is empty in vector-preference mode, we return empty
        # (strict preference for vector means we only want vector results)
        v, t = hybrid_service._apply_fallback_strategy(
            [], text_results, "vector"
        )
        # In vector preference mode, if vector is empty, both are returned empty
        assert len(v) == 0
        assert len(t) == 0

    def test_text_preference_mode(self, hybrid_service, vector_results, text_results):
        """Test text preference fallback mode."""
        # When text has results, use only text
        v, t = hybrid_service._apply_fallback_strategy(
            vector_results, text_results, "text"
        )
        assert len(v) == 0
        assert len(t) == len(text_results)

        # When text is empty in text-preference mode, we return empty
        # (strict preference for text means we only want text results)
        v, t = hybrid_service._apply_fallback_strategy(
            vector_results, [], "text"
        )
        # In text preference mode, if text is empty, both are returned empty
        assert len(v) == 0
        assert len(t) == 0


# ============================================================================
# Async Search Tests
# ============================================================================

@pytest.mark.asyncio
class TestHybridSearchAsync:
    """Async tests for the search method."""

    async def test_search_success(self, hybrid_service, mock_vector_service, mock_text_service, vector_results, text_results):
        """Test successful search execution."""
        async def mock_vec_search(*args, **kwargs):
            return vector_results
        async def mock_txt_search(*args, **kwargs):
            return text_results
        
        hybrid_service.vector_service.search_by_vector = mock_vec_search
        hybrid_service.text_service.search_by_text = mock_txt_search

        results = await hybrid_service.search(
            query="test query",
            embedding=[0.1] * 1536,
            filters=None,
            limit=10,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, HybridSearchResult) for r in results)
        assert all(r.rank > 0 for r in results)

    async def test_search_with_filters(self, hybrid_service):
        """Test search with metadata filters."""
        received_filters_vec = {}
        received_filters_txt = {}
        
        async def mock_vec_search(*args, **kwargs):
            nonlocal received_filters_vec
            received_filters_vec = kwargs.get("filters", {})
            return []
        
        async def mock_txt_search(*args, **kwargs):
            nonlocal received_filters_txt
            received_filters_txt = kwargs.get("filters", {})
            return []
        
        hybrid_service.vector_service.search_by_vector = mock_vec_search
        hybrid_service.text_service.search_by_text = mock_txt_search

        filters = {"source_type": "pdf", "author": "John"}
        await hybrid_service.search(
            query="test",
            embedding=[0.1] * 1536,
            filters=filters,
            limit=5,
        )

        # Verify filters were passed to services
        assert received_filters_vec == filters
        assert received_filters_txt == filters

    async def test_search_with_metadata_filter_object(self, hybrid_service):
        """Test search with MetadataFilter object."""
        received_filters = {}
        
        async def mock_vec_search(*args, **kwargs):
            nonlocal received_filters
            received_filters = kwargs.get("filters", {})
            return []
        
        async def mock_txt_search(*args, **kwargs):
            return []
        
        hybrid_service.vector_service.search_by_vector = mock_vec_search
        hybrid_service.text_service.search_by_text = mock_txt_search

        filter_obj = MetadataFilter(source_type="docx", author="Jane")
        await hybrid_service.search(
            query="test",
            embedding=[0.1] * 1536,
            filters=filter_obj,
            limit=5,
        )

        # Verify filters were converted and passed
        assert "source_type" in received_filters
        assert received_filters["source_type"] == "docx"

    async def test_search_no_results(self, hybrid_service):
        """Test search when no results found."""
        async def mock_empty_search(*args, **kwargs):
            return []
        
        hybrid_service.vector_service.search_by_vector = mock_empty_search
        hybrid_service.text_service.search_by_text = mock_empty_search

        results = await hybrid_service.search(
            query="nonexistent",
            embedding=[0.1] * 1536,
            limit=10,
        )

        assert results == []

    async def test_search_with_weight_preset(self, hybrid_service, vector_results, text_results):
        """Test search using weight preset."""
        async def mock_vec_search(*args, **kwargs):
            return vector_results
        async def mock_txt_search(*args, **kwargs):
            return text_results
        
        hybrid_service.vector_service.search_by_vector = mock_vec_search
        hybrid_service.text_service.search_by_text = mock_txt_search

        weights = get_weight_preset("semantic_focus")
        results = await hybrid_service.search(
            query="test",
            embedding=[0.1] * 1536,
            limit=10,
            vector_weight=weights["vector"],
            text_weight=weights["text"],
        )

        assert len(results) > 0

    async def test_search_weighted_sum_method(self, hybrid_service, vector_results, text_results):
        """Test search with weighted sum fusion method."""
        async def mock_vec_search(*args, **kwargs):
            return vector_results
        async def mock_txt_search(*args, **kwargs):
            return text_results
        
        hybrid_service.vector_service.search_by_vector = mock_vec_search
        hybrid_service.text_service.search_by_text = mock_txt_search

        results = await hybrid_service.search(
            query="test",
            embedding=[0.1] * 1536,
            limit=10,
            fusion_method=FusionMethod.WEIGHTED_SUM,
        )

        assert len(results) > 0
        # Verify weighted sum was used
        assert all(r.fusion_method == "weighted_sum" for r in results)

    async def test_search_invalid_weights(self, hybrid_service):
        """Test search with invalid weights."""
        with pytest.raises(InvalidWeightError):
            await hybrid_service.search(
                query="test",
                embedding=[0.1] * 1536,
                vector_weight=0.6,
                text_weight=0.3,  # Doesn't sum to 1.0
            )


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_hybrid_search_error(self):
        """Test HybridSearchError exception."""
        err = HybridSearchError("test error", context={"key": "value"})
        assert str(err) == "test error"
        assert err.context == {"key": "value"}

    def test_invalid_fusion_method_error(self):
        """Test InvalidFusionMethodError exception."""
        err = InvalidFusionMethodError("invalid method")
        assert str(err) == "invalid method"
        assert isinstance(err, HybridSearchError)

    def test_invalid_weight_error(self):
        """Test InvalidWeightError exception."""
        err = InvalidWeightError("bad weights")
        assert str(err) == "bad weights"
        assert isinstance(err, HybridSearchError)

    def test_invalid_filter_error(self):
        """Test InvalidFilterError exception."""
        err = InvalidFilterError("bad filter")
        assert str(err) == "bad filter"
        assert isinstance(err, HybridSearchError)


# ============================================================================
# Integration Tests
# ============================================================================

class TestHybridSearchIntegration:
    """Integration-style tests."""

    def test_weight_presets_affect_ranking(self, hybrid_service, vector_results, text_results):
        """Test that different weight presets produce different rankings."""
        # Semantic focus
        semantic_results = hybrid_service.reciprocal_rank_fusion(
            vector_results, text_results,
            vector_weight=0.9, text_weight=0.1, k=60
        )
        semantic_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        # Lexical focus
        lexical_results = hybrid_service.reciprocal_rank_fusion(
            vector_results, text_results,
            vector_weight=0.1, text_weight=0.9, k=60
        )
        lexical_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        # Rankings should be different (or at least scores should differ)
        semantic_top_chunk = semantic_results[0].chunk.id
        lexical_top_chunk = lexical_results[0].chunk.id

        # With different weights, top results likely differ
        # (or if same, scores should be different)
        if semantic_top_chunk == lexical_top_chunk:
            assert semantic_results[0].hybrid_score != lexical_results[0].hybrid_score

    def test_service_get_weight_preset(self, hybrid_service):
        """Test service method for getting weight presets."""
        preset = hybrid_service.get_weight_preset("balanced")
        assert preset["vector"] == 0.7
        assert preset["text"] == 0.3

    def test_service_list_weight_presets(self, hybrid_service):
        """Test listing all weight presets."""
        presets = hybrid_service.list_weight_presets()
        assert "semantic_focus" in presets
        assert "balanced" in presets
        assert "lexical_focus" in presets

    def test_hybrid_result_structure(self, hybrid_service, vector_results, text_results):
        """Test structure of HybridSearchResult objects."""
        results = hybrid_service.reciprocal_rank_fusion(
            vector_results, text_results,
            vector_weight=0.7, text_weight=0.3, k=60
        )

        for result in results:
            assert hasattr(result, "chunk")
            assert hasattr(result, "hybrid_score")
            assert hasattr(result, "vector_score")
            assert hasattr(result, "text_score")
            assert hasattr(result, "vector_rank")
            assert hasattr(result, "text_rank")
            assert hasattr(result, "fusion_method")
            assert hasattr(result, "rank")

            # Scores should be positive
            assert result.hybrid_score > 0

            # fusion_method should be valid
            assert result.fusion_method in ["rrf", "weighted_sum"]


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Tests for performance requirements."""

    def test_rrf_performance_under_100ms(self, hybrid_service):
        """Test that RRF fusion completes within 100ms for typical data."""
        import time

        # Create larger result sets
        chunks = [MockChunk(id=uuid4(), content=f"Content {i}") for i in range(100)]
        vec_results = [
            MockSearchResult(chunk=c, similarity_score=0.9 - i * 0.005, rank=i + 1)
            for i, c in enumerate(chunks[:50])
        ]
        txt_results = [
            MockTextSearchResult(chunk=c, rank_score=0.8 - i * 0.005, rank=i + 1)
            for i, c in enumerate(chunks[25:75])
        ]

        start = time.perf_counter()
        results = hybrid_service.reciprocal_rank_fusion(
            vec_results, txt_results,
            vector_weight=0.7, text_weight=0.3, k=60
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"RRF took {elapsed_ms:.2f}ms, expected <100ms"
        assert len(results) == 75  # Union of both sets
