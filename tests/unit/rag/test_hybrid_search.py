"""Unit tests for Enhanced Hybrid Search Strategy.

This module tests the EnhancedHybridSearch class and related components including
RRF fusion, weighted sum fusion, metadata filtering, weight presets, and query expansion.
"""

import asyncio
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

# Import models first
from src.db.models import DocumentChunkModel
from src.rag.models import (
    FusionMethod,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResultItem,
    WeightPreset,
)
from src.rag.strategies.hybrid_search import (
    EnhancedHybridSearch,
    HybridSearchError,
    HybridSearchResult,
    InvalidFilterError,
    InvalidFusionMethodError,
    InvalidPresetError,
    InvalidWeightError,
    MetadataFilter,
    QueryExpander,
    QueryExpansionResult,
    reciprocal_rank_fusion,
)
from src.rag.strategies.hybrid_search import (
    WeightPreset as StrategyWeightPreset,
)
from src.services.text_search_service import TextSearchResult, TextSearchService
from src.services.vector_search_service import SearchResult, VectorSearchService

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_service():
    """Create a mock VectorSearchService."""
    service = MagicMock(spec=VectorSearchService)
    # Use a coroutine mock instead of Future
    async def mock_search(*args, **kwargs):
        return []
    service.search_by_vector = mock_search
    return service


@pytest.fixture
def mock_text_service():
    """Create a mock TextSearchService."""
    service = MagicMock(spec=TextSearchService)
    # Use a coroutine mock instead of Future
    async def mock_search(*args, **kwargs):
        return []
    service.search_by_text = mock_search
    return service


@pytest.fixture
def mock_chunk():
    """Create a mock DocumentChunkModel."""
    chunk = MagicMock(spec=DocumentChunkModel)
    chunk.id = uuid4()
    chunk.content = "Test content"
    chunk.chunk_metadata = {"source_type": "pdf", "author": "Test Author"}
    return chunk


@pytest.fixture
def sample_vector_results(mock_chunk):
    """Create sample vector search results."""
    return [
        SearchResult(chunk=mock_chunk, similarity_score=0.95, rank=1),
        SearchResult(
            chunk=MagicMock(id=uuid4(), content="Content 2", chunk_metadata={}),
            similarity_score=0.85,
            rank=2,
        ),
        SearchResult(
            chunk=MagicMock(id=uuid4(), content="Content 3", chunk_metadata={}),
            similarity_score=0.75,
            rank=3,
        ),
    ]


@pytest.fixture
def sample_text_results(mock_chunk):
    """Create sample text search results."""
    return [
        TextSearchResult(chunk=mock_chunk, rank_score=0.9, rank=1),
        TextSearchResult(
            chunk=MagicMock(id=uuid4(), content="Text 2", chunk_metadata={}),
            rank_score=0.8,
            rank=2,
        ),
        TextSearchResult(
            chunk=MagicMock(id=uuid4(), content="Text 3", chunk_metadata={}),
            rank_score=0.7,
            rank=3,
        ),
    ]


@pytest.fixture
def hybrid_search():
    """Create an EnhancedHybridSearch instance with mocked services."""
    vector_service = MagicMock(spec=VectorSearchService)
    text_service = MagicMock(spec=TextSearchService)
    return EnhancedHybridSearch(
        vector_service=vector_service,
        text_service=text_service,
        config={"rrf_k": 60, "default_vector_weight": 0.7, "default_text_weight": 0.3},
    )


# ============================================================================
# Query Expansion Tests
# ============================================================================

class TestQueryExpander:
    """Tests for QueryExpander."""

    def test_default_creation(self):
        """Test creating expander with default values."""
        expander = QueryExpander()
        assert expander.max_expanded_terms == 5

    def test_custom_max_terms(self):
        """Test creating expander with custom max terms."""
        expander = QueryExpander(max_expanded_terms=10)
        assert expander.max_expanded_terms == 10

    def test_expand_with_acronym(self):
        """Test expanding query with known acronym."""
        expander = QueryExpander()
        result = expander.expand("What is NLP")

        assert "natural language processing" in result.expanded_query.lower()
        assert "nlp" in result.original_terms
        assert len(result.expanded_terms) > 0

    def test_expand_without_acronym(self):
        """Test expanding query without acronyms."""
        expander = QueryExpander()
        result = expander.expand("machine learning applications")

        # No acronym expansion, so query should remain similar
        assert result.expanded_query == "machine learning applications"
        assert len(result.expanded_terms) == 0

    def test_expand_multiple_acronyms(self):
        """Test expanding query with multiple acronyms."""
        expander = QueryExpander()
        result = expander.expand("AI and ML in healthcare")

        assert "artificial intelligence" in result.expanded_query.lower()
        assert "machine learning" in result.expanded_query.lower()

    def test_expand_with_patterns(self):
        """Test pattern-based expansion."""
        expander = QueryExpander()
        result = expander.expand_with_patterns("neural networks")

        # Should add plural/singular forms
        assert len(result.expanded_terms) > 0
        assert "networks" in result.original_terms

    def test_query_expansion_result_model(self):
        """Test QueryExpansionResult dataclass."""
        result = QueryExpansionResult(
            expanded_query="test query expanded",
            original_terms=["test", "query"],
            expanded_terms=["expanded"],
            expansion_method="synonym",
        )

        assert result.expanded_query == "test query expanded"
        assert result.original_terms == ["test", "query"]
        assert result.expanded_terms == ["expanded"]
        assert result.expansion_method == "synonym"


# ============================================================================
# Metadata Filter Tests
# ============================================================================

class TestMetadataFilter:
    """Tests for MetadataFilter."""

    def test_valid_filter_creation(self):
        """Test creating valid metadata filter."""
        filter_obj = MetadataFilter(field="source_type", value="pdf", operator="eq")
        assert filter_obj.field == "source_type"
        assert filter_obj.value == "pdf"
        assert filter_obj.operator == "eq"

    def test_default_operator(self):
        """Test that default operator is 'eq'."""
        filter_obj = MetadataFilter(field="author", value="John")
        assert filter_obj.operator == "eq"

    def test_invalid_field(self):
        """Test that invalid field raises error."""
        with pytest.raises(InvalidFilterError) as exc_info:
            MetadataFilter(field="invalid_field", value="test")
        assert "Invalid filter field" in str(exc_info.value)

    def test_invalid_operator(self):
        """Test that invalid operator raises error."""
        with pytest.raises(InvalidFilterError) as exc_info:
            MetadataFilter(field="source_type", value="pdf", operator="invalid")
        assert "Invalid filter operator" in str(exc_info.value)

    def test_all_valid_fields(self):
        """Test that all valid fields can be used."""
        valid_fields = MetadataFilter.VALID_FIELDS
        for field in valid_fields:
            filter_obj = MetadataFilter(field=field, value="test")
            assert filter_obj.field == field


# ============================================================================
# Weight Preset Tests
# ============================================================================

class TestWeightPreset:
    """Tests for WeightPreset enum and preset functionality."""

    def test_preset_values(self):
        """Test preset enum values."""
        assert StrategyWeightPreset.SEMANTIC_FOCUS.value == "semantic_focus"
        assert StrategyWeightPreset.BALANCED.value == "balanced"
        assert StrategyWeightPreset.LEXICAL_FOCUS.value == "lexical_focus"

    def test_semantic_focus_weights(self, hybrid_search):
        """Test semantic focus preset returns correct weights."""
        weights = hybrid_search._get_preset_weights("semantic_focus")
        assert weights["vector"] == 0.9
        assert weights["text"] == 0.1

    def test_balanced_weights(self, hybrid_search):
        """Test balanced preset returns correct weights."""
        weights = hybrid_search._get_preset_weights("balanced")
        assert weights["vector"] == 0.7
        assert weights["text"] == 0.3

    def test_lexical_focus_weights(self, hybrid_search):
        """Test lexical focus preset returns correct weights."""
        weights = hybrid_search._get_preset_weights("lexical_focus")
        assert weights["vector"] == 0.3
        assert weights["text"] == 0.7

    def test_invalid_preset(self, hybrid_search):
        """Test that invalid preset raises error."""
        with pytest.raises(InvalidPresetError) as exc_info:
            hybrid_search._get_preset_weights("invalid_preset")
        assert "Invalid weight preset" in str(exc_info.value)


# ============================================================================
# Weight Validation Tests
# ============================================================================

class TestWeightValidation:
    """Tests for weight validation."""

    def test_valid_weights(self, hybrid_search):
        """Test that valid weights pass validation."""
        # Should not raise
        hybrid_search._validate_weights(0.7, 0.3)
        hybrid_search._validate_weights(1.0, 0.0)
        hybrid_search._validate_weights(0.0, 1.0)
        hybrid_search._validate_weights(0.5, 0.5)

    def test_invalid_negative_weights(self, hybrid_search):
        """Test that negative weights raise error."""
        with pytest.raises(InvalidWeightError) as exc_info:
            hybrid_search._validate_weights(-0.1, 0.5)
        assert "non-negative" in str(exc_info.value)

    def test_weights_not_summing_to_one(self, hybrid_search):
        """Test that weights not summing to 1.0 raise error."""
        with pytest.raises(InvalidWeightError) as exc_info:
            hybrid_search._validate_weights(0.5, 0.3)
        assert "sum to 1.0" in str(exc_info.value)

    def test_weights_with_tolerance(self, hybrid_search):
        """Test that weights within floating point tolerance pass."""
        # Should not raise (within 0.01 tolerance)
        hybrid_search._validate_weights(0.701, 0.299)


# ============================================================================
# Filter Parsing Tests
# ============================================================================

class TestFilterParsing:
    """Tests for filter parsing functionality."""

    def test_parse_simple_equality_filter(self, hybrid_search):
        """Test parsing simple equality filter."""
        filters = {"source_type": "pdf"}
        parsed = hybrid_search._parse_filters(filters)

        assert len(parsed) == 1
        assert parsed[0].field == "source_type"
        assert parsed[0].value == "pdf"
        assert parsed[0].operator == "eq"

    def test_parse_nested_metadata_filter(self, hybrid_search):
        """Test parsing nested metadata filter."""
        filters = {"metadata": {"author": "John", "tags": "important"}}
        parsed = hybrid_search._parse_filters(filters)

        assert len(parsed) == 2
        fields = [f.field for f in parsed]
        assert "author" in fields
        assert "tags" in fields

    def test_parse_operator_filter(self, hybrid_search):
        """Test parsing filter with operator."""
        filters = {"created_date": {">=": "2024-01-01"}}
        parsed = hybrid_search._parse_filters(filters)

        assert len(parsed) == 1
        assert parsed[0].field == "created_date"
        assert parsed[0].value == "2024-01-01"
        assert parsed[0].operator == "gte"

    def test_parse_empty_filters(self, hybrid_search):
        """Test parsing empty filters."""
        parsed = hybrid_search._parse_filters({})
        assert len(parsed) == 0

    def test_parse_none_filters(self, hybrid_search):
        """Test parsing None filters."""
        parsed = hybrid_search._parse_filters(None)
        assert len(parsed) == 0

    def test_parse_multiple_filters(self, hybrid_search):
        """Test parsing multiple filters."""
        filters = {
            "source_type": "pdf",
            "author": "John",
            "created_date": {">": "2024-01-01"},
        }
        parsed = hybrid_search._parse_filters(filters)

        assert len(parsed) == 3


# ============================================================================
# RRF Fusion Tests
# ============================================================================

class TestReciprocalRankFusion:
    """Tests for Reciprocal Rank Fusion functionality."""

    def test_standalone_rrf_function(self, mock_chunk):
        """Test the standalone reciprocal_rank_fusion function."""
        vector_results = [
            SearchResult(chunk=mock_chunk, similarity_score=0.9, rank=1),
        ]
        text_results = [
            SearchResult(chunk=mock_chunk, similarity_score=0.8, rank=1),
        ]

        result = reciprocal_rank_fusion(
            vector_results, text_results, vector_weight=0.7, text_weight=0.3, k=60
        )

        assert len(result) == 1
        chunk_id, score = result[0]
        # Score should be weighted sum of contributions
        expected_score = (0.7 / 61) + (0.3 / 61)
        assert abs(score - expected_score) < 0.001

    def test_rrf_with_no_overlap(self):
        """Test RRF when results have no overlapping chunks."""
        chunk1 = MagicMock(id=uuid4())
        chunk2 = MagicMock(id=uuid4())

        vector_results = [SearchResult(chunk=chunk1, similarity_score=0.9, rank=1)]
        text_results = [TextSearchResult(chunk=chunk2, rank_score=0.8, rank=1)]

        result = reciprocal_rank_fusion(vector_results, text_results)

        assert len(result) == 2
        # Both should have scores
        assert all(score > 0 for _, score in result)

    def test_rrf_weights_affect_score(self, mock_chunk):
        """Test that weights affect RRF scores."""
        # Create lists with NO overlap - different chunks at rank 1
        chunk2 = MagicMock(id=uuid4())
        
        # Only mock_chunk in vector, only chunk2 in text
        vector_results = [
            SearchResult(chunk=mock_chunk, similarity_score=0.9, rank=1),
        ]
        text_results = [
            TextSearchResult(chunk=chunk2, rank_score=0.9, rank=1),
        ]

        # Test with different weights
        result1 = reciprocal_rank_fusion(
            vector_results, text_results, vector_weight=0.9, text_weight=0.1, k=60
        )
        result2 = reciprocal_rank_fusion(
            vector_results, text_results, vector_weight=0.1, text_weight=0.9, k=60
        )

        # With high vector weight, mock_chunk (rank 1 in vector) should be top
        # With high text weight, chunk2 (rank 1 in text) should be top
        # Scores:
        # result1: mock_chunk = 0.9/61 = 0.01475, chunk2 = 0.1/61 = 0.00164
        # result2: mock_chunk = 0.1/61 = 0.00164, chunk2 = 0.9/61 = 0.01475
        top_chunk_1 = result1[0][0]
        top_chunk_2 = result2[0][0]
        
        # The top chunk should be different with different weights
        assert top_chunk_1 != top_chunk_2, f"Top chunks should differ but both were {top_chunk_1}"

    def test_rrf_sorted_by_score(self, mock_chunk):
        """Test that RRF results are sorted by score descending."""
        chunk1 = MagicMock(id=uuid4())
        chunk2 = MagicMock(id=uuid4())
        chunk3 = MagicMock(id=uuid4())

        vector_results = [
            SearchResult(chunk=chunk1, similarity_score=0.9, rank=1),
            SearchResult(chunk=chunk2, similarity_score=0.8, rank=2),
        ]
        text_results = [
            TextSearchResult(chunk=chunk2, rank_score=0.9, rank=1),
            TextSearchResult(chunk=chunk3, rank_score=0.8, rank=2),
        ]

        result = reciprocal_rank_fusion(vector_results, text_results)

        # Check that results are sorted by score descending
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# Enhanced RRF Fusion Tests
# ============================================================================

class TestEnhancedRRFFusion:
    """Tests for EnhancedHybridSearch RRF fusion."""

    def test_fuse_rrf_weighted(self, hybrid_search, sample_vector_results, sample_text_results):
        """Test weighted RRF fusion."""
        results = hybrid_search._fuse_rrf_weighted(
            sample_vector_results,
            sample_text_results,
            vector_weight=0.7,
            text_weight=0.3,
            k=60,
        )

        assert len(results) > 0
        # All results should have HybridSearchResult type
        assert all(isinstance(r, HybridSearchResult) for r in results)
        # All should have hybrid scores
        assert all(r.hybrid_score > 0 for r in results)

    def test_fuse_rrf_preserves_chunk_data(
        self, hybrid_search, sample_vector_results, sample_text_results, mock_chunk
    ):
        """Test that RRF fusion preserves chunk data."""
        results = hybrid_search._fuse_rrf_weighted(
            sample_vector_results,
            sample_text_results,
            vector_weight=0.7,
            text_weight=0.3,
            k=60,
        )

        # Find result with our mock chunk
        mock_chunk_id = str(mock_chunk.id)
        result = next((r for r in results if str(r.chunk.id) == mock_chunk_id), None)
        assert result is not None
        assert result.vector_score is not None
        assert result.text_score is not None

    def test_fuse_rrf_with_vector_only(self, hybrid_search, sample_vector_results):
        """Test RRF fusion with only vector results."""
        results = hybrid_search._fuse_rrf_weighted(
            sample_vector_results,
            [],  # No text results
            vector_weight=1.0,
            text_weight=0.0,
            k=60,
        )

        assert len(results) == len(sample_vector_results)
        # All scores should come from vector only
        for result in results:
            assert result.vector_score is not None
            assert result.text_score is None

    def test_fuse_rrf_with_text_only(self, hybrid_search, sample_text_results):
        """Test RRF fusion with only text results."""
        results = hybrid_search._fuse_rrf_weighted(
            [],  # No vector results
            sample_text_results,
            vector_weight=0.0,
            text_weight=1.0,
            k=60,
        )

        assert len(results) == len(sample_text_results)
        # All scores should come from text only
        for result in results:
            assert result.vector_score is None
            assert result.text_score is not None


# ============================================================================
# Weighted Sum Fusion Tests
# ============================================================================

class TestWeightedSumFusion:
    """Tests for weighted sum fusion."""

    def test_fuse_weighted_sum(self, hybrid_search, sample_vector_results, sample_text_results):
        """Test weighted sum fusion."""
        results = hybrid_search._fuse_weighted_sum(
            sample_vector_results,
            sample_text_results,
            vector_weight=0.7,
            text_weight=0.3,
        )

        assert len(results) > 0
        # All results should have HybridSearchResult type
        assert all(isinstance(r, HybridSearchResult) for r in results)

    def test_fuse_weighted_sum_calculation(
        self, hybrid_search, sample_vector_results, sample_text_results
    ):
        """Test that weighted sum is calculated correctly."""
        # Use only first result from each for simplicity
        v_result = sample_vector_results[0]
        t_result = sample_text_results[0]

        results = hybrid_search._fuse_weighted_sum(
            [v_result],
            [t_result],
            vector_weight=0.7,
            text_weight=0.3,
        )

        # The same chunk should be in results
        result = results[0]
        # Text score is normalized to max (1.0) since there's only one text result
        expected_score = (0.7 * v_result.similarity_score) + (0.3 * 1.0)  # normalized text score
        assert abs(result.hybrid_score - expected_score) < 0.01

    def test_fuse_weighted_sum_single_source(self, hybrid_search, sample_vector_results):
        """Test weighted sum with only one source."""
        results = hybrid_search._fuse_weighted_sum(
            sample_vector_results,
            [],
            vector_weight=1.0,
            text_weight=0.0,
        )

        assert len(results) == len(sample_vector_results)
        # Results may be in different order, so check by chunk id
        result_scores = {str(r.chunk.id): r.hybrid_score for r in results}
        for v_result in sample_vector_results:
            chunk_id = str(v_result.chunk.id)
            # Score should be same as vector score when text_weight is 0
            assert abs(result_scores[chunk_id] - v_result.similarity_score) < 0.01


# ============================================================================
# Text Score Normalization Tests
# ============================================================================

class TestTextScoreNormalization:
    """Tests for text score normalization."""

    def test_normalize_scores(self, hybrid_search):
        """Test normalizing text scores."""
        chunk1 = MagicMock(id=uuid4())
        chunk2 = MagicMock(id=uuid4())

        results = [
            TextSearchResult(chunk=chunk1, rank_score=10.0, rank=1),
            TextSearchResult(chunk=chunk2, rank_score=5.0, rank=2),
        ]

        normalized = hybrid_search._normalize_text_scores(results)

        assert normalized[str(chunk1.id)] == 1.0  # Max score normalized to 1.0
        assert normalized[str(chunk2.id)] == 0.5  # Half of max

    def test_normalize_zero_scores(self, hybrid_search):
        """Test normalizing all zero scores."""
        chunk = MagicMock(id=uuid4())
        results = [TextSearchResult(chunk=chunk, rank_score=0.0, rank=1)]

        normalized = hybrid_search._normalize_text_scores(results)

        assert normalized[str(chunk.id)] == 0.0

    def test_normalize_empty_results(self, hybrid_search):
        """Test normalizing empty results."""
        normalized = hybrid_search._normalize_text_scores([])
        assert normalized == {}


# ============================================================================
# Async Search Tests
# ============================================================================

@pytest.mark.asyncio
class TestAsyncSearch:
    """Tests for async search functionality."""

    async def test_search_with_preset(
        self, sample_vector_results, sample_text_results
    ):
        """Test search with weight preset."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        
        async def mock_vector_search(*args, **kwargs):
            return sample_vector_results
        
        async def mock_text_search(*args, **kwargs):
            return sample_text_results
        
        vector_service.search_by_vector = mock_vector_search
        text_service.search_by_text = mock_text_search
        
        hybrid_search = EnhancedHybridSearch(vector_service, text_service)

        results = await hybrid_search.search(
            query="test query",
            embedding=[0.1] * 1536,
            limit=5,
            weight_preset="semantic_focus",
        )

        assert isinstance(results, list)
        assert len(results) <= 5

    async def test_search_with_explicit_weights(
        self, sample_vector_results, sample_text_results
    ):
        """Test search with explicit weights."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        
        async def mock_vector_search(*args, **kwargs):
            return sample_vector_results
        
        async def mock_text_search(*args, **kwargs):
            return sample_text_results
        
        vector_service.search_by_vector = mock_vector_search
        text_service.search_by_text = mock_text_search
        
        hybrid_search = EnhancedHybridSearch(vector_service, text_service)

        results = await hybrid_search.search(
            query="test query",
            embedding=[0.1] * 1536,
            limit=5,
            vector_weight=0.8,
            text_weight=0.2,
        )

        assert isinstance(results, list)

    async def test_search_with_filters(
        self, sample_vector_results, sample_text_results
    ):
        """Test search with metadata filters."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        
        async def mock_vector_search(*args, **kwargs):
            return sample_vector_results
        
        async def mock_text_search(*args, **kwargs):
            return sample_text_results
        
        vector_service.search_by_vector = mock_vector_search
        text_service.search_by_text = mock_text_search
        
        hybrid_search = EnhancedHybridSearch(vector_service, text_service)

        results = await hybrid_search.search(
            query="test query",
            embedding=[0.1] * 1536,
            filters={"source_type": "pdf", "author": "John"},
            limit=5,
        )

        assert isinstance(results, list)

    async def test_search_with_query_expansion(
        self, sample_vector_results, sample_text_results
    ):
        """Test search with query expansion enabled."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        
        async def mock_vector_search(*args, **kwargs):
            return sample_vector_results
        
        async def mock_text_search(*args, **kwargs):
            return sample_text_results
        
        vector_service.search_by_vector = mock_vector_search
        text_service.search_by_text = mock_text_search
        
        hybrid_search = EnhancedHybridSearch(vector_service, text_service)

        results = await hybrid_search.search(
            query="AI applications",
            embedding=[0.1] * 1536,
            limit=5,
            use_query_expansion=True,
        )

        assert isinstance(results, list)

    async def test_search_with_weighted_sum_fusion(
        self, sample_vector_results, sample_text_results
    ):
        """Test search with weighted sum fusion method."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        
        async def mock_vector_search(*args, **kwargs):
            return sample_vector_results
        
        async def mock_text_search(*args, **kwargs):
            return sample_text_results
        
        vector_service.search_by_vector = mock_vector_search
        text_service.search_by_text = mock_text_search
        
        hybrid_search = EnhancedHybridSearch(vector_service, text_service)

        results = await hybrid_search.search(
            query="test query",
            embedding=[0.1] * 1536,
            limit=5,
            fusion_method=FusionMethod.WEIGHTED_SUM,
        )

        assert isinstance(results, list)
        # All results should have fusion_method set correctly
        assert all(r.fusion_method == FusionMethod.WEIGHTED_SUM.value for r in results)

    async def test_search_empty_results(self):
        """Test search when both services return empty results."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        
        async def mock_empty_search(*args, **kwargs):
            return []
        
        vector_service.search_by_vector = mock_empty_search
        text_service.search_by_text = mock_empty_search
        
        hybrid_search = EnhancedHybridSearch(vector_service, text_service)

        results = await hybrid_search.search(
            query="test query",
            embedding=[0.1] * 1536,
            limit=5,
        )

        assert results == []

    async def test_search_invalid_weights(self):
        """Test that invalid weights raise error."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        hybrid_search = EnhancedHybridSearch(vector_service, text_service)
        
        with pytest.raises(InvalidWeightError):
            await hybrid_search.search(
                query="test",
                embedding=[0.1] * 1536,
                vector_weight=0.5,
                text_weight=0.3,  # Doesn't sum to 1.0
            )


# ============================================================================
# Filter Conversion Tests
# ============================================================================

class TestFilterConversion:
    """Tests for filter conversion to service formats."""

    def test_convert_filters_for_vector(self, hybrid_search):
        """Test converting filters for vector service."""
        filters = [
            MetadataFilter(field="source_type", value="pdf"),
            MetadataFilter(field="author", value="John"),
        ]

        result = hybrid_search._convert_filters_for_vector(filters)

        assert result is not None
        assert "metadata" in result
        assert result["metadata"]["source_type"] == "pdf"
        assert result["metadata"]["author"] == "John"

    def test_convert_filters_with_job_id(self, hybrid_search):
        """Test converting filters with job_id."""
        from uuid import UUID
        job_id = UUID("12345678-1234-5678-1234-567812345678")
        filters = [MetadataFilter(field="job_id", value=job_id)]

        result = hybrid_search._convert_filters_for_vector(filters)

        assert result is not None
        assert result["job_id"] == job_id

    def test_convert_empty_filters(self, hybrid_search):
        """Test converting empty filters."""
        result = hybrid_search._convert_filters_for_vector([])
        assert result is None

    def test_convert_filters_for_text(self, hybrid_search):
        """Test converting filters for text service (same as vector)."""
        filters = [MetadataFilter(field="source_type", value="pdf")]
        result = hybrid_search._convert_filters_for_text(filters)

        assert result is not None
        assert "metadata" in result


# ============================================================================
# Exception Classes Tests
# ============================================================================

class TestHybridSearchExceptions:
    """Tests for custom exception classes."""

    def test_hybrid_search_error_is_exception(self):
        """Test HybridSearchError is an Exception."""
        err = HybridSearchError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"

    def test_hybrid_search_error_with_context(self):
        """Test HybridSearchError with context."""
        context = {"query": "test", "field": "value"}
        err = HybridSearchError("error", context=context)
        assert err.context == context

    def test_invalid_weight_error_is_hybrid_search_error(self):
        """Test InvalidWeightError is a HybridSearchError."""
        err = InvalidWeightError("invalid weight")
        assert isinstance(err, HybridSearchError)

    def test_invalid_preset_error_is_hybrid_search_error(self):
        """Test InvalidPresetError is a HybridSearchError."""
        err = InvalidPresetError("invalid preset")
        assert isinstance(err, HybridSearchError)

    def test_invalid_filter_error_is_hybrid_search_error(self):
        """Test InvalidFilterError is a HybridSearchError."""
        err = InvalidFilterError("invalid filter")
        assert isinstance(err, HybridSearchError)

    def test_invalid_fusion_method_error_is_hybrid_search_error(self):
        """Test InvalidFusionMethodError is a HybridSearchError."""
        err = InvalidFusionMethodError("invalid method")
        assert isinstance(err, HybridSearchError)

    def test_exception_chaining(self):
        """Test that exceptions can be chained."""
        original = ValueError("original")
        try:
            raise HybridSearchError("wrapped") from original
        except HybridSearchError as err:
            assert err.__cause__ == original


# ============================================================================
# Pydantic Model Tests
# ============================================================================

class TestHybridSearchRequestModel:
    """Tests for HybridSearchRequest Pydantic model."""

    def test_valid_request_creation(self):
        """Test creating valid request."""
        request = HybridSearchRequest(
            query="test query",
            embedding=[0.1] * 1536,
            filters={"source_type": "pdf"},
            limit=10,
            weight_preset=WeightPreset.BALANCED,
            fusion_method=FusionMethod.RECIPROCAL_RANK_FUSION,
        )

        assert request.query == "test query"
        assert len(request.embedding) == 1536
        assert request.limit == 10

    def test_request_defaults(self):
        """Test request with default values."""
        request = HybridSearchRequest(query="test")

        assert request.limit == 10
        assert request.weight_preset == WeightPreset.BALANCED
        assert request.fusion_method == FusionMethod.RECIPROCAL_RANK_FUSION
        assert request.use_query_expansion is True

    def test_request_validation_invalid_weight(self):
        """Test that invalid weight range raises error."""
        with pytest.raises(ValueError):
            HybridSearchRequest(query="test", vector_weight=1.5)

    def test_request_validation_negative_weight(self):
        """Test that negative weight raises error."""
        with pytest.raises(ValueError):
            HybridSearchRequest(query="test", vector_weight=-0.1)


class TestHybridSearchResultItemModel:
    """Tests for HybridSearchResultItem Pydantic model."""

    def test_valid_result_creation(self):
        """Test creating valid result item."""
        result = HybridSearchResultItem(
            chunk_id=str(uuid4()),
            content="Test content",
            hybrid_score=0.85,
            vector_score=0.82,
            text_score=0.88,
            vector_rank=1,
            text_rank=2,
            metadata={"source": "test.pdf"},
        )

        assert result.hybrid_score == 0.85
        assert result.vector_score == 0.82
        assert result.text_score == 0.88

    def test_result_score_validation(self):
        """Test that scores must be in valid range."""
        with pytest.raises(ValueError):
            HybridSearchResultItem(
                chunk_id=str(uuid4()),
                content="test",
                hybrid_score=1.5,  # Invalid: > 1
            )


class TestHybridSearchResponseModel:
    """Tests for HybridSearchResponse Pydantic model."""

    def test_valid_response_creation(self):
        """Test creating valid response."""
        response = HybridSearchResponse(
            results=[
                HybridSearchResultItem(
                    chunk_id=str(uuid4()),
                    content="Test",
                    hybrid_score=0.85,
                )
            ],
            total_results=1,
            query="test query",
            fusion_method=FusionMethod.RECIPROCAL_RANK_FUSION,
            weight_preset=WeightPreset.BALANCED,
            vector_weight=0.7,
            text_weight=0.3,
            latency_ms=45.2,
        )

        assert response.total_results == 1
        assert response.latency_ms == 45.2


# ============================================================================
# Initialization Tests
# ============================================================================

class TestInitialization:
    """Tests for EnhancedHybridSearch initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        search = EnhancedHybridSearch(vector_service, text_service)

        assert search.vector_service == vector_service
        assert search.text_service == text_service
        assert search.rrf_k == 60  # Default from settings

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        config = {"rrf_k": 120, "default_vector_weight": 0.8}
        search = EnhancedHybridSearch(vector_service, text_service, config=config)

        assert search.rrf_k == 120
        assert search.default_vector_weight == 0.8

    def test_init_creates_query_expander(self):
        """Test that initialization creates query expander."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        search = EnhancedHybridSearch(vector_service, text_service)

        assert search.query_expander is not None
        assert isinstance(search.query_expander, QueryExpander)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
class TestIntegration:
    """Integration-style tests for hybrid search."""

    async def test_full_search_flow(
        self, sample_vector_results, sample_text_results
    ):
        """Test complete search flow with all features."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        
        async def mock_vector_search(*args, **kwargs):
            return sample_vector_results
        
        async def mock_text_search(*args, **kwargs):
            return sample_text_results
        
        vector_service.search_by_vector = mock_vector_search
        text_service.search_by_text = mock_text_search
        
        hybrid_search = EnhancedHybridSearch(vector_service, text_service)

        results = await hybrid_search.search(
            query="machine learning AI",
            embedding=[0.1] * 1536,
            filters={"source_type": "pdf"},
            limit=3,
            weight_preset="balanced",
            fusion_method=FusionMethod.RECIPROCAL_RANK_FUSION,
            use_query_expansion=True,
        )

        # Verify results structure
        assert isinstance(results, list)
        assert len(results) <= 3

        for result in results:
            assert isinstance(result, HybridSearchResult)
            assert result.chunk is not None
            assert result.hybrid_score > 0
            assert result.rank > 0

        # Verify results are sorted by score
        scores = [r.hybrid_score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_search_performance(
        self, sample_vector_results, sample_text_results
    ):
        """Test that search completes within acceptable time."""
        import time

        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        
        async def mock_vector_search(*args, **kwargs):
            return sample_vector_results
        
        async def mock_text_search(*args, **kwargs):
            return sample_text_results
        
        vector_service.search_by_vector = mock_vector_search
        text_service.search_by_text = mock_text_search
        
        hybrid_search = EnhancedHybridSearch(vector_service, text_service)

        start = time.perf_counter()
        results = await hybrid_search.search(
            query="test query",
            embedding=[0.1] * 1536,
            limit=10,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete within 100ms with mocked services
        assert elapsed_ms < 100, f"Search took {elapsed_ms:.2f}ms, expected <100ms"


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_search_with_empty_query(
        self, sample_vector_results, sample_text_results
    ):
        """Test search with empty query string."""
        vector_service = MagicMock(spec=VectorSearchService)
        text_service = MagicMock(spec=TextSearchService)
        
        async def mock_vector_search(*args, **kwargs):
            return sample_vector_results
        
        async def mock_text_search(*args, **kwargs):
            return sample_text_results
        
        vector_service.search_by_vector = mock_vector_search
        text_service.search_by_text = mock_text_search
        
        hybrid_search = EnhancedHybridSearch(vector_service, text_service)

        # Empty query should still work (just returns results)
        results = await hybrid_search.search(
            query="",
            embedding=[0.1] * 1536,
            limit=5,
        )

        # Should still return results
        assert isinstance(results, list)

    def test_rrf_with_large_k(self, hybrid_search, sample_vector_results, sample_text_results):
        """Test RRF with large k value."""
        results = hybrid_search._fuse_rrf_weighted(
            sample_vector_results,
            sample_text_results,
            vector_weight=0.5,
            text_weight=0.5,
            k=1000,  # Large k
        )

        assert len(results) > 0
        # With large k, scores should be smaller but still positive
        for result in results:
            assert result.hybrid_score > 0

    def test_weighted_sum_with_extreme_weights(self, hybrid_search, sample_vector_results):
        """Test weighted sum with extreme weight values."""
        results = hybrid_search._fuse_weighted_sum(
            sample_vector_results,
            [],
            vector_weight=1.0,
            text_weight=0.0,
        )

        # Results may be in different order, so check by chunk id
        result_scores = {str(r.chunk.id): r.hybrid_score for r in results}
        for v_result in sample_vector_results:
            chunk_id = str(v_result.chunk.id)
            # Score should be same as vector score when text_weight is 0
            assert abs(result_scores[chunk_id] - v_result.similarity_score) < 0.01

    def test_filter_with_special_characters(self, hybrid_search):
        """Test filters with special characters in values."""
        filters = {"author": "O'Connor", "source_type": "Test & Example"}
        parsed = hybrid_search._parse_filters(filters)

        assert len(parsed) == 2
        # Should preserve special characters
        values = [f.value for f in parsed]
        assert "O'Connor" in values
        assert "Test & Example" in values
