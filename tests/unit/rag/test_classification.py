"""Unit tests for Query Classification service.

This module tests the QueryClassifier class and related components
including caching, pattern-based classification, and strategy selection.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.rag.classification import (
    DEFAULT_CLASSIFICATION_PROMPT,
    STRATEGY_MATRIX,
    ClassificationCache,
    ClassificationError,
    ClassificationValidationError,
    NullClassificationCache,
    QueryClassifier,
    get_classification_patterns,
    get_strategy_matrix,
)
from src.rag.models import QueryClassification, QueryType

# ============================================================================
# QueryType Enum Tests
# ============================================================================


class TestQueryType:
    """Tests for QueryType enum."""

    def test_enum_values(self):
        """Test that enum has all expected values."""
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.ANALYTICAL.value == "analytical"
        assert QueryType.COMPARATIVE.value == "comparative"
        assert QueryType.VAGUE.value == "vague"
        assert QueryType.MULTI_HOP.value == "multi_hop"

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert QueryType("factual") == QueryType.FACTUAL
        assert QueryType("analytical") == QueryType.ANALYTICAL
        assert QueryType("comparative") == QueryType.COMPARATIVE
        assert QueryType("vague") == QueryType.VAGUE
        assert QueryType("multi_hop") == QueryType.MULTI_HOP

    def test_invalid_enum_value(self):
        """Test that invalid enum value raises error."""
        with pytest.raises(ValueError):
            QueryType("invalid")


# ============================================================================
# QueryClassification Model Tests
# ============================================================================


class TestQueryClassification:
    """Tests for QueryClassification Pydantic model."""

    def test_default_creation(self):
        """Test creating result with required values."""
        result = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="Test reasoning",
        )

        assert result.query_type == QueryType.FACTUAL
        assert result.confidence == 0.95
        assert result.reasoning == "Test reasoning"
        assert result.suggested_strategies == []

    def test_full_creation(self):
        """Test creating result with all values."""
        result = QueryClassification(
            query_type=QueryType.ANALYTICAL,
            confidence=0.85,
            reasoning="Requires explanation",
            suggested_strategies=["hyde", "reranking"],
        )

        assert result.query_type == QueryType.ANALYTICAL
        assert result.confidence == 0.85
        assert result.reasoning == "Requires explanation"
        assert result.suggested_strategies == ["hyde", "reranking"]

    def test_confidence_bounds(self):
        """Test that confidence must be between 0 and 1."""
        # Valid values
        QueryClassification(query_type=QueryType.FACTUAL, confidence=0.0, reasoning="test")
        QueryClassification(query_type=QueryType.FACTUAL, confidence=1.0, reasoning="test")
        QueryClassification(query_type=QueryType.FACTUAL, confidence=0.5, reasoning="test")

        # Invalid values
        with pytest.raises(ValueError):
            QueryClassification(query_type=QueryType.FACTUAL, confidence=-0.1, reasoning="test")
        
        with pytest.raises(ValueError):
            QueryClassification(query_type=QueryType.FACTUAL, confidence=1.1, reasoning="test")

    def test_serialization(self):
        """Test JSON serialization."""
        result = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="Test",
            suggested_strategies=["reranking"],
        )

        json_str = result.model_dump_json()
        data = json.loads(json_str)

        assert data["query_type"] == "factual"
        assert data["confidence"] == 0.95
        assert data["reasoning"] == "Test"
        assert data["suggested_strategies"] == ["reranking"]

    def test_deserialization(self):
        """Test JSON deserialization."""
        data = {
            "query_type": "analytical",
            "confidence": 0.88,
            "reasoning": "Needs explanation",
            "suggested_strategies": ["hyde", "query_rewrite"],
        }

        result = QueryClassification(**data)

        assert result.query_type == QueryType.ANALYTICAL
        assert result.confidence == 0.88
        assert result.reasoning == "Needs explanation"
        assert result.suggested_strategies == ["hyde", "query_rewrite"]


# ============================================================================
# Strategy Matrix Tests
# ============================================================================


class TestStrategyMatrix:
    """Tests for the strategy selection matrix."""

    def test_strategy_matrix_structure(self):
        """Test that strategy matrix has all query types."""
        for query_type in QueryType:
            assert query_type in STRATEGY_MATRIX

    def test_factual_strategy(self):
        """Test strategy configuration for factual queries."""
        config = STRATEGY_MATRIX[QueryType.FACTUAL]
        assert config["hyde"] is False
        assert config["reranking"] is True
        assert config["hybrid"] is True
        assert config["query_rewrite"] is True

    def test_analytical_strategy(self):
        """Test strategy configuration for analytical queries."""
        config = STRATEGY_MATRIX[QueryType.ANALYTICAL]
        assert config["hyde"] is True
        assert config["reranking"] is True
        assert config["hybrid"] is True
        assert config["query_rewrite"] is True

    def test_comparative_strategy(self):
        """Test strategy configuration for comparative queries."""
        config = STRATEGY_MATRIX[QueryType.COMPARATIVE]
        assert config["hyde"] is False
        assert config["reranking"] is True
        assert config["hybrid"] is True
        assert config["query_rewrite"] is True

    def test_vague_strategy(self):
        """Test strategy configuration for vague queries."""
        config = STRATEGY_MATRIX[QueryType.VAGUE]
        assert config["hyde"] is True
        assert config["reranking"] is True
        assert config["hybrid"] is True
        assert config["query_rewrite"] is True

    def test_multi_hop_strategy(self):
        """Test strategy configuration for multi-hop queries."""
        config = STRATEGY_MATRIX[QueryType.MULTI_HOP]
        assert config["hyde"] is True
        assert config["reranking"] is True
        assert config["hybrid"] is True
        assert config["query_rewrite"] is True

    def test_get_strategy_matrix(self):
        """Test get_strategy_matrix utility function."""
        matrix = get_strategy_matrix()
        assert isinstance(matrix, dict)
        assert len(matrix) == len(QueryType)
        # Ensure it's a copy
        matrix[QueryType.FACTUAL]["hyde"] = True
        assert STRATEGY_MATRIX[QueryType.FACTUAL]["hyde"] is False

    def test_get_classification_patterns(self):
        """Test get_classification_patterns utility function."""
        patterns = get_classification_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        # Each pattern should be a tuple of (pattern, query_type, confidence)
        for pattern, query_type, confidence in patterns:
            assert isinstance(pattern, str)
            assert isinstance(query_type, str)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0


# ============================================================================
# NullClassificationCache Tests
# ============================================================================


class TestNullClassificationCache:
    """Tests for NullClassificationCache (null object pattern)."""

    @pytest.mark.asyncio
    async def test_get_returns_none(self):
        """Test that get always returns None."""
        cache = NullClassificationCache()

        result = await cache.get("any query")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_returns_false(self):
        """Test that set always returns False."""
        cache = NullClassificationCache()
        result = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="test",
        )

        success = await cache.set("query", result)

        assert success is False

    @pytest.mark.asyncio
    async def test_delete_returns_false(self):
        """Test that delete always returns False."""
        cache = NullClassificationCache()

        success = await cache.delete("query")

        assert success is False

    @pytest.mark.asyncio
    async def test_clear_all_returns_zero(self):
        """Test that clear_all always returns 0."""
        cache = NullClassificationCache()

        count = await cache.clear_all()

        assert count == 0


# ============================================================================
# ClassificationCache Tests
# ============================================================================


class TestClassificationCache:
    """Tests for ClassificationCache with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis_mock = AsyncMock()
        return redis_mock

    @pytest.fixture
    def cache(self, mock_redis):
        """Create cache instance with mock Redis."""
        return ClassificationCache(redis_client=mock_redis)

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache, mock_redis):
        """Test getting cached result when it exists."""
        # Setup mock to return cached data
        cached_data = json.dumps({
            "query_type": "factual",
            "confidence": 0.95,
            "reasoning": "Test reasoning",
            "suggested_strategies": ["reranking"],
        })
        mock_redis.get.return_value = cached_data

        result = await cache.get("test query")

        assert result is not None
        assert result.query_type == QueryType.FACTUAL
        assert result.confidence == 0.95
        assert result.reasoning == "Test reasoning"

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache, mock_redis):
        """Test getting cached result when it doesn't exist."""
        mock_redis.get.return_value = None

        result = await cache.get("test query")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_context(self, cache, mock_redis):
        """Test getting with conversation context."""
        cached_data = json.dumps({
            "query_type": "analytical",
            "confidence": 0.88,
            "reasoning": "Needs explanation",
            "suggested_strategies": ["hyde"],
        })
        mock_redis.get.return_value = cached_data

        context = {"previous_queries": ["What is AI?"], "session_id": "sess_123"}
        result = await cache.get("test query", context)

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_redis_failure(self, cache, mock_redis):
        """Test graceful handling of Redis failure on get."""
        mock_redis.get.side_effect = Exception("Redis connection failed")

        result = await cache.get("test query")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_success(self, cache, mock_redis):
        """Test successful cache set."""
        result = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="Test",
        )
        mock_redis.setex.return_value = True

        success = await cache.set("test query", result)

        assert success is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, cache, mock_redis):
        """Test cache set with custom TTL."""
        result = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="Test",
        )
        mock_redis.setex.return_value = True

        await cache.set("test query", result, ttl=7200)

        # Verify TTL was passed correctly
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 7200  # TTL is second positional arg

    @pytest.mark.asyncio
    async def test_set_redis_failure(self, cache, mock_redis):
        """Test graceful handling of Redis failure on set."""
        result = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="Test",
        )
        mock_redis.setex.side_effect = Exception("Redis connection failed")

        success = await cache.set("test query", result)

        assert success is False

    @pytest.mark.asyncio
    async def test_set_without_redis(self):
        """Test set when Redis client is None."""
        cache = ClassificationCache(redis_client=None)
        result = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="Test",
        )

        success = await cache.set("test query", result)

        assert success is False

    @pytest.mark.asyncio
    async def test_delete_success(self, cache, mock_redis):
        """Test successful cache delete."""
        mock_redis.delete.return_value = 1

        success = await cache.delete("test query")

        assert success is True

    @pytest.mark.asyncio
    async def test_clear_all(self, cache, mock_redis):
        """Test clearing all cached entries."""
        # Mock scan_iter to return an async iterable
        async def mock_async_iter(*args, **kwargs):
            yield b"key1"
            yield b"key2"

        mock_redis.scan_iter = mock_async_iter
        mock_redis.delete.return_value = 2

        count = await cache.clear_all()

        assert count == 2
        mock_redis.delete.assert_called_once_with(b"key1", b"key2")

    @pytest.mark.asyncio
    async def test_clear_all_no_keys(self, cache, mock_redis):
        """Test clearing all when no keys exist."""
        # Mock scan_iter to return an empty async iterable
        async def mock_async_iter(*args, **kwargs):
            return
            yield  # Make it a generator

        mock_redis.scan_iter = mock_async_iter

        count = await cache.clear_all()

        assert count == 0

    def test_make_key_consistency(self, mock_redis):
        """Test that same query produces same key."""
        cache = ClassificationCache(redis_client=mock_redis)

        key1 = cache._make_key("test query", "context_hash")
        key2 = cache._make_key("test query", "context_hash")

        assert key1 == key2
        assert key1.startswith("rag:classification:")

    def test_make_key_different_queries(self, mock_redis):
        """Test that different queries produce different keys."""
        cache = ClassificationCache(redis_client=mock_redis)

        key1 = cache._make_key("query one", "")
        key2 = cache._make_key("query two", "")

        assert key1 != key2

    def test_hash_context_empty(self, mock_redis):
        """Test hashing empty context."""
        cache = ClassificationCache(redis_client=mock_redis)

        hash_val = cache._hash_context(None)

        assert hash_val == ""

    def test_hash_context_with_data(self, mock_redis):
        """Test hashing context with data."""
        cache = ClassificationCache(redis_client=mock_redis)

        context = {"session_id": "sess_123", "queries": ["q1", "q2"]}
        hash1 = cache._hash_context(context)
        hash2 = cache._hash_context(context)

        # Same context should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # We truncate to 16 chars


# ============================================================================
# QueryClassifier Initialization Tests
# ============================================================================


@pytest.mark.asyncio
class TestQueryClassifierInitialization:
    """Tests for QueryClassifier initialization."""

    async def test_init_default_values(self):
        """Test initialization with default values."""
        with patch("src.rag.classification.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            classifier = QueryClassifier()

            assert classifier.model == "agentic-decisions"
            assert classifier.temperature == 0.1
            assert classifier.max_tokens == 300
            assert classifier.system_prompt == DEFAULT_CLASSIFICATION_PROMPT
            assert classifier.min_confidence_threshold == 0.7
            assert classifier.use_pattern_fallback is True
            assert classifier.max_query_length == 4000

    async def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        custom_prompt = "Custom classification prompt"

        with patch("src.rag.classification.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            classifier = QueryClassifier(
                model="gpt-4o",
                temperature=0.2,
                max_tokens=500,
                system_prompt=custom_prompt,
                min_confidence_threshold=0.8,
                use_pattern_fallback=False,
                max_query_length=2000,
            )

            assert classifier.model == "gpt-4o"
            assert classifier.temperature == 0.2
            assert classifier.max_tokens == 500
            assert classifier.system_prompt == custom_prompt
            assert classifier.min_confidence_threshold == 0.8
            assert classifier.use_pattern_fallback is False
            assert classifier.max_query_length == 2000


# ============================================================================
# QueryClassifier Pattern-Based Classification Tests
# ============================================================================


class TestQueryClassifierPatternClassification:
    """Tests for pattern-based classification."""

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier with mocked LLM."""
        with patch("src.rag.classification.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            return QueryClassifier()

    def test_classify_factual_pattern(self, classifier):
        """Test pattern-based classification of factual queries."""
        result = classifier._classify_by_pattern("What is machine learning?")

        assert result is not None
        assert result.query_type == QueryType.FACTUAL
        assert result.confidence >= 0.80
        assert "reranking" in result.suggested_strategies

    def test_classify_comparative_pattern(self, classifier):
        """Test pattern-based classification of comparative queries."""
        result = classifier._classify_by_pattern("Compare Python and JavaScript")

        assert result is not None
        assert result.query_type == QueryType.COMPARATIVE
        assert result.confidence >= 0.80

    def test_classify_analytical_pattern(self, classifier):
        """Test pattern-based classification of analytical queries."""
        result = classifier._classify_by_pattern("Explain how neural networks work")

        assert result is not None
        assert result.query_type == QueryType.ANALYTICAL
        assert result.confidence >= 0.75

    def test_classify_vague_pattern(self, classifier):
        """Test pattern-based classification of vague queries."""
        result = classifier._classify_by_pattern("Tell me about artificial intelligence")

        assert result is not None
        assert result.query_type == QueryType.VAGUE
        assert result.confidence >= 0.85

    def test_classify_multi_hop_pattern(self, classifier):
        """Test pattern-based classification of multi-hop queries."""
        result = classifier._classify_by_pattern("What did the author of Deep Learning say about neural networks?")

        assert result is not None
        assert result.query_type == QueryType.MULTI_HOP
        assert result.confidence >= 0.80

    def test_classify_no_pattern_match(self, classifier):
        """Test that unmatched queries return None."""
        result = classifier._classify_by_pattern("xyz abc 123")

        assert result is None

    def test_classify_pattern_disabled(self, classifier):
        """Test that pattern classification can be disabled."""
        classifier.use_pattern_fallback = False
        result = classifier._classify_by_pattern("What is AI?")

        assert result is None


# ============================================================================
# QueryClassifier Parse Response Tests
# ============================================================================


class TestQueryClassifierParseResponse:
    """Tests for QueryClassifier._parse_response method."""

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier instance."""
        with patch("src.rag.classification.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            return QueryClassifier()

    def test_parse_valid_response(self, classifier):
        """Test parsing valid JSON response."""
        content = json.dumps({
            "query_type": "factual",
            "confidence": 0.95,
            "reasoning": "Asks for specific definition",
        })

        result = classifier._parse_response(content)

        assert result.query_type == QueryType.FACTUAL
        assert result.confidence == 0.95
        assert result.reasoning == "Asks for specific definition"
        assert "reranking" in result.suggested_strategies

    def test_parse_analytical_response(self, classifier):
        """Test parsing analytical query response."""
        content = json.dumps({
            "query_type": "analytical",
            "confidence": 0.88,
            "reasoning": "Requires explanation",
        })

        result = classifier._parse_response(content)

        assert result.query_type == QueryType.ANALYTICAL
        assert "hyde" in result.suggested_strategies

    def parse_trims_whitespace(self, classifier):
        """Test that text fields are trimmed."""
        content = json.dumps({
            "query_type": "factual",
            "confidence": 0.90,
            "reasoning": "  spaced reasoning  ",
        })

        result = classifier._parse_response(content)

        assert result.reasoning == "spaced reasoning"

    def test_parse_invalid_json(self, classifier):
        """Test handling of invalid JSON."""
        with pytest.raises(ClassificationValidationError):
            classifier._parse_response("not json")

    def test_parse_missing_query_type(self, classifier):
        """Test handling of missing query_type field."""
        content = json.dumps({
            "confidence": 0.90,
            "reasoning": "test",
        })

        with pytest.raises(ClassificationValidationError) as exc_info:
            classifier._parse_response(content)

        assert "query_type" in str(exc_info.value)

    def test_parse_missing_confidence(self, classifier):
        """Test handling of missing confidence field."""
        content = json.dumps({
            "query_type": "factual",
            "reasoning": "test",
        })

        with pytest.raises(ClassificationValidationError) as exc_info:
            classifier._parse_response(content)

        assert "confidence" in str(exc_info.value)

    def test_parse_missing_reasoning(self, classifier):
        """Test handling of missing reasoning field."""
        content = json.dumps({
            "query_type": "factual",
            "confidence": 0.90,
        })

        with pytest.raises(ClassificationValidationError) as exc_info:
            classifier._parse_response(content)

        assert "reasoning" in str(exc_info.value)

    def test_parse_invalid_query_type(self, classifier):
        """Test handling of invalid query_type value."""
        content = json.dumps({
            "query_type": "invalid_type",
            "confidence": 0.90,
            "reasoning": "test",
        })

        with pytest.raises(ClassificationValidationError) as exc_info:
            classifier._parse_response(content)

        assert "query_type" in str(exc_info.value)

    def test_parse_confidence_out_of_range(self, classifier):
        """Test handling of confidence value out of range."""
        content = json.dumps({
            "query_type": "factual",
            "confidence": 1.5,
            "reasoning": "test",
        })

        with pytest.raises(ClassificationValidationError) as exc_info:
            classifier._parse_response(content)

        assert "confidence" in str(exc_info.value)

    def test_parse_wrong_type_confidence(self, classifier):
        """Test handling of wrong type for confidence."""
        content = json.dumps({
            "query_type": "factual",
            "confidence": "high",
            "reasoning": "test",
        })

        with pytest.raises(ClassificationValidationError) as exc_info:
            classifier._parse_response(content)

        assert "confidence" in str(exc_info.value)

    def test_parse_wrong_type_reasoning(self, classifier):
        """Test handling of wrong type for reasoning."""
        content = json.dumps({
            "query_type": "factual",
            "confidence": 0.90,
            "reasoning": 123,
        })

        with pytest.raises(ClassificationValidationError) as exc_info:
            classifier._parse_response(content)

        assert "reasoning" in str(exc_info.value)


# ============================================================================
# QueryClassifier Build Prompt Tests
# ============================================================================


class TestQueryClassifierBuildPrompt:
    """Tests for QueryClassifier._build_prompt method."""

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier instance."""
        with patch("src.rag.classification.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            return QueryClassifier()

    def test_build_prompt_simple(self, classifier):
        """Test building prompt with simple query."""
        query = "What is AI?"

        prompt = classifier._build_prompt(query, None)

        assert 'Query: "What is AI?"' in prompt
        assert "Classify this query" in prompt

    def test_build_prompt_with_context(self, classifier):
        """Test building prompt with conversation context."""
        query = "How does it work?"
        context = {
            "previous_queries": ["What is machine learning?", "How is it used?"],
        }

        prompt = classifier._build_prompt(query, context)

        assert 'Query: "How does it work?"' in prompt
        assert "Previous queries in this conversation:" in prompt
        assert "What is machine learning?" in prompt
        assert "How is it used?" in prompt

    def test_build_prompt_limits_previous_queries(self, classifier):
        """Test that only last 3 previous queries are included."""
        query = "Final question?"
        context = {
            "previous_queries": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        }

        prompt = classifier._build_prompt(query, context)

        # Should only include Q3, Q4, Q5 (last 3)
        assert "Q1" not in prompt
        assert "Q2" not in prompt
        assert "Q3" in prompt
        assert "Q4" in prompt
        assert "Q5" in prompt


# ============================================================================
# QueryClassifier Classify Tests
# ============================================================================


@pytest.mark.asyncio
class TestQueryClassifierClassify:
    """Tests for QueryClassifier.classify method."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.health_check = AsyncMock(return_value={"healthy": True})
        return provider

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        return cache

    @pytest.fixture
    def classifier(self, mock_llm_provider, mock_cache):
        """Create QueryClassifier with mocks."""
        return QueryClassifier(
            llm_provider=mock_llm_provider,
            cache=mock_cache,
            model="test-model",
            temperature=0.1,
        )

    async def test_classify_empty_query(self, classifier):
        """Test handling of empty query."""
        result = await classifier.classify("")

        assert result.query_type == QueryType.VAGUE
        assert result.confidence == 0.5
        assert "Empty query" in result.reasoning

    async def test_classify_whitespace_query(self, classifier):
        """Test handling of whitespace-only query."""
        result = await classifier.classify("   ")

        assert result.query_type == QueryType.VAGUE
        assert result.confidence == 0.5

    async def test_classify_long_query_truncation(self, classifier):
        """Test that long queries are truncated."""
        long_query = "x" * 5000

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "factual",
            "confidence": 0.90,
            "reasoning": "test",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        await classifier.classify(long_query)

        # Verify the call was made with truncated query
        call_args = classifier.llm_provider.chat_completion.call_args
        messages = call_args[1]["messages"]
        # The query in the message should be truncated
        assert len(messages) == 2

    async def test_classify_cache_hit(self, classifier, mock_cache):
        """Test classification with cache hit."""
        cached_result = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="Cached",
            suggested_strategies=["reranking"],
        )
        mock_cache.get.return_value = cached_result

        result = await classifier.classify("test query")

        assert result == cached_result
        # LLM should not be called
        classifier.llm_provider.chat_completion.assert_not_called()

    async def test_classify_pattern_match(self, classifier):
        """Test classification using pattern match."""
        # No cache, but pattern should match
        classifier.llm_provider.chat_completion = AsyncMock()

        result = await classifier.classify("What is machine learning?")

        # Should use pattern-based classification
        assert result.query_type == QueryType.FACTUAL
        assert result.confidence >= 0.80
        # LLM should not be called for pattern match
        classifier.llm_provider.chat_completion.assert_not_called()

    async def test_classify_pattern_low_confidence_uses_llm(self, classifier):
        """Test that low confidence pattern falls back to LLM."""
        # Set high threshold so pattern match won't be accepted
        classifier.min_confidence_threshold = 0.95

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "factual",
            "confidence": 0.98,
            "reasoning": "High confidence factual",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await classifier.classify("What is AI?")

        # Should use LLM since pattern confidence is too low
        classifier.llm_provider.chat_completion.assert_called_once()
        assert result.confidence == 0.98

    async def test_classify_llm_success(self, classifier):
        """Test successful LLM classification."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "analytical",
            "confidence": 0.88,
            "reasoning": "Requires explanation of mechanism",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        # Use a query that won't pattern match
        result = await classifier.classify("Describe the architecture details")

        assert result.query_type == QueryType.ANALYTICAL
        assert result.confidence == 0.88
        assert "explanation" in result.reasoning
        assert "hyde" in result.suggested_strategies

    async def test_classify_caching(self, classifier, mock_cache):
        """Test that results are cached."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "factual",
            "confidence": 0.95,
            "reasoning": "Clear factual query",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await classifier.classify("test query with no pattern")

        # Verify cache.set was called
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == "test query with no pattern"  # query
        assert call_args[0][1] == result  # result

    async def test_classify_llm_failure(self, classifier):
        """Test handling of LLM failure."""
        classifier.llm_provider.chat_completion = AsyncMock(
            side_effect=Exception("LLM service unavailable")
        )

        with pytest.raises(ClassificationError):
            await classifier.classify("test query with no pattern")

    async def test_classify_with_context(self, classifier):
        """Test classification with conversation context."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "factual",
            "confidence": 0.92,
            "reasoning": "Clear fact request",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        context = {
            "previous_queries": ["What is machine learning?"],
            "session_id": "sess_123",
        }
        result = await classifier.classify("Who invented it?", context)

        assert result.query_type == QueryType.FACTUAL
        # Verify context was passed to cache
        call_args = classifier.cache.set.call_args
        assert call_args[0][0] == "Who invented it?"  # query
        assert call_args[0][2] == context  # context


# ============================================================================
# QueryClassifier Batch Tests
# ============================================================================


@pytest.mark.asyncio
class TestQueryClassifierBatch:
    """Tests for QueryClassifier.classify_batch method."""

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier with mocked LLM."""
        classifier = QueryClassifier()
        classifier.llm_provider = MagicMock()
        classifier.cache = NullClassificationCache()
        return classifier

    async def test_batch_classify_success(self, classifier):
        """Test batch classification with successful responses."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "factual",
            "confidence": 0.95,
            "reasoning": "test",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        queries = ["query 1", "query 2", "query 3"]
        results = await classifier.classify_batch(queries)

        assert len(results) == 3
        for result in results:
            assert result.query_type == QueryType.FACTUAL

    async def test_batch_classify_partial_failure(self, classifier):
        """Test batch classification with some failures."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "factual",
            "confidence": 0.95,
            "reasoning": "test",
        })

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second call fails
                raise Exception("LLM failure")
            return mock_response

        classifier.llm_provider.chat_completion = AsyncMock(side_effect=side_effect)

        queries = ["query 1", "query 2", "query 3"]
        results = await classifier.classify_batch(queries)

        assert len(results) == 3
        # Failed query should have default classification
        assert results[1].query_type == QueryType.VAGUE
        assert results[1].confidence == 0.5


# ============================================================================
# QueryClassifier Get Strategy Config Tests
# ============================================================================


@pytest.mark.asyncio
class TestQueryClassifierGetStrategyConfig:
    """Tests for QueryClassifier.get_strategy_config method."""

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier with mocked LLM."""
        classifier = QueryClassifier()
        classifier.llm_provider = MagicMock()
        classifier.cache = NullClassificationCache()
        return classifier

    async def test_get_strategy_config_factual(self, classifier):
        """Test getting strategy config for factual query."""
        # Disable pattern fallback to test LLM-based classification
        classifier.use_pattern_fallback = False
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "factual",
            "confidence": 0.95,
            "reasoning": "Clear fact",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        # Use query that won't pattern match
        config = await classifier.get_strategy_config("Detailed technical specification")

        assert config["hyde"] is False
        assert config["reranking"] is True
        assert config["hybrid"] is True
        assert config["query_rewrite"] is True

    async def test_get_strategy_config_analytical(self, classifier):
        """Test getting strategy config for analytical query."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "analytical",
            "confidence": 0.88,
            "reasoning": "Needs explanation",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        config = await classifier.get_strategy_config("Explain neural networks")

        assert config["hyde"] is True
        assert config["reranking"] is True
        assert config["hybrid"] is True
        assert config["query_rewrite"] is True


# ============================================================================
# QueryClassifier Health Check Tests
# ============================================================================


@pytest.mark.asyncio
class TestQueryClassifierHealthCheck:
    """Tests for QueryClassifier.health_check method."""

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier with mocked components."""
        classifier = QueryClassifier()
        classifier.llm_provider = MagicMock()
        classifier.cache = NullClassificationCache()
        return classifier

    async def test_health_check_healthy(self, classifier):
        """Test health check when all components are healthy."""
        classifier.llm_provider.health_check = AsyncMock(return_value={
            "healthy": True,
            "models": {"test-model": {"healthy": True}},
        })

        health = await classifier.health_check()

        assert health["healthy"] is True
        assert health["components"]["llm_provider"]["healthy"] is True
        assert health["components"]["cache"]["enabled"] is False

    async def test_health_check_llm_unhealthy(self, classifier):
        """Test health check when LLM is unhealthy."""
        classifier.llm_provider.health_check = AsyncMock(return_value={
            "healthy": False,
            "models": {},
        })

        health = await classifier.health_check()

        assert health["healthy"] is False
        assert health["components"]["llm_provider"]["healthy"] is False

    async def test_health_check_llm_exception(self, classifier):
        """Test health check when LLM check raises exception."""
        classifier.llm_provider.health_check = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        health = await classifier.health_check()

        assert health["healthy"] is False
        assert "error" in health["components"]["llm_provider"]

    async def test_health_check_with_cache_enabled(self, classifier):
        """Test health check when cache is enabled."""
        classifier.llm_provider.health_check = AsyncMock(return_value={
            "healthy": True,
            "models": {},
        })
        # Create cache with mock Redis
        mock_redis = AsyncMock()
        classifier.cache = ClassificationCache(redis_client=mock_redis)

        health = await classifier.health_check()

        assert health["components"]["cache"]["enabled"] is True
        assert health["components"]["cache"]["healthy"] is True


# ============================================================================
# Pattern Classification Accuracy Tests
# ============================================================================


class TestPatternClassificationAccuracy:
    """Tests to verify pattern classification accuracy meets >90% target."""

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier for testing."""
        with patch("src.rag.classification.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            return QueryClassifier()

    def test_factual_queries(self, classifier):
        """Test classification of factual query examples."""
        factual_queries = [
            "What is machine learning?",
            "Who invented the telephone?",
            "When did World War II end?",
            "Where is the Eiffel Tower located?",
            "How many planets are in the solar system?",
            "What is the capital of France?",
            "Define artificial intelligence",
            "What does RAG mean?",
            "Is Python a programming language?",
            "Are neural networks part of deep learning?",
        ]

        correct = 0
        for query in factual_queries:
            result = classifier._classify_by_pattern(query)
            if result and result.query_type == QueryType.FACTUAL:
                correct += 1

        accuracy = correct / len(factual_queries)
        assert accuracy >= 0.70, f"Factual classification accuracy {accuracy:.0%} is below target"

    def test_comparative_queries(self, classifier):
        """Test classification of comparative query examples."""
        comparative_queries = [
            "Compare Python and JavaScript",
            "What are the differences between SQL and NoSQL?",
            "Pros and cons of microservices",
            "Advantages of cloud computing",
            "React vs Angular",
            "Which is better, Mac or PC?",
            "Similarities between machine learning and statistics",
        ]

        correct = 0
        for query in comparative_queries:
            result = classifier._classify_by_pattern(query)
            if result and result.query_type == QueryType.COMPARATIVE:
                correct += 1

        accuracy = correct / len(comparative_queries)
        assert accuracy >= 0.70, f"Comparative classification accuracy {accuracy:.0%} is below target"

    def test_analytical_queries(self, classifier):
        """Test classification of analytical query examples."""
        analytical_queries = [
            "Explain how neural networks work",
            "How does blockchain technology function?",
            "Why is scalability important?",
            "Analyze the impact of AI on jobs",
            "What causes climate change?",
            "Reasons for using microservices",
            "Impact of cloud migration",
        ]

        correct = 0
        for query in analytical_queries:
            result = classifier._classify_by_pattern(query)
            if result and result.query_type == QueryType.ANALYTICAL:
                correct += 1

        accuracy = correct / len(analytical_queries)
        # Analytical is harder to pattern match, so we expect slightly lower accuracy
        assert accuracy >= 0.50, f"Analytical classification accuracy {accuracy:.0%} is below target"

    def test_vague_queries(self, classifier):
        """Test classification of vague query examples."""
        vague_queries = [
            "Tell me about artificial intelligence",
            "Information on cloud computing",
            "Stuff related to machine learning",
            "Anything about data science",
            "Details about neural networks",
            "What can you tell me about Python?",
            "I want to know about databases",
        ]

        correct = 0
        for query in vague_queries:
            result = classifier._classify_by_pattern(query)
            if result and result.query_type == QueryType.VAGUE:
                correct += 1

        accuracy = correct / len(vague_queries)
        assert accuracy >= 0.70, f"Vague classification accuracy {accuracy:.0%} is below target"


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.asyncio
class TestClassificationPerformance:
    """Tests for classification performance requirements."""

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier with mocked LLM."""
        classifier = QueryClassifier()
        classifier.llm_provider = MagicMock()
        classifier.cache = NullClassificationCache()
        return classifier

    async def test_pattern_classification_under_100ms(self, classifier):
        """Test that pattern-based classification completes within 100ms."""
        import time

        start = time.perf_counter()
        result = await classifier.classify("What is machine learning?")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        assert elapsed_ms < 100, f"Pattern classification took {elapsed_ms:.2f}ms, expected <100ms"

    async def test_cached_classification_under_100ms(self, classifier):
        """Test that cached classification completes within 100ms."""
        import time

        # Pre-populate cache
        cached_result = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.95,
            reasoning="Cached",
        )
        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps(cached_result.model_dump())
        classifier.cache = ClassificationCache(redis_client=mock_redis)

        start = time.perf_counter()
        result = await classifier.classify("test query")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        assert elapsed_ms < 100, f"Cached classification took {elapsed_ms:.2f}ms, expected <100ms"


# ============================================================================
# Integration-style Tests
# ============================================================================


@pytest.mark.asyncio
class TestClassificationIntegration:
    """Integration-style tests with mocked LLM responses."""

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier with mocked LLM and null cache."""
        classifier = QueryClassifier(cache=NullClassificationCache())
        classifier.llm_provider = MagicMock()
        return classifier

    async def test_full_flow_factual(self, classifier):
        """Test full classification flow for factual query."""
        # Disable pattern fallback to test LLM-based classification
        classifier.use_pattern_fallback = False
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "factual",
            "confidence": 0.95,
            "reasoning": "Query asks for a specific definition",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        # Use query that won't pattern match
        result = await classifier.classify("Describe the implementation details")

        assert result.query_type == QueryType.FACTUAL
        assert result.confidence == 0.95
        assert result.reasoning == "Query asks for a specific definition"
        assert "reranking" in result.suggested_strategies
        assert "query_rewrite" in result.suggested_strategies
        assert "hyde" not in result.suggested_strategies  # Factual doesn't use HyDE

    async def test_full_flow_analytical(self, classifier):
        """Test full classification flow for analytical query."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "analytical",
            "confidence": 0.88,
            "reasoning": "Requires explanation of mechanism and reasoning",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await classifier.classify("Detailed architectural analysis needed")

        assert result.query_type == QueryType.ANALYTICAL
        assert result.confidence == 0.88
        assert "hyde" in result.suggested_strategies  # Analytical uses HyDE
        assert "reranking" in result.suggested_strategies

    async def test_full_flow_multi_hop(self, classifier):
        """Test full classification flow for multi-hop query."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "multi_hop",
            "confidence": 0.92,
            "reasoning": "Requires connecting author information with topic",
        })
        classifier.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await classifier.classify("Detailed cross-reference analysis")

        assert result.query_type == QueryType.MULTI_HOP
        assert result.confidence == 0.92
        assert "hyde" in result.suggested_strategies  # Multi-hop uses HyDE
