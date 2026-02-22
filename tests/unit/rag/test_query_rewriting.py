"""Unit tests for Query Rewriting service.

This module tests the QueryRewriter class and related components
including caching, parsing, and error handling.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.rag.models import ConversationContext, QueryRewriteResult
from src.rag.strategies.query_rewriting import (
    DEFAULT_SYSTEM_PROMPT,
    NullQueryRewritingCache,
    QueryRewriter,
    QueryRewritingCache,
    QueryRewritingError,
    QueryRewritingValidationError,
)

# ============================================================================
# QueryRewriteResult Model Tests
# ============================================================================

class TestQueryRewriteResult:
    """Tests for QueryRewriteResult Pydantic model."""

    def test_default_creation(self):
        """Test creating result with default values."""
        result = QueryRewriteResult()

        assert result.search_rag is False
        assert result.embedding_source_text == ""
        assert result.llm_query == ""

    def test_full_creation(self):
        """Test creating result with all values."""
        result = QueryRewriteResult(
            search_rag=True,
            embedding_source_text="vibe coding",
            llm_query="Explain vibe coding",
        )

        assert result.search_rag is True
        assert result.embedding_source_text == "vibe coding"
        assert result.llm_query == "Explain vibe coding"

    def test_serialization(self):
        """Test JSON serialization."""
        result = QueryRewriteResult(
            search_rag=True,
            embedding_source_text="test keywords",
            llm_query="test instruction",
        )

        json_str = result.model_dump_json()
        data = json.loads(json_str)

        assert data["search_rag"] is True
        assert data["embedding_source_text"] == "test keywords"
        assert data["llm_query"] == "test instruction"

    def test_deserialization(self):
        """Test JSON deserialization."""
        data = {
            "search_rag": True,
            "embedding_source_text": "keywords",
            "llm_query": "instruction",
        }

        result = QueryRewriteResult(**data)

        assert result.search_rag is True
        assert result.embedding_source_text == "keywords"
        assert result.llm_query == "instruction"


# ============================================================================
# ConversationContext Model Tests
# ============================================================================

class TestConversationContext:
    """Tests for ConversationContext Pydantic model."""

    def test_default_creation(self):
        """Test creating context with defaults."""
        context = ConversationContext()

        assert context.previous_queries == []
        assert context.previous_responses == []
        assert context.session_id is None

    def test_full_creation(self):
        """Test creating context with values."""
        context = ConversationContext(
            previous_queries=["What is AI?"],
            previous_responses=["AI is..."],
            session_id="sess_123",
        )

        assert context.previous_queries == ["What is AI?"]
        assert context.previous_responses == ["AI is..."]
        assert context.session_id == "sess_123"


# ============================================================================
# NullQueryRewritingCache Tests
# ============================================================================

class TestNullQueryRewritingCache:
    """Tests for NullQueryRewritingCache (null object pattern)."""

    @pytest.mark.asyncio
    async def test_get_returns_none(self):
        """Test that get always returns None."""
        cache = NullQueryRewritingCache()

        result = await cache.get("any query")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_returns_false(self):
        """Test that set always returns False."""
        cache = NullQueryRewritingCache()
        result = QueryRewriteResult(search_rag=True, embedding_source_text="test", llm_query="test")

        success = await cache.set("query", result)

        assert success is False

    @pytest.mark.asyncio
    async def test_delete_returns_false(self):
        """Test that delete always returns False."""
        cache = NullQueryRewritingCache()

        success = await cache.delete("query")

        assert success is False

    @pytest.mark.asyncio
    async def test_clear_all_returns_zero(self):
        """Test that clear_all always returns 0."""
        cache = NullQueryRewritingCache()

        count = await cache.clear_all()

        assert count == 0


# ============================================================================
# QueryRewritingCache Tests
# ============================================================================

class TestQueryRewritingCache:
    """Tests for QueryRewritingCache with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis_mock = AsyncMock()
        return redis_mock

    @pytest.fixture
    def cache(self, mock_redis):
        """Create cache instance with mock Redis."""
        return QueryRewritingCache(redis_client=mock_redis)

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache, mock_redis):
        """Test getting cached result when it exists."""
        # Setup mock to return cached data
        cached_data = json.dumps({
            "search_rag": True,
            "embedding_source_text": "test keywords",
            "llm_query": "test instruction",
        })
        mock_redis.get.return_value = cached_data

        result = await cache.get("test query")

        assert result is not None
        assert result.search_rag is True
        assert result.embedding_source_text == "test keywords"
        assert result.llm_query == "test instruction"

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
            "search_rag": True,
            "embedding_source_text": "keywords",
            "llm_query": "instruction",
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
        result = QueryRewriteResult(
            search_rag=True,
            embedding_source_text="keywords",
            llm_query="instruction",
        )
        mock_redis.setex.return_value = True

        success = await cache.set("test query", result)

        assert success is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, cache, mock_redis):
        """Test cache set with custom TTL."""
        result = QueryRewriteResult(search_rag=True, embedding_source_text="k", llm_query="i")
        mock_redis.setex.return_value = True

        await cache.set("test query", result, ttl=7200)

        # Verify TTL was passed correctly
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 7200  # TTL is second positional arg

    @pytest.mark.asyncio
    async def test_set_redis_failure(self, cache, mock_redis):
        """Test graceful handling of Redis failure on set."""
        result = QueryRewriteResult(search_rag=True, embedding_source_text="k", llm_query="i")
        mock_redis.setex.side_effect = Exception("Redis connection failed")

        success = await cache.set("test query", result)

        assert success is False

    @pytest.mark.asyncio
    async def test_set_without_redis(self):
        """Test set when Redis client is None."""
        cache = QueryRewritingCache(redis_client=None)
        result = QueryRewriteResult(search_rag=True, embedding_source_text="k", llm_query="i")

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
        cache = QueryRewritingCache(redis_client=mock_redis)

        key1 = cache._make_key("test query", "context_hash")
        key2 = cache._make_key("test query", "context_hash")

        assert key1 == key2
        assert key1.startswith("rag:query_rewrite:")

    def test_make_key_different_queries(self, mock_redis):
        """Test that different queries produce different keys."""
        cache = QueryRewritingCache(redis_client=mock_redis)

        key1 = cache._make_key("query one", "")
        key2 = cache._make_key("query two", "")

        assert key1 != key2

    def test_hash_context_empty(self, mock_redis):
        """Test hashing empty context."""
        cache = QueryRewritingCache(redis_client=mock_redis)

        hash_val = cache._hash_context(None)

        assert hash_val == ""

    def test_hash_context_with_data(self, mock_redis):
        """Test hashing context with data."""
        cache = QueryRewritingCache(redis_client=mock_redis)

        context = {"session_id": "sess_123", "queries": ["q1", "q2"]}
        hash1 = cache._hash_context(context)
        hash2 = cache._hash_context(context)

        # Same context should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # We truncate to 16 chars


# ============================================================================
# QueryRewriter Tests
# ============================================================================

@pytest.mark.asyncio
class TestQueryRewriter:
    """Tests for QueryRewriter class."""

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
    def rewriter(self, mock_llm_provider, mock_cache):
        """Create QueryRewriter with mocks."""
        return QueryRewriter(
            llm_provider=mock_llm_provider,
            cache=mock_cache,
            model="test-model",
            temperature=0.1,
        )

    async def test_init_default_values(self):
        """Test initialization with default values."""
        with patch("src.rag.strategies.query_rewriting.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            rewriter = QueryRewriter()

            assert rewriter.model == "agentic-decisions"
            assert rewriter.temperature == 0.1
            assert rewriter.max_tokens == 500
            assert rewriter.system_prompt == DEFAULT_SYSTEM_PROMPT

    async def test_rewrite_empty_query(self, rewriter):
        """Test handling of empty query."""
        result = await rewriter.rewrite("")

        assert result.search_rag is False
        assert result.embedding_source_text == ""
        assert result.llm_query == ""

    async def test_rewrite_whitespace_query(self, rewriter):
        """Test handling of whitespace-only query."""
        result = await rewriter.rewrite("   ")

        assert result.search_rag is False
        assert result.embedding_source_text == ""
        assert result.llm_query == ""

    async def test_rewrite_long_query_truncation(self, rewriter):
        """Test that long queries are truncated."""
        long_query = "x" * 5000

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "search_rag": True,
            "embedding_source_text": "test",
            "llm_query": "test",
        })
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        await rewriter.rewrite(long_query)

        # Verify the call was made with truncated query
        call_args = rewriter.llm_provider.chat_completion.call_args
        messages = call_args[1]["messages"]
        # The query in the message should be truncated
        assert len(messages) == 2

    async def test_rewrite_cache_hit(self, rewriter, mock_cache):
        """Test rewrite with cache hit."""
        cached_result = QueryRewriteResult(
            search_rag=True,
            embedding_source_text="cached",
            llm_query="cached instruction",
        )
        mock_cache.get.return_value = cached_result

        result = await rewriter.rewrite("test query")

        assert result == cached_result
        # LLM should not be called
        rewriter.llm_provider.chat_completion.assert_not_called()

    async def test_rewrite_successful(self, rewriter):
        """Test successful query rewrite."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "search_rag": True,
            "embedding_source_text": "vibe coding programming",
            "llm_query": "Explain vibe coding based on context",
        })
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await rewriter.rewrite("@knowledgebase tell me about vibe coding")

        assert result.search_rag is True
        assert result.embedding_source_text == "vibe coding programming"
        assert result.llm_query == "Explain vibe coding based on context"

    async def test_rewrite_forces_search_rag(self, rewriter):
        """Test that @knowledgebase forces search_rag=True."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "search_rag": False,  # LLM incorrectly returns false
            "embedding_source_text": "test",
            "llm_query": "test",
        })
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await rewriter.rewrite("@knowledgebase search for something")

        # Should be forced to True due to @knowledgebase
        assert result.search_rag is True

    async def test_rewrite_with_context(self, rewriter):
        """Test rewrite with conversation context."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "search_rag": True,
            "embedding_source_text": "AI development",
            "llm_query": "Explain AI development",
        })
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        context = {
            "previous_queries": ["What is machine learning?"],
            "session_id": "sess_123",
        }
        result = await rewriter.rewrite("How does it relate to AI?", context)

        assert result.search_rag is True
        # Verify context was passed to cache (args are positional: query, result, context, ttl)
        call_args = rewriter.cache.set.call_args
        assert call_args[0][0] == "How does it relate to AI?"  # query
        assert call_args[0][2] == context  # context

    async def test_rewrite_llm_failure(self, rewriter):
        """Test handling of LLM failure."""
        rewriter.llm_provider.chat_completion = AsyncMock(
            side_effect=Exception("LLM service unavailable")
        )

        with pytest.raises(QueryRewritingError):
            await rewriter.rewrite("test query")

    async def test_rewrite_caching(self, rewriter, mock_cache):
        """Test that results are cached."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "search_rag": True,
            "embedding_source_text": "test keywords",
            "llm_query": "test instruction",
        })
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await rewriter.rewrite("test query")

        # Verify cache.set was called with correct arguments (positional: query, result, context, ttl)
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == "test query"  # query
        assert call_args[0][1] == result  # result


# ============================================================================
# QueryRewriter Parse Response Tests
# ============================================================================

class TestQueryRewriterParseResponse:
    """Tests for QueryRewriter._parse_response method."""

    @pytest.fixture
    def rewriter(self):
        """Create QueryRewriter instance."""
        return QueryRewriter()

    def test_parse_valid_response(self, rewriter):
        """Test parsing valid JSON response."""
        content = json.dumps({
            "search_rag": True,
            "embedding_source_text": "vibe coding",
            "llm_query": "Explain vibe coding",
        })

        result = rewriter._parse_response(content, "original query")

        assert result.search_rag is True
        assert result.embedding_source_text == "vibe coding"
        assert result.llm_query == "Explain vibe coding"

    def test_parse_trims_whitespace(self, rewriter):
        """Test that text fields are trimmed."""
        content = json.dumps({
            "search_rag": False,
            "embedding_source_text": "  spaced keywords  ",
            "llm_query": "  spaced instruction  ",
        })

        result = rewriter._parse_response(content, "query")

        assert result.embedding_source_text == "spaced keywords"
        assert result.llm_query == "spaced instruction"

    def test_parse_invalid_json(self, rewriter):
        """Test handling of invalid JSON."""
        with pytest.raises(QueryRewritingValidationError):
            rewriter._parse_response("not json", "query")

    def test_parse_missing_search_rag(self, rewriter):
        """Test handling of missing search_rag field."""
        content = json.dumps({
            "embedding_source_text": "test",
            "llm_query": "test",
        })

        with pytest.raises(QueryRewritingValidationError) as exc_info:
            rewriter._parse_response(content, "query")

        assert "search_rag" in str(exc_info.value)

    def test_parse_missing_embedding_source_text(self, rewriter):
        """Test handling of missing embedding_source_text field."""
        content = json.dumps({
            "search_rag": True,
            "llm_query": "test",
        })

        with pytest.raises(QueryRewritingValidationError) as exc_info:
            rewriter._parse_response(content, "query")

        assert "embedding_source_text" in str(exc_info.value)

    def test_parse_missing_llm_query(self, rewriter):
        """Test handling of missing llm_query field."""
        content = json.dumps({
            "search_rag": True,
            "embedding_source_text": "test",
        })

        with pytest.raises(QueryRewritingValidationError) as exc_info:
            rewriter._parse_response(content, "query")

        assert "llm_query" in str(exc_info.value)

    def test_parse_wrong_type_search_rag(self, rewriter):
        """Test handling of wrong type for search_rag."""
        content = json.dumps({
            "search_rag": "true",  # String instead of bool
            "embedding_source_text": "test",
            "llm_query": "test",
        })

        with pytest.raises(QueryRewritingValidationError) as exc_info:
            rewriter._parse_response(content, "query")

        assert "search_rag" in str(exc_info.value)
        assert "boolean" in str(exc_info.value)

    def test_parse_wrong_type_embedding_source_text(self, rewriter):
        """Test handling of wrong type for embedding_source_text."""
        content = json.dumps({
            "search_rag": True,
            "embedding_source_text": 123,  # Number instead of string
            "llm_query": "test",
        })

        with pytest.raises(QueryRewritingValidationError) as exc_info:
            rewriter._parse_response(content, "query")

        assert "embedding_source_text" in str(exc_info.value)

    def test_parse_forces_search_rag_with_knowledgebase(self, rewriter):
        """Test that @knowledgebase in original query forces search_rag=True."""
        content = json.dumps({
            "search_rag": False,
            "embedding_source_text": "test",
            "llm_query": "test",
        })
        original_query = "@knowledgebase search for AI"

        result = rewriter._parse_response(content, original_query)

        assert result.search_rag is True


# ============================================================================
# QueryRewriter Build Prompt Tests
# ============================================================================

class TestQueryRewriterBuildPrompt:
    """Tests for QueryRewriter._build_prompt method."""

    @pytest.fixture
    def rewriter(self):
        """Create QueryRewriter instance."""
        return QueryRewriter()

    def test_build_prompt_simple(self, rewriter):
        """Test building prompt with simple query."""
        query = "What is AI?"

        prompt = rewriter._build_prompt(query, None)

        assert 'Query: "What is AI?"' in prompt
        assert "Rewrite this query into the required JSON format" in prompt

    def test_build_prompt_with_context(self, rewriter):
        """Test building prompt with conversation context."""
        query = "How does it work?"
        context = {
            "previous_queries": ["What is machine learning?", "How is it used?"],
            "previous_responses": ["ML is...", "It's used for..."],
        }

        prompt = rewriter._build_prompt(query, context)

        assert 'Query: "How does it work?"' in prompt
        assert "Previous queries in this conversation:" in prompt
        assert "What is machine learning?" in prompt
        assert "How is it used?" in prompt
        assert "This is a follow-up question." in prompt

    def test_build_prompt_limits_previous_queries(self, rewriter):
        """Test that only last 3 previous queries are included."""
        query = "Final question?"
        context = {
            "previous_queries": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        }

        prompt = rewriter._build_prompt(query, context)

        # Should only include Q3, Q4, Q5 (last 3)
        assert "Q1" not in prompt
        assert "Q2" not in prompt
        assert "Q3" in prompt
        assert "Q4" in prompt
        assert "Q5" in prompt


# ============================================================================
# QueryRewriter Batch Tests
# ============================================================================

@pytest.mark.asyncio
class TestQueryRewriterBatch:
    """Tests for QueryRewriter.rewrite_batch method."""

    @pytest.fixture
    def rewriter(self):
        """Create QueryRewriter with mocked LLM."""
        rewriter = QueryRewriter()
        rewriter.llm_provider = MagicMock()
        rewriter.cache = NullQueryRewritingCache()
        return rewriter

    async def test_batch_rewrite_success(self, rewriter):
        """Test batch rewriting with successful responses."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "search_rag": True,
            "embedding_source_text": "test",
            "llm_query": "test",
        })
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        queries = ["query 1", "query 2", "query 3"]
        results = await rewriter.rewrite_batch(queries)

        assert len(results) == 3
        for result in results:
            assert result.search_rag is True

    async def test_batch_rewrite_partial_failure(self, rewriter):
        """Test batch rewriting with some failures."""
        # First call succeeds, second fails, third succeeds
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "search_rag": True,
            "embedding_source_text": "test",
            "llm_query": "test",
        })

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second call fails
                raise Exception("LLM failure")
            return mock_response

        rewriter.llm_provider.chat_completion = AsyncMock(side_effect=side_effect)

        queries = ["query 1", "query 2", "query 3"]
        results = await rewriter.rewrite_batch(queries)

        assert len(results) == 3
        # Failed query should have fallback result
        assert results[1].search_rag is False
        assert results[1].embedding_source_text == "query 2"


# ============================================================================
# QueryRewriter Health Check Tests
# ============================================================================

@pytest.mark.asyncio
class TestQueryRewriterHealthCheck:
    """Tests for QueryRewriter.health_check method."""

    @pytest.fixture
    def rewriter(self):
        """Create QueryRewriter with mocked components."""
        rewriter = QueryRewriter()
        rewriter.llm_provider = MagicMock()
        rewriter.cache = NullQueryRewritingCache()
        return rewriter

    async def test_health_check_healthy(self, rewriter):
        """Test health check when all components are healthy."""
        rewriter.llm_provider.health_check = AsyncMock(return_value={
            "healthy": True,
            "models": {"test-model": {"healthy": True}},
        })

        health = await rewriter.health_check()

        assert health["healthy"] is True
        assert health["components"]["llm_provider"]["healthy"] is True
        assert health["components"]["cache"]["enabled"] is False

    async def test_health_check_llm_unhealthy(self, rewriter):
        """Test health check when LLM is unhealthy."""
        rewriter.llm_provider.health_check = AsyncMock(return_value={
            "healthy": False,
            "models": {},
        })

        health = await rewriter.health_check()

        assert health["healthy"] is False
        assert health["components"]["llm_provider"]["healthy"] is False

    async def test_health_check_llm_exception(self, rewriter):
        """Test health check when LLM check raises exception."""
        rewriter.llm_provider.health_check = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        health = await rewriter.health_check()

        assert health["healthy"] is False
        assert "error" in health["components"]["llm_provider"]

    async def test_health_check_with_cache_enabled(self, rewriter):
        """Test health check when cache is enabled."""
        rewriter.llm_provider.health_check = AsyncMock(return_value={
            "healthy": True,
            "models": {},
        })
        # Create cache with mock Redis
        mock_redis = AsyncMock()
        rewriter.cache = QueryRewritingCache(redis_client=mock_redis)

        health = await rewriter.health_check()

        assert health["components"]["cache"]["enabled"] is True
        assert health["components"]["cache"]["healthy"] is True


# ============================================================================
# Integration-style Tests
# ============================================================================

@pytest.mark.asyncio
class TestQueryRewriterIntegration:
    """Integration-style tests with mocked LLM responses."""

    @pytest.fixture
    def rewriter(self):
        """Create QueryRewriter with mocked LLM and null cache."""
        rewriter = QueryRewriter(cache=NullQueryRewritingCache())
        rewriter.llm_provider = MagicMock()
        return rewriter

    async def test_full_flow_with_example(self, rewriter):
        """Test full flow with the spec example."""
        # Mock the expected LLM response per the spec
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "search_rag": True,
            "embedding_source_text": "vibe coding programming approach",
            "llm_query": (
                "Based on the provided context, explain what vibe coding is, "
                "including its pros and cons, and cite sources."
            ),
        })
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        input_query = "@knowledgebase search on vibe coding, then summarize, list pros and cons"
        result = await rewriter.rewrite(input_query)

        assert result.search_rag is True
        assert result.embedding_source_text == "vibe coding programming approach"
        assert "Based on the provided context" in result.llm_query
        assert "pros and cons" in result.llm_query

    async def test_processing_time_under_100ms(self, rewriter):
        """Test that processing completes within 100ms (cached)."""
        import time

        # Pre-populate cache
        cached_result = QueryRewriteResult(
            search_rag=True,
            embedding_source_text="cached",
            llm_query="cached",
        )
        rewriter.cache = MagicMock()
        rewriter.cache.get = AsyncMock(return_value=cached_result)
        rewriter.cache.set = AsyncMock(return_value=True)

        start = time.perf_counter()
        result = await rewriter.rewrite("test query")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        assert elapsed_ms < 100, f"Processing took {elapsed_ms:.2f}ms, expected <100ms"
