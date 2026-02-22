"""Unit tests for HyDE (Hypothetical Document Embeddings) service.

This module tests the HyDERewriter class and related components
including hypothetical document generation, caching, and fallback behavior.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.rag.models import QueryRewriteResult
from src.rag.strategies.hyde import (
    DEFAULT_HYDE_SYSTEM_PROMPT,
    HyDERewriter,
    HyDERewritingError,
)
from src.rag.strategies.query_rewriting import NullQueryRewritingCache, QueryRewritingCache

# ============================================================================
# HyDERewriter Initialization Tests
# ============================================================================


class TestHyDERewriterInitialization:
    """Tests for HyDERewriter initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            rewriter = HyDERewriter()

            assert rewriter.model == "agentic-decisions"
            assert rewriter.temperature == 0.7
            assert rewriter.max_tokens == 300
            assert rewriter.max_hypothetical_length == 512
            assert rewriter.cache_ttl == 7200
            assert rewriter.system_prompt == DEFAULT_HYDE_SYSTEM_PROMPT
            assert rewriter.fallback_to_standard is True
            assert rewriter.max_processing_time_ms == 500

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        custom_system_prompt = "Custom prompt for testing"

        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            rewriter = HyDERewriter(
                model="gpt-4o",
                temperature=0.5,
                max_tokens=200,
                max_hypothetical_length=256,
                cache_ttl=3600,
                system_prompt=custom_system_prompt,
                fallback_to_standard=False,
                max_processing_time_ms=300,
            )

            assert rewriter.model == "gpt-4o"
            assert rewriter.temperature == 0.5
            assert rewriter.max_tokens == 200
            assert rewriter.max_hypothetical_length == 256
            assert rewriter.cache_ttl == 3600
            assert rewriter.system_prompt == custom_system_prompt
            assert rewriter.fallback_to_standard is False
            assert rewriter.max_processing_time_ms == 300

    def test_init_with_query_types(self):
        """Test initialization with specific query types."""
        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            rewriter = HyDERewriter(
                enable_for_query_types=["complex_questions", "vague_queries"]
            )

            assert rewriter.enable_for_query_types == [
                "complex_questions",
                "vague_queries",
            ]


# ============================================================================
# HyDERewriter from_settings Tests
# ============================================================================


class TestHyDERewriterFromSettings:
    """Tests for HyDERewriter.from_settings factory method."""

    def test_from_settings_with_defaults(self):
        """Test creating from default settings."""
        mock_settings = MagicMock()
        mock_settings.hyde.enabled = True
        mock_settings.hyde.model = "agentic-decisions"
        mock_settings.hyde.temperature = 0.7
        mock_settings.hyde.max_tokens = 300
        mock_settings.hyde.max_hypothetical_length = 512
        mock_settings.hyde.system_prompt = "Test prompt"
        mock_settings.hyde.cache_enabled = False
        mock_settings.hyde.cache_ttl = 7200
        mock_settings.hyde.enable_for_query_types = ["complex_questions"]
        mock_settings.hyde.fallback_to_standard = True
        mock_settings.hyde.max_processing_time_ms = 500
        mock_settings.redis.url = "redis://localhost:6379/0"

        with patch(
            "src.rag.strategies.hyde.get_settings", return_value=mock_settings
        ), patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            rewriter = HyDERewriter.from_settings(mock_settings)

            assert rewriter.model == "agentic-decisions"
            assert rewriter.temperature == 0.7
            assert isinstance(rewriter.cache, NullQueryRewritingCache)


# ============================================================================
# HyDERewriter Helper Method Tests
# ============================================================================


class TestHyDERewriterHelpers:
    """Tests for HyDERewriter helper methods."""

    @pytest.fixture
    def rewriter(self):
        """Create HyDERewriter instance with mocked LLM."""
        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            return HyDERewriter()

    def test_make_cache_key_consistency(self, rewriter):
        """Test that same query produces same cache key."""
        key1 = rewriter._make_cache_key("test query")
        key2 = rewriter._make_cache_key("test query")

        assert key1 == key2
        assert key1.startswith("rag:hyde:")

    def test_make_cache_key_different_queries(self, rewriter):
        """Test that different queries produce different keys."""
        key1 = rewriter._make_cache_key("query one")
        key2 = rewriter._make_cache_key("query two")

        assert key1 != key2

    def test_should_use_hyde_with_knowledgebase(self, rewriter):
        """Test that @knowledgebase always enables HyDE."""
        assert rewriter._should_use_hyde("@knowledgebase what is AI") is True
        assert rewriter._should_use_hyde("Tell me @knowledgebase about ML") is True

    def test_should_use_hyde_without_knowledgebase(self, rewriter):
        """Test HyDE decision without knowledgebase trigger."""
        # With default enable_for_query_types, should return True
        assert rewriter._should_use_hyde("What is AI?") is True

    def test_truncate_hypothetical_under_limit(self, rewriter):
        """Test that short text is not truncated."""
        text = "This is a short hypothetical document."
        result = rewriter._truncate_hypothetical(text)
        assert result == text

    def test_truncate_hypothetical_over_limit(self, rewriter):
        """Test that long text is truncated."""
        rewriter.max_hypothetical_length = 50
        text = "This is a very long hypothetical document that should be truncated to fit the limit."
        result = rewriter._truncate_hypothetical(text)
        assert len(result) <= 50

    def test_truncate_hypothetical_at_sentence_boundary(self, rewriter):
        """Test truncation at sentence boundary when possible."""
        rewriter.max_hypothetical_length = 60
        text = "First sentence. Second sentence. Third sentence that is very long."
        result = rewriter._truncate_hypothetical(text)
        # Should end at a period if one exists in the first half
        assert result.endswith(".")

    def test_create_llm_prompt(self, rewriter):
        """Test LLM prompt creation."""
        query = "What is vibe coding?"
        prompt = rewriter._create_llm_prompt(query)

        assert "Based on the provided context" in prompt
        assert query in prompt


# ============================================================================
# HyDERewriter generate_hypothetical_document Tests
# ============================================================================


@pytest.mark.asyncio
class TestGenerateHypotheticalDocument:
    """Tests for generate_hypothetical_document method."""

    @pytest.fixture
    def rewriter(self):
        """Create HyDERewriter with mocked LLM."""
        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_llm = MagicMock()
            mock_provider_cls.return_value = mock_llm
            rewriter = HyDERewriter()
            rewriter.llm_provider = mock_llm
            return rewriter

    async def test_generate_success(self, rewriter):
        """Test successful hypothetical document generation."""
        mock_response = MagicMock()
        mock_response.content = (
            "Vibe coding is a programming approach emphasizing intuition and flow state."
        )
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await rewriter.generate_hypothetical_document(
            "What is vibe coding?"
        )

        assert "Vibe coding" in result
        assert "intuition" in result
        rewriter.llm_provider.chat_completion.assert_called_once()

    async def test_generate_with_context(self, rewriter):
        """Test generation with conversation context."""
        mock_response = MagicMock()
        mock_response.content = "AI is artificial intelligence technology."
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        context = {"previous_queries": ["What is ML?"], "session_id": "sess_123"}
        result = await rewriter.generate_hypothetical_document(
            "What is AI?", context
        )

        assert "AI" in result

    async def test_generate_truncates_long_response(self, rewriter):
        """Test that long responses are truncated."""
        rewriter.max_hypothetical_length = 50
        mock_response = MagicMock()
        mock_response.content = "a" * 100  # Very long response
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await rewriter.generate_hypothetical_document("test query")

        assert len(result) <= 50

    async def test_generate_llm_failure(self, rewriter):
        """Test handling of LLM failure."""
        rewriter.llm_provider.chat_completion = AsyncMock(
            side_effect=Exception("LLM service unavailable")
        )

        with pytest.raises(HyDERewritingError):
            await rewriter.generate_hypothetical_document("test query")


# ============================================================================
# HyDERewriter rewrite Tests
# ============================================================================


@pytest.mark.asyncio
class TestHyDERewriterRewrite:
    """Tests for rewrite method."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        return provider

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        cache.redis = AsyncMock()
        return cache

    @pytest.fixture
    def rewriter(self, mock_llm_provider, mock_cache):
        """Create HyDERewriter with mocks."""
        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_provider_cls.return_value = mock_llm_provider
            return HyDERewriter(
                llm_provider=mock_llm_provider,
                cache=mock_cache,
                model="test-model",
                temperature=0.7,
            )

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

    async def test_rewrite_success_with_knowledgebase(self, rewriter):
        """Test successful rewrite with @knowledgebase trigger."""
        mock_response = MagicMock()
        mock_response.content = (
            "Vibe coding is a programming approach that emphasizes intuition and flow state. "
            "This coding style prioritizes developer comfort and creativity."
        )
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await rewriter.rewrite("@knowledgebase What is vibe coding?")

        assert result.search_rag is True
        assert "Vibe coding" in result.embedding_source_text
        assert "intuition" in result.embedding_source_text
        assert "Based on the provided context" in result.llm_query

    async def test_rewrite_success_without_knowledgebase(self, rewriter):
        """Test successful rewrite without @knowledgebase trigger."""
        mock_response = MagicMock()
        mock_response.content = "Machine learning is a subset of AI."
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await rewriter.rewrite("What is machine learning?")

        assert result.search_rag is False  # No @knowledgebase
        assert "Machine learning" in result.embedding_source_text

    async def test_rewrite_uses_cache(self, rewriter, mock_cache):
        """Test that rewrite uses cached results."""
        # Setup cache hit
        cached_data = json.dumps(
            {
                "hypothetical": "Cached hypothetical document about vibe coding.",
            }
        )
        mock_cache.redis.get = AsyncMock(return_value=cached_data)

        result = await rewriter.rewrite("@knowledgebase What is vibe coding?")

        assert result.search_rag is True
        assert result.embedding_source_text == "Cached hypothetical document about vibe coding."
        # LLM should not be called when cache hit
        rewriter.llm_provider.chat_completion.assert_not_called()

    async def test_rewrite_caches_result(self, rewriter, mock_cache):
        """Test that successful rewrite caches the result."""
        mock_response = MagicMock()
        mock_response.content = "Generated hypothetical document."
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)
        mock_cache.redis.setex = AsyncMock(return_value=True)

        await rewriter.rewrite("@knowledgebase Test query")

        # Verify cache was set
        mock_cache.redis.setex.assert_called_once()
        call_args = mock_cache.redis.setex.call_args
        assert call_args[0][1] == 7200  # TTL

    async def test_rewrite_fallback_on_failure(self, rewriter):
        """Test fallback to standard rewriting on HyDE failure."""
        rewriter.fallback_to_standard = True

        # Mock parent class rewrite
        fallback_result = QueryRewriteResult(
            search_rag=True,
            embedding_source_text="fallback keywords",
            llm_query="fallback prompt",
        )

        # Make HyDE generation fail
        rewriter.llm_provider.chat_completion = AsyncMock(
            side_effect=Exception("LLM failure")
        )

        # Mock the parent rewrite method
        with patch.object(
            rewriter.__class__.__bases__[0],
            "rewrite",
            AsyncMock(return_value=fallback_result),
        ) as mock_parent_rewrite:
            result = await rewriter.rewrite("@knowledgebase Test query")

            assert result == fallback_result
            mock_parent_rewrite.assert_called_once()

    async def test_rewrite_no_fallback_on_failure(self, rewriter):
        """Test no fallback when fallback_to_standard is False."""
        rewriter.fallback_to_standard = False

        rewriter.llm_provider.chat_completion = AsyncMock(
            side_effect=Exception("LLM failure")
        )

        with pytest.raises(HyDERewritingError):
            await rewriter.rewrite("@knowledgebase Test query")

    async def test_rewrite_with_conversation_context(self, rewriter):
        """Test rewrite with conversation context."""
        mock_response = MagicMock()
        mock_response.content = "AI is artificial intelligence."
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        context = {
            "previous_queries": ["What is ML?"],
            "previous_responses": ["ML is machine learning."],
            "session_id": "sess_123",
        }

        result = await rewriter.rewrite("@knowledgebase How is AI different?", context)

        assert result.search_rag is True
        assert "AI" in result.embedding_source_text


# ============================================================================
# HyDERewriter Batch Tests
# ============================================================================


@pytest.mark.asyncio
class TestHyDERewriterBatch:
    """Tests for rewrite_batch method."""

    @pytest.fixture
    def rewriter(self):
        """Create HyDERewriter with mocked LLM."""
        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_llm = MagicMock()
            mock_provider_cls.return_value = mock_llm
            rewriter = HyDERewriter()
            rewriter.llm_provider = mock_llm
            rewriter.cache = NullQueryRewritingCache()
            return rewriter

    async def test_batch_rewrite_success(self, rewriter):
        """Test batch rewriting with successful responses."""
        mock_response = MagicMock()
        mock_response.content = "Hypothetical document."
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        queries = [
            "@knowledgebase What is AI?",
            "@knowledgebase What is ML?",
            "@knowledgebase What is DL?",
        ]
        results = await rewriter.rewrite_batch(queries)

        assert len(results) == 3
        for result in results:
            assert result.search_rag is True
            assert "Hypothetical" in result.embedding_source_text

    async def test_batch_rewrite_partial_failure(self, rewriter):
        """Test batch rewriting with some failures."""
        # Mock HyDE generation responses (plain text)
        hyde_response = MagicMock()
        hyde_response.content = (
            "Vibe coding is a programming approach that emphasizes intuition."
        )

        # Mock standard query rewriting response (JSON format for parent class)
        standard_response = MagicMock()
        standard_response.content = json.dumps({
            "search_rag": True,
            "embedding_source_text": "fallback keywords",
            "llm_query": "fallback prompt",
        })

        # Track calls:
        # Call 1: HyDE for Query 1 (success) -> hyde_response
        # Call 2: HyDE for Query 2 (failure) -> raises Exception
        # Call 3: Fallback for Query 2 (success) -> standard_response (JSON!)
        # Call 4: HyDE for Query 3 (success) -> hyde_response
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return hyde_response  # Query 1 HyDE success
            elif call_count == 2:
                raise Exception("LLM failure")  # Query 2 HyDE failure
            elif call_count == 3:
                return standard_response  # Query 2 fallback (must be JSON!)
            elif call_count == 4:
                return hyde_response  # Query 3 HyDE success
            return hyde_response

        rewriter.llm_provider.chat_completion = AsyncMock(side_effect=side_effect)

        queries = [
            "@knowledgebase Query 1",
            "@knowledgebase Query 2",
            "@knowledgebase Query 3",
        ]
        results = await rewriter.rewrite_batch(queries)

        assert len(results) == 3
        # Failed query should have fallback result
        assert results[1].embedding_source_text == "fallback keywords"


# ============================================================================
# HyDE Cache Integration Tests
# ============================================================================


@pytest.mark.asyncio
class TestHyDECacheIntegration:
    """Tests for HyDE caching with Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis_mock = AsyncMock()
        return redis_mock

    @pytest.fixture
    def rewriter_with_cache(self, mock_redis):
        """Create HyDERewriter with mock Redis cache."""
        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_llm = MagicMock()
            mock_provider_cls.return_value = mock_llm

            cache = QueryRewritingCache(redis_client=mock_redis)
            rewriter = HyDERewriter(
                llm_provider=mock_llm,
                cache=cache,
                cache_ttl=7200,
            )
            return rewriter

    async def test_cache_hit_returns_cached_hypothetical(self, rewriter_with_cache, mock_redis):
        """Test that cache hit returns cached hypothetical document."""
        cached_data = json.dumps(
            {
                "hypothetical": "Cached vibe coding explanation.",
            }
        )
        mock_redis.get.return_value = cached_data

        result = await rewriter_with_cache.rewrite("@knowledgebase What is vibe coding?")

        assert result.embedding_source_text == "Cached vibe coding explanation."
        # LLM should not be called
        rewriter_with_cache.llm_provider.chat_completion.assert_not_called()

    async def test_cache_miss_generates_new(self, rewriter_with_cache, mock_redis):
        """Test that cache miss generates new hypothetical document."""
        mock_redis.get.return_value = None

        mock_response = MagicMock()
        mock_response.content = "Newly generated hypothetical."
        rewriter_with_cache.llm_provider.chat_completion = AsyncMock(
            return_value=mock_response
        )

        result = await rewriter_with_cache.rewrite("@knowledgebase Test query")

        assert result.embedding_source_text == "Newly generated hypothetical."
        rewriter_with_cache.llm_provider.chat_completion.assert_called_once()

    async def test_cache_set_after_generation(self, rewriter_with_cache, mock_redis):
        """Test that generated document is cached."""
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True

        mock_response = MagicMock()
        mock_response.content = "Generated document."
        rewriter_with_cache.llm_provider.chat_completion = AsyncMock(
            return_value=mock_response
        )

        await rewriter_with_cache.rewrite("@knowledgebase Test query")

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 7200  # TTL is 2 hours


# ============================================================================
# HyDE Performance Tests
# ============================================================================


@pytest.mark.asyncio
class TestHyDEPerformance:
    """Tests for HyDE performance requirements."""

    @pytest.fixture
    def rewriter(self):
        """Create HyDERewriter with mocked LLM."""
        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_llm = MagicMock()
            mock_provider_cls.return_value = mock_llm
            rewriter = HyDERewriter()
            rewriter.llm_provider = mock_llm
            rewriter.cache = NullQueryRewritingCache()
            return rewriter

    async def test_processing_time_under_500ms_cached(self, rewriter):
        """Test that cached processing completes within 500ms."""
        import time

        # Pre-populate cache by mocking Redis
        cached_data = json.dumps({"hypothetical": "Cached response."})

        mock_redis = AsyncMock()
        mock_redis.get.return_value = cached_data
        rewriter.cache = QueryRewritingCache(redis_client=mock_redis)

        start = time.perf_counter()
        result = await rewriter.rewrite("@knowledgebase Test query")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        assert elapsed_ms < 100, f"Cached processing took {elapsed_ms:.2f}ms, expected <100ms"


# ============================================================================
# HyDE Example from Spec Tests
# ============================================================================


@pytest.mark.asyncio
class TestHyDESpecExample:
    """Tests matching the spec examples."""

    @pytest.fixture
    def rewriter(self):
        """Create HyDERewriter with mocked LLM."""
        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_llm = MagicMock()
            mock_provider_cls.return_value = mock_llm
            rewriter = HyDERewriter()
            rewriter.llm_provider = mock_llm
            rewriter.cache = NullQueryRewritingCache()
            return rewriter

    async def test_vibe_coding_example(self, rewriter):
        """Test the vibe coding example from the spec."""
        mock_response = MagicMock()
        mock_response.content = (
            "Vibe coding is a programming approach that emphasizes writing code based on "
            "intuition, flow state, and personal rhythm rather than strict methodologies. "
            "This coding style prioritizes developer comfort and creativity."
        )
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await rewriter.rewrite("@knowledgebase What is vibe coding?")

        assert result.search_rag is True
        assert "Vibe coding" in result.embedding_source_text
        assert "intuition" in result.embedding_source_text
        assert "flow state" in result.embedding_source_text
        assert "Based on the provided context" in result.llm_query

    async def test_llm_query_format(self, rewriter):
        """Test that llm_query has correct format."""
        mock_response = MagicMock()
        mock_response.content = "Hypothetical document."
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        result = await rewriter.rewrite("@knowledgebase Explain quantum computing")

        assert "Based on the provided context" in result.llm_query
        assert "Explain quantum computing" in result.llm_query


# ============================================================================
# HyDE Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
class TestHyDEErrorHandling:
    """Tests for HyDE error handling."""

    @pytest.fixture
    def rewriter(self):
        """Create HyDERewriter with mocked LLM."""
        with patch("src.rag.strategies.hyde.LLMProvider") as mock_provider_cls:
            mock_llm = MagicMock()
            mock_provider_cls.return_value = mock_llm
            rewriter = HyDERewriter()
            rewriter.llm_provider = mock_llm
            rewriter.cache = NullQueryRewritingCache()
            return rewriter

    async def test_handles_cache_parse_error(self, rewriter):
        """Test graceful handling of cache parse errors."""
        # Setup malformed cache data
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "not valid json"
        rewriter.cache = QueryRewritingCache(redis_client=mock_redis)

        mock_response = MagicMock()
        mock_response.content = "Generated document."
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        # Should not raise, should generate new document
        result = await rewriter.rewrite("@knowledgebase Test query")
        assert result.embedding_source_text == "Generated document."

    async def test_handles_cache_set_error(self, rewriter):
        """Test graceful handling of cache set errors."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.setex.side_effect = Exception("Redis error")
        rewriter.cache = QueryRewritingCache(redis_client=mock_redis)

        mock_response = MagicMock()
        mock_response.content = "Generated document."
        rewriter.llm_provider.chat_completion = AsyncMock(return_value=mock_response)

        # Should not raise, should return result even if caching fails
        result = await rewriter.rewrite("@knowledgebase Test query")
        assert result.embedding_source_text == "Generated document."
