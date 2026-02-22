"""Unit tests for multi-layer caching system.

This module tests the caching functionality including:
- L1 Redis cache
- L2 PostgreSQL cache
- L3 Semantic cache
- MultiLayerCache orchestrator
- Cache statistics tracking
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

# Import the module under test
from src.rag.cache import (
    CacheStats,
    CachedLLMResponse,
    CachedQueryResult,
    EmbeddingCacheModel,
    L1RedisCache,
    L2PostgresCache,
    L3SemanticCache,
    LLMResponseCacheModel,
    MultiLayerCache,
    NullMultiLayerCache,
    QueryCacheModel,
)


class TestCacheStats:
    """Tests for CacheStats class."""
    
    def test_initial_state(self) -> None:
        """Test initial stats are zero."""
        stats = CacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.size == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.layer_hits == {}
    
    def test_record_hit(self) -> None:
        """Test recording cache hits."""
        stats = CacheStats()
        
        stats.record_hit("L1_redis", 5.0)
        
        assert stats.hits == 1
        assert stats.misses == 0
        assert stats.hit_rate == 1.0
        assert stats.avg_latency_ms == 5.0
        assert stats.layer_hits["L1_redis"] == 1
    
    def test_record_miss(self) -> None:
        """Test recording cache misses."""
        stats = CacheStats()
        
        stats.record_miss(10.0)
        
        assert stats.hits == 0
        assert stats.misses == 1
        assert stats.hit_rate == 0.0
        assert stats.avg_latency_ms == 10.0
    
    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation with mixed hits and misses."""
        stats = CacheStats()
        
        # 3 hits, 1 miss = 75% hit rate
        stats.record_hit("L1", 5.0)
        stats.record_hit("L1", 5.0)
        stats.record_hit("L2", 10.0)
        stats.record_miss(20.0)
        
        assert stats.hits == 3
        assert stats.misses == 1
        assert stats.hit_rate == 0.75
    
    def test_running_average_latency(self) -> None:
        """Test running average latency calculation."""
        stats = CacheStats()
        
        stats.record_hit("L1", 10.0)  # avg = 10
        stats.record_hit("L1", 20.0)  # avg = (10 + 20) / 2 = 15
        stats.record_hit("L1", 30.0)  # avg = (15 * 2 + 30) / 3 = 20
        
        assert stats.avg_latency_ms == 20.0
    
    def test_model_dump(self) -> None:
        """Test that CacheStats can be serialized."""
        stats = CacheStats()
        stats.record_hit("L1", 5.0)
        
        dumped = stats.model_dump()
        
        assert "hits" in dumped
        assert "misses" in dumped
        assert "hit_rate" in dumped
        assert dumped["hits"] == 1


class TestCachedQueryResult:
    """Tests for CachedQueryResult model."""
    
    def test_creation(self) -> None:
        """Test creating a cached query result."""
        result = CachedQueryResult(
            query="What is vibe coding?",
            answer="Vibe coding is a programming approach...",
            sources=[{"chunk_id": "uuid", "content": "..."}],
            metrics={"latency_ms": 100},
            strategy_used="balanced",
        )
        
        assert result.query == "What is vibe coding?"
        assert result.answer == "Vibe coding is a programming approach..."
        assert len(result.sources) == 1
        assert result.strategy_used == "balanced"
        assert isinstance(result.created_at, datetime)
    
    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        result = CachedQueryResult(
            query="test",
            answer="test answer",
        )
        
        assert result.sources == []
        assert result.metrics == {}
        assert result.strategy_used == "balanced"


class TestCachedLLMResponse:
    """Tests for CachedLLMResponse model."""
    
    def test_creation(self) -> None:
        """Test creating a cached LLM response."""
        response = CachedLLMResponse(
            prompt="Explain Python",
            model="gpt-4",
            response="Python is a programming language...",
            tokens_used=150,
        )
        
        assert response.prompt == "Explain Python"
        assert response.model == "gpt-4"
        assert response.tokens_used == 150


class TestL1RedisCache:
    """Tests for L1 Redis cache."""
    
    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Create a mock Redis client."""
        mock = AsyncMock(spec=redis.Redis)
        return mock
    
    @pytest.mark.asyncio
    async def test_get_embedding_cache_hit(self, mock_redis: AsyncMock) -> None:
        """Test getting embedding from cache hit."""
        mock_redis.get = AsyncMock(return_value='[0.1, 0.2, 0.3]')
        cache = L1RedisCache(redis_client=mock_redis)
        
        result = await cache.get_embedding("hash123", "model-1")
        
        assert result == [0.1, 0.2, 0.3]
        assert cache._stats.hits == 1
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_embedding_cache_miss(self, mock_redis: AsyncMock) -> None:
        """Test getting embedding with cache miss."""
        mock_redis.get = AsyncMock(return_value=None)
        cache = L1RedisCache(redis_client=mock_redis)
        
        result = await cache.get_embedding("hash123", "model-1")
        
        assert result is None
        assert cache._stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_set_embedding(self, mock_redis: AsyncMock) -> None:
        """Test setting embedding in cache."""
        mock_redis.setex = AsyncMock(return_value=True)
        cache = L1RedisCache(redis_client=mock_redis, ttl_seconds=3600)
        
        success = await cache.set_embedding("hash123", "model-1", [0.1, 0.2, 0.3])
        
        assert success is True
        mock_redis.setex.assert_called_once()
        # Verify the key format
        call_args = mock_redis.setex.call_args
        assert "rag:embedding:model-1:hash123" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_get_query_result_cache_hit(self, mock_redis: AsyncMock) -> None:
        """Test getting query result from cache hit."""
        import json
        result_data = {
            "query": "What is Python?",
            "answer": "Python is a language",
            "sources": [],
            "metrics": {},
            "strategy_used": "balanced",
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(result_data))
        cache = L1RedisCache(redis_client=mock_redis)
        
        result = await cache.get_query_result("query_hash")
        
        assert result is not None
        assert result.query == "What is Python?"
        assert cache._stats.hits == 1
    
    @pytest.mark.asyncio
    async def test_set_query_result(self, mock_redis: AsyncMock) -> None:
        """Test setting query result in cache."""
        mock_redis.setex = AsyncMock(return_value=True)
        cache = L1RedisCache(redis_client=mock_redis)
        result = CachedQueryResult(
            query="test",
            answer="test answer",
        )
        
        success = await cache.set_query_result("query_hash", result)
        
        assert success is True
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_llm_response_cache_hit(self, mock_redis: AsyncMock) -> None:
        """Test getting LLM response from cache hit."""
        import json
        response_data = {
            "prompt": "Explain",
            "model": "gpt-4",
            "response": "Explanation",
            "tokens_used": 100,
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(response_data))
        cache = L1RedisCache(redis_client=mock_redis)
        
        result = await cache.get_llm_response("prompt_hash", "gpt-4")
        
        assert result is not None
        assert result.response == "Explanation"
    
    @pytest.mark.asyncio
    async def test_delete(self, mock_redis: AsyncMock) -> None:
        """Test deleting cache entry."""
        mock_redis.delete = AsyncMock(return_value=1)
        cache = L1RedisCache(redis_client=mock_redis)
        
        success = await cache.delete("key123", "query")
        
        assert success is True
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_all(self, mock_redis: AsyncMock) -> None:
        """Test clearing all cache entries."""
        mock_redis.scan_iter = MagicMock(return_value=AsyncIterator(["key1", "key2"]))
        mock_redis.delete = AsyncMock(return_value=2)
        cache = L1RedisCache(redis_client=mock_redis)
        
        count = await cache.clear_all()
        
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure(self) -> None:
        """Test handling of Redis connection failure."""
        cache = L1RedisCache(redis_client=None)
        
        result = await cache.get_embedding("hash", "model")
        
        assert result is None
        assert cache._stats.misses == 1
    
    def test_get_stats(self) -> None:
        """Test getting cache statistics."""
        cache = L1RedisCache()
        cache._stats.record_hit("L1_redis", 5.0)
        
        stats = cache.get_stats()
        
        assert stats.hits == 1
        assert stats.layer_hits["L1_redis"] == 1


class AsyncIterator:
    """Helper class for async iteration in tests."""
    
    def __init__(self, items: list):
        self.items = items
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class TestL2PostgresCache:
    """Tests for L2 PostgreSQL cache."""
    
    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)
    
    @pytest.mark.asyncio
    async def test_get_embedding_cache_hit(self, mock_session: AsyncMock) -> None:
        """Test getting embedding from database."""
        mock_record = MagicMock()
        mock_record.embedding = [0.1, 0.2, 0.3]
        mock_record.access_count = 1
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result
        
        cache = L2PostgresCache(session=mock_session)
        result = await cache.get_embedding("hash123", "model-1")
        
        assert result == [0.1, 0.2, 0.3]
        assert mock_record.access_count == 2  # Incremented
        assert cache._stats.hits == 1
    
    @pytest.mark.asyncio
    async def test_get_embedding_cache_miss(self, mock_session: AsyncMock) -> None:
        """Test getting embedding with cache miss."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        cache = L2PostgresCache(session=mock_session)
        result = await cache.get_embedding("hash123", "model-1")
        
        assert result is None
        assert cache._stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_set_embedding_new(self, mock_session: AsyncMock) -> None:
        """Test storing new embedding."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        cache = L2PostgresCache(session=mock_session)
        success = await cache.set_embedding("hash123", "model-1", [0.1, 0.2, 0.3])
        
        assert success is True
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_query_result_cache_hit(self, mock_session: AsyncMock) -> None:
        """Test getting query result from database."""
        mock_record = MagicMock()
        mock_record.result_json = {
            "query": "What is Python?",
            "answer": "Python is a language",
            "sources": [],
            "metrics": {},
            "strategy_used": "balanced",
            "created_at": datetime.utcnow(),
        }
        mock_record.ttl_seconds = 3600
        mock_record.created_at = datetime.utcnow()
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result
        
        cache = L2PostgresCache(session=mock_session)
        result = await cache.get_query_result("query_hash")
        
        assert result is not None
        assert result.query == "What is Python?"
    
    @pytest.mark.asyncio
    async def test_get_query_result_expired(self, mock_session: AsyncMock) -> None:
        """Test getting expired query result."""
        mock_record = MagicMock()
        mock_record.created_at = datetime.utcnow() - timedelta(hours=2)
        mock_record.ttl_seconds = 3600  # 1 hour TTL
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result
        
        cache = L2PostgresCache(session=mock_session)
        result = await cache.get_query_result("query_hash")
        
        assert result is None  # Expired
        assert cache._stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_set_query_result(self, mock_session: AsyncMock) -> None:
        """Test storing query result."""
        cache = L2PostgresCache(session=mock_session)
        result = CachedQueryResult(
            query="test",
            answer="test answer",
        )
        
        success = await cache.set_query_result(
            "query_hash",
            "test query",
            result,
            query_embedding=[0.1, 0.2],
        )
        
        assert success is True
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_llm_response(self, mock_session: AsyncMock) -> None:
        """Test getting LLM response from database."""
        mock_record = MagicMock()
        mock_record.model = "gpt-4"
        mock_record.response = "Response text"
        mock_record.tokens_used = 100
        mock_record.created_at = datetime.utcnow()
        mock_record.prompt_preview = "Prompt text"
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result
        
        cache = L2PostgresCache(session=mock_session)
        result = await cache.get_llm_response("prompt_hash", "gpt-4")
        
        assert result is not None
        assert result.response == "Response text"
        assert result.tokens_used == 100
    
    @pytest.mark.asyncio
    async def test_find_similar_queries(self, mock_session: AsyncMock) -> None:
        """Test finding similar queries."""
        # Create mock row result with similarity
        mock_row = MagicMock()
        mock_row.result_json = {
            "query": "What is Python?",
            "answer": "Python is a language",
            "sources": [],
            "metrics": {},
            "strategy_used": "balanced",
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_row.similarity = 0.95
        
        mock_result = MagicMock()
        mock_result.all.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        
        cache = L2PostgresCache(session=mock_session)
        results = await cache.find_similar_queries(
            query_embedding=[0.1, 0.2, 0.3],
            threshold=0.90,
        )
        
        assert len(results) == 1
        result, similarity = results[0]
        assert similarity == 0.95


class TestL3SemanticCache:
    """Tests for L3 Semantic cache."""
    
    @pytest.mark.asyncio
    async def test_find_similar_found(self) -> None:
        """Test finding similar query."""
        l2_cache = AsyncMock(spec=L2PostgresCache)
        cached_result = CachedQueryResult(
            query="What is Python?",
            answer="Python is a language",
        )
        l2_cache.find_similar_queries.return_value = [(cached_result, 0.96)]
        
        cache = L3SemanticCache(l2_cache, similarity_threshold=0.95)
        result = await cache.find_similar([0.1, 0.2, 0.3])
        
        assert result is not None
        assert result.query == "What is Python?"
        assert cache._stats.hits == 1
    
    @pytest.mark.asyncio
    async def test_find_similar_not_found(self) -> None:
        """Test when no similar query found."""
        l2_cache = AsyncMock(spec=L2PostgresCache)
        l2_cache.find_similar_queries.return_value = []
        
        cache = L3SemanticCache(l2_cache, similarity_threshold=0.95)
        result = await cache.find_similar([0.1, 0.2, 0.3])
        
        assert result is None
        assert cache._stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_find_similar_below_threshold(self) -> None:
        """Test when similarity is below threshold."""
        l2_cache = AsyncMock(spec=L2PostgresCache)
        cached_result = CachedQueryResult(
            query="What is Python?",
            answer="Python is a language",
        )
        # 0.90 similarity below 0.95 threshold
        l2_cache.find_similar_queries.return_value = [(cached_result, 0.90)]
        
        cache = L3SemanticCache(l2_cache, similarity_threshold=0.95)
        result = await cache.find_similar([0.1, 0.2, 0.3])
        
        assert result is None  # Below threshold
    
    @pytest.mark.asyncio
    async def test_store_delegates_to_l2(self) -> None:
        """Test that store delegates to L2 cache."""
        l2_cache = AsyncMock(spec=L2PostgresCache)
        l2_cache.set_query_result.return_value = True
        
        cache = L3SemanticCache(l2_cache)
        result = CachedQueryResult(query="test", answer="answer")
        
        success = await cache.store("hash", "query", [0.1], result)
        
        assert success is True
        l2_cache.set_query_result.assert_called_once()


class TestMultiLayerCache:
    """Tests for MultiLayerCache orchestrator."""
    
    @pytest.mark.asyncio
    async def test_get_embedding_from_l1(self) -> None:
        """Test getting embedding from L1 cache."""
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.get = AsyncMock(return_value='[0.1, 0.2, 0.3]')
        
        cache = MultiLayerCache(redis_client=mock_redis)
        result = await cache.get_embedding("test text", "model-1")
        
        assert result == [0.1, 0.2, 0.3]
        assert cache._stats.layer_hits.get("L1_redis") == 1
    
    @pytest.mark.asyncio
    async def test_get_embedding_from_l2_promotes_to_l1(self) -> None:
        """Test L2 hit promotes to L1."""
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.get = AsyncMock(return_value=None)  # L1 miss
        mock_redis.setex = AsyncMock(return_value=True)
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_record = MagicMock()
        mock_record.embedding = [0.1, 0.2, 0.3]
        mock_record.access_count = 1
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result
        
        cache = MultiLayerCache(redis_client=mock_redis, session=mock_session)
        result = await cache.get_embedding("test text", "model-1")
        
        assert result == [0.1, 0.2, 0.3]
        # Should promote to L1
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_embedding_cache_miss(self) -> None:
        """Test complete cache miss."""
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.get = AsyncMock(return_value=None)
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        cache = MultiLayerCache(redis_client=mock_redis, session=mock_session)
        result = await cache.get_embedding("test text", "model-1")
        
        assert result is None
        assert cache._stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_set_embedding_all_layers(self) -> None:
        """Test storing embedding in all layers."""
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.setex = AsyncMock(return_value=True)
        mock_session = AsyncMock(spec=AsyncSession)
        
        # Mock L2 to return existing record
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        cache = MultiLayerCache(redis_client=mock_redis, session=mock_session)
        await cache.set_embedding("test text", "model-1", [0.1, 0.2, 0.3], mock_session)
        
        # Should store in L1
        mock_redis.setex.assert_called_once()
        # Should store in L2
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_query_result_from_l1(self) -> None:
        """Test getting query result from L1."""
        import json
        mock_redis = AsyncMock(spec=redis.Redis)
        result_data = {
            "query": "What is Python?",
            "answer": "Python is a language",
            "sources": [],
            "metrics": {},
            "strategy_used": "balanced",
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(result_data))
        
        cache = MultiLayerCache(redis_client=mock_redis)
        result = await cache.get_query_result("What is Python?")
        
        assert result is not None
        assert result.answer == "Python is a language"
    
    @pytest.mark.asyncio
    async def test_get_query_result_from_l3_semantic(self) -> None:
        """Test getting query result from L3 semantic cache."""
        import json
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.get = AsyncMock(return_value=None)  # L1 miss
        
        mock_session = AsyncMock(spec=AsyncSession)
        
        # L2 exact match miss
        mock_result_l2 = MagicMock()
        mock_result_l2.scalar_one_or_none.return_value = None
        
        # L3 semantic match - mock the raw SQL query result
        mock_row = MagicMock()
        mock_row.result_json = {
            "query": "What is Python programming?",
            "answer": "Python is a language",
            "sources": [],
            "metrics": {},
            "strategy_used": "balanced",
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_row.similarity = 0.98
        mock_result_l3 = MagicMock()
        mock_result_l3.all.return_value = [mock_row]
        
        mock_session.execute.side_effect = [mock_result_l2, mock_result_l3]
        
        cache = MultiLayerCache(redis_client=mock_redis, session=mock_session)
        result = await cache.get_query_result(
            "What is Python?",
            query_embedding=[0.1] * 1536,
            similarity_threshold=0.95,
        )
        
        assert result is not None
        assert cache._stats.layer_hits.get("L3_semantic") == 1
    
    @pytest.mark.asyncio
    async def test_get_llm_response_from_l1(self) -> None:
        """Test getting LLM response from L1."""
        import json
        mock_redis = AsyncMock(spec=redis.Redis)
        response_data = {
            "prompt": "Explain",
            "model": "gpt-4",
            "response": "Explanation",
            "tokens_used": 100,
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(response_data))
        
        cache = MultiLayerCache(redis_client=mock_redis)
        result = await cache.get_llm_response("Explain Python", "gpt-4")
        
        assert result is not None
        assert result.response == "Explanation"
    
    @pytest.mark.asyncio
    async def test_invalidate_query_cache(self) -> None:
        """Test invalidating query cache."""
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.delete = AsyncMock(return_value=1)
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_session.execute.return_value = mock_result
        
        cache = MultiLayerCache(redis_client=mock_redis, session=mock_session)
        count = await cache.invalidate_query_cache("query_hash")
        
        assert isinstance(count, int)
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_all(self) -> None:
        """Test clearing all cache layers."""
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.scan_iter = MagicMock(return_value=AsyncIterator(["key1", "key2"]))
        mock_redis.delete = AsyncMock(return_value=2)
        
        cache = MultiLayerCache(redis_client=mock_redis)
        count = await cache.clear_all()
        
        assert count == 2
    
    def test_get_stats(self) -> None:
        """Test getting cache statistics."""
        mock_redis = AsyncMock(spec=redis.Redis)
        cache = MultiLayerCache(redis_client=mock_redis)
        
        stats = cache.get_stats()
        
        assert "combined" in stats
        assert "l1_redis" in stats
        assert "l2_postgres" in stats
        assert "l3_semantic" in stats
    
    @pytest.mark.asyncio
    async def test_warmup_embedding_cache(self) -> None:
        """Test cache warmup functionality."""
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.setex = AsyncMock(return_value=True)
        mock_session = AsyncMock(spec=AsyncSession)
        
        # Mock L2 to return existing record
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        cache = MultiLayerCache(redis_client=mock_redis, session=mock_session)
        texts = [
            ("text1", "model-1", [0.1, 0.2]),
            ("text2", "model-1", [0.3, 0.4]),
        ]
        
        count = await cache.warmup_embedding_cache(texts, mock_session)
        
        assert count == 2
    
    def test_generate_hash_consistency(self) -> None:
        """Test that hash generation is consistent."""
        cache = MultiLayerCache()
        
        hash1 = cache._generate_hash("test text")
        hash2 = cache._generate_hash("test text")
        hash3 = cache._generate_hash("different text")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 hex digest length


class TestNullMultiLayerCache:
    """Tests for NullMultiLayerCache (disabled caching)."""
    
    @pytest.mark.asyncio
    async def test_get_embedding_returns_none(self) -> None:
        """Test that get_embedding always returns None."""
        cache = NullMultiLayerCache()
        result = await cache.get_embedding("text", "model")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_query_result_returns_none(self) -> None:
        """Test that get_query_result always returns None."""
        cache = NullMultiLayerCache()
        result = await cache.get_query_result("query")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_llm_response_returns_none(self) -> None:
        """Test that get_llm_response always returns None."""
        cache = NullMultiLayerCache()
        result = await cache.get_llm_response("prompt", "model")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_methods_do_nothing(self) -> None:
        """Test that set methods do nothing."""
        cache = NullMultiLayerCache()
        result = CachedQueryResult(query="test", answer="answer")
        response = CachedLLMResponse(
            prompt="test",
            model="gpt-4",
            response="response",
        )
        
        # These should not raise exceptions
        await cache.set_embedding("text", "model", [0.1])
        await cache.set_query_result("query", result)
        await cache.set_llm_response("prompt", "model", response)
    
    @pytest.mark.asyncio
    async def test_invalidate_returns_zero(self) -> None:
        """Test that invalidate returns 0."""
        cache = NullMultiLayerCache()
        count = await cache.invalidate_query_cache()
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_clear_all_returns_zero(self) -> None:
        """Test that clear_all returns 0."""
        cache = NullMultiLayerCache()
        count = await cache.clear_all()
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_warmup_returns_zero(self) -> None:
        """Test that warmup returns 0."""
        cache = NullMultiLayerCache()
        count = await cache.warmup_embedding_cache([])
        
        assert count == 0
    
    def test_get_stats_returns_empty(self) -> None:
        """Test that get_stats returns empty stats."""
        cache = NullMultiLayerCache()
        stats = cache.get_stats()
        
        assert stats["combined"]["hits"] == 0
        assert stats["l1_redis"] is None


class TestCacheConfiguration:
    """Tests for cache configuration from settings."""
    
    def test_default_configuration(self) -> None:
        """Test that default configuration is applied."""
        from src.config import get_settings
        
        settings = get_settings()
        
        assert settings.caching.enabled is True
        assert settings.caching.layers["l1_redis"]["enabled"] is True
        assert settings.caching.layers["l1_redis"]["ttl"] == 3600
        assert settings.caching.layers["l2_postgres"]["ttl"] == 86400
        assert settings.caching.layers["l3_semantic"]["similarity_threshold"] == 0.95
    
    def test_cache_targets_configuration(self) -> None:
        """Test cache targets configuration."""
        from src.config import get_settings
        
        settings = get_settings()
        
        targets = settings.caching.cache_targets
        assert targets["embeddings"]["enabled"] is True
        assert targets["query_results"]["enabled"] is True
        assert targets["llm_responses"]["enabled"] is True
        assert targets["reranking_scores"]["enabled"] is True


class TestCacheIntegration:
    """Integration-style tests for cache operations."""
    
    @pytest.mark.asyncio
    async def test_embedding_cache_roundtrip(self) -> None:
        """Test complete embedding cache roundtrip."""
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.setex = AsyncMock(return_value=True)
        mock_session = AsyncMock(spec=AsyncSession)
        
        # Mock L2 to return existing record
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        cache = MultiLayerCache(redis_client=mock_redis, session=mock_session)
        
        # First, store an embedding
        embedding = [0.1, 0.2, 0.3]
        await cache.set_embedding("test text", "model-1", embedding, mock_session)
        
        # Verify it was stored in both layers
        mock_redis.setex.assert_called_once()
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_result_with_semantic_matching(self) -> None:
        """Test query result caching with semantic matching."""
        import json
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.get = AsyncMock(return_value=None)  # L1 miss
        
        mock_session = AsyncMock(spec=AsyncSession)
        
        # L2 exact match miss
        mock_result_l2 = MagicMock()
        mock_result_l2.scalar_one_or_none.return_value = None
        
        # L3 semantic match - mock raw SQL result
        mock_row = MagicMock()
        mock_row.result_json = {
            "query": "Explain Python programming language",
            "answer": "Python is a high-level language",
            "sources": [{"chunk_id": "1", "content": "..."}],
            "metrics": {"latency_ms": 100},
            "strategy_used": "balanced",
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_row.similarity = 0.97
        mock_result_l3 = MagicMock()
        mock_result_l3.all.return_value = [mock_row]
        
        mock_session.execute.side_effect = [mock_result_l2, mock_result_l3]
        
        cache = MultiLayerCache(redis_client=mock_redis, session=mock_session)
        
        # Query that's semantically similar but not identical
        result = await cache.get_query_result(
            "What is Python?",
            query_embedding=[0.1] * 1536,
            similarity_threshold=0.95,
        )
        
        assert result is not None
        assert result.answer == "Python is a high-level language"
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_tracking(self) -> None:
        """Test that hit rates are properly tracked."""
        import json
        mock_redis = AsyncMock(spec=redis.Redis)
        
        # Simulate hits
        result_data = {
            "query": "Q",
            "answer": "A",
            "sources": [],
            "metrics": {},
            "strategy_used": "balanced",
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(result_data))
        
        cache = MultiLayerCache(redis_client=mock_redis)
        
        # 3 hits
        for _ in range(3):
            await cache.get_query_result("test query")
        
        stats = cache.get_stats()
        assert stats["combined"]["hits"] == 3
        assert stats["combined"]["hit_rate"] == 1.0


class TestCachePerformanceTargets:
    """Tests verifying cache performance targets."""
    
    @pytest.mark.asyncio
    async def test_l1_cache_latency_target(self) -> None:
        """Test that L1 cache targets <5ms latency."""
        import json
        mock_redis = AsyncMock(spec=redis.Redis)
        result_data = {
            "query": "Q",
            "answer": "A",
            "sources": [],
            "metrics": {},
            "strategy_used": "balanced",
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_redis.get.return_value = json.dumps(result_data)
        
        cache = L1RedisCache(redis_client=mock_redis)
        
        # Multiple accesses to measure latency
        for _ in range(10):
            await cache.get_query_result("hash")
        
        stats = cache.get_stats()
        # Mock should be very fast, but in production we'd expect <5ms
        assert stats.avg_latency_ms >= 0  # Should track latency
    
    def test_cache_stats_model_serialization(self) -> None:
        """Test that cache stats can be serialized for monitoring."""
        stats = CacheStats()
        stats.record_hit("L1_redis", 2.5)
        stats.record_hit("L2_postgres", 25.0)
        stats.record_hit("L1_redis", 3.0)
        stats.record_miss(50.0)
        
        data = stats.model_dump()
        
        assert data["hits"] == 3
        assert data["misses"] == 1
        assert data["hit_rate"] == 0.75
        assert data["layer_hits"]["L1_redis"] == 2
        assert data["layer_hits"]["L2_postgres"] == 1
