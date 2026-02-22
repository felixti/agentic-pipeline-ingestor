"""Multi-layer caching system for RAG operations.

This module provides a comprehensive caching solution with three layers:
- L1: Redis (in-memory) - Hot data with fast access
- L2: PostgreSQL (persistent) - All cached data with longer TTL
- L3: Semantic (vector similarity) - Similar query matching

The cache supports embeddings, query results, and LLM responses with
configurable TTLs and hit rate tracking.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import redis.asyncio as redis
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Column, DateTime, Integer, String, Text, and_, delete, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base

from src.config import get_settings
from src.db.models import Vector
from src.observability.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

# SQLAlchemy base for cache models
CacheBase: Any = declarative_base()


class EmbeddingCacheModel(CacheBase):  # type: ignore[misc]
    """Database model for embedding cache (L2)."""
    
    __tablename__ = "embedding_cache"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    text_hash = Column(String(64), unique=True, nullable=False, index=True)
    text_preview = Column(String(200), nullable=True)
    model = Column(String(100), nullable=False, index=True)
    embedding = Column(Vector(dimensions=1536), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    accessed_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    access_count = Column(Integer, default=1, nullable=False)


class QueryCacheModel(CacheBase):  # type: ignore[misc]
    """Database model for query result cache (L2/L3)."""
    
    __tablename__ = "query_cache"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_hash = Column(String(64), unique=True, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    query_embedding = Column(Vector(dimensions=1536), nullable=True)
    result_json = Column(JSONB, nullable=False)
    strategy_config = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    ttl_seconds = Column(Integer, default=3600, nullable=False)


class LLMResponseCacheModel(CacheBase):  # type: ignore[misc]
    """Database model for LLM response cache (L2)."""
    
    __tablename__ = "llm_response_cache"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    prompt_hash = Column(String(64), nullable=False, index=True)
    prompt_preview = Column(Text, nullable=True)
    model = Column(String(100), nullable=False, index=True)
    response = Column(Text, nullable=False)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class CacheStats(BaseModel):
    """Cache statistics for monitoring.
    
    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        hit_rate: Cache hit rate (0-1)
        size: Current cache size
        avg_latency_ms: Average access latency in milliseconds
        layer_hits: Breakdown of hits by cache layer
    """
    
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    size: int = 0
    avg_latency_ms: float = 0.0
    layer_hits: dict[str, int] = Field(default_factory=dict)
    
    def record_hit(self, layer: str, latency_ms: float) -> None:
        """Record a cache hit.
        
        Args:
            layer: Cache layer that served the hit
            latency_ms: Access latency in milliseconds
        """
        self.hits += 1
        self.layer_hits[layer] = self.layer_hits.get(layer, 0) + 1
        self._update_hit_rate()
        self._update_latency(latency_ms)
    
    def record_miss(self, latency_ms: float) -> None:
        """Record a cache miss.
        
        Args:
            latency_ms: Access latency in milliseconds
        """
        self.misses += 1
        self._update_hit_rate()
        self._update_latency(latency_ms)
    
    def _update_hit_rate(self) -> None:
        """Update hit rate based on current hits and misses."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0
    
    def _update_latency(self, latency_ms: float) -> None:
        """Update average latency using running average."""
        total = self.hits + self.misses
        if total == 1:
            self.avg_latency_ms = latency_ms
        else:
            # Running average
            self.avg_latency_ms = (
                (self.avg_latency_ms * (total - 1)) + latency_ms
            ) / total


class CachedQueryResult(BaseModel):
    """Cached query result with metadata.
    
    Attributes:
        query: Original query text
        answer: Generated answer
        sources: List of source documents
        metrics: Query execution metrics
        strategy_used: RAG strategy used
        created_at: Timestamp of cache entry
    """
    
    query: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    strategy_used: str = "balanced"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What is vibe coding?",
                    "answer": "Vibe coding is a programming approach...",
                    "sources": [{"chunk_id": "uuid", "content": "..."}],
                    "strategy_used": "balanced",
                }
            ]
        }
    )


class CachedLLMResponse(BaseModel):
    """Cached LLM response with metadata.
    
    Attributes:
        prompt: Original prompt
        model: Model used for generation
        response: Generated response text
        tokens_used: Number of tokens consumed
        created_at: Timestamp of cache entry
    """
    
    prompt: str
    model: str
    response: str
    tokens_used: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class L1RedisCache:
    """L1 Cache: Redis-based in-memory cache for hot data.
    
    Provides fast access to frequently used data with configurable TTL.
    """
    
    CACHE_PREFIX = "rag:"
    
    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        ttl_seconds: int = 3600,
        max_size: int = 10000,
    ):
        """Initialize L1 Redis cache.
        
        Args:
            redis_client: Redis async client (created if None)
            ttl_seconds: Default TTL for cache entries
            max_size: Maximum number of entries (for stats)
        """
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._stats = CacheStats()
        self._enabled = redis_client is not None
    
    def _make_key(self, key: str, key_type: str) -> str:
        """Create prefixed cache key.
        
        Args:
            key: Base cache key
            key_type: Type of cached data (embedding, query, llm)
            
        Returns:
            Prefixed cache key
        """
        return f"{self.CACHE_PREFIX}{key_type}:{key}"
    
    async def _get_redis(self) -> redis.Redis | None:
        """Get Redis client, initializing if needed."""
        if self.redis is None and self._enabled:
            try:
                settings = get_settings()
                self.redis = await redis.from_url(
                    str(settings.redis.url),
                    encoding="utf-8",
                    decode_responses=True,
                )
            except Exception as e:
                logger.warning("redis_connection_failed", error=str(e))
                self._enabled = False
        return self.redis
    
    async def get_embedding(self, text_hash: str, model: str) -> list[float] | None:
        """Get cached embedding from Redis.
        
        Args:
            text_hash: Hash of the text
            model: Model identifier
            
        Returns:
            Embedding vector or None if not found
        """
        start_time = time.monotonic()
        key = self._make_key(f"{model}:{text_hash}", "embedding")
        
        try:
            r = await self._get_redis()
            if r is None:
                self._stats.record_miss(0)
                return None
            
            data = await r.get(key)
            latency_ms = (time.monotonic() - start_time) * 1000
            
            if data is None:
                self._stats.record_miss(latency_ms)
                return None
            
            embedding: list[float] = json.loads(data)
            self._stats.record_hit("L1_redis", latency_ms)
            return embedding
            
        except Exception as e:
            logger.warning("l1_get_embedding_failed", error=str(e))
            self._stats.record_miss((time.monotonic() - start_time) * 1000)
            return None
    
    async def set_embedding(
        self,
        text_hash: str,
        model: str,
        embedding: list[float],
        ttl: int | None = None,
    ) -> bool:
        """Store embedding in Redis.
        
        Args:
            text_hash: Hash of the text
            model: Model identifier
            embedding: Embedding vector
            ttl: Optional custom TTL
            
        Returns:
            True if stored successfully
        """
        key = self._make_key(f"{model}:{text_hash}", "embedding")
        
        try:
            r = await self._get_redis()
            if r is None:
                return False
            
            await r.setex(
                key,
                ttl or self.ttl_seconds,
                json.dumps(embedding),
            )
            return True
            
        except Exception as e:
            logger.warning("l1_set_embedding_failed", error=str(e))
            return False
    
    async def get_query_result(self, query_hash: str) -> CachedQueryResult | None:
        """Get cached query result from Redis.
        
        Args:
            query_hash: Hash of the query
            
        Returns:
            Cached result or None if not found
        """
        start_time = time.monotonic()
        key = self._make_key(query_hash, "query")
        
        try:
            r = await self._get_redis()
            if r is None:
                self._stats.record_miss(0)
                return None
            
            data = await r.get(key)
            latency_ms = (time.monotonic() - start_time) * 1000
            
            if data is None:
                self._stats.record_miss(latency_ms)
                return None
            
            result_dict = json.loads(data)
            # Convert ISO timestamp back to datetime
            if "created_at" in result_dict and isinstance(result_dict["created_at"], str):
                result_dict["created_at"] = datetime.fromisoformat(result_dict["created_at"])
            
            self._stats.record_hit("L1_redis", latency_ms)
            return CachedQueryResult(**result_dict)
            
        except Exception as e:
            logger.warning("l1_get_query_failed", error=str(e))
            self._stats.record_miss((time.monotonic() - start_time) * 1000)
            return None
    
    async def set_query_result(
        self,
        query_hash: str,
        result: CachedQueryResult,
        ttl: int | None = None,
    ) -> bool:
        """Store query result in Redis.
        
        Args:
            query_hash: Hash of the query
            result: Query result to cache
            ttl: Optional custom TTL
            
        Returns:
            True if stored successfully
        """
        key = self._make_key(query_hash, "query")
        
        try:
            r = await self._get_redis()
            if r is None:
                return False
            
            await r.setex(
                key,
                ttl or self.ttl_seconds,
                json.dumps(result.model_dump(mode="json")),
            )
            return True
            
        except Exception as e:
            logger.warning("l1_set_query_failed", error=str(e))
            return False
    
    async def get_llm_response(self, prompt_hash: str, model: str) -> CachedLLMResponse | None:
        """Get cached LLM response from Redis.
        
        Args:
            prompt_hash: Hash of the prompt
            model: Model identifier
            
        Returns:
            Cached response or None if not found
        """
        start_time = time.monotonic()
        key = self._make_key(f"{model}:{prompt_hash}", "llm")
        
        try:
            r = await self._get_redis()
            if r is None:
                self._stats.record_miss(0)
                return None
            
            data = await r.get(key)
            latency_ms = (time.monotonic() - start_time) * 1000
            
            if data is None:
                self._stats.record_miss(latency_ms)
                return None
            
            result_dict = json.loads(data)
            if "created_at" in result_dict and isinstance(result_dict["created_at"], str):
                result_dict["created_at"] = datetime.fromisoformat(result_dict["created_at"])
            
            self._stats.record_hit("L1_redis", latency_ms)
            return CachedLLMResponse(**result_dict)
            
        except Exception as e:
            logger.warning("l1_get_llm_failed", error=str(e))
            self._stats.record_miss((time.monotonic() - start_time) * 1000)
            return None
    
    async def set_llm_response(
        self,
        prompt_hash: str,
        model: str,
        response: CachedLLMResponse,
        ttl: int | None = None,
    ) -> bool:
        """Store LLM response in Redis.
        
        Args:
            prompt_hash: Hash of the prompt
            model: Model identifier
            response: LLM response to cache
            ttl: Optional custom TTL
            
        Returns:
            True if stored successfully
        """
        key = self._make_key(f"{model}:{prompt_hash}", "llm")
        
        try:
            r = await self._get_redis()
            if r is None:
                return False
            
            await r.setex(
                key,
                ttl or self.ttl_seconds,
                json.dumps(response.model_dump(mode="json")),
            )
            return True
            
        except Exception as e:
            logger.warning("l1_set_llm_failed", error=str(e))
            return False
    
    async def delete(self, key: str, key_type: str) -> bool:
        """Delete a cache entry.
        
        Args:
            key: Cache key
            key_type: Type of cached data
            
        Returns:
            True if deleted
        """
        full_key = self._make_key(key, key_type)
        
        try:
            r = await self._get_redis()
            if r is None:
                return False
            
            result = await r.delete(full_key)
            return bool(result > 0)
            
        except Exception as e:
            logger.warning("l1_delete_failed", error=str(e))
            return False
    
    async def clear_all(self) -> int:
        """Clear all RAG cache entries from Redis.
        
        Returns:
            Number of keys deleted
        """
        try:
            r = await self._get_redis()
            if r is None:
                return 0
            
            pattern = f"{self.CACHE_PREFIX}*"
            keys = []
            async for key in r.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return int(await r.delete(*keys))
            return 0
            
        except Exception as e:
            logger.warning("l1_clear_all_failed", error=str(e))
            return 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class L2PostgresCache:
    """L2 Cache: PostgreSQL-based persistent cache.
    
    Stores all cached data with longer TTL, survives restarts.
    """
    
    def __init__(
        self,
        session: AsyncSession | None = None,
        ttl_seconds: int = 86400,
    ):
        """Initialize L2 PostgreSQL cache.
        
        Args:
            session: Database session (optional)
            ttl_seconds: Default TTL for cache entries
        """
        self.session = session
        self.ttl_seconds = ttl_seconds
        self._stats = CacheStats()
    
    async def get_embedding(
        self,
        text_hash: str,
        model: str,
        session: AsyncSession | None = None,
    ) -> list[float] | None:
        """Get cached embedding from PostgreSQL.
        
        Args:
            text_hash: Hash of the text
            model: Model identifier
            session: Optional database session
            
        Returns:
            Embedding vector or None if not found
        """
        start_time = time.monotonic()
        sess = session or self.session
        
        if sess is None:
            self._stats.record_miss(0)
            return None
        
        try:
            result = await sess.execute(
                select(EmbeddingCacheModel).where(
                    and_(
                        EmbeddingCacheModel.text_hash == text_hash,
                        EmbeddingCacheModel.model == model,
                    )
                )
            )
            record = result.scalar_one_or_none()
            latency_ms = (time.monotonic() - start_time) * 1000
            
            if record is None:
                self._stats.record_miss(latency_ms)
                return None
            
            # Update access metadata
            record.access_count = int(record.access_count) + 1  # type: ignore
            record.accessed_at = datetime.utcnow()  # type: ignore
            
            self._stats.record_hit("L2_postgres", latency_ms)
            return record.embedding  # type: ignore
            
        except Exception as e:
            logger.warning("l2_get_embedding_failed", error=str(e))
            self._stats.record_miss((time.monotonic() - start_time) * 1000)
            return None
    
    async def set_embedding(
        self,
        text_hash: str,
        model: str,
        embedding: list[float],
        text_preview: str | None = None,
        session: AsyncSession | None = None,
    ) -> bool:
        """Store embedding in PostgreSQL.
        
        Args:
            text_hash: Hash of the text
            model: Model identifier
            embedding: Embedding vector
            text_preview: Preview of original text
            session: Optional database session
            
        Returns:
            True if stored successfully
        """
        sess = session or self.session
        if sess is None:
            return False
        
        try:
            # Check if exists
            result = await sess.execute(
                select(EmbeddingCacheModel).where(
                    and_(
                        EmbeddingCacheModel.text_hash == text_hash,
                        EmbeddingCacheModel.model == model,
                    )
                )
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing
                existing.embedding = embedding  # type: ignore
                existing.accessed_at = datetime.utcnow()  # type: ignore
                existing.access_count = int(existing.access_count) + 1  # type: ignore
            else:
                # Create new
                record = EmbeddingCacheModel(
                    text_hash=text_hash,
                    text_preview=text_preview or text_hash[:200],
                    model=model,
                    embedding=embedding,
                )
                sess.add(record)
            
            return True
            
        except Exception as e:
            logger.warning("l2_set_embedding_failed", error=str(e))
            return False
    
    async def get_query_result(
        self,
        query_hash: str,
        session: AsyncSession | None = None,
    ) -> CachedQueryResult | None:
        """Get cached query result from PostgreSQL.
        
        Args:
            query_hash: Hash of the query
            session: Optional database session
            
        Returns:
            Cached result or None if not found/expired
        """
        start_time = time.monotonic()
        sess = session or self.session
        
        if sess is None:
            self._stats.record_miss(0)
            return None
        
        try:
            result = await sess.execute(
                select(QueryCacheModel).where(
                    QueryCacheModel.query_hash == query_hash
                )
            )
            record = result.scalar_one_or_none()
            latency_ms = (time.monotonic() - start_time) * 1000
            
            if record is None:
                self._stats.record_miss(latency_ms)
                return None
            
            # Check TTL
            age = datetime.utcnow() - record.created_at
            if age.total_seconds() > record.ttl_seconds:
                self._stats.record_miss(latency_ms)
                return None
            
            self._stats.record_hit("L2_postgres", latency_ms)
            return CachedQueryResult(**record.result_json)
            
        except Exception as e:
            logger.warning("l2_get_query_failed", error=str(e))
            self._stats.record_miss((time.monotonic() - start_time) * 1000)
            return None
    
    async def set_query_result(
        self,
        query_hash: str,
        query_text: str,
        result: CachedQueryResult,
        query_embedding: list[float] | None = None,
        ttl: int | None = None,
        session: AsyncSession | None = None,
    ) -> bool:
        """Store query result in PostgreSQL.
        
        Args:
            query_hash: Hash of the query
            query_text: Original query text
            result: Query result to cache
            query_embedding: Optional embedding for semantic search
            ttl: Optional custom TTL
            session: Optional database session
            
        Returns:
            True if stored successfully
        """
        sess = session or self.session
        if sess is None:
            return False
        
        try:
            record = QueryCacheModel(
                query_hash=query_hash,
                query_text=query_text,
                query_embedding=query_embedding,
                result_json=result.model_dump(mode="json"),
                ttl_seconds=ttl or self.ttl_seconds,
            )
            sess.add(record)
            return True
            
        except Exception as e:
            logger.warning("l2_set_query_failed", error=str(e))
            return False
    
    async def get_llm_response(
        self,
        prompt_hash: str,
        model: str,
        session: AsyncSession | None = None,
    ) -> CachedLLMResponse | None:
        """Get cached LLM response from PostgreSQL.
        
        Args:
            prompt_hash: Hash of the prompt
            model: Model identifier
            session: Optional database session
            
        Returns:
            Cached response or None if not found
        """
        start_time = time.monotonic()
        sess = session or self.session
        
        if sess is None:
            self._stats.record_miss(0)
            return None
        
        try:
            result = await sess.execute(
                select(LLMResponseCacheModel).where(
                    and_(
                        LLMResponseCacheModel.prompt_hash == prompt_hash,
                        LLMResponseCacheModel.model == model,
                    )
                )
            )
            record = result.scalar_one_or_none()
            latency_ms = (time.monotonic() - start_time) * 1000
            
            if record is None:
                self._stats.record_miss(latency_ms)
                return None
            
            self._stats.record_hit("L2_postgres", latency_ms)
            return CachedLLMResponse(
                prompt=str(record.prompt_preview or ""),
                model=str(record.model),
                response=str(record.response),
                tokens_used=int(record.tokens_used or 0),
                created_at=record.created_at,  # type: ignore
            )
            
        except Exception as e:
            logger.warning("l2_get_llm_failed", error=str(e))
            self._stats.record_miss((time.monotonic() - start_time) * 1000)
            return None
    
    async def set_llm_response(
        self,
        prompt_hash: str,
        model: str,
        response: CachedLLMResponse,
        session: AsyncSession | None = None,
    ) -> bool:
        """Store LLM response in PostgreSQL.
        
        Args:
            prompt_hash: Hash of the prompt
            model: Model identifier
            response: LLM response to cache
            session: Optional database session
            
        Returns:
            True if stored successfully
        """
        sess = session or self.session
        if sess is None:
            return False
        
        try:
            record = LLMResponseCacheModel(
                prompt_hash=prompt_hash,
                prompt_preview=response.prompt[:500] if len(response.prompt) > 500 else response.prompt,
                model=model,
                response=response.response,
                tokens_used=response.tokens_used,
            )
            sess.add(record)
            return True
            
        except Exception as e:
            logger.warning("l2_set_llm_failed", error=str(e))
            return False
    
    async def find_similar_queries(
        self,
        query_embedding: list[float],
        threshold: float = 0.95,
        limit: int = 1,
        session: AsyncSession | None = None,
    ) -> list[tuple[CachedQueryResult, float]]:
        """Find semantically similar cached queries.
        
        Args:
            query_embedding: Embedding of the query to match
            threshold: Minimum similarity threshold (0-1)
            limit: Maximum number of results
            session: Optional database session
            
        Returns:
            List of (result, similarity) tuples
        """
        start_time = time.monotonic()
        sess = session or self.session
        
        if sess is None:
            return []
        
        try:
            # Use raw SQL for pgvector cosine similarity
            # cosine_distance = 1 - cosine_similarity
            max_distance = 1.0 - threshold
            
            from sqlalchemy import text
            
            query = text("""
                SELECT id, result_json, 
                       1 - (query_embedding <=> :embedding) as similarity
                FROM query_cache
                WHERE query_embedding IS NOT NULL
                AND 1 - (query_embedding <=> :embedding) >= :threshold
                ORDER BY query_embedding <=> :embedding
                LIMIT :limit
            """)
            
            result = await sess.execute(
                query,
                {
                    "embedding": f"[{','.join(str(x) for x in query_embedding)}]",
                    "threshold": threshold,
                    "limit": limit,
                }
            )
            
            records = result.all()
            latency_ms = (time.monotonic() - start_time) * 1000
            
            results = []
            for row in records:
                similarity = float(row.similarity)
                if similarity >= threshold:
                    results.append((CachedQueryResult(**row.result_json), similarity))
            
            if results:
                self._stats.record_hit("L3_semantic", latency_ms)
            
            return results
            
        except Exception as e:
            logger.warning("l2_find_similar_failed", error=str(e))
            return []
    
    async def invalidate_expired(self, session: AsyncSession | None = None) -> int:
        """Remove expired cache entries.
        
        Args:
            session: Optional database session
            
        Returns:
            Number of entries removed
        """
        sess = session or self.session
        if sess is None:
            return 0
        
        try:
            # Delete expired query cache entries
            result = await sess.execute(
                delete(QueryCacheModel).where(
                    QueryCacheModel.created_at < datetime.utcnow() - timedelta(seconds=self.ttl_seconds)
                )
            )
            return int(getattr(result, "rowcount", 0) or 0)
            
        except Exception as e:
            logger.warning("l2_invalidate_failed", error=str(e))
            return 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class L3SemanticCache:
    """L3 Cache: Semantic similarity-based cache.
    
    Uses vector similarity to find cached results for semantically
    similar queries, even if they don't match exactly.
    """
    
    def __init__(
        self,
        l2_cache: L2PostgresCache,
        similarity_threshold: float = 0.95,
    ):
        """Initialize L3 semantic cache.
        
        Args:
            l2_cache: L2 cache instance for database access
            similarity_threshold: Minimum similarity threshold (0-1)
        """
        self.l2_cache = l2_cache
        self.similarity_threshold = similarity_threshold
        self._stats = CacheStats()
    
    async def find_similar(
        self,
        query_embedding: list[float],
        threshold: float | None = None,
        session: AsyncSession | None = None,
    ) -> CachedQueryResult | None:
        """Find cached result for semantically similar query.
        
        Args:
            query_embedding: Embedding of the query
            threshold: Optional override for similarity threshold
            session: Optional database session
            
        Returns:
            Cached result or None if no similar query found
        """
        start_time = time.monotonic()
        threshold = threshold or self.similarity_threshold
        
        try:
            results = await self.l2_cache.find_similar_queries(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=1,
                session=session,
            )
            
            latency_ms = (time.monotonic() - start_time) * 1000
            
            if results:
                result, similarity = results[0]
                # Double-check threshold
                if similarity >= threshold:
                    self._stats.record_hit("L3_semantic", latency_ms)
                    logger.debug(
                        "semantic_cache_hit",
                        similarity=round(similarity, 4),
                    )
                    return result
            
            self._stats.record_miss(latency_ms)
            return None
            
        except Exception as e:
            logger.warning("l3_find_similar_failed", error=str(e))
            self._stats.record_miss((time.monotonic() - start_time) * 1000)
            return None
    
    async def store(
        self,
        query_hash: str,
        query_text: str,
        query_embedding: list[float],
        result: CachedQueryResult,
        session: AsyncSession | None = None,
    ) -> bool:
        """Store query result with embedding for semantic search.
        
        Args:
            query_hash: Hash of the query
            query_text: Original query text
            query_embedding: Embedding of the query
            result: Query result to cache
            session: Optional database session
            
        Returns:
            True if stored successfully
        """
        return await self.l2_cache.set_query_result(
            query_hash=query_hash,
            query_text=query_text,
            result=result,
            query_embedding=query_embedding,
            session=session,
        )
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class MultiLayerCache:
    """Multi-layer cache orchestrator for RAG operations.
    
    Coordinates L1 (Redis), L2 (PostgreSQL), and L3 (Semantic) caches
    to provide fast, persistent, and intelligent caching.
    
    Example:
        >>> cache = MultiLayerCache()
        >>> # Get embedding from cache hierarchy
        >>> embedding = await cache.get_embedding("text", "model")
        >>> # Store embedding in all layers
        >>> await cache.set_embedding("text", "model", embedding)
    """
    
    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        session: AsyncSession | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize multi-layer cache.
        
        Args:
            redis_client: Optional Redis client
            session: Optional database session
            config: Optional configuration overrides
        """
        settings = get_settings()
        cache_config = config or {}
        
        # Get configuration from settings or use defaults
        l1_config = cache_config.get("l1_redis", {})
        l2_config = cache_config.get("l2_postgres", {})
        l3_config = cache_config.get("l3_semantic", {})
        
        # Initialize L1 Redis cache
        l1_enabled = l1_config.get("enabled", True)
        self.l1_cache = L1RedisCache(
            redis_client=redis_client if l1_enabled else None,
            ttl_seconds=l1_config.get("ttl", 3600),
            max_size=l1_config.get("max_size", 10000),
        )
        
        # Initialize L2 PostgreSQL cache
        l2_enabled = l2_config.get("enabled", True)
        self.l2_cache = L2PostgresCache(
            session=session if l2_enabled else None,
            ttl_seconds=l2_config.get("ttl", 86400),
        )
        
        # Initialize L3 Semantic cache
        l3_enabled = l3_config.get("enabled", True)
        self.l3_cache = L3SemanticCache(
            l2_cache=self.l2_cache,
            similarity_threshold=l3_config.get("similarity_threshold", 0.95),
        ) if l3_enabled else None
        
        # Overall stats
        self._stats = CacheStats()
    
    def _generate_hash(self, text: str) -> str:
        """Generate SHA-256 hash for text.
        
        Args:
            text: Text to hash
            
        Returns:
            Hex digest of hash
        """
        return hashlib.sha256(text.encode()).hexdigest()
    
    async def get_embedding(
        self,
        text: str,
        model: str,
        session: AsyncSession | None = None,
    ) -> list[float] | None:
        """Get embedding from cache hierarchy.
        
        Checks L1 (Redis) first, then L2 (PostgreSQL).
        If found in L2, promotes to L1.
        
        Args:
            text: Original text
            model: Model identifier
            session: Optional database session
            
        Returns:
            Embedding vector or None if not found
        """
        text_hash = self._generate_hash(text)
        
        # L1: Try Redis first
        embedding = await self.l1_cache.get_embedding(text_hash, model)
        if embedding is not None:
            self._stats.record_hit("L1_redis", 0)
            return embedding
        
        # L2: Try PostgreSQL
        embedding = await self.l2_cache.get_embedding(text_hash, model, session)
        if embedding is not None:
            self._stats.record_hit("L2_postgres", 0)
            # Promote to L1
            await self.l1_cache.set_embedding(text_hash, model, embedding)
            return embedding
        
        self._stats.record_miss(0)
        return None
    
    async def set_embedding(
        self,
        text: str,
        model: str,
        embedding: list[float],
        session: AsyncSession | None = None,
    ) -> None:
        """Store embedding in all cache layers.
        
        Args:
            text: Original text
            model: Model identifier
            embedding: Embedding vector
            session: Optional database session
        """
        text_hash = self._generate_hash(text)
        text_preview = text[:200] if len(text) > 200 else text
        
        # Store in both L1 and L2 concurrently
        await self.l1_cache.set_embedding(text_hash, model, embedding)
        await self.l2_cache.set_embedding(
            text_hash, model, embedding, text_preview, session
        )
    
    async def get_query_result(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        similarity_threshold: float = 0.95,
        session: AsyncSession | None = None,
    ) -> CachedQueryResult | None:
        """Get cached query result from hierarchy.
        
        Checks L1 (Redis) first, then L3 (Semantic) if embedding provided.
        
        Args:
            query: Query text
            query_embedding: Optional query embedding for semantic search
            similarity_threshold: Minimum similarity for semantic match
            session: Optional database session
            
        Returns:
            Cached result or None if not found
        """
        query_hash = self._generate_hash(query)
        
        # L1: Try Redis first
        result = await self.l1_cache.get_query_result(query_hash)
        if result is not None:
            self._stats.record_hit("L1_redis", 0)
            return result
        
        # L2: Try exact match in PostgreSQL
        result = await self.l2_cache.get_query_result(query_hash, session)
        if result is not None:
            self._stats.record_hit("L2_postgres", 0)
            # Promote to L1
            await self.l1_cache.set_query_result(query_hash, result)
            return result
        
        # L3: Try semantic similarity
        if self.l3_cache is not None and query_embedding is not None:
            result = await self.l3_cache.find_similar(
                query_embedding, similarity_threshold, session
            )
            if result is not None:
                self._stats.record_hit("L3_semantic", 0)
                return result
        
        self._stats.record_miss(0)
        return None
    
    async def set_query_result(
        self,
        query: str,
        result: CachedQueryResult,
        query_embedding: list[float] | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Store query result in cache.
        
        Args:
            query: Query text
            result: Query result to cache
            query_embedding: Optional embedding for semantic search
            session: Optional database session
        """
        query_hash = self._generate_hash(query)
        
        # Store in L1
        await self.l1_cache.set_query_result(query_hash, result)
        
        # Store in L2/L3 with embedding
        await self.l2_cache.set_query_result(
            query_hash=query_hash,
            query_text=query,
            result=result,
            query_embedding=query_embedding,
            session=session,
        )
    
    async def get_llm_response(
        self,
        prompt: str,
        model: str,
        session: AsyncSession | None = None,
    ) -> CachedLLMResponse | None:
        """Get cached LLM response from hierarchy.
        
        Args:
            prompt: Prompt text
            model: Model identifier
            session: Optional database session
            
        Returns:
            Cached response or None if not found
        """
        prompt_hash = self._generate_hash(prompt)
        
        # L1: Try Redis first
        response = await self.l1_cache.get_llm_response(prompt_hash, model)
        if response is not None:
            self._stats.record_hit("L1_redis", 0)
            return response
        
        # L2: Try PostgreSQL
        response = await self.l2_cache.get_llm_response(prompt_hash, model, session)
        if response is not None:
            self._stats.record_hit("L2_postgres", 0)
            # Promote to L1
            await self.l1_cache.set_llm_response(prompt_hash, model, response)
            return response
        
        self._stats.record_miss(0)
        return None
    
    async def set_llm_response(
        self,
        prompt: str,
        model: str,
        response: CachedLLMResponse,
        session: AsyncSession | None = None,
    ) -> None:
        """Store LLM response in cache.
        
        Args:
            prompt: Prompt text
            model: Model identifier
            response: LLM response to cache
            session: Optional database session
        """
        prompt_hash = self._generate_hash(prompt)
        
        # Store in both L1 and L2
        await self.l1_cache.set_llm_response(prompt_hash, model, response)
        await self.l2_cache.set_llm_response(prompt_hash, model, response, session)
    
    async def invalidate_query_cache(
        self,
        query_hash: str | None = None,
        session: AsyncSession | None = None,
    ) -> int:
        """Invalidate query cache entries.
        
        Args:
            query_hash: Specific query hash to invalidate (None = all)
            session: Optional database session
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        # Clear L1
        if query_hash:
            await self.l1_cache.delete(query_hash, "query")
            count += 1
        else:
            count += await self.l1_cache.clear_all()
        
        # Clear expired from L2
        count += await self.l2_cache.invalidate_expired(session)
        
        return count
    
    async def clear_all(self) -> int:
        """Clear all cache layers.
        
        Returns:
            Total number of entries cleared
        """
        count = await self.l1_cache.clear_all()
        logger.info("cache_cleared", l1_cleared=count)
        return count
    
    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics.
        
        Returns:
            Dictionary with stats for all layers and combined
        """
        return {
            "combined": self._stats.model_dump(),
            "l1_redis": self.l1_cache.get_stats().model_dump(),
            "l2_postgres": self.l2_cache.get_stats().model_dump(),
            "l3_semantic": self.l3_cache.get_stats().model_dump() if self.l3_cache else None,
        }
    
    async def warmup_embedding_cache(
        self,
        texts: list[tuple[str, str, list[float]]],
        session: AsyncSession | None = None,
    ) -> int:
        """Pre-populate embedding cache.
        
        Args:
            texts: List of (text, model, embedding) tuples
            session: Optional database session
            
        Returns:
            Number of entries cached
        """
        count = 0
        for text, model, embedding in texts:
            await self.set_embedding(text, model, embedding, session)
            count += 1
        
        logger.info("cache_warmup_complete", entries_cached=count)
        return count


class NullMultiLayerCache:
    """Null object pattern for when caching is disabled.
    
    Provides the same interface as MultiLayerCache but performs no
    caching operations. Useful for testing or when caching is disabled.
    """
    
    async def get_embedding(
        self,
        text: str,
        model: str,
        session: AsyncSession | None = None,
    ) -> None:
        """Always returns None."""
        return None
    
    async def set_embedding(
        self,
        text: str,
        model: str,
        embedding: list[float],
        session: AsyncSession | None = None,
    ) -> None:
        """Does nothing."""
        pass
    
    async def get_query_result(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        similarity_threshold: float = 0.95,
        session: AsyncSession | None = None,
    ) -> None:
        """Always returns None."""
        return None
    
    async def set_query_result(
        self,
        query: str,
        result: CachedQueryResult,
        query_embedding: list[float] | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Does nothing."""
        pass
    
    async def get_llm_response(
        self,
        prompt: str,
        model: str,
        session: AsyncSession | None = None,
    ) -> None:
        """Always returns None."""
        return None
    
    async def set_llm_response(
        self,
        prompt: str,
        model: str,
        response: CachedLLMResponse,
        session: AsyncSession | None = None,
    ) -> None:
        """Does nothing."""
        pass
    
    async def invalidate_query_cache(
        self,
        query_hash: str | None = None,
        session: AsyncSession | None = None,
    ) -> int:
        """Always returns 0."""
        return 0
    
    async def clear_all(self) -> int:
        """Always returns 0."""
        return 0
    
    def get_stats(self) -> dict[str, Any]:
        """Returns empty stats."""
        return {
            "combined": {"hits": 0, "misses": 0, "hit_rate": 0.0},
            "l1_redis": None,
            "l2_postgres": None,
            "l3_semantic": None,
        }
    
    async def warmup_embedding_cache(
        self,
        texts: list[tuple[str, str, list[float]]],
        session: AsyncSession | None = None,
    ) -> int:
        """Always returns 0."""
        return 0
