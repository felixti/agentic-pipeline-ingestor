"""Embedding service for text vectorization using LLM adapters.

This module provides a high-level service interface for generating text embeddings
using litellm-compatible embedding models. It supports batch processing, caching,
and integration with the existing LLM provider infrastructure.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

# Try to import litellm for embeddings
try:
    import litellm
    from litellm import aembedding
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None  # type: ignore

from src.observability.logging import get_logger
from src.vector_store_config import VectorStoreConfig, get_vector_store_config

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.
    
    Attributes:
        text: Original text that was embedded
        embedding: List of float values representing the embedding vector
        model: Model used for embedding
        dimensions: Number of dimensions in the embedding
        tokens_used: Number of tokens consumed (if available)
        latency_ms: Time taken to generate embedding in milliseconds
    """
    text: str
    embedding: list[float]
    model: str
    dimensions: int
    tokens_used: int | None = None
    latency_ms: float | None = None


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    
    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}


class EmbeddingProviderError(EmbeddingError):
    """Exception raised when the embedding provider fails."""
    pass


class EmbeddingDimensionError(EmbeddingError):
    """Exception raised when embedding dimensions don't match expected size."""
    pass


class EmbeddingCache:
    """Simple in-memory cache for embedding results.
    
    Uses content hash as the cache key to avoid storing large text strings.
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """Initialize the cache.
        
        Args:
            max_size: Maximum number of entries in cache
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[list[float], float]] = {}
        self._access_times: dict[str, float] = {}
    
    def _generate_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model.
        
        Args:
            text: Text content
            model: Model identifier
            
        Returns:
            SHA-256 hash as cache key
        """
        content = f"{text}:{model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> list[float] | None:
        """Get cached embedding if available and not expired.
        
        Args:
            text: Text content
            model: Model identifier
            
        Returns:
            Cached embedding vector or None
        """
        key = self._generate_key(text, model)
        
        if key not in self._cache:
            return None
        
        # Check if expired
        cached_time = self._access_times.get(key, 0)
        if time.time() - cached_time > self.ttl_seconds:
            # Remove expired entry
            del self._cache[key]
            del self._access_times[key]
            return None
        
        # Update access time for LRU behavior
        self._access_times[key] = time.time()
        return self._cache[key][0]
    
    def set(self, text: str, model: str, embedding: list[float]) -> None:
        """Cache an embedding result.
        
        Args:
            text: Text content
            model: Model identifier
            embedding: Embedding vector to cache
        """
        # Evict oldest entries if cache is full
        while len(self._cache) >= self.max_size:
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        key = self._generate_key(text, model)
        self._cache[key] = (embedding, time.time())
        self._access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_times.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


class EmbeddingService:
    """Service for generating text embeddings.
    
    Provides methods for generating embeddings with support for:
    - Batch processing
    - Result caching
    - Error handling and retries
    - Integration with litellm for multi-provider support
    
    Example:
        >>> service = EmbeddingService()
        >>> result = await service.embed_text("Hello world")
        >>> print(f"Embedding dimensions: {result.dimensions}")
        
        >>> # Batch processing
        >>> texts = ["Text 1", "Text 2", "Text 3"]
        >>> results = await service.embed_batch(texts)
    """
    
    def __init__(self, config: VectorStoreConfig | None = None):
        """Initialize the embedding service.
        
        Args:
            config: Vector store configuration. If None, loads from default config file.
            
        Raises:
            ImportError: If litellm is not installed
            EmbeddingError: If configuration is invalid
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not installed. "
                "Install with: pip install litellm"
            )
        
        self.config = config or get_vector_store_config()
        
        if not self.config.enabled:
            logger.warning("Vector store is disabled in configuration")
        
        # Initialize cache if enabled
        self._cache: EmbeddingCache | None = None
        if self.config.cache.enabled:
            self._cache = EmbeddingCache(
                max_size=self.config.cache.max_size,
                ttl_seconds=self.config.cache.ttl_seconds,
            )
        
        self.logger = logger
    
    async def embed_text(
        self,
        text: str,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use caching (if enabled)
            
        Returns:
            EmbeddingResult with embedding vector and metadata
            
        Raises:
            EmbeddingProviderError: If the embedding provider fails
            EmbeddingDimensionError: If embedding dimensions don't match config
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        # Check cache first
        if use_cache and self._cache is not None:
            cached = self._cache.get(text, self.config.embedding.model)
            if cached is not None:
                self.logger.debug("embedding_cache_hit", text_length=len(text))
                return EmbeddingResult(
                    text=text,
                    embedding=cached,
                    model=self.config.embedding.model,
                    dimensions=len(cached),
                    latency_ms=0.0,
                )
        
        start_time = time.monotonic()
        
        try:
            # Prepare request parameters
            params = self.config.to_litellm_params()
            params["input"] = text
            
            self.logger.debug(
                "embedding_request",
                model=self.config.embedding.model,
                text_length=len(text),
            )
            
            # Call litellm for embedding
            response = await aembedding(**params)
            
            # Extract embedding from response
            embedding_data = response.data[0]["embedding"]
            
            # Validate dimensions
            expected_dims = self.config.embedding.dimensions
            if len(embedding_data) != expected_dims:
                raise EmbeddingDimensionError(
                    f"Embedding dimension mismatch: expected {expected_dims}, "
                    f"got {len(embedding_data)}",
                    context={
                        "expected": expected_dims,
                        "actual": len(embedding_data),
                        "model": self.config.embedding.model,
                    }
                )
            
            latency_ms = (time.monotonic() - start_time) * 1000
            
            # Get token usage if available
            tokens_used = None
            if hasattr(response, "usage") and response.usage:
                tokens_used = response.usage.get("prompt_tokens")
            
            result = EmbeddingResult(
                text=text,
                embedding=embedding_data,
                model=self.config.embedding.model,
                dimensions=len(embedding_data),
                tokens_used=tokens_used,
                latency_ms=round(latency_ms, 2),
            )
            
            # Cache the result
            if use_cache and self._cache is not None:
                self._cache.set(text, self.config.embedding.model, embedding_data)
            
            self.logger.info(
                "embedding_generated",
                model=result.model,
                dimensions=result.dimensions,
                latency_ms=result.latency_ms,
                tokens_used=result.tokens_used,
            )
            
            return result
            
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            
            self.logger.error(
                "embedding_failed",
                error=str(e),
                latency_ms=round(latency_ms, 2),
                text_length=len(text),
            )
            
            raise EmbeddingProviderError(
                f"Failed to generate embedding: {e}",
                context={
                    "model": self.config.embedding.model,
                    "text_length": len(text),
                    "latency_ms": round(latency_ms, 2),
                }
            ) from e
    
    async def embed_batch(
        self,
        texts: list[str],
        use_cache: bool = True,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts in batches.
        
        Processes texts in batches according to the configured batch_size.
        Checks cache for each text before making API calls.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching (if enabled)
            
        Returns:
            List of EmbeddingResult objects in same order as input
            
        Raises:
            EmbeddingProviderError: If the embedding provider fails
        """
        if not texts:
            return []
        
        results: list[EmbeddingResult | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []
        
        # Check cache first
        if use_cache and self._cache is not None:
            for i, text in enumerate(texts):
                cached = self._cache.get(text, self.config.embedding.model)
                if cached is not None:
                    results[i] = EmbeddingResult(
                        text=text,
                        embedding=cached,
                        model=self.config.embedding.model,
                        dimensions=len(cached),
                        latency_ms=0.0,
                    )
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = [(i, text) for i, text in enumerate(texts)]
        
        if not texts_to_embed:
            self.logger.debug("batch_all_cached", total_texts=len(texts))
            return [r for r in results if r is not None]
        
        self.logger.info(
            "batch_embedding_start",
            total_texts=len(texts),
            cached=len(texts) - len(texts_to_embed),
            to_embed=len(texts_to_embed),
        )
        
        # Process in batches
        batch_size = self.config.embedding.batch_size
        total_latency = 0.0
        
        for batch_start in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[batch_start:batch_start + batch_size]
            batch_indices = [idx for idx, _ in batch]
            batch_texts = [text for _, text in batch]
            
            try:
                batch_results = await self._embed_batch_raw(batch_texts)
                
                # Store results and update cache
                for idx, result in zip(batch_indices, batch_results):
                    results[idx] = result
                    total_latency += result.latency_ms or 0
                    
                    if use_cache and self._cache is not None:
                        self._cache.set(result.text, result.model, result.embedding)
                
            except Exception as e:
                self.logger.error(
                    "batch_embedding_failed",
                    batch_start=batch_start,
                    batch_size=len(batch),
                    error=str(e),
                )
                raise EmbeddingProviderError(
                    f"Failed to embed batch starting at index {batch_start}: {e}",
                    context={"batch_start": batch_start, "batch_size": len(batch)},
                ) from e
        
        self.logger.info(
            "batch_embedding_completed",
            total_texts=len(texts),
            avg_latency_ms=round(total_latency / len(texts_to_embed), 2) if texts_to_embed else 0,
        )
        
        return [r for r in results if r is not None]
    
    async def _embed_batch_raw(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts using litellm.
        
        Args:
            texts: List of texts to embed (batch size must be <= config limit)
            
        Returns:
            List of EmbeddingResult objects
        """
        start_time = time.monotonic()
        
        # Prepare request parameters
        params = self.config.to_litellm_params()
        params["input"] = texts
        
        # Call litellm for batch embedding
        response = await aembedding(**params)
        
        latency_ms = (time.monotonic() - start_time) * 1000
        
        # Process results
        results = []
        for i, item in enumerate(response.data):
            embedding_data = item["embedding"]
            
            # Validate dimensions
            expected_dims = self.config.embedding.dimensions
            if len(embedding_data) != expected_dims:
                raise EmbeddingDimensionError(
                    f"Embedding dimension mismatch at index {i}: expected {expected_dims}, "
                    f"got {len(embedding_data)}",
                    context={
                        "index": i,
                        "expected": expected_dims,
                        "actual": len(embedding_data),
                    }
                )
            
            results.append(EmbeddingResult(
                text=texts[i],
                embedding=embedding_data,
                model=self.config.embedding.model,
                dimensions=len(embedding_data),
                latency_ms=round(latency_ms / len(texts), 2),  # Approximate per-item latency
            ))
        
        return results
    
    async def health_check(self) -> dict[str, Any]:
        """Check health of embedding service.
        
        Returns:
            Dictionary with health status information
        """
        result = {
            "healthy": False,
            "model": self.config.embedding.model,
            "dimensions": self.config.embedding.dimensions,
            "enabled": self.config.enabled,
        }
        
        if not self.config.enabled:
            result["message"] = "Vector store is disabled"
            return result
        
        if not LITELLM_AVAILABLE:
            result["error"] = "litellm not installed"
            return result
        
        # Test embedding generation
        start_time = time.monotonic()
        try:
            test_result = await self.embed_text("Health check test", use_cache=False)
            latency_ms = (time.monotonic() - start_time) * 1000
            
            result["healthy"] = True
            result["latency_ms"] = round(latency_ms, 2)
            result["actual_dimensions"] = len(test_result.embedding)
            result["message"] = "Embedding service is operational"
            
        except Exception as e:
            result["error"] = str(e)
            result["message"] = f"Health check failed: {e}"
        
        return result
    
    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics if caching is enabled.
        
        Returns:
            Cache statistics or None if caching is disabled
        """
        if self._cache is None:
            return None
        return self._cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear the embedding cache if caching is enabled."""
        if self._cache is not None:
            self._cache.clear()
            self.logger.info("embedding_cache_cleared")


# Convenience functions for simple use cases

async def embed_text(text: str, config: VectorStoreConfig | None = None) -> list[float]:
    """Generate embedding for a single text (convenience function).
    
    Args:
        text: Text to embed
        config: Optional configuration override
        
    Returns:
        Embedding vector as list of floats
    """
    service = EmbeddingService(config)
    result = await service.embed_text(text)
    return result.embedding


async def embed_batch(
    texts: list[str],
    config: VectorStoreConfig | None = None,
) -> list[list[float]]:
    """Generate embeddings for multiple texts (convenience function).
    
    Args:
        texts: List of texts to embed
        config: Optional configuration override
        
    Returns:
        List of embedding vectors
    """
    service = EmbeddingService(config)
    results = await service.embed_batch(texts)
    return [r.embedding for r in results]
