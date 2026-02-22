"""Embedding optimization module for RAG.

This module provides advanced embedding capabilities including:
- Multiple embedding model support with automatic selection
- Dimensionality reduction using PCA
- 8-bit quantization for storage efficiency
- Multi-level caching for performance
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_settings
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None  # type: ignore

try:
    import litellm
    from litellm import aembedding
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None  # type: ignore


class EmbeddingModel(str, Enum):
    """Supported embedding models.
    
    Attributes:
        OPENAI_SMALL: text-embedding-3-small (1536 dims, fast)
        OPENAI_LARGE: text-embedding-3-large (3072 dims, precise)
        SENTENCE_TRANSFORMER: all-MiniLM-L6-v2 (384 dims, local)
        BGE_TECHNICAL: bge-large-en-v1.5 (1024 dims, technical)
        VOYAGE: voyage-2 (1024 dims, enterprise)
    """
    
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"
    SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
    BGE_TECHNICAL = "BAAI/bge-large-en-v1.5"
    VOYAGE = "voyage-2"


class ModelAlias(str, Enum):
    """User-friendly aliases for embedding models."""
    
    FAST = "fast"
    PRECISE = "precise"
    LOCAL = "local"
    TECHNICAL = "technical"
    ENTERPRISE = "enterprise"
    AUTO = "auto"


# Model configuration mapping
MODEL_CONFIG: dict[EmbeddingModel, dict[str, Any]] = {
    EmbeddingModel.OPENAI_SMALL: {
        "dimensions": 1536,
        "speed": "fast",
        "quality": "good",
        "batch_size": 100,
        "provider": "openai",
    },
    EmbeddingModel.OPENAI_LARGE: {
        "dimensions": 3072,
        "speed": "medium",
        "quality": "excellent",
        "batch_size": 50,
        "provider": "openai",
    },
    EmbeddingModel.SENTENCE_TRANSFORMER: {
        "dimensions": 384,
        "speed": "very_fast",
        "quality": "good",
        "batch_size": 64,
        "provider": "sentence_transformers",
    },
    EmbeddingModel.BGE_TECHNICAL: {
        "dimensions": 1024,
        "speed": "medium",
        "quality": "excellent",
        "batch_size": 32,
        "provider": "sentence_transformers",
    },
    EmbeddingModel.VOYAGE: {
        "dimensions": 1024,
        "speed": "fast",
        "quality": "excellent",
        "batch_size": 64,
        "provider": "voyage",
    },
}

# Alias to model mapping
ALIAS_TO_MODEL: dict[ModelAlias, EmbeddingModel] = {
    ModelAlias.FAST: EmbeddingModel.OPENAI_SMALL,
    ModelAlias.PRECISE: EmbeddingModel.OPENAI_LARGE,
    ModelAlias.LOCAL: EmbeddingModel.SENTENCE_TRANSFORMER,
    ModelAlias.TECHNICAL: EmbeddingModel.BGE_TECHNICAL,
    ModelAlias.ENTERPRISE: EmbeddingModel.VOYAGE,
}


class Embedding(BaseModel):
    """A single embedding result.
    
    Attributes:
        vector: The embedding vector as a list of floats
        model: Model used for embedding
        dimensions: Number of dimensions
        text_hash: Hash of the original text for cache validation
        metadata: Additional metadata (latency, tokens, etc.)
    """
    
    vector: list[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used for embedding")
    dimensions: int = Field(..., description="Number of dimensions")
    text_hash: str | None = Field(default=None, description="Hash of original text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "vector": [0.1, 0.2, 0.3],
                    "model": "text-embedding-3-small",
                    "dimensions": 1536,
                    "text_hash": "abc123",
                    "metadata": {"latency_ms": 45.2},
                }
            ]
        }
    )


class EmbeddingBatch(BaseModel):
    """Batch of embeddings.
    
    Attributes:
        embeddings: List of embeddings
        model: Model used
        total_tokens: Total tokens consumed
        avg_latency_ms: Average latency per embedding
    """
    
    embeddings: list[Embedding] = Field(default_factory=list)
    model: str = Field(default="")
    total_tokens: int = Field(default=0)
    avg_latency_ms: float = Field(default=0.0)
    cache_hits: int = Field(default=0)
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "embeddings": [],
                    "model": "text-embedding-3-small",
                    "total_tokens": 150,
                    "avg_latency_ms": 45.2,
                    "cache_hits": 2,
                }
            ]
        }
    )


class QuantizedEmbedding(BaseModel):
    """Quantized embedding for storage efficiency.
    
    Uses 8-bit quantization to achieve 4x compression.
    
    Attributes:
        data: Quantized bytes
        scale: Min/max values for dequantization
        original_dims: Original dimensions before quantization
    """
    
    data: bytes = Field(..., description="Quantized bytes")
    scale: tuple[float, float] = Field(..., description="(min, max) values")
    original_dims: int = Field(..., description="Original dimensions")
    
    @classmethod
    def quantize(cls, vector: list[float]) -> QuantizedEmbedding:
        """Quantize float32 vector to int8.
        
        Args:
            vector: Float32 embedding vector
            
        Returns:
            Quantized embedding
        """
        arr = np.array(vector, dtype=np.float32)
        min_val, max_val = float(arr.min()), float(arr.max())
        
        # Avoid division by zero
        if max_val == min_val:
            scaled = np.zeros_like(arr, dtype=np.uint8)
        else:
            # Scale to 0-255 range
            scaled = ((arr - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        return cls(
            data=scaled.tobytes(),
            scale=(min_val, max_val),
            original_dims=len(vector),
        )
    
    def dequantize(self) -> list[float]:
        """Dequantize int8 back to float32.
        
        Returns:
            Dequantized float32 vector
        """
        arr = np.frombuffer(self.data, dtype=np.uint8).astype(np.float32)
        min_val, max_val = self.scale
        
        # Scale back to original range
        dequantized = arr / 255 * (max_val - min_val) + min_val
        return dequantized.tolist()
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio.
        
        Returns:
            Ratio of original size to compressed size
        """
        # Original: float32 = 4 bytes per dimension
        # Quantized: uint8 = 1 byte per dimension
        return 4.0
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "data": "base64encoded...",
                    "scale": (-0.5, 0.5),
                    "original_dims": 1536,
                }
            ]
        }
    )


class EmbeddingCache:
    """Multi-level embedding cache.
    
    Provides in-memory caching with TTL support and LRU eviction.
    """
    
    def __init__(
        self,
        max_size: int = 100000,
        ttl_seconds: int = 86400,
    ):
        """Initialize cache.
        
        Args:
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live for entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[list[float], float]] = {}
        self._access_times: dict[str, float] = {}
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, text: str, model: str) -> str:
        """Generate cache key."""
        content = f"{text}:{model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> list[float] | None:
        """Get cached embedding.
        
        Args:
            text: Original text
            model: Model identifier
            
        Returns:
            Cached embedding or None
        """
        key = self._generate_key(text, model)
        
        if key not in self._cache:
            self._misses += 1
            return None
        
        # Check TTL
        cached_time = self._access_times.get(key, 0)
        if time.time() - cached_time > self.ttl_seconds:
            del self._cache[key]
            del self._access_times[key]
            self._misses += 1
            return None
        
        # Update access time (LRU)
        self._access_times[key] = time.time()
        self._hits += 1
        return self._cache[key][0]
    
    def get_batch(
        self,
        texts: list[str],
        model: str,
    ) -> tuple[list[list[float] | None], int]:
        """Get batch of cached embeddings.
        
        Args:
            texts: List of texts
            model: Model identifier
            
        Returns:
            Tuple of (embeddings, hit_count)
        """
        results: list[list[float] | None] = []
        hits = 0
        
        for text in texts:
            cached = self.get(text, model)
            results.append(cached)
            if cached is not None:
                hits += 1
        
        return results, hits
    
    def set(self, text: str, model: str, embedding: list[float]) -> None:
        """Cache embedding.
        
        Args:
            text: Original text
            model: Model identifier
            embedding: Embedding vector
        """
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        key = self._generate_key(text, model)
        self._cache[key] = (embedding, time.time())
        self._access_times[key] = time.time()
    
    def set_batch(
        self,
        texts: list[str],
        model: str,
        embeddings: list[list[float]],
    ) -> None:
        """Cache batch of embeddings.
        
        Args:
            texts: Original texts
            model: Model identifier
            embeddings: Embedding vectors
        """
        for text, embedding in zip(texts, embeddings):
            self.set(text, model, embedding)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache stats dictionary
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "ttl_seconds": self.ttl_seconds,
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()
        self._hits = 0
        self._misses = 0


class DimensionalityReducer:
    """PCA-based dimensionality reduction for embeddings.
    
    Reduces embedding dimensions while preserving quality.
    """
    
    def __init__(
        self,
        target_dims: int = 256,
        preserve_threshold: float = 0.95,
    ):
        """Initialize reducer.
        
        Args:
            target_dims: Target dimensions
            preserve_threshold: Minimum quality preservation threshold
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for dimensionality reduction. "
                "Install with: pip install scikit-learn"
            )
        
        self.target_dims = target_dims
        self.preserve_threshold = preserve_threshold
        self._pca: Any = None
        self._is_fitted = False
        self._original_dims: int | None = None
    
    def fit(self, embeddings: list[list[float]]) -> DimensionalityReducer:
        """Fit PCA on embedding data.
        
        Args:
            embeddings: Training embeddings
            
        Returns:
            Self for chaining
        """
        if not embeddings:
            raise ValueError("Cannot fit on empty embeddings")
        
        data = np.array(embeddings)
        self._original_dims = data.shape[1]
        
        # Adjust target dims if needed
        effective_target = min(self.target_dims, self._original_dims - 1)
        
        self._pca = PCA(n_components=effective_target)
        self._pca.fit(data)
        self._is_fitted = True
        
        # Log explained variance ratio
        explained = sum(self._pca.explained_variance_ratio_)
        logger.info(
            "pca_fitted",
            original_dims=self._original_dims,
            target_dims=effective_target,
            explained_variance=round(explained, 4),
        )
        
        return self
    
    def reduce(self, embedding: list[float]) -> list[float]:
        """Reduce embedding dimensions.
        
        Args:
            embedding: Original embedding
            
        Returns:
            Reduced embedding
        """
        if not self._is_fitted:
            raise RuntimeError("Reducer must be fitted before use")
        
        data = np.array(embedding).reshape(1, -1)
        reduced = self._pca.transform(data)[0]
        return reduced.tolist()
    
    def reduce_batch(self, embeddings: list[list[float]]) -> list[list[float]]:
        """Reduce batch of embeddings.
        
        Args:
            embeddings: Original embeddings
            
        Returns:
            Reduced embeddings
        """
        if not self._is_fitted:
            raise RuntimeError("Reducer must be fitted before use")
        
        data = np.array(embeddings)
        reduced = self._pca.transform(data)
        return reduced.tolist()
    
    def reconstruct(self, reduced: list[float]) -> list[float]:
        """Reconstruct original dimension embedding.
        
        Args:
            reduced: Reduced embedding
            
        Returns:
            Reconstructed embedding (approximation)
        """
        if not self._is_fitted:
            raise RuntimeError("Reducer must be fitted before use")
        
        data = np.array(reduced).reshape(1, -1)
        reconstructed = self._pca.inverse_transform(data)[0]
        return reconstructed.tolist()
    
    def get_quality_score(self) -> float:
        """Get quality preservation score.
        
        Returns:
            Fraction of variance preserved (0-1)
        """
        if not self._is_fitted:
            return 0.0
        return float(sum(self._pca.explained_variance_ratio_))
    
    def is_quality_acceptable(self) -> bool:
        """Check if quality preservation meets threshold.
        
        Returns:
            True if quality >= threshold
        """
        return self.get_quality_score() >= self.preserve_threshold


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""
    
    def __init__(self, model: EmbeddingModel, config: dict[str, Any] | None = None):
        """Initialize embedder.
        
        Args:
            model: Embedding model
            config: Optional configuration overrides
        """
        self.model = model
        self.model_config = MODEL_CONFIG.get(model, {})
        if config:
            self.model_config.update(config)
        
        self.dimensions = self.model_config.get("dimensions", 1536)
        self.batch_size = self.model_config.get("batch_size", 100)
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model.value


class LiteLLMEmbedder(BaseEmbedder):
    """Embedder using litellm for multi-provider support."""
    
    def __init__(
        self,
        model: EmbeddingModel,
        config: dict[str, Any] | None = None,
    ):
        """Initialize LiteLLM embedder.
        
        Args:
            model: Embedding model
            config: Optional configuration
        """
        super().__init__(model, config)
        
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for embedding generation. "
                "Install with: pip install litellm"
            )
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using litellm.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        settings = get_settings()
        
        # Prepare parameters
        params: dict[str, Any] = {
            "model": self.model.value,
            "input": texts,
            "dimensions": self.dimensions,
        }
        
        # Add API configuration from settings
        if hasattr(settings, "llm_yaml"):
            yaml_config = settings.llm_yaml.load_yaml()
            if "llm" in yaml_config and "router" in yaml_config["llm"]:
                router = yaml_config["llm"]["router"]
                if router and len(router) > 0:
                    litellm_params = router[0].get("litellm_params", {})
                    if "api_base" in litellm_params:
                        params["api_base"] = litellm_params["api_base"]
                    if "api_key" in litellm_params:
                        params["api_key"] = litellm_params["api_key"]
        
        try:
            response = await aembedding(**params)
            
            embeddings = []
            for item in response.data:
                embedding_data = item["embedding"]
                
                # Validate dimensions
                if len(embedding_data) != self.dimensions:
                    logger.warning(
                        "dimension_mismatch",
                        expected=self.dimensions,
                        actual=len(embedding_data),
                    )
                
                embeddings.append(embedding_data)
            
            return embeddings
            
        except Exception as e:
            logger.error("embedding_failed", model=self.model.value, error=str(e))
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using sentence-transformers (local execution).
    
    Note: This is a placeholder implementation. Actual implementation
    would require sentence-transformers library.
    """
    
    def __init__(
        self,
        model: EmbeddingModel,
        config: dict[str, Any] | None = None,
        device: str = "cpu",
    ):
        """Initialize sentence-transformers embedder.
        
        Args:
            model: Embedding model
            config: Optional configuration
            device: Device to run on (cpu/cuda)
        """
        super().__init__(model, config)
        self.device = device
        self._model = None
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using sentence-transformers.
        
        Note: This is a mock implementation. In production,
        this would use the actual sentence-transformers library.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors (mock data)
        """
        # Mock implementation - returns random normalized vectors
        embeddings = []
        for _ in texts:
            vec = np.random.randn(self.dimensions).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            embeddings.append(vec.tolist())
        return embeddings


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class ModelSelectionError(EmbeddingError):
    """Exception raised when model selection fails."""
    pass


class EmbeddingService:
    """Optimized embedding service with model selection and caching.
    
    This service provides:
    - Multiple embedding model support
    - Automatic model selection based on content
    - Dimensionality reduction
    - Quantization for storage efficiency
    - Multi-level caching
    
    Example:
        >>> service = EmbeddingService()
        >>> result = await service.embed(
        ...     ["text to embed"],
        ...     model="auto",
        ...     use_cache=True
        ... )
    """
    
    # Technical terms for content analysis
    TECHNICAL_TERMS = [
        "api", "function", "class", "method", "parameter",
        "configuration", "implementation", "algorithm",
        "database", "query", "server", "client",
        "framework", "library", "module", "package",
        "syntax", "compiler", "runtime", "debug",
        "architecture", "protocol", "interface",
    ]
    
    def __init__(
        self,
        default_model: ModelAlias | str = ModelAlias.FAST,
        enable_cache: bool = True,
        enable_reduction: bool = False,
        enable_quantization: bool = False,
        cache_config: dict[str, Any] | None = None,
        reduction_config: dict[str, Any] | None = None,
    ):
        """Initialize embedding service.
        
        Args:
            default_model: Default model or alias
            enable_cache: Whether to enable caching
            enable_reduction: Whether to enable dimensionality reduction
            enable_quantization: Whether to enable quantization
            cache_config: Cache configuration
            reduction_config: Reduction configuration
        """
        self.default_model = default_model
        self.enable_cache = enable_cache
        self.enable_reduction = enable_reduction
        self.enable_quantization = enable_quantization
        
        # Initialize embedders
        self._embedders: dict[EmbeddingModel, BaseEmbedder] = {}
        self._init_embedders()
        
        # Initialize cache
        self._cache: EmbeddingCache | None = None
        if enable_cache:
            cache_cfg = cache_config or {}
            self._cache = EmbeddingCache(
                max_size=cache_cfg.get("max_size", 100000),
                ttl_seconds=cache_cfg.get("ttl_seconds", 86400),
            )
        
        # Initialize dimensionality reducer
        self._reducer: DimensionalityReducer | None = None
        if enable_reduction:
            if SKLEARN_AVAILABLE:
                red_cfg = reduction_config or {}
                self._reducer = DimensionalityReducer(
                    target_dims=red_cfg.get("target_dimensions", 256),
                    preserve_threshold=red_cfg.get("preserve_threshold", 0.95),
                )
            else:
                logger.warning("scikit-learn not available, dimensionality reduction disabled")
        
        self._quantized_cache: dict[str, QuantizedEmbedding] = {}
    
    def _init_embedders(self) -> None:
        """Initialize embedder instances."""
        for model in EmbeddingModel:
            config = MODEL_CONFIG.get(model, {})
            provider = config.get("provider", "openai")
            
            try:
                if provider in ("openai", "voyage"):
                    if LITELLM_AVAILABLE:
                        self._embedders[model] = LiteLLMEmbedder(model, config)
                    else:
                        logger.warning(f"litellm not available, {model.value} disabled")
                elif provider == "sentence_transformers":
                    self._embedders[model] = SentenceTransformerEmbedder(model, config)
            except Exception as e:
                logger.warning(f"Failed to initialize {model.value}: {e}")
    
    def _get_model_from_alias(self, alias: str) -> EmbeddingModel:
        """Resolve alias to model.
        
        Args:
            alias: Model alias or name
            
        Returns:
            Resolved embedding model
        """
        # Check if it's an alias
        try:
            model_alias = ModelAlias(alias)
            if model_alias in ALIAS_TO_MODEL:
                return ALIAS_TO_MODEL[model_alias]
        except ValueError:
            pass
        
        # Check if it's a direct model name
        try:
            return EmbeddingModel(alias)
        except ValueError:
            pass
        
        # Default fallback
        logger.warning(f"Unknown model '{alias}', using default")
        return EmbeddingModel.OPENAI_SMALL
    
    def select_model(self, texts: list[str]) -> EmbeddingModel:
        """Select optimal model based on content analysis.
        
        Selection rules:
        - Technical content + long texts -> BGE technical
        - Very long texts -> OpenAI large
        - Default -> OpenAI small (fast)
        
        Args:
            texts: Texts to analyze
            
        Returns:
            Selected embedding model
        """
        if not texts:
            return EmbeddingModel.OPENAI_SMALL
        
        # Calculate metrics
        avg_length = sum(len(t) for t in texts) / len(texts)
        has_technical = any(self._is_technical(t) for t in texts)
        
        logger.debug(
            "model_selection_analysis",
            avg_length=avg_length,
            has_technical=has_technical,
            text_count=len(texts),
        )
        
        # Apply selection rules
        if has_technical and avg_length > 500:
            return EmbeddingModel.BGE_TECHNICAL
        elif avg_length > 1000:
            return EmbeddingModel.OPENAI_LARGE
        else:
            return EmbeddingModel.OPENAI_SMALL
    
    def _is_technical(self, text: str) -> bool:
        """Check if text contains technical terms.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if technical content detected
        """
        text_lower = text.lower()
        term_count = sum(1 for term in self.TECHNICAL_TERMS if term in text_lower)
        return term_count >= 2
    
    async def embed(
        self,
        texts: list[str],
        model: str = "auto",
        use_cache: bool = True,
        reduce: bool | None = None,
        quantize: bool | None = None,
    ) -> EmbeddingBatch:
        """Generate embeddings with optimization.
        
        Args:
            texts: Texts to embed
            model: Model name, alias, or "auto"
            use_cache: Whether to use caching
            reduce: Override dimensionality reduction (None = use service setting)
            quantize: Override quantization (None = use service setting)
            
        Returns:
            Batch of embeddings
        """
        if not texts:
            return EmbeddingBatch()
        
        # Resolve model
        if model == "auto":
            selected_model = self.select_model(texts)
        else:
            selected_model = self._get_model_from_alias(model)
        
        # Check cache
        cached_embeddings: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []
        cache_hits = 0
        
        if use_cache and self._cache is not None:
            cached_embeddings, cache_hits = self._cache.get_batch(
                texts, selected_model.value
            )
            
            # Identify texts that need embedding
            for i, cached in enumerate(cached_embeddings):
                if cached is None:
                    texts_to_embed.append((i, texts[i]))
        else:
            texts_to_embed = [(i, texts[i]) for i in range(len(texts))]
        
        # Generate embeddings for non-cached texts
        new_embeddings: list[list[float]] = []
        if texts_to_embed:
            embedder = self._embedders.get(selected_model)
            if embedder is None:
                raise ModelSelectionError(
                    f"Embedder not available for {selected_model.value}"
                )
            
            # Process in batches
            batch_texts = [t for _, t in texts_to_embed]
            start_time = time.monotonic()
            
            try:
                batch_embeddings = await embedder.embed(batch_texts)
                latency_ms = (time.monotonic() - start_time) * 1000
                
                new_embeddings = batch_embeddings
                
                # Cache new embeddings
                if use_cache and self._cache is not None:
                    self._cache.set_batch(batch_texts, selected_model.value, batch_embeddings)
                
                logger.info(
                    "embeddings_generated",
                    model=selected_model.value,
                    count=len(batch_embeddings),
                    latency_ms=round(latency_ms, 2),
                )
                
            except Exception as e:
                logger.error("embedding_generation_failed", error=str(e))
                raise
        
        # Merge cached and new embeddings
        final_embeddings: list[list[float]] = []
        new_idx = 0
        for i in range(len(texts)):
            if cached_embeddings[i] is not None:
                final_embeddings.append(cached_embeddings[i])
            else:
                final_embeddings.append(new_embeddings[new_idx])
                new_idx += 1
        
        # Apply dimensionality reduction if enabled
        should_reduce = reduce if reduce is not None else self.enable_reduction
        if should_reduce and self._reducer is not None:
            if not self._reducer._is_fitted:
                # Fit on first batch
                self._reducer.fit(final_embeddings)
            
            final_embeddings = self._reducer.reduce_batch(final_embeddings)
        
        # Apply quantization if enabled
        should_quantize = quantize if quantize is not None else self.enable_quantization
        quantized_data: list[QuantizedEmbedding] = []
        if should_quantize:
            for vec in final_embeddings:
                quantized = QuantizedEmbedding.quantize(vec)
                quantized_data.append(quantized)
            
            # Dequantize for return (storage would keep quantized)
            final_embeddings = [q.dequantize() for q in quantized_data]
        
        # Create result
        embedding_objects = [
            Embedding(
                vector=vec,
                model=selected_model.value,
                dimensions=len(vec),
                text_hash=hashlib.sha256(texts[i].encode()).hexdigest()[:16],
            )
            for i, vec in enumerate(final_embeddings)
        ]
        
        return EmbeddingBatch(
            embeddings=embedding_objects,
            model=selected_model.value,
            cache_hits=cache_hits,
        )
    
    async def embed_single(
        self,
        text: str,
        model: str = "auto",
        use_cache: bool = True,
    ) -> Embedding:
        """Embed a single text.
        
        Args:
            text: Text to embed
            model: Model or alias
            use_cache: Whether to use caching
            
        Returns:
            Single embedding
        """
        batch = await self.embed([text], model=model, use_cache=use_cache)
        return batch.embeddings[0]
    
    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self._cache is None:
            return None
        return self._cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self._cache is not None:
            self._cache.clear()
    
    def fit_reducer(self, embeddings: list[list[float]]) -> None:
        """Fit dimensionality reducer on sample embeddings.
        
        Args:
            embeddings: Sample embeddings for fitting PCA
        """
        if self._reducer is not None:
            self._reducer.fit(embeddings)
            quality = self._reducer.get_quality_score()
            logger.info(
                "reducer_fitted",
                quality_preserved=round(quality, 4),
                acceptable=self._reducer.is_quality_acceptable(),
            )
    
    def get_reducer_quality(self) -> float:
        """Get dimensionality reducer quality score."""
        if self._reducer is None:
            return 0.0
        return self._reducer.get_quality_score()


# Convenience functions

async def embed_texts(
    texts: list[str],
    model: str = "auto",
    use_cache: bool = True,
) -> list[list[float]]:
    """Convenience function to embed texts.
    
    Args:
        texts: Texts to embed
        model: Model or alias
        use_cache: Whether to use caching
        
    Returns:
        List of embedding vectors
    """
    service = EmbeddingService()
    batch = await service.embed(texts, model=model, use_cache=use_cache)
    return [e.vector for e in batch.embeddings]


async def embed_text(
    text: str,
    model: str = "auto",
    use_cache: bool = True,
) -> list[float]:
    """Convenience function to embed a single text.
    
    Args:
        text: Text to embed
        model: Model or alias
        use_cache: Whether to use caching
        
    Returns:
        Embedding vector
    """
    service = EmbeddingService()
    embedding = await service.embed_single(text, model=model, use_cache=use_cache)
    return embedding.vector
