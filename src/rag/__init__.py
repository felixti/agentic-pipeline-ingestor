"""RAG (Retrieval-Augmented Generation) module.

This module provides RAG-related functionality including query rewriting,
strategies for retrieval optimization, and generation enhancement.
"""

from src.rag.cache import (
    CachedLLMResponse,
    CachedQueryResult,
    CacheStats,
    L1RedisCache,
    L2PostgresCache,
    L3SemanticCache,
    MultiLayerCache,
    NullMultiLayerCache,
)
from src.rag.embeddings import (
    Embedding,
    EmbeddingBatch,
    EmbeddingCache,
    EmbeddingModel,
    EmbeddingService,
    ModelAlias,
    QuantizedEmbedding,
    embed_text,
    embed_texts,
)
from src.rag.models import (
    EmbeddingBatchResult,
    EmbeddingCacheStats,
    EmbeddingModelInfo,
    EmbeddingRequest,
    EmbeddingResult,
    QueryRewriteResult,
    RAGConfig,
    RAGMetrics,
    RAGResult,
    Source,
)
from src.rag.router import AgenticRAG, AgenticRAGError, get_strategy_presets
from src.rag.strategies.query_rewriting import QueryRewriter

__all__ = [
    "AgenticRAG",
    "AgenticRAGError",
    "CacheStats",
    "CachedLLMResponse",
    "CachedQueryResult",
    "Embedding",
    "EmbeddingBatch",
    "EmbeddingBatchResult",
    "EmbeddingCache",
    "EmbeddingCacheStats",
    "EmbeddingModel",
    "EmbeddingModelInfo",
    "EmbeddingRequest",
    "EmbeddingResult",
    "EmbeddingService",
    "L1RedisCache",
    "L2PostgresCache",
    "L3SemanticCache",
    "ModelAlias",
    "MultiLayerCache",
    "NullMultiLayerCache",
    "QuantizedEmbedding",
    "QueryRewriteResult",
    "QueryRewriter",
    "RAGConfig",
    "RAGMetrics",
    "RAGResult",
    "Source",
    "embed_text",
    "embed_texts",
    "get_strategy_presets",
]
