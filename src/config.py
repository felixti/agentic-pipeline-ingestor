"""Configuration management for the Agentic Data Pipeline Ingestor.

This module provides centralized configuration management using Pydantic Settings,
supporting environment variables, .env files, and YAML configuration files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline",
        description="PostgreSQL connection URL",
    )
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=100)
    pool_timeout: int = Field(default=30, ge=1)
    echo: bool = Field(default=False, description="Enable SQL echo for debugging")

    model_config = SettingsConfigDict(env_prefix="DB_")


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    url: RedisDsn = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    password: str | None = Field(default=None)
    ssl: bool = Field(default=False)
    socket_timeout: int = Field(default=5)
    socket_connect_timeout: int = Field(default=5)
    retry_on_timeout: bool = Field(default=True)

    model_config = SettingsConfigDict(env_prefix="REDIS_")


class OpenSearchSettings(BaseSettings):
    """OpenSearch connection settings for audit logs."""

    hosts: list[str] = Field(default=["http://localhost:9200"])
    username: str | None = Field(default=None)
    password: str | None = Field(default=None)
    use_ssl: bool = Field(default=False)
    verify_certs: bool = Field(default=True)
    index_prefix: str = Field(default="pipeline-audit")

    model_config = SettingsConfigDict(env_prefix="OPENSEARCH_")


class AzureSettings(BaseSettings):
    """Azure service settings."""

    tenant_id: str | None = Field(default=None)
    client_id: str | None = Field(default=None)
    client_secret: str | None = Field(default=None)
    subscription_id: str | None = Field(default=None)
    storage_account: str | None = Field(default=None)
    storage_key: str | None = Field(default=None)
    queue_connection_string: str | None = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="AZURE_")


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""

    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for JWT signing",
    )
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, ge=1)
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_default: int = Field(default=100, ge=1)
    rate_limit_window: int = Field(default=60, ge=1)
    allowed_hosts: list[str] = Field(default=["*"])
    cors_origins: list[str] = Field(default=["http://localhost:3000"])
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: list[str] = Field(default=["*"])
    cors_allow_headers: list[str] = Field(default=["*"])

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate that secret key is not the default in production."""
        if v == "change-me-in-production":
            import warnings
            warnings.warn(
                "Using default secret key. Please set a secure SECRET_KEY in production!",
                stacklevel=2,
            )
        return v

    model_config = SettingsConfigDict(env_prefix="SECURITY_")


class ProcessingSettings(BaseSettings):
    """Document processing settings."""

    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    max_pages_per_document: int = Field(default=1000, ge=1)
    allowed_mime_types: list[str] = Field(
        default=[
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "image/jpeg",
            "image/png",
            "image/tiff",
            "text/csv",
            "text/plain",
            "application/zip",
        ]
    )
    temp_dir: Path = Field(default=Path("/tmp/pipeline"))
    cleanup_temp_files: bool = Field(default=True)
    default_timeout_seconds: int = Field(default=300, ge=30)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=5, ge=1)

    model_config = SettingsConfigDict(env_prefix="PROCESSING_")


class ObservabilitySettings(BaseSettings):
    """Observability and monitoring settings."""

    service_name: str = Field(default="pipeline-ingestor")
    service_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    otlp_endpoint: str | None = Field(default=None)
    otlp_insecure: bool = Field(default=True)
    jaeger_enabled: bool = Field(default=False)
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_request_body: bool = Field(default=False)
    log_response_body: bool = Field(default=False)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    model_config = SettingsConfigDict(env_prefix="OTEL_")


class QueryRewritingSettings(BaseSettings):
    """Query Rewriting service settings for RAG optimization."""

    enabled: bool = Field(default=True, description="Enable query rewriting")
    model: str = Field(
        default="agentic-decisions",
        description="Model group to use for query rewriting",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM",
    )
    max_tokens: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum tokens for rewrite response",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching of rewrite results",
    )
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retry attempts for LLM calls",
    )
    timeout_ms: int = Field(
        default=2000,
        ge=500,
        le=10000,
        description="Timeout for LLM calls in milliseconds",
    )
    max_query_length: int = Field(
        default=4000,
        ge=100,
        le=10000,
        description="Maximum allowed query length",
    )

    model_config = SettingsConfigDict(env_prefix="QUERY_REWRITE_")


class HyDESettings(BaseSettings):
    """HyDE (Hypothetical Document Embeddings) settings for RAG optimization.
    
    HyDE generates hypothetical documents that answer the user's query,
    then uses these documents for vector search instead of the raw query.
    This improves retrieval quality for vague or out-of-domain queries.
    """

    enabled: bool = Field(
        default=True,
        description="Enable HyDE query rewriting",
    )
    model: str = Field(
        default="agentic-decisions",
        description="Model group to use for hypothetical document generation",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for hypothetical document generation",
    )
    max_tokens: int = Field(
        default=300,
        ge=50,
        le=1000,
        description="Maximum tokens for hypothetical document",
    )
    max_hypothetical_length: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Maximum character length for hypothetical document",
    )
    system_prompt: str = Field(
        default=(
            "Generate a focused 2-4 sentence hypothetical document that answers the "
            "user's question. This will be used for semantic search, so include key "
            "concepts and terminology that would appear in relevant documents."
        ),
        description="System prompt for hypothetical document generation",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching of hypothetical documents",
    )
    cache_ttl: int = Field(
        default=7200,
        ge=300,
        le=86400,
        description="Cache time-to-live in seconds (default: 2 hours)",
    )
    enable_for_query_types: list[str] = Field(
        default=["complex_questions", "vague_queries", "out_of_domain"],
        description="Query types to enable HyDE for",
    )
    fallback_to_standard: bool = Field(
        default=True,
        description="Fallback to standard query rewriting if HyDE fails",
    )
    max_processing_time_ms: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Maximum processing time in milliseconds",
    )

    model_config = SettingsConfigDict(env_prefix="HYDE_")


class ReRankingSettings(BaseSettings):
    """Re-ranking settings for RAG optimization.
    
    Re-ranking uses cross-encoder models to score and re-order retrieved chunks
    for significantly improved relevance. Cross-encoders encode query and document
    together, capturing finer relevance signals than bi-encoders.
    
    Attributes:
        enabled: Whether re-ranking is enabled
        models: Available model presets (default, high_precision, fast, alternative)
        initial_retrieval_k: Number of chunks to retrieve for re-ranking
        final_k: Number of chunks to return after re-ranking
        batch_size: Batch size for cross-encoder inference
        async_mode: Whether to use async processing
        timeout_ms: Timeout for re-ranking operations in milliseconds
    """

    enabled: bool = Field(
        default=True,
        description="Enable cross-encoder re-ranking",
    )
    models: dict[str, str] = Field(
        default={
            "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "high_precision": "cross-encoder/ms-marco-electra-base",
            "fast": "cross-encoder/ms-marco-TinyBERT-L-2",
            "alternative": "BAAI/bge-reranker-base",
        },
        description="Available cross-encoder model presets",
    )
    initial_retrieval_k: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of chunks to retrieve for re-ranking",
    )
    final_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to return after re-ranking",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Batch size for cross-encoder inference",
    )
    async_mode: bool = Field(
        default=True,
        description="Use async processing for re-ranking",
    )
    timeout_ms: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Timeout for re-ranking operations in milliseconds",
    )

    model_config = SettingsConfigDict(env_prefix="RERANKING_")


class ClassificationSettings(BaseSettings):
    """Query Classification settings for RAG strategy selection.
    
    Classification determines the optimal RAG strategy based on query type
    (factual, analytical, comparative, vague, multi_hop).
    
    Attributes:
        enabled: Whether classification is enabled
        model: Model group to use for classification
        temperature: Sampling temperature for LLM classification
        max_tokens: Maximum tokens for classification response
        cache_enabled: Whether to cache classification results
        cache_ttl: Cache time-to-live in seconds
        min_confidence_threshold: Minimum confidence for accepting classification
        use_pattern_fallback: Whether to use pattern-based fallback classification
        timeout_ms: Timeout for classification operations in milliseconds
    """

    enabled: bool = Field(
        default=True,
        description="Enable query classification",
    )
    model: str = Field(
        default="agentic-decisions",
        description="Model group to use for classification",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM classification",
    )
    max_tokens: int = Field(
        default=300,
        ge=100,
        le=1000,
        description="Maximum tokens for classification response",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching of classification results",
    )
    cache_ttl: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Cache time-to-live in seconds",
    )
    min_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for accepting classification",
    )
    use_pattern_fallback: bool = Field(
        default=True,
        description="Use pattern-based fallback for fast classification",
    )
    timeout_ms: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Timeout for classification operations in milliseconds",
    )

    model_config = SettingsConfigDict(env_prefix="CLASSIFICATION_")


class HybridSearchSettings(BaseSettings):
    """Hybrid Search settings for RAG retrieval.
    
    This settings class configures the hybrid search service which combines
    vector similarity and full-text search using Reciprocal Rank Fusion (RRF)
    and weighted sum methods.
    
    Attributes:
        enabled: Whether hybrid search is enabled
        default_vector_weight: Default weight for vector search scores
        default_text_weight: Default weight for text search scores
        rrf_k: RRF constant k value (typically 60)
        default_fusion_method: Default fusion method (rrf or weighted_sum)
        min_similarity: Default minimum similarity threshold
        max_top_k: Maximum number of results to return
        fallback_mode: Fallback strategy when one search returns empty
        apply_weights_to_rrf: Whether to apply weights in RRF formula
    """

    enabled: bool = Field(
        default=True,
        description="Enable hybrid search",
    )
    default_vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default weight for vector search scores",
    )
    default_text_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Default weight for text search scores",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="RRF constant k value",
    )
    default_fusion_method: str = Field(
        default="rrf",
        description="Default fusion method: rrf or weighted_sum",
    )
    min_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default minimum similarity threshold",
    )
    max_top_k: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of results to return",
    )
    fallback_mode: str = Field(
        default="auto",
        description="Fallback strategy: auto, vector, text, or strict",
    )
    apply_weights_to_rrf: bool = Field(
        default=True,
        description="Apply weights to RRF formula",
    )
    query_expansion_enabled: bool = Field(
        default=True,
        description="Enable query expansion for better lexical matching",
    )
    query_expansion_max_terms: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum expanded terms for query expansion",
    )
    weight_presets: dict[str, dict[str, float]] = Field(
        default={
            "semantic_focus": {"vector": 0.9, "text": 0.1},
            "balanced": {"vector": 0.7, "text": 0.3},
            "lexical_focus": {"vector": 0.3, "text": 0.7},
        },
        description="Predefined weight configurations for different search focuses",
    )
    filterable_fields: list[str] = Field(
        default=[
            "source_type",
            "document_type",
            "created_date",
            "author",
            "tags",
        ],
        description="Fields that can be used for metadata filtering",
    )

    @field_validator("default_fusion_method")
    @classmethod
    def validate_fusion_method(cls, v: str) -> str:
        """Validate fusion method."""
        valid_methods = {"rrf", "weighted_sum"}
        if v.lower() not in valid_methods:
            raise ValueError(f"Invalid fusion method: {v}. Must be one of {valid_methods}")
        return v.lower()

    @field_validator("fallback_mode")
    @classmethod
    def validate_fallback_mode(cls, v: str) -> str:
        """Validate fallback mode."""
        valid_modes = {"auto", "vector", "text", "strict"}
        if v.lower() not in valid_modes:
            raise ValueError(f"Invalid fallback mode: {v}. Must be one of {valid_modes}")
        return v.lower()

    model_config = SettingsConfigDict(env_prefix="HYBRID_SEARCH_")


class AgenticRAGSettings(BaseSettings):
    """Agentic RAG Router settings for orchestrating RAG strategies.
    
    This settings class configures the AgenticRAG router which orchestrates
    all RAG strategies based on query classification.
    
    Attributes:
        enabled: Whether agentic RAG is enabled
        quality_threshold: Minimum retrieval quality score (0-1)
        max_iterations: Maximum self-correction iterations
        default_preset: Default strategy preset (fast, balanced, thorough, auto)
        multi_hop_enabled: Whether to enable multi-hop query processing
        latency_target_ms: Target latency in milliseconds
    """

    enabled: bool = Field(
        default=True,
        description="Enable agentic RAG orchestration",
    )
    quality_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum retrieval quality score to accept results",
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum self-correction iterations",
    )
    default_preset: str = Field(
        default="auto",
        description="Default strategy preset: fast, balanced, thorough, or auto",
    )
    multi_hop_enabled: bool = Field(
        default=True,
        description="Enable multi-hop query processing",
    )
    latency_target_ms: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Target latency in milliseconds",
    )

    model_config = SettingsConfigDict(env_prefix="AGENTIC_RAG_")


class ContextualRetrievalSettings(BaseSettings):
    """Contextual Retrieval settings for enhancing chunks with context.
    
    This settings class configures the ContextualRetrieval service which
    enhances document chunks with surrounding context before embedding.
    This improves semantic understanding and retrieval quality.
    
    Attributes:
        enabled: Whether contextual retrieval is enabled
        default_strategy: Default context enhancement strategy
        strategies: Configuration for each strategy
        enable_embedding: Whether to generate embeddings for enhanced text
    """

    enabled: bool = Field(
        default=True,
        description="Enable contextual retrieval enhancement",
    )
    default_strategy: str = Field(
        default="parent_document",
        description="Default context strategy: parent_document, window, or hierarchical",
    )
    enable_embedding: bool = Field(
        default=False,
        description="Generate embeddings for enhanced text",
    )
    strategies: dict[str, dict[str, Any]] = Field(
        default={
            "parent_document": {
                "include_metadata": True,
                "metadata_fields": ["title", "author", "category"],
                "max_context_length": 256,
            },
            "window": {
                "window_size": 1,
                "separator": " | ",
            },
            "hierarchical": {
                "max_depth": 3,
                "include_path": True,
            },
        },
        description="Configuration for each context strategy",
    )

    @field_validator("default_strategy")
    @classmethod
    def validate_default_strategy(cls, v: str) -> str:
        """Validate default strategy."""
        valid_strategies = {"parent_document", "window", "hierarchical"}
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy: {v}. Must be one of: {valid_strategies}")
        return v

    model_config = SettingsConfigDict(env_prefix="CONTEXTUAL_RETRIEVAL_")


class ChunkingSettings(BaseSettings):
    """Chunking strategy settings for document segmentation.
    
    This settings class configures the various chunking strategies available
    for document segmentation including semantic, hierarchical, fixed-size,
    and agentic selection.
    
    Attributes:
        default_strategy: Default chunking strategy to use
        semantic: Configuration for semantic chunking
        hierarchical: Configuration for hierarchical chunking
        fixed: Configuration for fixed-size chunking
        agentic: Configuration for agentic strategy selection
        special_elements: Special handling for code blocks, tables, etc.
    """

    default_strategy: str = Field(
        default="agentic",
        description="Default chunking strategy: semantic, hierarchical, fixed, or agentic",
    )
    
    semantic: dict[str, Any] = Field(
        default={
            "similarity_threshold": 0.85,
            "min_chunk_size": 100,
            "max_chunk_size": 512,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        description="Semantic chunking configuration",
    )
    
    hierarchical: dict[str, Any] = Field(
        default={
            "max_depth": 4,
            "respect_headers": True,
            "preserve_code_blocks": True,
        },
        description="Hierarchical chunking configuration",
    )
    
    fixed: dict[str, Any] = Field(
        default={
            "chunk_size": 512,
            "overlap": 50,
            "tokenizer": "cl100k_base",
        },
        description="Fixed-size chunking configuration",
    )
    
    agentic: dict[str, Any] = Field(
        default={
            "selection_model": "gpt-4o-mini",
            "decision_prompt": (
                "Analyze this document and select the best chunking strategy:\n"
                '- "hierarchical" for structured docs with clear sections\n'
                '- "semantic" for technical content with logical breaks\n'
                '- "fixed" for narrative or mixed content'
            ),
        },
        description="Agentic strategy selection configuration",
    )
    
    special_elements: dict[str, Any] = Field(
        default={
            "code_blocks": {
                "preserve_integrity": True,
                "max_chunk_size": 1024,
            },
            "tables": {
                "preserve_rows": True,
                "max_rows_per_chunk": 10,
            },
        },
        description="Special element handling configuration",
    )

    @field_validator("default_strategy")
    @classmethod
    def validate_default_strategy(cls, v: str) -> str:
        """Validate default strategy."""
        valid_strategies = {"semantic", "hierarchical", "fixed", "agentic"}
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy: {v}. Must be one of: {valid_strategies}")
        return v

    model_config = SettingsConfigDict(env_prefix="CHUNKING_")


class EmbeddingOptimizationSettings(BaseSettings):
    """Embedding optimization settings for RAG.
    
    This settings class configures embedding model selection, dimensionality
    reduction, quantization, and caching for optimized embedding generation.
    
    Attributes:
        default_model: Default embedding model to use
        models: Configuration for available models
        optimization: Optimization settings (reduction, quantization, caching)
        auto_selection: Automatic model selection configuration
    """

    default_model: str = Field(
        default="text-embedding-3-small",
        description="Default embedding model identifier",
    )
    
    models: dict[str, dict[str, Any]] = Field(
        default={
            "text-embedding-3-small": {
                "provider": "openai",
                "dimensions": 1536,
                "batch_size": 100,
                "speed": "fast",
                "quality": "good",
            },
            "text-embedding-3-large": {
                "provider": "openai",
                "dimensions": 3072,
                "batch_size": 50,
                "speed": "medium",
                "quality": "excellent",
            },
            "all-MiniLM-L6-v2": {
                "provider": "sentence-transformers",
                "dimensions": 384,
                "batch_size": 64,
                "device": "cuda",
                "speed": "very_fast",
                "quality": "good",
            },
            "bge-large-en-v1.5": {
                "provider": "sentence-transformers",
                "dimensions": 1024,
                "batch_size": 32,
                "speed": "medium",
                "quality": "excellent",
            },
            "voyage-2": {
                "provider": "voyage",
                "dimensions": 1024,
                "batch_size": 64,
                "speed": "fast",
                "quality": "excellent",
            },
        },
        description="Configuration for available embedding models",
    )
    
    optimization: dict[str, Any] = Field(
        default={
            "dimensionality_reduction": {
                "enabled": True,
                "target_dimensions": 256,
                "preserve_quality_threshold": 0.95,
                "method": "pca",
            },
            "quantization": {
                "enabled": True,
                "bits": 8,
                "compression_ratio": 4.0,
            },
            "caching": {
                "enabled": True,
                "ttl": 86400,
                "max_size": 100000,
            },
        },
        description="Optimization settings",
    )
    
    auto_selection: dict[str, Any] = Field(
        default={
            "enabled": True,
            "rules": [
                {
                    "condition": "technical_content AND length > 500",
                    "model": "bge-large-en-v1.5",
                    "priority": 10,
                },
                {
                    "condition": "length > 2000",
                    "model": "text-embedding-3-large",
                    "priority": 5,
                },
                {
                    "condition": "default",
                    "model": "text-embedding-3-small",
                    "priority": 0,
                },
            ],
        },
        description="Automatic model selection configuration",
    )
    
    performance_targets: dict[str, float] = Field(
        default={
            "max_latency_ms": 100.0,
            "min_cache_hit_rate": 0.70,
            "min_quality_retention": 0.95,
            "target_compression_ratio": 4.0,
        },
        description="Performance targets for embedding optimization",
    )
    
    @field_validator("default_model")
    @classmethod
    def validate_default_model(cls, v: str) -> str:
        """Validate default model."""
        valid_models = {
            "text-embedding-3-small",
            "text-embedding-3-large",
            "all-MiniLM-L6-v2",
            "bge-large-en-v1.5",
            "voyage-2",
            "fast",
            "precise",
            "local",
            "technical",
            "enterprise",
            "auto",
        }
        if v not in valid_models:
            raise ValueError(f"Invalid default model: {v}")
        return v
    
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")


class CachingSettings(BaseSettings):
    """Multi-layer caching settings for RAG operations.
    
    Configures the three-layer caching system:
    - L1: Redis (in-memory, fast access)
    - L2: PostgreSQL (persistent, longer TTL)
    - L3: Semantic (vector similarity matching)
    
    Attributes:
        enabled: Master switch for caching
        layers: Configuration for each cache layer
        cache_targets: What types of data to cache
        invalidation: Cache invalidation strategies
        monitoring: Hit rate tracking and alerting
    """
    
    enabled: bool = Field(
        default=True,
        description="Enable multi-layer caching",
    )
    
    layers: dict[str, Any] = Field(
        default={
            "l1_redis": {
                "enabled": True,
                "ttl": 3600,  # 1 hour
                "max_size": 10000,
            },
            "l2_postgres": {
                "enabled": True,
                "ttl": 86400,  # 24 hours
            },
            "l3_semantic": {
                "enabled": True,
                "ttl": 604800,  # 7 days
                "similarity_threshold": 0.95,
            },
        },
        description="Configuration for each cache layer",
    )
    
    cache_targets: dict[str, Any] = Field(
        default={
            "embeddings": {
                "enabled": True,
                "ttl": 86400,  # 24 hours
            },
            "query_results": {
                "enabled": True,
                "ttl": 3600,  # 1 hour
                "min_query_length": 10,
            },
            "llm_responses": {
                "enabled": True,
                "ttl": 7200,  # 2 hours
                "cacheable_patterns": ["what is", "how to", "explain"],
            },
            "reranking_scores": {
                "enabled": True,
                "ttl": 1800,  # 30 minutes
            },
        },
        description="What types of data to cache",
    )
    
    invalidation: dict[str, Any] = Field(
        default={
            "strategies": ["ttl_based", "manual_flush", "document_update"],
            "flush_on_document_update": True,
        },
        description="Cache invalidation strategies",
    )
    
    monitoring: dict[str, Any] = Field(
        default={
            "track_hit_rates": True,
            "alert_on_low_hit_rate": {
                "threshold": 0.5,
                "window": "1h",
            },
        },
        description="Monitoring configuration",
    )
    
    model_config = SettingsConfigDict(env_prefix="CACHING_")


class EvaluationSettings(BaseSettings):
    """RAG Evaluation Framework settings.
    
    This settings class configures the evaluation framework for measuring
    retrieval and generation quality in RAG systems.
    
    Attributes:
        enabled: Master switch for evaluation
        auto_evaluate_enabled: Whether to automatically evaluate queries
        sample_rate: Rate at which to sample queries for evaluation (0-1)
        metrics: Which metrics to compute
        benchmarks: Benchmark dataset configurations
        alerting_enabled: Whether to enable alerting
        thresholds: Metric thresholds for alerting
        alert_channels: Channels for sending alerts
    """
    
    enabled: bool = Field(
        default=True,
        description="Enable RAG evaluation framework",
    )
    
    auto_evaluate_enabled: bool = Field(
        default=True,
        description="Automatically evaluate a sample of queries",
    )
    
    sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Rate at which to sample queries for evaluation (0-1)",
    )
    
    metrics: dict[str, list[str]] = Field(
        default={
            "retrieval": ["mrr", "ndcg@10", "recall@5", "precision@5", "hit_rate@10"],
            "generation": ["bertscore", "faithfulness", "answer_relevance"],
        },
        description="Metrics to compute by category",
    )
    
    benchmarks: list[dict[str, Any]] = Field(
        default=[
            {
                "name": "ms_marco",
                "dataset": "microsoft/ms_marco",
                "max_queries": 1000,
            },
        ],
        description="Benchmark dataset configurations",
    )
    
    alerting_enabled: bool = Field(
        default=True,
        description="Enable alerting on threshold violations",
    )
    
    thresholds: dict[str, float] = Field(
        default={
            "mrr_min": 0.70,
            "ndcg_at_10_min": 0.75,
            "recall_at_5_min": 0.60,
            "precision_at_5_min": 0.40,
            "bertscore_f1_min": 0.85,
            "faithfulness_min": 0.80,
            "answer_relevance_min": 0.80,
            "latency_p95_max": 1000.0,
            "mrr_drop_threshold": 0.05,
        },
        description="Metric thresholds for alerting",
    )
    
    alert_channels: list[str] = Field(
        default=["log"],
        description="Channels for alerts (log, slack, email)",
    )
    
    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: float) -> float:
        """Validate sample rate."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        return v
    
    @field_validator("alert_channels")
    @classmethod
    def validate_alert_channels(cls, v: list[str]) -> list[str]:
        """Validate alert channels."""
        valid_channels = {"log", "slack", "email", "webhook"}
        for channel in v:
            if channel not in valid_channels:
                raise ValueError(
                    f"Invalid alert channel: {channel}. "
                    f"Must be one of: {valid_channels}"
                )
        return v
    
    model_config = SettingsConfigDict(env_prefix="EVALUATION_")


class LLMYamlConfig(BaseSettings):
    """LLM configuration loaded from YAML file."""

    config_path: Path = Field(default=Path("config/llm.yaml"))
    _config_cache: dict[str, Any] | None = None

    def load_yaml(self) -> dict[str, Any]:
        """Load LLM configuration from YAML file.
        
        Returns:
            Dictionary containing LLM configuration
        """
        if self._config_cache is not None:
            return self._config_cache

        if not self.config_path.exists():
            return self._default_config()

        with open(self.config_path) as f:
            self._config_cache = yaml.safe_load(f)
        return self._config_cache or self._default_config()

    def _default_config(self) -> dict[str, Any]:
        """Return default LLM configuration."""
        return {
            "llm": {
                "router": [
                    {
                        "model_name": "agentic-decisions",
                        "litellm_params": {
                            "model": "azure/gpt-4",
                            "api_base": "${AZURE_OPENAI_API_BASE}",
                            "api_key": "${AZURE_OPENAI_API_KEY}",
                            "api_version": "2024-02-01",
                            "tpm": 10000,
                            "rpm": 60,
                        },
                        "fallback_models": [
                            {
                                "model": "openrouter/anthropic/claude-3-opus",
                                "api_key": "${OPENROUTER_API_KEY}",
                                "api_base": "https://openrouter.ai/api/v1",
                                "tpm": 5000,
                            }
                        ],
                    }
                ],
                "proxy": {"host": "0.0.0.0", "port": 4000},
                "retry": {"num_retries": 3, "timeout": 30, "backoff_factor": 2},
                "defaults": {"temperature": 0.3, "max_tokens": 2000},
            }
        }

    model_config = SettingsConfigDict(env_prefix="LLM_")


class Settings(BaseSettings):
    """Main application settings."""

    # Application
    app_name: str = Field(default="Agentic Data Pipeline Ingestor")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    env: str = Field(default="development")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    azure: AzureSettings = Field(default_factory=AzureSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    llm_yaml: LLMYamlConfig = Field(default_factory=LLMYamlConfig)
    query_rewriting: QueryRewritingSettings = Field(default_factory=QueryRewritingSettings)
    hyde: HyDESettings = Field(default_factory=HyDESettings)
    reranking: ReRankingSettings = Field(default_factory=ReRankingSettings)
    classification: ClassificationSettings = Field(default_factory=ClassificationSettings)
    hybrid_search: HybridSearchSettings = Field(default_factory=HybridSearchSettings)
    agentic_rag: AgenticRAGSettings = Field(default_factory=AgenticRAGSettings)
    contextual_retrieval: ContextualRetrievalSettings = Field(default_factory=ContextualRetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding_optimization: EmbeddingOptimizationSettings = Field(default_factory=EmbeddingOptimizationSettings)
    caching: CachingSettings = Field(default_factory=CachingSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.
    
    Returns:
        Settings instance with loaded configuration
    """
    return Settings()


def reload_settings() -> Settings:
    """Reload settings from environment.
    
    Returns:
        Fresh Settings instance
    """
    get_settings.cache_clear()
    return get_settings()


# Global settings instance for convenient access
settings = get_settings()
