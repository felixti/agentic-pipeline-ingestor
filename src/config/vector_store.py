"""Vector store configuration management for pgvector integration.

This module provides configuration structures for the vector storage and search
system, including embedding settings, HNSW index parameters, and search defaults.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for text embedding generation.
    
    Attributes:
        model: Model identifier for embeddings (litellm format)
        dimensions: Expected embedding dimensions (must match model output)
        batch_size: Maximum number of texts per embedding batch
        max_tokens: Maximum tokens per chunk for truncation
        provider_params: Provider-specific parameters (timeout, retries, etc.)
    """
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    max_tokens: int = 8192
    provider_params: dict[str, Any] = field(default_factory=lambda: {
        "timeout": 30,
        "retries": 3,
        "backoff_factor": 2.0,
    })

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate model dimensions compatibility
        model_dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-large": 3072,
        }
        
        # Check if known model has correct dimensions
        for model_name, expected_dims in model_dimension_map.items():
            if model_name in self.model and self.dimensions != expected_dims:
                logger.warning(
                    f"Model {self.model} typically has {expected_dims} dimensions, "
                    f"but config specifies {self.dimensions}"
                )
                break


@dataclass
class SearchConfig:
    """Configuration for vector search operations.
    
    Attributes:
        default_top_k: Default number of results to return
        max_top_k: Maximum allowed top_k to prevent abuse
        default_min_similarity: Default minimum similarity threshold (0-1)
        query_timeout_ms: Query timeout in milliseconds
    """
    default_top_k: int = 10
    max_top_k: int = 100
    default_min_similarity: float = 0.7
    query_timeout_ms: int = 5000


@dataclass
class HNSWIndexConfig:
    """Configuration for HNSW (Hierarchical Navigable Small World) index.
    
    These parameters affect the quality, build time, and search performance
    of the approximate nearest neighbor index used by pgvector.
    
    Attributes:
        hnsw_m: Number of bi-directional links per node (higher = better recall)
        hnsw_ef_construction: Dynamic candidate list size during build
        hnsw_ef_search: Dynamic candidate list size during search
        distance_metric: Distance metric for indexing (cosine, l2, inner_product)
    """
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 32
    distance_metric: str = "cosine"

    def __post_init__(self) -> None:
        """Validate HNSW parameters."""
        valid_metrics = ["cosine", "l2", "inner_product"]
        if self.distance_metric not in valid_metrics:
            raise ValueError(
                f"Invalid distance_metric: {self.distance_metric}. "
                f"Must be one of: {valid_metrics}"
            )


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid (vector + text) search.
    
    Attributes:
        default_vector_weight: Default weight for vector search scores (0-1)
        default_text_weight: Default weight for text search scores (0-1)
        rrf_k: RRF (Reciprocal Rank Fusion) constant
        fallback_strategy: Strategy when one search type fails
    """
    default_vector_weight: float = 0.7
    default_text_weight: float = 0.3
    rrf_k: int = 60
    fallback_strategy: str = "text_only"

    def __post_init__(self) -> None:
        """Validate hybrid search parameters."""
        valid_strategies = ["vector_only", "text_only", "error"]
        if self.fallback_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid fallback_strategy: {self.fallback_strategy}. "
                f"Must be one of: {valid_strategies}"
            )
        
        # Validate weights sum to approximately 1.0
        total_weight = self.default_vector_weight + self.default_text_weight
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(
                f"Vector weight ({self.default_vector_weight}) + text weight "
                f"({self.default_text_weight}) = {total_weight}, should equal 1.0"
            )


@dataclass
class CacheConfig:
    """Configuration for embedding result caching.
    
    Attributes:
        enabled: Whether caching is enabled
        provider: Cache provider (memory, redis)
        ttl_seconds: Time-to-live for cached entries
        max_size: Maximum cache size (for memory provider)
    """
    enabled: bool = True
    provider: str = "memory"
    ttl_seconds: int = 3600
    max_size: int = 10000

    def __post_init__(self) -> None:
        """Validate cache configuration."""
        valid_providers = ["memory", "redis"]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {self.provider}. "
                f"Must be one of: {valid_providers}"
            )


@dataclass
class PipelineConfig:
    """Configuration for pipeline integration.
    
    Attributes:
        auto_generate_embeddings: Auto-generate embeddings during pipeline
        chunking_strategy: Strategy for text chunking
        chunk_size: Chunk size in tokens/characters
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size (smaller chunks filtered)
        store_chunks: Store chunks in database during processing
    """
    auto_generate_embeddings: bool = True
    chunking_strategy: str = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    store_chunks: bool = True

    def __post_init__(self) -> None:
        """Validate pipeline configuration."""
        valid_strategies = ["fixed", "semantic", "hierarchical"]
        if self.chunking_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid chunking_strategy: {self.chunking_strategy}. "
                f"Must be one of: {valid_strategies}"
            )


@dataclass
class VectorStoreConfig:
    """Complete vector store configuration.
    
    This is the main configuration class that aggregates all vector store
    settings including embedding, search, index, and pipeline integration.
    
    Attributes:
        enabled: Whether vector store functionality is enabled
        embedding: Embedding generation configuration
        search: Search operation configuration
        index: HNSW index configuration
        hybrid: Hybrid search configuration
        cache: Embedding cache configuration
        pipeline: Pipeline integration configuration
    """
    enabled: bool = True
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    index: HNSWIndexConfig = field(default_factory=HNSWIndexConfig)
    hybrid: HybridSearchConfig = field(default_factory=HybridSearchConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "VectorStoreConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Loaded VectorStoreConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls._apply_env_overrides(cls())

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise ValueError(f"Invalid YAML config: {e}") from e

        if not data or "vector_store" not in data:
            logger.warning("Invalid config format, using defaults")
            return cls._apply_env_overrides(cls())

        vs_data = data["vector_store"]

        # Parse embedding config
        embedding_data = vs_data.get("embedding", {})
        embedding = EmbeddingConfig(
            model=embedding_data.get("model", "text-embedding-3-small"),
            dimensions=embedding_data.get("dimensions", 1536),
            batch_size=embedding_data.get("batch_size", 100),
            max_tokens=embedding_data.get("max_tokens", 8192),
            provider_params=embedding_data.get("provider_params", {
                "timeout": 30,
                "retries": 3,
                "backoff_factor": 2.0,
            }),
        )

        # Parse search config
        search_data = vs_data.get("search", {})
        search = SearchConfig(
            default_top_k=search_data.get("default_top_k", 10),
            max_top_k=search_data.get("max_top_k", 100),
            default_min_similarity=search_data.get("default_min_similarity", 0.7),
            query_timeout_ms=search_data.get("query_timeout_ms", 5000),
        )

        # Parse index config
        index_data = vs_data.get("index", {})
        index = HNSWIndexConfig(
            hnsw_m=index_data.get("hnsw_m", 16),
            hnsw_ef_construction=index_data.get("hnsw_ef_construction", 64),
            hnsw_ef_search=index_data.get("hnsw_ef_search", 32),
            distance_metric=index_data.get("distance_metric", "cosine"),
        )

        # Parse hybrid config
        hybrid_data = vs_data.get("hybrid", {})
        hybrid = HybridSearchConfig(
            default_vector_weight=hybrid_data.get("default_vector_weight", 0.7),
            default_text_weight=hybrid_data.get("default_text_weight", 0.3),
            rrf_k=hybrid_data.get("rrf_k", 60),
            fallback_strategy=hybrid_data.get("fallback_strategy", "text_only"),
        )

        # Parse cache config
        cache_data = vs_data.get("cache", {})
        cache = CacheConfig(
            enabled=cache_data.get("enabled", True),
            provider=cache_data.get("provider", "memory"),
            ttl_seconds=cache_data.get("ttl_seconds", 3600),
            max_size=cache_data.get("max_size", 10000),
        )

        # Parse pipeline config
        pipeline_data = vs_data.get("pipeline", {})
        pipeline = PipelineConfig(
            auto_generate_embeddings=pipeline_data.get("auto_generate_embeddings", True),
            chunking_strategy=pipeline_data.get("chunking_strategy", "semantic"),
            chunk_size=pipeline_data.get("chunk_size", 1000),
            chunk_overlap=pipeline_data.get("chunk_overlap", 200),
            min_chunk_size=pipeline_data.get("min_chunk_size", 100),
            store_chunks=pipeline_data.get("store_chunks", True),
        )

        config = cls(
            enabled=vs_data.get("enabled", True),
            embedding=embedding,
            search=search,
            index=index,
            hybrid=hybrid,
            cache=cache,
            pipeline=pipeline,
        )

        return cls._apply_env_overrides(config)

    @classmethod
    def _apply_env_overrides(cls, config: "VectorStoreConfig") -> "VectorStoreConfig":
        """Apply environment variable overrides to configuration.
        
        Args:
            config: Configuration to modify
            
        Returns:
            Modified configuration with env overrides applied
        """
        # Override enabled status
        if "VECTOR_STORE_ENABLED" in os.environ:
            config.enabled = os.environ["VECTOR_STORE_ENABLED"].lower() in ("true", "1", "yes")
        
        # Override embedding model
        if "EMBEDDING_MODEL" in os.environ:
            config.embedding.model = os.environ["EMBEDDING_MODEL"]
        
        # Override embedding dimensions
        if "EMBEDDING_DIMENSIONS" in os.environ:
            try:
                config.embedding.dimensions = int(os.environ["EMBEDDING_DIMENSIONS"])
            except ValueError:
                logger.warning("Invalid EMBEDDING_DIMENSIONS value, using default")
        
        # Override API settings in provider_params
        if "EMBEDDING_API_KEY" in os.environ:
            config.embedding.provider_params["api_key"] = os.environ["EMBEDDING_API_KEY"]
        
        if "EMBEDDING_API_BASE" in os.environ:
            config.embedding.provider_params["api_base"] = os.environ["EMBEDDING_API_BASE"]
        
        return config

    def to_litellm_params(self) -> dict[str, Any]:
        """Convert embedding config to litellm-compatible parameters.
        
        Returns:
            Dictionary of parameters for litellm embedding calls
        """
        params: dict[str, Any] = {
            "model": self.embedding.model,
            "dimensions": self.embedding.dimensions,
        }
        
        # Add provider params
        params.update(self.embedding.provider_params)
        
        return params

    def get_index_sql(self, table_name: str = "document_chunks") -> str:
        """Generate SQL for creating HNSW index based on configuration.
        
        Args:
            table_name: Name of the table to index
            
        Returns:
            SQL statement for creating HNSW index
        """
        metric_ops = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "inner_product": "vector_ip_ops",
        }
        
        op_class = metric_ops.get(self.index.distance_metric, "vector_cosine_ops")
        
        return f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding_hnsw 
        ON {table_name} 
        USING hnsw (embedding {op_class})
        WITH (m = {self.index.hnsw_m}, ef_construction = {self.index.hnsw_ef_construction});
        """


def load_vector_store_config(config_path: str | Path | None = None) -> VectorStoreConfig:
    """Load vector store configuration from file or return defaults.
    
    Args:
        config_path: Path to configuration file. If None, uses
                     default path "config/vector_store.yaml"
    
    Returns:
        Loaded VectorStoreConfig
    """
    if config_path is None:
        config_path = "config/vector_store.yaml"
    
    return VectorStoreConfig.from_yaml(config_path)


# Global config instance for convenient access
_vector_store_config: VectorStoreConfig | None = None


def get_vector_store_config() -> VectorStoreConfig:
    """Get cached vector store configuration.
    
    Returns:
        VectorStoreConfig instance (cached)
    """
    global _vector_store_config
    if _vector_store_config is None:
        _vector_store_config = load_vector_store_config()
    return _vector_store_config


def reload_vector_store_config() -> VectorStoreConfig:
    """Reload vector store configuration from file.
    
    Returns:
        Fresh VectorStoreConfig instance
    """
    global _vector_store_config
    _vector_store_config = load_vector_store_config()
    return _vector_store_config
