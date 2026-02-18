"""Validators for search-related API requests.

This module provides Pydantic validators and validation functions
for search endpoints, including embedding dimension validation,
query length limits, weight validation, and language code validation.
"""

from typing import Any

from pydantic import field_validator, model_validator

__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "MAX_QUERY_LENGTH",
    "SUPPORTED_DIMENSIONS",
    "VALID_FUSION_METHODS",
    "VALID_LANGUAGES",
    "HybridSearchValidatorsMixin",
    "SearchValidatorsMixin",
    "SemanticSearchValidatorsMixin",
    "TextSearchValidatorsMixin",
    "normalize_language",
    "sanitize_search_query",
    "validate_embedding",
    "validate_embedding_dimension",
    "validate_embedding_values",
    "validate_fusion_method",
    "validate_language",
    "validate_query_length",
    "validate_similarity_threshold",
    "validate_top_k",
    "validate_weight_range",
    "validate_weights",
]


# ============================================================================
# Constants
# ============================================================================

# Default embedding dimension (OpenAI text-embedding-3-small)
DEFAULT_EMBEDDING_DIMENSION = 1536

# Supported embedding dimensions
SUPPORTED_DIMENSIONS = {384, 512, 768, 1024, 1536, 2048, 3072}

# Maximum query length
MAX_QUERY_LENGTH = 1024

# Minimum query length
MIN_QUERY_LENGTH = 1

# Valid PostgreSQL text search languages
VALID_LANGUAGES = {
    "simple",      # Simple catalog (no stemming)
    "arabic",
    "armenian",
    "basque",
    "catalan",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "greek",
    "hindi",
    "hungarian",
    "indonesian",
    "irish",
    "italian",
    "lithuanian",
    "nepali",
    "norwegian",
    "portuguese",
    "romanian",
    "russian",
    "serbian",
    "spanish",
    "swedish",
    "tamil",
    "turkish",
    "yiddish",
}

# Default fusion methods
VALID_FUSION_METHODS = {"weighted_sum", "rrf"}

# Weight tolerance for sum validation
WEIGHT_TOLERANCE = 0.01


# ============================================================================
# Embedding Validation
# ============================================================================

def validate_embedding_dimension(
    embedding: list[float],
    expected_dim: int = DEFAULT_EMBEDDING_DIMENSION,
) -> None:
    """Validate embedding vector dimension.
    
    Args:
        embedding: Embedding vector to validate
        expected_dim: Expected dimension size
        
    Raises:
        ValueError: If dimension doesn't match or is invalid
    """
    if not isinstance(embedding, list):
        raise ValueError("Embedding must be a list of floats")
    
    if len(embedding) == 0:
        raise ValueError("Embedding vector cannot be empty")
    
    actual_dim = len(embedding)
    
    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}. "
            f"Supported dimensions: {sorted(SUPPORTED_DIMENSIONS)}"
        )


def validate_embedding_values(embedding: list[float]) -> None:
    """Validate embedding vector values.
    
    Args:
        embedding: Embedding vector to validate
        
    Raises:
        ValueError: If values are invalid
    """
    if not isinstance(embedding, list):
        raise ValueError("Embedding must be a list of floats")
    
    for i, value in enumerate(embedding):
        if not isinstance(value, (int, float)):
            raise ValueError(f"Embedding value at index {i} is not a number")
        
        # Check for NaN or Inf
        import math
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Embedding value at index {i} is NaN or infinite")


def validate_embedding(
    embedding: list[float],
    expected_dim: int = DEFAULT_EMBEDDING_DIMENSION,
) -> list[float]:
    """Validate embedding vector dimension and values.
    
    Args:
        embedding: Embedding vector to validate
        expected_dim: Expected dimension size
        
    Returns:
        Validated embedding vector
        
    Raises:
        ValueError: If embedding is invalid
    """
    validate_embedding_values(embedding)
    validate_embedding_dimension(embedding, expected_dim)
    return embedding


# ============================================================================
# Query Validation
# ============================================================================

def validate_query_length(
    query: str,
    min_length: int = MIN_QUERY_LENGTH,
    max_length: int = MAX_QUERY_LENGTH,
) -> str:
    """Validate search query length.
    
    Args:
        query: Search query to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        Validated query string
        
    Raises:
        ValueError: If query length is invalid
    """
    if query is None:
        raise ValueError("Query cannot be None")
    
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    
    query_len = len(query.strip())
    
    if query_len < min_length:
        raise ValueError(f"Query must be at least {min_length} character(s)")
    
    if query_len > max_length:
        raise ValueError(
            f"Query exceeds maximum length of {max_length} characters "
            f"(got {query_len})"
        )
    
    return query.strip()


def sanitize_search_query(query: str) -> str:
    """Sanitize search query by removing control characters.
    
    Args:
        query: Raw search query
        
    Returns:
        Sanitized query string
    """
    if not query:
        return query
    
    # Remove null bytes and other control characters
    sanitized = "".join(char for char in query if char.isprintable() or char in "\t\n\r")
    
    # Normalize whitespace
    sanitized = " ".join(sanitized.split())
    
    return sanitized


# ============================================================================
# Weight Validation
# ============================================================================

def validate_weights(
    vector_weight: float,
    text_weight: float,
    tolerance: float = WEIGHT_TOLERANCE,
) -> None:
    """Validate that weights sum to approximately 1.0.
    
    Args:
        vector_weight: Weight for vector search component
        text_weight: Weight for text search component
        tolerance: Allowed deviation from 1.0
        
    Raises:
        ValueError: If weights don't sum to ~1.0
    """
    total = vector_weight + text_weight
    
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"Weights must sum to 1.0 (within ±{tolerance}), "
            f"got vector_weight={vector_weight}, text_weight={text_weight}, "
            f"sum={total:.4f}"
        )


def validate_weight_range(weight: float, name: str = "weight") -> float:
    """Validate that a weight is within valid range [0, 1].
    
    Args:
        weight: Weight value to validate
        name: Name of the weight for error messages
        
    Returns:
        Validated weight
        
    Raises:
        ValueError: If weight is out of range
    """
    if not isinstance(weight, (int, float)):
        raise ValueError(f"{name} must be a number")
    
    if weight < 0.0 or weight > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {weight}")
    
    return float(weight)


# ============================================================================
# Language Validation
# ============================================================================

def validate_language(language: str) -> str:
    """Validate PostgreSQL text search language.
    
    Args:
        language: Language code to validate
        
    Returns:
        Validated language code
        
    Raises:
        ValueError: If language is not supported
    """
    if not language:
        raise ValueError("Language cannot be empty")
    
    if not isinstance(language, str):
        raise ValueError("Language must be a string")
    
    lang_lower = language.lower().strip()
    
    if lang_lower not in VALID_LANGUAGES:
        valid_langs = ", ".join(sorted(VALID_LANGUAGES))
        raise ValueError(
            f"Unsupported language: '{language}'. "
            f"Valid languages: {valid_langs}"
        )
    
    return lang_lower


def normalize_language(language: str | None, default: str = "english") -> str:
    """Normalize language code to valid PostgreSQL text search language.
    
    Args:
        language: Language code to normalize
        default: Default language if None or invalid
        
    Returns:
        Normalized language code
    """
    if language is None:
        return default
    
    try:
        return validate_language(language)
    except ValueError:
        return default


# ============================================================================
# Fusion Method Validation
# ============================================================================

def validate_fusion_method(method: str) -> str:
    """Validate fusion method for hybrid search.
    
    Args:
        method: Fusion method to validate
        
    Returns:
        Validated fusion method
        
    Raises:
        ValueError: If method is invalid
    """
    if not method:
        raise ValueError("Fusion method cannot be empty")
    
    if not isinstance(method, str):
        raise ValueError("Fusion method must be a string")
    
    method_lower = method.lower().strip()
    
    if method_lower not in VALID_FUSION_METHODS:
        valid_methods = ", ".join(sorted(VALID_FUSION_METHODS))
        raise ValueError(
            f"Invalid fusion method: '{method}'. "
            f"Valid methods: {valid_methods}"
        )
    
    return method_lower


# ============================================================================
# Top-K Validation
# ============================================================================

def validate_top_k(
    top_k: int,
    min_value: int = 1,
    max_value: int = 100,
) -> int:
    """Validate top_k parameter for search results.
    
    Args:
        top_k: Number of results to return
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated top_k value
        
    Raises:
        ValueError: If top_k is invalid
    """
    if not isinstance(top_k, int):
        raise ValueError("top_k must be an integer")
    
    if top_k < min_value:
        raise ValueError(f"top_k must be at least {min_value}")
    
    if top_k > max_value:
        raise ValueError(f"top_k cannot exceed {max_value}")
    
    return top_k


# ============================================================================
# Similarity Threshold Validation
# ============================================================================

def validate_similarity_threshold(
    threshold: float,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    """Validate similarity threshold for search.
    
    Args:
        threshold: Similarity threshold value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated threshold
        
    Raises:
        ValueError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError("Similarity threshold must be a number")
    
    threshold = float(threshold)
    
    if threshold < min_value or threshold > max_value:
        raise ValueError(
            f"Similarity threshold must be between {min_value} and {max_value}, "
            f"got {threshold}"
        )
    
    return threshold


# ============================================================================
# Pydantic Model Validators
# ============================================================================

class SearchValidatorsMixin:
    """Mixin class providing Pydantic validators for search request models."""
    
    @field_validator("query", mode="before")
    @classmethod
    def validate_query_field(cls, v: Any) -> Any:
        """Validate and sanitize query field."""
        if v is None:
            return v
        
        if not isinstance(v, str):
            raise ValueError("Query must be a string")
        
        # Sanitize
        v = sanitize_search_query(v)
        
        # Validate length
        return validate_query_length(v)
    
    @field_validator("query_embedding", mode="before")
    @classmethod
    def validate_query_embedding_field(cls, v: Any) -> Any:
        """Validate embedding field."""
        if v is None:
            return v
        
        if not isinstance(v, list):
            raise ValueError("Embedding must be a list of floats")
        
        # Validate values
        validate_embedding_values(v)
        
        return v
    
    @field_validator("language", mode="before")
    @classmethod
    def validate_language_field(cls, v: Any) -> Any:
        """Validate language field."""
        if v is None:
            return "english"
        
        return validate_language(v)
    
    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k_field(cls, v: Any) -> Any:
        """Validate top_k field."""
        if v is None:
            return 10
        
        return validate_top_k(v)
    
    @field_validator("fusion_method", mode="before")
    @classmethod
    def validate_fusion_method_field(cls, v: Any) -> Any:
        """Validate fusion_method field."""
        if v is None:
            return "weighted_sum"
        
        return validate_fusion_method(v)
    
    @field_validator("vector_weight", "text_weight", mode="before")
    @classmethod
    def validate_weight_field(cls, v: Any) -> Any:
        """Validate weight fields."""
        if v is None:
            return v
        
        return validate_weight_range(v, "weight")
    
    @model_validator(mode="after")
    def validate_weights_sum(self) -> "SearchValidatorsMixin":
        """Validate that weights sum to 1.0 for hybrid search."""
        # Only validate if both weights are present
        if hasattr(self, "vector_weight") and hasattr(self, "text_weight"):
            vector_weight = getattr(self, "vector_weight", None)
            text_weight = getattr(self, "text_weight", None)
            
            if vector_weight is not None and text_weight is not None:
                validate_weights(vector_weight, text_weight)
        
        return self


class SemanticSearchValidatorsMixin:
    """Mixin class for semantic search request validators."""
    
    @field_validator("query_embedding", mode="before")
    @classmethod
    def validate_embedding_field(cls, v: Any) -> Any:
        """Validate embedding vector."""
        if v is None:
            raise ValueError("query_embedding is required for semantic search")
        
        validate_embedding_values(v)
        return v
    
    @field_validator("min_similarity", mode="before")
    @classmethod
    def validate_min_similarity_field(cls, v: Any) -> Any:
        """Validate min_similarity field."""
        if v is None:
            return 0.7
        
        return validate_similarity_threshold(v)


class TextSearchValidatorsMixin:
    """Mixin class for text search request validators."""
    
    @field_validator("query", mode="before")
    @classmethod
    def validate_text_query_field(cls, v: Any) -> Any:
        """Validate query text."""
        if v is None:
            raise ValueError("query is required for text search")
        
        if not isinstance(v, str):
            raise ValueError("Query must be a string")
        
        # Sanitize
        v = sanitize_search_query(v)
        
        # Validate length
        return validate_query_length(v)
    
    @field_validator("language", mode="before")
    @classmethod
    def validate_text_language_field(cls, v: Any) -> Any:
        """Validate language field."""
        if v is None:
            return "english"
        
        return validate_language(v)


class HybridSearchValidatorsMixin:
    """Mixin class for hybrid search request validators."""
    
    @field_validator("query", mode="before")
    @classmethod
    def validate_hybrid_query_field(cls, v: Any) -> Any:
        """Validate query text."""
        if v is None:
            raise ValueError("query is required for hybrid search")
        
        if not isinstance(v, str):
            raise ValueError("Query must be a string")
        
        # Sanitize
        v = sanitize_search_query(v)
        
        # Validate length
        return validate_query_length(v)
    
    @field_validator("vector_weight", "text_weight", mode="before")
    @classmethod
    def validate_hybrid_weight_field(cls, v: Any, info: Any) -> Any:
        """Validate individual weight fields."""
        if v is None:
            return v
        
        field_name = info.field_name if hasattr(info, "field_name") else "weight"
        return validate_weight_range(v, field_name)
    
    @model_validator(mode="after")
    def validate_hybrid_weights_sum(self) -> "HybridSearchValidatorsMixin":
        """Validate that vector_weight + text_weight ≈ 1.0."""
        vector_weight = getattr(self, "vector_weight", 0.7)
        text_weight = getattr(self, "text_weight", 0.3)
        
        validate_weights(vector_weight, text_weight)
        
        return self
