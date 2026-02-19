"""API validators for input validation and sanitization.

This package provides Pydantic validators and validation functions
for API request models, including UUID validation, content sanitization,
and search parameter validation.
"""

from src.api.validators.chunk_validators import (
    ChunkValidatorsMixin,
    PaginationValidatorsMixin,
    html_escape,
    sanitize_content,
    sanitize_query,
    sanitize_uuid,
    strip_html_tags,
    validate_chunk_id,
    validate_content_field,
    validate_job_id,
    validate_limit,
    validate_offset,
    validate_uuid,
)
from src.api.validators.search_validators import (
    HybridSearchValidatorsMixin,
    SemanticSearchValidatorsMixin,
    TextSearchValidatorsMixin,
    normalize_language,
    sanitize_search_query,
    validate_embedding,
    validate_embedding_dimension,
    validate_embedding_values,
    validate_fusion_method,
    validate_language,
    validate_query_length,
    validate_similarity_threshold,
    validate_top_k,
    validate_weight_range,
    validate_weights,
)

__all__ = [
    # Chunk validators
    "ChunkValidatorsMixin",
    "HybridSearchValidatorsMixin",
    "PaginationValidatorsMixin",
    # Search validators
    "SemanticSearchValidatorsMixin",
    "TextSearchValidatorsMixin",
    "html_escape",
    "normalize_language",
    "sanitize_content",
    "sanitize_query",
    "sanitize_search_query",
    "sanitize_uuid",
    "strip_html_tags",
    "validate_chunk_id",
    "validate_content_field",
    "validate_embedding",
    "validate_embedding_dimension",
    "validate_embedding_values",
    "validate_fusion_method",
    "validate_job_id",
    "validate_language",
    "validate_limit",
    "validate_offset",
    "validate_query_length",
    "validate_similarity_threshold",
    "validate_top_k",
    "validate_uuid",
    "validate_weight_range",
    "validate_weights",
]
