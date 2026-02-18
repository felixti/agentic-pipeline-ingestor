"""Validators for chunk-related API requests.

This module provides Pydantic validators and sanitization functions
for document chunk API endpoints, including UUID validation,
content sanitization, and input length limits.
"""

import html
import re
from typing import Any
from uuid import UUID

from pydantic import field_validator

__all__ = [
    "ChunkValidatorsMixin",
    "PaginationValidatorsMixin",
    "html_escape",
    "sanitize_content",
    "sanitize_query",
    "sanitize_uuid",
    "strip_html_tags",
    "validate_chunk_id",
    "validate_content_field",
    "validate_job_id",
    "validate_limit",
    "validate_offset",
    "validate_uuid",
]


# ============================================================================
# Constants
# ============================================================================

MAX_QUERY_LENGTH = 1024
MAX_CONTENT_LENGTH = 100_000  # 100KB max content size
ALLOWED_HTML_TAGS: set[str] = set()  # No HTML tags allowed by default

# Regex patterns for validation
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Pattern to detect script tags and event handlers
SCRIPT_PATTERN = re.compile(
    r"<script[^>]*>.*?</script>|"
    r"<[^>]+\s+(?:on\w+|javascript:)\s*=",
    re.IGNORECASE | re.DOTALL,
)

# Pattern to strip all HTML tags
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


# ============================================================================
# UUID Validation
# ============================================================================

def validate_uuid(uuid_str: str, field_name: str = "uuid") -> UUID:
    """Validate and parse a UUID string.
    
    Args:
        uuid_str: UUID string to validate
        field_name: Name of the field for error messages
        
    Returns:
        Parsed UUID object
        
    Raises:
        ValueError: If UUID format is invalid
    """
    if not uuid_str or not isinstance(uuid_str, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    
    # Check pattern first for better error message
    if not UUID_PATTERN.match(uuid_str.strip()):
        raise ValueError(
            f"{field_name} must be a valid UUID (e.g., '123e4567-e89b-12d3-a456-426614174000')"
        )
    
    try:
        return UUID(uuid_str.strip())
    except ValueError as e:
        raise ValueError(f"Invalid {field_name}: {e}") from e


def sanitize_uuid(uuid_str: str | None, field_name: str = "uuid") -> str | None:
    """Sanitize a UUID string by stripping whitespace.
    
    Args:
        uuid_str: UUID string to sanitize
        field_name: Name of the field for error messages
        
    Returns:
        Sanitized UUID string or None if input is None
        
    Raises:
        ValueError: If UUID format is invalid
    """
    if uuid_str is None:
        return None
    
    if not isinstance(uuid_str, str):
        raise ValueError(f"{field_name} must be a string")
    
    sanitized = uuid_str.strip()
    
    if not sanitized:
        raise ValueError(f"{field_name} cannot be empty")
    
    return sanitized


# ============================================================================
# Content Sanitization
# ============================================================================

def strip_html_tags(content: str) -> str:
    """Remove all HTML tags from content.
    
    Args:
        content: Content that may contain HTML tags
        
    Returns:
        Content with HTML tags removed
    """
    if not content:
        return content
    
    # Remove script tags and event handlers first
    content = SCRIPT_PATTERN.sub("", content)
    
    # Remove all remaining HTML tags
    content = HTML_TAG_PATTERN.sub("", content)
    
    return content


def html_escape(content: str) -> str:
    """Escape HTML special characters in content.
    
    Args:
        content: Content to escape
        
    Returns:
        Content with HTML special characters escaped
    """
    if not content:
        return content
    
    return html.escape(content)


def sanitize_content(content: str | None, max_length: int = MAX_CONTENT_LENGTH) -> str | None:
    """Sanitize content by stripping HTML and escaping special characters.
    
    This function:
    1. Strips all HTML tags
    2. Escapes remaining HTML special characters
    3. Truncates to max_length
    4. Normalizes whitespace
    
    Args:
        content: Content to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized content or None if input is None
    """
    if content is None:
        return None
    
    if not isinstance(content, str):
        raise ValueError("Content must be a string")
    
    # Strip HTML tags
    content = strip_html_tags(content)
    
    # Escape HTML special characters
    content = html_escape(content)
    
    # Normalize whitespace
    content = " ".join(content.split())
    
    # Truncate if too long
    if len(content) > max_length:
        content = content[:max_length].rsplit(" ", 1)[0] + "..."
    
    return content


def sanitize_query(query: str | None, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Sanitize a search query.
    
    This function:
    1. Validates query is not empty
    2. Strips HTML tags
    3. Escapes special characters
    4. Validates length
    5. Normalizes whitespace
    
    Args:
        query: Search query to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized query string
        
    Raises:
        ValueError: If query is invalid
    """
    if query is None:
        raise ValueError("Query cannot be None")
    
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    
    # Strip and normalize
    query = query.strip()
    
    if not query:
        raise ValueError("Query cannot be empty")
    
    # Strip HTML tags
    query = strip_html_tags(query)
    
    # Escape HTML special characters
    query = html_escape(query)
    
    # Check length after sanitization
    if len(query) > max_length:
        raise ValueError(f"Query exceeds maximum length of {max_length} characters")
    
    return query


# ============================================================================
# Pydantic Field Validators
# ============================================================================

def validate_job_id(v: str | UUID | None) -> UUID | None:
    """Validate job_id field.
    
    Args:
        v: Job ID value (string, UUID, or None)
        
    Returns:
        Validated UUID or None
        
    Raises:
        ValueError: If job_id format is invalid
    """
    if v is None:
        return None
    
    if isinstance(v, UUID):
        return v
    
    return validate_uuid(v, "job_id")


def validate_chunk_id(v: str | UUID | None) -> UUID | None:
    """Validate chunk_id field.
    
    Args:
        v: Chunk ID value (string, UUID, or None)
        
    Returns:
        Validated UUID or None
        
    Raises:
        ValueError: If chunk_id format is invalid
    """
    if v is None:
        return None
    
    if isinstance(v, UUID):
        return v
    
    return validate_uuid(v, "chunk_id")


def validate_content_field(v: str | None, max_length: int = MAX_CONTENT_LENGTH) -> str | None:
    """Validate and sanitize content field.
    
    Args:
        v: Content value
        max_length: Maximum allowed length
        
    Returns:
        Sanitized content or None
        
    Raises:
        ValueError: If content is invalid
    """
    if v is None:
        return None
    
    if not isinstance(v, str):
        raise ValueError("Content must be a string")
    
    if len(v) > max_length:
        raise ValueError(f"Content exceeds maximum length of {max_length} characters")
    
    return sanitize_content(v, max_length)


def validate_limit(v: int | None, default: int = 100, max_value: int = 1000) -> int:
    """Validate limit parameter for pagination.
    
    Args:
        v: Limit value
        default: Default value if None
        max_value: Maximum allowed value
        
    Returns:
        Validated limit
        
    Raises:
        ValueError: If limit is invalid
    """
    if v is None:
        return default
    
    if not isinstance(v, int):
        raise ValueError("Limit must be an integer")
    
    if v < 1:
        raise ValueError("Limit must be at least 1")
    
    if v > max_value:
        raise ValueError(f"Limit cannot exceed {max_value}")
    
    return v


def validate_offset(v: int | None, default: int = 0) -> int:
    """Validate offset parameter for pagination.
    
    Args:
        v: Offset value
        default: Default value if None
        
    Returns:
        Validated offset
        
    Raises:
        ValueError: If offset is invalid
    """
    if v is None:
        return default
    
    if not isinstance(v, int):
        raise ValueError("Offset must be an integer")
    
    if v < 0:
        raise ValueError("Offset cannot be negative")
    
    return v


# ============================================================================
# Model Validators (for use with Pydantic models)
# ============================================================================

class ChunkValidatorsMixin:
    """Mixin class providing Pydantic validators for chunk-related models."""
    
    @field_validator("job_id", mode="before")
    @classmethod
    def validate_job_id_field(cls, v: Any) -> Any:
        """Validate job_id field in Pydantic models."""
        if v is None:
            return v
        
        if isinstance(v, UUID):
            return v
        
        # Validate as string UUID
        return validate_uuid(v, "job_id")
    
    @field_validator("chunk_id", mode="before")
    @classmethod
    def validate_chunk_id_field(cls, v: Any) -> Any:
        """Validate chunk_id field in Pydantic models."""
        if v is None:
            return v
        
        if isinstance(v, UUID):
            return v
        
        # Validate as string UUID
        return validate_uuid(v, "chunk_id")
    
    @field_validator("content", mode="before")
    @classmethod
    def validate_content_field(cls, v: Any) -> Any:
        """Validate and sanitize content field in Pydantic models."""
        if v is None:
            return v
        
        if not isinstance(v, str):
            raise ValueError("Content must be a string")
        
        # Strip HTML tags
        v = strip_html_tags(v)
        
        # Escape HTML special characters
        v = html_escape(v)
        
        return v


class PaginationValidatorsMixin:
    """Mixin class providing Pydantic validators for pagination parameters."""
    
    @field_validator("limit", mode="before")
    @classmethod
    def validate_limit_field(cls, v: Any) -> Any:
        """Validate limit field."""
        if v is None:
            return 100
        
        if not isinstance(v, int):
            raise ValueError("Limit must be an integer")
        
        if v < 1:
            raise ValueError("Limit must be at least 1")
        
        if v > 1000:
            raise ValueError("Limit cannot exceed 1000")
        
        return v
    
    @field_validator("offset", mode="before")
    @classmethod
    def validate_offset_field(cls, v: Any) -> Any:
        """Validate offset field."""
        if v is None:
            return 0
        
        if not isinstance(v, int):
            raise ValueError("Offset must be an integer")
        
        if v < 0:
            raise ValueError("Offset cannot be negative")
        
        return v
