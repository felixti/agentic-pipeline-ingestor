"""API middleware for cross-cutting concerns.

This package provides middleware components for the API layer,
including rate limiting, authentication, logging, and error handling.
"""

from src.api.middleware.rate_limiter import (
    DEFAULT_RATE_LIMITS,
    RateLimitConfig,
    RateLimitHeadersMiddleware,
    RateLimitTier,
    RedisRateLimiter,
    rate_limit,
    rate_limit_exception_handler,
    rate_limit_get_chunk,
    rate_limit_hybrid_search,
    rate_limit_list_chunks,
    rate_limit_semantic_search,
    rate_limit_similar_chunks,
    rate_limit_text_search,
)

__all__ = [
    "DEFAULT_RATE_LIMITS",
    "RateLimitConfig",
    "RateLimitHeadersMiddleware",
    # Rate limiting
    "RateLimitTier",
    "RedisRateLimiter",
    "rate_limit",
    "rate_limit_exception_handler",
    "rate_limit_get_chunk",
    "rate_limit_hybrid_search",
    "rate_limit_list_chunks",
    "rate_limit_semantic_search",
    "rate_limit_similar_chunks",
    "rate_limit_text_search",
]
