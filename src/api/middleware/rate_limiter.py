"""Redis-based rate limiting middleware for API endpoints.

This module provides rate limiting functionality using Redis as the backend store.
It supports different rate limits per endpoint type and returns proper 429 responses
with Retry-After headers.
"""

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.observability.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Rate Limit Configuration
# ============================================================================

class RateLimitTier(str, Enum):
    """Rate limit tiers for different endpoint types."""
    
    LIST_CHUNKS = "list_chunks"           # 100/min
    GET_CHUNK = "get_chunk"               # 200/min
    SEMANTIC_SEARCH = "semantic_search"   # 30/min
    TEXT_SEARCH = "text_search"           # 60/min
    HYBRID_SEARCH = "hybrid_search"       # 20/min
    SIMILAR_CHUNKS = "similar_chunks"     # 40/min
    DEFAULT = "default"                   # 100/min


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit tier."""
    
    requests: int
    window: int  # seconds
    
    
# Default rate limits per tier
DEFAULT_RATE_LIMITS: dict[RateLimitTier, RateLimitConfig] = {
    RateLimitTier.LIST_CHUNKS: RateLimitConfig(requests=100, window=60),
    RateLimitTier.GET_CHUNK: RateLimitConfig(requests=200, window=60),
    RateLimitTier.SEMANTIC_SEARCH: RateLimitConfig(requests=30, window=60),
    RateLimitTier.TEXT_SEARCH: RateLimitConfig(requests=60, window=60),
    RateLimitTier.HYBRID_SEARCH: RateLimitConfig(requests=20, window=60),
    RateLimitTier.SIMILAR_CHUNKS: RateLimitConfig(requests=40, window=60),
    RateLimitTier.DEFAULT: RateLimitConfig(requests=100, window=60),
}


# ============================================================================
# Redis Client Helper
# ============================================================================

class RedisRateLimiter:
    """Redis-based rate limiter implementation.
    
    Uses Redis to track request counts within time windows using
    sliding window algorithm with sorted sets.
    """
    
    def __init__(self) -> None:
        """Initialize the rate limiter."""
        self._redis: Any | None = None
        self._initialized = False
    
    async def _get_redis(self) -> Any | None:
        """Get or create Redis connection.
        
        Returns:
            Redis client or None if not available
        """
        if self._initialized:
            return self._redis
        
        try:
            import redis.asyncio as aioredis
            
            settings = get_settings()
            redis_url = str(settings.redis.url)
            
            self._redis = await aioredis.from_url(
                redis_url,
                password=settings.redis.password,
                socket_timeout=settings.redis.socket_timeout,
                socket_connect_timeout=settings.redis.socket_connect_timeout,
                retry_on_timeout=settings.redis.retry_on_timeout,
                decode_responses=True,
            )
            
            # Test connection
            await self._redis.ping()
            self._initialized = True
            
            logger.info("redis_rate_limiter_connected", url=redis_url)
            
        except Exception as e:
            logger.warning("redis_rate_limiter_connection_failed", error=str(e))
            self._redis = None
            self._initialized = True
        
        return self._redis
    
    async def is_allowed(
        self,
        key: str,
        requests: int,
        window: int,
    ) -> tuple[bool, int, int]:
        """Check if a request is allowed under rate limit.
        
        Uses sliding window algorithm with Redis sorted sets.
        
        Args:
            key: Unique identifier for the rate limit bucket
            requests: Maximum number of requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (allowed, remaining_requests, reset_time)
        """
        redis = await self._get_redis()
        
        if redis is None:
            # Redis not available, allow request
            return True, requests - 1, int(time.time()) + window
        
        try:
            now = time.time()
            window_start = now - window
            
            # Use pipeline for atomic operations
            pipe = redis.pipeline()
            
            # Remove old entries outside the window
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current entries in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiry on the key
            pipe.expire(key, window + 1)
            
            results = await pipe.execute()
            current_count = results[1]  # zcard result
            
            # Check if allowed
            allowed = current_count <= requests
            remaining = max(0, requests - current_count)
            reset_time = int(now) + window
            
            if not allowed:
                # Remove the request we just added
                await redis.zrem(key, str(now))
            
            return allowed, remaining, reset_time
            
        except Exception as e:
            logger.error("rate_limit_check_error", error=str(e), key=key)
            # Fail open - allow request if Redis error
            return True, requests - 1, int(time.time()) + window
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._initialized = False
            self._redis = None


# Global rate limiter instance
_rate_limiter = RedisRateLimiter()


async def get_rate_limiter() -> RedisRateLimiter:
    """Get the global rate limiter instance.
    
    Returns:
        RedisRateLimiter instance
    """
    return _rate_limiter


# ============================================================================
# Rate Limit Decorator
# ============================================================================

def rate_limit(
    requests: int | None = None,
    window: int = 60,
    tier: RateLimitTier | None = None,
    key_func: Callable[[Request], str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to apply rate limiting to an endpoint.
    
    Args:
        requests: Maximum number of requests allowed in window
        window: Time window in seconds
        tier: Rate limit tier to use (overrides requests/window)
        key_func: Function to generate rate limit key from request
        
    Returns:
        Decorator function
        
    Example:
        @router.get("/items")
        @rate_limit(requests=100, window=60)
        async def list_items(request: Request):
            return {"items": []}
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Find request object in args or kwargs
            request: Request | None = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request is None:
                request = kwargs.get("request")
            
            # Get rate limit config
            if tier is not None:
                config = DEFAULT_RATE_LIMITS.get(tier, DEFAULT_RATE_LIMITS[RateLimitTier.DEFAULT])
                limit_requests = config.requests
                limit_window = config.window
            else:
                limit_requests = requests or DEFAULT_RATE_LIMITS[RateLimitTier.DEFAULT].requests
                limit_window = window
            
            # Generate rate limit key
            if key_func is not None:
                key = key_func(request) if request else "default"
            else:
                key = _generate_rate_limit_key(request, func.__name__)
            
            # Check rate limit
            allowed, remaining, reset_time = await _rate_limiter.is_allowed(
                key, limit_requests, limit_window
            )
            
            # Log rate limit check
            logger.debug(
                "rate_limit_check",
                key=key,
                allowed=allowed,
                remaining=remaining,
                limit=limit_requests,
                window=limit_window,
            )
            
            if not allowed:
                logger.warning(
                    "rate_limit_exceeded",
                    key=key,
                    limit=limit_requests,
                    window=limit_window,
                )
                raise _create_rate_limit_exception(reset_time)
            
            # Store rate limit info in request state for response headers
            if request is not None:
                request.state.rate_limit_remaining = remaining
                request.state.rate_limit_reset = reset_time
                request.state.rate_limit_limit = limit_requests
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def _generate_rate_limit_key(request: Request | None, endpoint: str) -> str:
    """Generate a rate limit key from request.
    
    Uses client IP + endpoint name as the key.
    
    Args:
        request: FastAPI request object
        endpoint: Endpoint function name
        
    Returns:
        Rate limit key string
    """
    if request is None:
        return f"ratelimit:default:{endpoint}"
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Check for forwarded IP (if behind proxy)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    
    # Create key
    return f"ratelimit:{client_ip}:{endpoint}"


def _create_rate_limit_exception(reset_time: int) -> HTTPException:
    """Create a rate limit exceeded exception.
    
    Args:
        reset_time: Unix timestamp when the rate limit resets
        
    Returns:
        HTTPException with 429 status and Retry-After header
    """
    retry_after = max(1, reset_time - int(time.time()))
    
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "error": "Rate limit exceeded",
            "code": "RATE_LIMIT_EXCEEDED",
            "message": f"Too many requests. Please try again in {retry_after} seconds.",
            "retry_after": retry_after,
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": "0",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(reset_time),
        },
    )


# ============================================================================
# Convenience Decorators for Endpoint Types
# ============================================================================

def rate_limit_list_chunks(func: Callable[..., Any]) -> Callable[..., Any]:
    """Rate limit decorator for list chunks endpoint (100/min)."""
    return rate_limit(tier=RateLimitTier.LIST_CHUNKS)(func)


def rate_limit_get_chunk(func: Callable[..., Any]) -> Callable[..., Any]:
    """Rate limit decorator for get chunk endpoint (200/min)."""
    return rate_limit(tier=RateLimitTier.GET_CHUNK)(func)


def rate_limit_semantic_search(func: Callable[..., Any]) -> Callable[..., Any]:
    """Rate limit decorator for semantic search endpoint (30/min)."""
    return rate_limit(tier=RateLimitTier.SEMANTIC_SEARCH)(func)


def rate_limit_text_search(func: Callable[..., Any]) -> Callable[..., Any]:
    """Rate limit decorator for text search endpoint (60/min)."""
    return rate_limit(tier=RateLimitTier.TEXT_SEARCH)(func)


def rate_limit_hybrid_search(func: Callable[..., Any]) -> Callable[..., Any]:
    """Rate limit decorator for hybrid search endpoint (20/min)."""
    return rate_limit(tier=RateLimitTier.HYBRID_SEARCH)(func)


def rate_limit_similar_chunks(func: Callable[..., Any]) -> Callable[..., Any]:
    """Rate limit decorator for similar chunks endpoint (40/min)."""
    return rate_limit(tier=RateLimitTier.SIMILAR_CHUNKS)(func)


# ============================================================================
# Rate Limit Headers Middleware
# ============================================================================

class RateLimitHeadersMiddleware:
    """Middleware to add rate limit headers to responses.
    
    This middleware adds X-RateLimit-* headers to responses for
    endpoints that use rate limiting.
    """
    
    async def __call__(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        """Process request and add rate limit headers to response.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response with rate limit headers
        """
        response = await call_next(request)
        
        # Check if rate limit info was set by decorator
        if hasattr(request.state, "rate_limit_remaining"):
            response.headers["X-RateLimit-Limit"] = str(
                getattr(request.state, "rate_limit_limit", 0)
            )
            response.headers["X-RateLimit-Remaining"] = str(
                getattr(request.state, "rate_limit_remaining", 0)
            )
            response.headers["X-RateLimit-Reset"] = str(
                getattr(request.state, "rate_limit_reset", 0)
            )
        
        return response


# ============================================================================
# Custom Exception Handler
# ============================================================================

async def rate_limit_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Custom exception handler for rate limit errors.
    
    Args:
        request: FastAPI request
        exc: HTTP exception
        
    Returns:
        JSON response with rate limit details
    """
    if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        headers = dict(exc.headers or {})
        
        # Handle both dict and str detail formats
        detail: dict[str, Any] | str = exc.detail
        if isinstance(detail, dict):
            content: dict[str, Any] = detail
        else:
            content = {"error": detail}
        
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=headers,
        )
    
    # Re-raise for other handlers
    raise exc
