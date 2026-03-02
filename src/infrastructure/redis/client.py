"""Redis client management.

This module provides Redis client initialization and management,
supporting both singleton and per-request client patterns.
"""

from __future__ import annotations

import os
from typing import Any

import redis.asyncio as redis

from src.observability.logging import get_logger

logger = get_logger(__name__)

# Global client instance for singleton pattern
_global_client: redis.Redis | None = None


async def get_redis_client(
    url: str | None = None,
    force_new: bool = False,
    **kwargs: Any,
) -> redis.Redis:
    """Get or create a Redis client instance.
    
    This function provides a singleton pattern for the Redis client.
    If a global client exists, it returns that instance.
    Otherwise, it creates a new client.
    
    Args:
        url: Redis connection URL (default: REDIS_URL env var or redis://localhost:6379/0)
        force_new: If True, always create a new client instance
        **kwargs: Additional arguments passed to redis.Redis
        
    Returns:
        Redis client instance
        
    Example:
        >>> client = await get_redis_client()
        >>> await client.set("key", "value")
        >>> value = await client.get("key")
        
        >>> # Force new client
        >>> client2 = await get_redis_client(force_new=True)
    """
    global _global_client
    
    if not force_new and _global_client is not None:
        logger.debug("redis_client_reusing_existing")
        return _global_client
    
    redis_url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    try:
        client = redis.from_url(redis_url, decode_responses=True, **kwargs)
        
        # Test connection
        await client.ping()
        
        logger.info(
            "redis_client_connected",
            url=redis_url.replace("://", "://***@") if "@" in redis_url else redis_url,
        )
        
        if not force_new:
            _global_client = client
        
        return client
        
    except Exception as e:
        logger.error(
            "redis_client_connection_failed",
            url=redis_url.replace("://", "://***@") if "@" in redis_url else redis_url,
            error=str(e),
        )
        raise ConnectionError(f"Failed to connect to Redis: {e}") from e


async def close_redis_client() -> None:
    """Close the global Redis client instance.
    
    This function should be called during application shutdown to ensure
    proper cleanup of Redis connections.
    
    Example:
        >>> await close_redis_client()
    """
    global _global_client
    
    if _global_client is not None:
        await _global_client.close()
        _global_client = None
        logger.info("redis_global_client_closed")


async def get_redis_lock(
    lock_name: str,
    timeout: int = 60,
    blocking: bool = True,
    blocking_timeout: float = 30.0,
) -> Any:
    """Get a Redis distributed lock.
    
    This is a convenience function for acquiring a Redis lock.
    For more control, use DistributedLock directly.
    
    Args:
        lock_name: Unique name for the lock
        timeout: Lock timeout in seconds (default: 60)
        blocking: Whether to block until lock is acquired (default: True)
        blocking_timeout: Maximum time to wait for lock (default: 30.0)
        
    Returns:
        Redis lock object that can be used as a context manager
        
    Example:
        >>> lock = await get_redis_lock("my_resource")
        >>> async with lock:
        ...     # Critical section
        ...     await process_data()
    """
    client = await get_redis_client()
    return client.lock(
        lock_name,
        timeout=timeout,
        blocking=blocking,
        blocking_timeout=blocking_timeout,
    )
