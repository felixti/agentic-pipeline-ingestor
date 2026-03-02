"""Distributed locking using Redis.

This module provides distributed locking mechanisms to coordinate
operations across multiple workers or processes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import redis.asyncio as redis

from src.observability.logging import get_logger

logger = get_logger(__name__)


class LockError(Exception):
    """Exception raised for lock-related errors."""
    pass


class LockAcquisitionError(LockError):
    """Exception raised when lock acquisition fails."""
    pass


class DistributedLock:
    """Distributed lock using Redis.
    
    This class provides a high-level interface for distributed locking
    that can be used to prevent concurrent execution of critical sections
    across multiple workers.
    
    Attributes:
        name: Unique name for the lock
        redis_client: Redis client instance
        timeout: Lock timeout in seconds
        blocking: Whether to block until lock is acquired
        blocking_timeout: Maximum time to wait for lock
        
    Example:
        >>> lock = DistributedLock("graph_write_lock", redis_client)
        >>> async with lock:
        ...     # Only one worker can execute this at a time
        ...     await update_graph()
        
        >>> # With custom timeout
        >>> lock = DistributedLock(
        ...     "graph_write_lock",
        ...     redis_client,
        ...     timeout=120,
        ...     blocking_timeout=10.0
        ... )
        >>> if await lock.acquire():
        ...     try:
        ...         await update_graph()
        ...     finally:
        ...         await lock.release()
    """

    def __init__(
        self,
        name: str,
        redis_client: redis.Redis | None = None,
        timeout: int = 60,
        blocking: bool = True,
        blocking_timeout: float = 30.0,
    ) -> None:
        """Initialize distributed lock.
        
        Args:
            name: Unique name for the lock
            redis_client: Redis client instance (optional, will create if not provided)
            timeout: Lock timeout in seconds (default: 60)
            blocking: Whether to block until lock is acquired (default: True)
            blocking_timeout: Maximum time to wait for lock (default: 30.0)
        """
        self.name = name
        self._redis = redis_client
        self.timeout = timeout
        self.blocking = blocking
        self.blocking_timeout = blocking_timeout
        self._lock: Any = None
        self._is_acquired = False

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._redis is None:
            from src.infrastructure.redis.client import get_redis_client
            self._redis = await get_redis_client()
        return self._redis

    async def acquire(self) -> bool:
        """Acquire the distributed lock.
        
        Returns:
            True if lock was acquired, False otherwise
            
        Raises:
            LockAcquisitionError: If blocking is True and lock cannot be acquired
        """
        redis_client = await self._get_redis()
        
        self._lock = redis_client.lock(
            self.name,
            timeout=self.timeout,
        )
        
        try:
            acquired = await self._lock.acquire(
                blocking=self.blocking,
                blocking_timeout=self.blocking_timeout,
            )
            
            if acquired:
                self._is_acquired = True
                logger.debug(
                    "distributed_lock_acquired",
                    lock_name=self.name,
                    timeout=self.timeout,
                )
                return True
            else:
                logger.warning(
                    "distributed_lock_not_acquired",
                    lock_name=self.name,
                    blocking_timeout=self.blocking_timeout,
                )
                if self.blocking:
                    raise LockAcquisitionError(
                        f"Failed to acquire lock '{self.name}' within {self.blocking_timeout}s"
                    )
                return False
                
        except Exception as e:
            logger.error(
                "distributed_lock_acquisition_error",
                lock_name=self.name,
                error=str(e),
            )
            raise LockAcquisitionError(f"Error acquiring lock '{self.name}': {e}") from e

    async def release(self) -> None:
        """Release the distributed lock.
        
        This method is safe to call even if the lock was not acquired.
        """
        if self._lock is not None and self._is_acquired:
            try:
                await self._lock.release()
                self._is_acquired = False
                logger.debug(
                    "distributed_lock_released",
                    lock_name=self.name,
                )
            except Exception as e:
                # Log but don't raise - lock may have expired
                logger.warning(
                    "distributed_lock_release_error",
                    lock_name=self.name,
                    error=str(e),
                )

    async def __aenter__(self) -> DistributedLock:
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.release()

    @property
    def is_acquired(self) -> bool:
        """Check if lock is currently acquired."""
        return self._is_acquired


@asynccontextmanager
async def distributed_lock(
    name: str,
    redis_client: redis.Redis | None = None,
    timeout: int = 60,
    blocking: bool = True,
    blocking_timeout: float = 30.0,
) -> AsyncGenerator[DistributedLock, None]:
    """Context manager for distributed locking.
    
    This is a convenience function that provides a simpler interface
    for using distributed locks.
    
    Args:
        name: Unique name for the lock
        redis_client: Redis client instance (optional)
        timeout: Lock timeout in seconds (default: 60)
        blocking: Whether to block until lock is acquired (default: True)
        blocking_timeout: Maximum time to wait for lock (default: 30.0)
        
    Yields:
        DistributedLock instance
        
    Raises:
        LockAcquisitionError: If lock cannot be acquired
        
    Example:
        >>> async with distributed_lock("graph_write", timeout=120) as lock:
        ...     # Critical section - only one worker at a time
        ...     await update_graph()
    """
    lock = DistributedLock(
        name=name,
        redis_client=redis_client,
        timeout=timeout,
        blocking=blocking,
        blocking_timeout=blocking_timeout,
    )
    
    try:
        await lock.acquire()
        yield lock
    finally:
        await lock.release()


class CogneeWriteLock:
    """Specialized lock for Cognee write operations.
    
    This lock is designed specifically for Cognee graph write operations
    to prevent deadlocks when multiple workers try to write to the graph
    simultaneously.
    
    Example:
        >>> lock = CogneeWriteLock(dataset_id="my_dataset")
        >>> async with lock:
        ...     await cognee.cognify(datasets=["my_dataset"])
    """

    def __init__(
        self,
        dataset_id: str,
        redis_client: redis.Redis | None = None,
        timeout: int = 300,  # 5 minutes default for graph operations
        blocking_timeout: float = 60.0,
    ) -> None:
        """Initialize Cognee write lock.
        
        Args:
            dataset_id: Dataset ID for the lock
            redis_client: Redis client instance (optional)
            timeout: Lock timeout in seconds (default: 300)
            blocking_timeout: Maximum time to wait for lock (default: 60.0)
        """
        lock_name = f"cognee_write_lock:{dataset_id}"
        self._lock = DistributedLock(
            name=lock_name,
            redis_client=redis_client,
            timeout=timeout,
            blocking=True,
            blocking_timeout=blocking_timeout,
        )
        self.dataset_id = dataset_id

    async def acquire(self) -> bool:
        """Acquire the lock."""
        logger.info(
            "cognee_write_lock_acquiring",
            dataset_id=self.dataset_id,
        )
        result = await self._lock.acquire()
        if result:
            logger.info(
                "cognee_write_lock_acquired",
                dataset_id=self.dataset_id,
            )
        return result

    async def release(self) -> None:
        """Release the lock."""
        await self._lock.release()
        logger.info(
            "cognee_write_lock_released",
            dataset_id=self.dataset_id,
        )

    async def __aenter__(self) -> CogneeWriteLock:
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.release()
