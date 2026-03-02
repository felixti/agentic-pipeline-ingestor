"""Redis infrastructure module.

This module provides Redis connectivity and utilities including:
- Redis client management
- Distributed locking for coordinating workers
- Cache operations
"""

from src.infrastructure.redis.client import (
    close_redis_client,
    get_redis_client,
    get_redis_lock,
)
from src.infrastructure.redis.lock import (
    CogneeWriteLock,
    DistributedLock,
    LockAcquisitionError,
    LockError,
    distributed_lock,
)

__all__ = [
    "get_redis_client",
    "close_redis_client",
    "get_redis_lock",
    "DistributedLock",
    "LockError",
    "LockAcquisitionError",
    "distributed_lock",
    "CogneeWriteLock",
]
