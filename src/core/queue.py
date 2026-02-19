"""Redis queue abstraction for job distribution.

This module provides a queue abstraction layer using Redis
for distributing jobs across multiple workers.
"""

import json
import logging
from typing import Any

import redis.asyncio as redis

from src.config import settings

logger = logging.getLogger(__name__)


class QueuePriority:
    """Queue priority levels."""
    
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    
    ALL = [HIGH, NORMAL, LOW]


class JobQueue:
    """Redis-based job queue.
    
    Provides priority queue support and job distribution across workers.
    
    Example:
        >>> queue = JobQueue()
        >>> await queue.enqueue("job-123", priority="high")
        >>> job_id = await queue.dequeue()
    """
    
    def __init__(
        self,
        redis_url: str | None = None,
        prefix: str = "pipeline",
    ) -> None:
        """Initialize the job queue.
        
        Args:
            redis_url: Redis connection URL (uses settings if not provided)
            prefix: Key prefix for queue names
        """
        self.redis_url = redis_url or str(settings.redis.url)
        self.prefix = prefix
        self._redis: redis.Redis | None = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
            )
            logger.info("connected_to_redis")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("disconnected_from_redis")
    
    def _queue_key(self, priority: str) -> str:
        """Get the Redis key for a priority queue.
        
        Args:
            priority: Priority level
            
        Returns:
            Redis key
        """
        return f"{self.prefix}:jobs:{priority}"
    
    def _processing_key(self, worker_id: str) -> str:
        """Get the Redis key for processing jobs.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            Redis key
        """
        return f"{self.prefix}:processing:{worker_id}"
    
    async def enqueue(
        self,
        job_id: str,
        priority: str = QueuePriority.NORMAL,
        ttl: int = 86400,  # 24 hours
    ) -> bool:
        """Enqueue a job.
        
        Args:
            job_id: Job ID to enqueue
            priority: Priority level (high, normal, low)
            ttl: Time-to-live in seconds
            
        Returns:
            True if enqueued successfully
        """
        await self.connect()
        
        if priority not in QueuePriority.ALL:
            priority = QueuePriority.NORMAL
        
        queue_key = self._queue_key(priority)
        
        try:
            assert self._redis is not None, "Redis not connected"
            # Add job to priority queue (using LPUSH for FIFO)
            await self._redis.lpush(queue_key, job_id)  # type: ignore[misc]
            
            # Set TTL on queue
            await self._redis.expire(queue_key, ttl)
            
            logger.debug("job_enqueued: job_id=%s priority=%s", job_id, priority)
            
            return True
            
        except Exception as e:
            logger.error("failed_to_enqueue_job: job_id=%s error=%s", job_id, e)
            return False
    
    async def dequeue(
        self,
        worker_id: str,
        timeout: int = 5,
    ) -> str | None:
        """Dequeue a job.
        
        Checks queues in priority order (high -> normal -> low).
        Uses blocking pop to wait for jobs.
        
        Args:
            worker_id: Worker identifier for tracking
            timeout: Block timeout in seconds
            
        Returns:
            Job ID if available, None otherwise
        """
        await self.connect()
        
        # Check queues in priority order
        queue_keys = [self._queue_key(p) for p in QueuePriority.ALL]
        
        try:
            assert self._redis is not None, "Redis not connected"
            # Use BRPOP to block and pop from the right (FIFO)
            result = await self._redis.brpop(  # type: ignore[misc]
                queue_keys,
                timeout=timeout,
            )
            
            if result:
                queue_name, job_id = result
                
                # Track job as processing
                processing_key = self._processing_key(worker_id)
                await self._redis.sadd(processing_key, job_id)  # type: ignore[misc]
                await self._redis.expire(processing_key, 3600)  # 1 hour
                
                logger.debug("job_dequeued: job_id=%s worker_id=%s", job_id, worker_id)
                
                return job_id  # type: ignore[no-any-return]
            
            return None
            
        except Exception as e:
            logger.error("failed_to_dequeue_job: worker_id=%s error=%s", worker_id, e)
            return None
    
    async def ack(
        self,
        job_id: str,
        worker_id: str,
    ) -> bool:
        """Acknowledge job completion.
        
        Args:
            job_id: Job ID
            worker_id: Worker identifier
            
        Returns:
            True if acknowledged
        """
        await self.connect()
        
        try:
            assert self._redis is not None, "Redis not connected"
            processing_key = self._processing_key(worker_id)
            await self._redis.srem(processing_key, job_id)  # type: ignore[misc]
            
            logger.debug("job_acknowledged: job_id=%s worker_id=%s", job_id, worker_id)
            
            return True
            
        except Exception as e:
            logger.error("failed_to_ack_job: job_id=%s error=%s", job_id, e)
            return False
    
    async def nack(
        self,
        job_id: str,
        worker_id: str,
        priority: str = QueuePriority.NORMAL,
    ) -> bool:
        """Negative acknowledge - requeue job.
        
        Args:
            job_id: Job ID
            worker_id: Worker identifier
            priority: Priority for requeue
            
        Returns:
            True if requeued
        """
        await self.connect()
        
        try:
            assert self._redis is not None, "Redis not connected"
            # Remove from processing
            processing_key = self._processing_key(worker_id)
            await self._redis.srem(processing_key, job_id)  # type: ignore[misc]
            
            # Requeue
            queue_key = self._queue_key(priority)
            await self._redis.lpush(queue_key, job_id)  # type: ignore[misc]
            
            logger.debug("job_requeued: job_id=%s worker_id=%s", job_id, worker_id)
            
            return True
            
        except Exception as e:
            logger.error("failed_to_nack_job: job_id=%s error=%s", job_id, e)
            return False
    
    async def get_queue_depths(self) -> dict[str, int]:
        """Get the depth of each queue.
        
        Returns:
            Dictionary of priority -> count
        """
        await self.connect()
        
        depths = {}
        
        try:
            assert self._redis is not None, "Redis not connected"
            for priority in QueuePriority.ALL:
                queue_key = self._queue_key(priority)
                count = await self._redis.llen(queue_key)  # type: ignore[misc]
                depths[priority] = count
                
        except Exception as e:
            logger.error("failed_to_get_queue_depths: %s", e)
            
        return depths
    
    async def get_processing_count(self, worker_id: str | None = None) -> int | dict[str, int]:
        """Get count of jobs being processed.
        
        Args:
            worker_id: Specific worker, or all if None
            
        Returns:
            Count or dictionary of worker_id -> count
        """
        await self.connect()
        
        try:
            assert self._redis is not None, "Redis not connected"
            if worker_id:
                processing_key = self._processing_key(worker_id)
                return await self._redis.scard(processing_key)  # type: ignore[no-any-return,misc]
            else:
                # Get all workers
                pattern = f"{self.prefix}:processing:*"
                keys = await self._redis.keys(pattern)
                
                counts: dict[str, int] = {}
                for key in keys:
                    worker = key.split(":")[-1]
                    counts[worker] = await self._redis.scard(key)  # type: ignore[misc]
                    
                return counts
                
        except Exception as e:
            logger.error("failed_to_get_processing_count: %s", e)
            return {} if worker_id is None else 0  # type: ignore[return]
    
    async def clear_queue(self, priority: str | None = None) -> int:
        """Clear queue(s).
        
        Args:
            priority: Specific priority queue, or all if None
            
        Returns:
            Number of jobs cleared
        """
        await self.connect()
        
        try:
            assert self._redis is not None, "Redis not connected"
            if priority:
                queue_key = self._queue_key(priority)
                count = await self._redis.llen(queue_key)  # type: ignore[misc]
                await self._redis.delete(queue_key)
                return count  # type: ignore[no-any-return]
            else:
                # Clear all queues
                total = 0
                for p in QueuePriority.ALL:
                    queue_key = self._queue_key(p)
                    count = await self._redis.llen(queue_key)  # type: ignore[misc]
                    await self._redis.delete(queue_key)  # type: ignore[misc]
                    total += count
                return total
                
        except Exception as e:
            logger.error("failed_to_clear_queue: %s", e)
            return 0


# Global queue instance
_queue_instance: JobQueue | None = None


def get_queue() -> JobQueue:
    """Get or create the global queue instance.
    
    Returns:
        JobQueue instance
    """
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = JobQueue()
    return _queue_instance
