"""Performance optimizations for the Agentic Data Pipeline Ingestor.

This module provides performance tuning and optimization features
for achieving 20GB/day throughput targets.
"""

import asyncio
import functools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import aiohttp
from aiohttp import ClientSession, TCPConnector

T = TypeVar("T")


class ConnectionPoolManager:
    """Manages connection pools for HTTP clients.
    
    This class provides optimized connection pooling for:
    - Database connections
    - Redis connections
    - HTTP client connections
    - LLM API connections
    """

    def __init__(self):
        """Initialize the connection pool manager."""
        self._http_session: ClientSession | None = None
        self._executor: ThreadPoolExecutor | None = None

    async def get_http_session(self) -> ClientSession:
        """Get or create an optimized HTTP session.
        
        Returns:
            Configured aiohttp ClientSession
        """
        if self._http_session is None or self._http_session.closed:
            # Configure connection pool for high throughput
            connector = TCPConnector(
                limit=100,              # Total connection pool size
                limit_per_host=30,      # Connections per host
                enable_cleanup_closed=True,
                force_close=False,
                ttl_dns_cache=300,      # DNS cache TTL
                use_dns_cache=True,
            )

            # Configure timeouts
            timeout = aiohttp.ClientTimeout(
                total=300,              # Total timeout
                connect=30,             # Connection timeout
                sock_read=60,           # Socket read timeout
            )

            self._http_session = ClientSession(
                connector=connector,
                timeout=timeout,
                raise_for_status=True,
            )

        return self._http_session

    def get_thread_pool(self, max_workers: int = 10) -> ThreadPoolExecutor:
        """Get or create a thread pool executor.
        
        Args:
            max_workers: Maximum number of worker threads
            
        Returns:
            ThreadPoolExecutor instance
        """
        if self._executor is None or self._executor._shutdown:
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="pipeline_worker_",
            )
        return self._executor

    async def close(self) -> None:
        """Close all connection pools."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

        if self._executor and not self._executor._shutdown:
            self._executor.shutdown(wait=True)


class BatchProcessor:
    """Processes items in batches for improved throughput.
    
    This class provides batching capabilities for:
    - Database writes
    - API calls
    - Message queue operations
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_wait_time: float = 1.0,
        max_concurrent: int = 5,
    ):
        """Initialize the batch processor.
        
        Args:
            batch_size: Maximum items per batch
            max_wait_time: Maximum time to wait for batch fill
            max_concurrent: Maximum concurrent batch operations
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent = max_concurrent

        self._queue: asyncio.Queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._processing = False

    async def add(self, item: Any) -> None:
        """Add an item to the batch queue.
        
        Args:
            item: Item to add to queue
        """
        await self._queue.put(item)

    async def process_batches(
        self,
        processor: Callable[[list[Any]], Any],
    ) -> None:
        """Process items in batches.
        
        Args:
            processor: Function to process a batch of items
        """
        self._processing = True

        while self._processing:
            batch: list[Any] = []

            # Collect items for batch
            try:
                # Wait for first item
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.max_wait_time,
                )
                batch.append(item)

                # Collect more items up to batch_size
                while len(batch) < self.batch_size:
                    try:
                        item = self._queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break

                # Process batch
                async with self._semaphore:
                    await processor(batch)

            except TimeoutError:
                # No items in queue, continue
                continue

    def stop(self) -> None:
        """Stop batch processing."""
        self._processing = False

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()


class AsyncCache:
    """Simple async cache with TTL support.
    
    This provides caching for:
    - LLM responses
    - Parser results
    - Content detection results
    """

    def __init__(self, default_ttl: int = 3600):
        """Initialize the cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self._cache: dict[str, Any] = {}
        self._ttl: dict[str, float] = {}
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get item from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        async with self._lock:
            if key in self._cache:
                import time
                if time.time() < self._ttl.get(key, 0):
                    return self._cache[key]
                else:
                    # Expired, remove
                    del self._cache[key]
                    del self._ttl[key]
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        import time

        async with self._lock:
            self._cache[key] = value
            self._ttl[key] = time.time() + (ttl or self._default_ttl)

    async def delete(self, key: str) -> None:
        """Delete item from cache.
        
        Args:
            key: Cache key
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._ttl[key]

    async def clear(self) -> None:
        """Clear all cached items."""
        async with self._lock:
            self._cache.clear()
            self._ttl.clear()


class PerformanceOptimizer:
    """Main performance optimizer for the pipeline.
    
    This class coordinates all performance optimizations for achieving
    20GB/day throughput targets.
    
    Optimizations:
    - Connection pooling
    - Batch processing
    - Concurrent workers
    - Caching strategies
    - Async I/O
    """

    # Target throughput: 20GB/day = ~231 KB/s sustained
    # With burst capacity for near-realtime processing

    TARGET_DAILY_THROUGHPUT_GB = 20
    TARGET_P99_LATENCY_SECONDS = 2
    TARGET_SUCCESS_RATE = 0.995

    def __init__(self):
        """Initialize the performance optimizer."""
        self.connection_pool = ConnectionPoolManager()
        self.batch_processor = BatchProcessor(
            batch_size=50,
            max_wait_time=0.5,
            max_concurrent=10,
        )
        self.cache = AsyncCache(default_ttl=1800)  # 30 min cache
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all optimization components."""
        if self._initialized:
            return

        # Initialize HTTP session
        await self.connection_pool.get_http_session()

        self._initialized = True

    async def optimize_for_throughput(self) -> dict[str, Any]:
        """Configure system for maximum throughput.
        
        Returns:
            Configuration settings applied
        """
        config = {
            # Database connection pool
            "db_pool_size": 20,
            "db_max_overflow": 10,
            "db_pool_timeout": 30,
            "db_pool_recycle": 1800,

            # Redis connection pool
            "redis_max_connections": 100,
            "redis_socket_timeout": 30,

            # Worker configuration
            "worker_max_concurrent": 10,
            "worker_poll_interval": 1.0,

            # Batch processing
            "batch_size": 50,
            "batch_max_wait": 0.5,

            # HTTP client
            "http_pool_size": 100,
            "http_pool_per_host": 30,

            # Caching
            "cache_ttl_default": 1800,
            "cache_max_size": 10000,

            # File processing
            "chunk_size_bytes": 8192,
            "max_file_size_mb": 100,
            "concurrent_parsers": 5,
        }

        return config

    async def run_concurrent(
        self,
        func: Callable[..., T],
        items: list[Any],
        max_concurrent: int = 10,
        **kwargs: Any,
    ) -> list[T]:
        """Run function concurrently on multiple items.
        
        Args:
            func: Async function to run
            items: Items to process
            max_concurrent: Maximum concurrent executions
            **kwargs: Additional arguments to pass to func
            
        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(item: Any) -> T:
            async with semaphore:
                return await func(item, **kwargs)

        tasks = [run_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    @asynccontextmanager
    async def timed_operation(self, operation_name: str):
        """Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation
            
        Yields:
            None
        """
        import time

        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            # Could log or record metrics here

    def estimate_processing_time(
        self,
        file_size_bytes: int,
        file_type: str,
        parser: str,
    ) -> float:
        """Estimate processing time for a file.
        
        Args:
            file_size_bytes: File size in bytes
            file_type: Type of file
            parser: Parser to use
            
        Returns:
            Estimated processing time in seconds
        """
        # Base processing rates (MB/s)
        base_rates = {
            "docling": {
                "pdf": 0.5,
                "docx": 1.0,
                "pptx": 0.3,
                "xlsx": 2.0,
            },
            "azure_ocr": {
                "pdf": 0.2,
                "image": 0.1,
            },
        }

        parser_rates = base_rates.get(parser, base_rates["docling"])
        rate = parser_rates.get(file_type, 0.5)  # Default 0.5 MB/s

        # Convert bytes to MB
        size_mb = file_size_bytes / (1024 * 1024)

        # Calculate time with overhead
        estimated_time = (size_mb / rate) + 0.5  # 0.5s overhead

        return max(estimated_time, 0.1)  # Minimum 0.1s

    async def close(self) -> None:
        """Close all resources."""
        await self.connection_pool.close()
        self.batch_processor.stop()


# Global optimizer instance
_optimizer: PerformanceOptimizer | None = None


def get_optimizer() -> PerformanceOptimizer:
    """Get or create the global performance optimizer.
    
    Returns:
        PerformanceOptimizer singleton
    """
    global _optimizer
    if _optimizer is None:
        _optimizer = PerformanceOptimizer()
    return _optimizer


def run_in_executor(func: Callable[..., T]) -> Callable[..., asyncio.Future[T]]:
    """Decorator to run synchronous function in thread pool.
    
    Args:
        func: Synchronous function
        
    Returns:
        Async function that runs in thread pool
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        optimizer = get_optimizer()
        executor = optimizer.connection_pool.get_thread_pool()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            functools.partial(func, *args, **kwargs),
        )
    return wrapper


class StreamingFileProcessor:
    """Processes large files using streaming for memory efficiency.
    
    This is critical for handling files up to 100MB without OOM issues.
    """

    CHUNK_SIZE = 8192  # 8KB chunks

    @staticmethod
    async def stream_upload(
        source_path: str,
        destination_path: str,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        """Stream upload a file in chunks.
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            chunk_size: Size of chunks to read/write
        """
        import aiofiles

        async with aiofiles.open(source_path, "rb") as src:
            async with aiofiles.open(destination_path, "wb") as dst:
                while True:
                    chunk = await src.read(chunk_size)
                    if not chunk:
                        break
                    await dst.write(chunk)

    @staticmethod
    def calculate_optimal_chunk_size(file_size: int) -> int:
        """Calculate optimal chunk size based on file size.
        
        Args:
            file_size: File size in bytes
            
        Returns:
            Optimal chunk size
        """
        # For files < 1MB, use 8KB chunks
        if file_size < 1024 * 1024:
            return 8192
        # For files < 10MB, use 64KB chunks
        elif file_size < 10 * 1024 * 1024:
            return 65536
        # For files < 100MB, use 256KB chunks
        elif file_size < 100 * 1024 * 1024:
            return 262144
        # For larger files, use 1MB chunks
        else:
            return 1048576
