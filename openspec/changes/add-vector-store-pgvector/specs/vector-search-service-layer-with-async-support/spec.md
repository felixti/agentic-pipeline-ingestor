# Spec: Vector Search Service Layer with Async Support

## Purpose
Provide a high-level, asynchronous service interface for vector search operations that abstracts database complexity and supports concurrent, non-blocking queries.

## Interface

### Service Class Definition
```python
from typing import List, Optional, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

@dataclass
class DocumentChunk:
    id: str
    job_id: str
    chunk_index: int
    content: str
    embedding: Optional[list[float]]
    metadata: dict
    created_at: datetime

@dataclass
class SearchResult:
    chunk: DocumentChunk
    similarity: float  # 0.0 to 1.0
    distance: float    # Raw distance from query

@dataclass
class SearchParams:
    query_vector: list[float] | np.ndarray
    top_k: int = 10
    min_similarity: float = 0.0
    metadata_filter: Optional[dict] = None
    job_id: Optional[str] = None

class VectorSearchService:
    """
    Async service for vector similarity search operations.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        config: VectorStoreConfig,
    ):
        self.db = db_session
        self.config = config
        self.logger = get_logger(__name__)
    
    # Core search methods
    async def search_by_vector(
        self,
        params: SearchParams,
    ) -> List[SearchResult]:
        """
        Search for similar chunks by vector similarity.
        
        Args:
            params: Search parameters including query vector and filters
            
        Returns:
            List of SearchResult ordered by similarity (highest first)
        """
        pass
    
    async def find_similar_chunks(
        self,
        chunk_id: str,
        top_k: int = 10,
        exclude_same_job: bool = False,
    ) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk by ID.
        
        Args:
            chunk_id: UUID of the reference chunk
            top_k: Number of similar chunks to return
            exclude_same_job: If True, exclude chunks from the same job
            
        Returns:
            List of SearchResult excluding the reference chunk
        """
        pass
    
    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        embed_model: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search by text - generates embedding and performs vector search.
        
        Args:
            query_text: Natural language query
            top_k: Number of results
            embed_model: Optional override for embedding model
            
        Returns:
            List of SearchResult
        """
        pass
    
    # Bulk operations
    async def batch_search(
        self,
        queries: List[SearchParams],
        max_concurrency: int = 5,
    ) -> List[List[SearchResult]]:
        """
        Execute multiple searches concurrently.
        
        Args:
            queries: List of search parameter sets
            max_concurrency: Maximum parallel searches
            
        Returns:
            List of results matching input order
        """
        pass
    
    # Streaming for large result sets
    async def search_streaming(
        self,
        params: SearchParams,
    ) -> AsyncIterator[SearchResult]:
        """
        Stream search results as they are retrieved.
        
        Yields:
            SearchResult objects ordered by similarity
        """
        pass
    
    # Utility methods
    async def get_chunk_by_id(
        self,
        chunk_id: str,
    ) -> Optional[DocumentChunk]:
        """Retrieve a single chunk by ID."""
        pass
    
    async def get_chunks_by_job(
        self,
        job_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DocumentChunk]:
        """List all chunks for a specific job."""
        pass
    
    async def get_chunk_count(
        self,
        job_id: Optional[str] = None,
    ) -> int:
        """Get total chunk count, optionally filtered by job."""
        pass
```

### Factory and Dependency Injection
```python
async def get_vector_search_service(
    db: AsyncSession = Depends(get_db_session),
) -> VectorSearchService:
    """FastAPI dependency for service injection."""
    config = load_vector_store_config()
    return VectorSearchService(db, config)

# Usage in API endpoint
@router.post("/search/semantic")
async def semantic_search(
    request: SemanticSearchRequest,
    service: VectorSearchService = Depends(get_vector_search_service),
):
    results = await service.search_by_vector(
        SearchParams(
            query_vector=request.query_vector,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )
    )
    return {"results": results}
```

## Behavior

### Async Database Operations
- All database queries use `asyncpg` via SQLAlchemy async session
- Connection pooling managed by SQLAlchemy engine
- Proper session cleanup with context managers

### Session Lifecycle
```python
async def search_by_vector(self, params: SearchParams) -> List[SearchResult]:
    start_time = time.monotonic()
    
    try:
        # Validate inputs
        self._validate_query_vector(params.query_vector)
        
        # Build and execute query
        query = self._build_similarity_query(params)
        result = await self.db.execute(query)
        rows = result.fetchall()
        
        # Transform to domain objects
        results = [self._row_to_search_result(r) for r in rows]
        
        # Log performance metrics
        duration = time.monotonic() - start_time
        self.logger.info(
            "vector_search_completed",
            top_k=params.top_k,
            result_count=len(results),
            duration_ms=duration * 1000,
        )
        
        return results
        
    except SQLAlchemyError as e:
        self.logger.error("database_error", error=str(e))
        raise SearchServiceError(f"Search failed: {e}") from e
```

### Concurrency Control
- `batch_search` uses `asyncio.gather()` with semaphore-based limiting
- Default `max_concurrency=5` prevents database overload
- Individual query timeouts prevent runaway operations

### Connection Pooling
```python
# Engine configuration for async operations
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False,
)
```

### Retry Logic
- Automatic retry on transient failures (connection drops, timeouts)
- Exponential backoff: 100ms, 200ms, 400ms
- Max 3 retries for idempotent operations

## Error Handling

| Exception | Cause | Handling |
|-----------|-------|----------|
| `InvalidVectorError` | Dimension mismatch, NaN values | 400 Bad Request |
| `ChunkNotFoundError` | chunk_id does not exist | 404 Not Found |
| `SearchServiceError` | Database failure | 500 Internal Error |
| `TimeoutError` | Query exceeds timeout | 504 Gateway Timeout |
| `ConfigurationError` | Missing/invalid config | 500 Internal Error |

### Error Context
```python
class SearchServiceError(Exception):
    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}
        
# Usage
except SQLAlchemyError as e:
    raise SearchServiceError(
        "Database query failed",
        context={
            "query_type": "cosine_similarity",
            "top_k": params.top_k,
            "table": "document_chunks",
        }
    ) from e
```

## Performance Considerations

### Query Building
- Use SQLAlchemy 2.0 style queries for type safety
- Compile queries once, execute with different parameters
- Leverage prepared statements for repeated queries

### Caching Strategy
- Cache embedding model instances (thread-safe)
- Consider result caching for identical queries (configurable TTL)
- Cache metadata lookups (job_id validation)

### Memory Management
- Stream large result sets instead of loading all into memory
- Use `yield_per()` for bulk operations
- Clear result sets after processing

### Performance Monitoring
```python
# Metrics to track
search_duration_histogram.observe(duration)
search_result_counter.inc(len(results))
search_error_counter.inc(exception_type)
db_query_duration_histogram.observe(query_time)
```

### Async Best Practices
- Never use sync I/O in async methods (blocks event loop)
- Use `asyncio.to_thread()` for CPU-bound operations (embedding generation)
- Proper cancellation handling for long-running queries
- Context manager support for resource cleanup

```python
async def search_by_text(self, query_text: str, ...) -> List[SearchResult]:
    # CPU-bound: run in thread pool
    query_vector = await asyncio.to_thread(
        self.embedder.encode,
        query_text
    )
    
    # I/O-bound: native async
    return await self.search_by_vector(
        SearchParams(query_vector=query_vector, ...)
    )
```
