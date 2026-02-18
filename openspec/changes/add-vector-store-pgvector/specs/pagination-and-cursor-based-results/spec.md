# Spec: Pagination and Cursor-Based Results

## Purpose
Enable efficient handling of large vector search result sets through offset-based and cursor-based pagination mechanisms.

## Interface

### Service Interface
```python
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class PaginationParams:
    limit: int = 20
    offset: Optional[int] = None  # For offset-based
    cursor: Optional[str] = None  # For cursor-based
    cursor_field: str = "similarity"  # Field to use for cursor

@dataclass
class PaginatedResults:
    results: List[SearchResult]
    total: int  # Total matching results (may be estimated)
    limit: int
    offset: Optional[int]
    next_cursor: Optional[str]
    has_more: bool
    
class VectorSearchService:
    async def search_paginated(
        self,
        query_vector: list[float],
        pagination: PaginationParams,
        params: SearchParams,
    ) -> PaginatedResults:
        """
        Execute paginated vector search.
        
        Supports both offset-based (simple) and cursor-based (efficient)
        pagination strategies.
        """
        pass
    
    async def search_with_cursor(
        self,
        query_vector: list[float],
        cursor: Optional[str] = None,
        limit: int = 20,
        params: SearchParams,
    ) -> CursorPaginatedResults:
        """
        Cursor-based search for efficient deep pagination.
        
        Cursor format: base64-encoded "similarity:id" tuple
        """
        pass
```

### API Interface

#### Offset-Based Pagination
```http
POST /api/v1/search/semantic
Content-Type: application/json

{
  "query_vector": [0.023, -0.156, 0.892, ...],
  "top_k": 100,
  "pagination": {
    "type": "offset",
    "limit": 20,
    "offset": 40
  }
}

# Response
{
  "success": true,
  "data": {
    "results": [...],
    "pagination": {
      "total": 156,
      "limit": 20,
      "offset": 40,
      "has_more": true,
      "next_offset": 60,
      "prev_offset": 20
    }
  }
}
```

#### Cursor-Based Pagination
```http
POST /api/v1/search/semantic
Content-Type: application/json

{
  "query_vector": [0.023, -0.156, 0.892, ...],
  "top_k": 100,
  "pagination": {
    "type": "cursor",
    "limit": 20,
    "cursor": "eyJzaW1pbGFyaXR5IjogMC44NSwgImlkIjogInV1aWQtMTIzIn0="  # base64
  }
}

# Response
{
  "success": true,
  "data": {
    "results": [...],
    "pagination": {
      "limit": 20,
      "next_cursor": "eyJzaW1pbGFyaXR5IjogMC43MiwgImlkIjogInV1aWQtNDU2In0=",
      "has_more": true,
      "total_estimated": 156
    }
  }
}
```

### SQL Interface

#### Offset-Based Query
```sql
-- Offset-based pagination (simple, slower for deep pages)
SELECT id, content, metadata,
       1 - (embedding <=> :query_vector) AS similarity
FROM document_chunks
WHERE job_id = :job_id
ORDER BY embedding <=> :query_vector ASC
LIMIT :limit OFFSET :offset;

-- Count query for total
SELECT COUNT(*) 
FROM document_chunks
WHERE job_id = :job_id;
```

#### Cursor-Based Query
```sql
-- Cursor-based pagination (efficient for deep pages)
-- Cursor decodes to: (similarity=0.85, id='uuid-123')
SELECT id, content, metadata,
       1 - (embedding <=> :query_vector) AS similarity
FROM document_chunks
WHERE job_id = :job_id
  AND (
    1 - (embedding <=> :query_vector) < :cursor_similarity
    OR (
      1 - (embedding <=> :query_vector) = :cursor_similarity 
      AND id > :cursor_id
    )
  )
ORDER BY embedding <=> :query_vector ASC, id ASC
LIMIT :limit;
```

## Behavior

### Pagination Strategies

| Strategy | Best For | Performance | Consistency |
|----------|----------|-------------|-------------|
| **Offset** | Small result sets (<1000), simple UI | Degrades with offset | May miss/duplicate during updates |
| **Cursor** | Large result sets, infinite scroll | Constant time | Stable across updates |
| **Seek** | Very large datasets, sorted data | Fast | Requires unique sort field |

### Offset-Based Behavior
1. Execute search query with `LIMIT :limit OFFSET :offset`
2. Optionally execute `COUNT(*)` for total (expensive on large tables)
3. Return results with navigation metadata
4. Client increments `offset` by `limit` for next page

### Cursor-Based Behavior
1. Decode cursor to get `(similarity, id)` tuple
2. Filter results: `similarity < cursor_similarity OR (similarity = cursor_similarity AND id > cursor_id)`
3. Order by `similarity DESC, id ASC` for stability
4. Encode last result's `(similarity, id)` as next cursor
5. Return null cursor when no more results

### Cursor Encoding
```python
import base64
import json

def encode_cursor(similarity: float, chunk_id: str) -> str:
    """Encode pagination cursor."""
    data = {"s": round(similarity, 6), "id": chunk_id}
    json_bytes = json.dumps(data, separators=(',', ':')).encode()
    return base64.urlsafe_b64encode(json_bytes).decode().rstrip('=')

def decode_cursor(cursor: str) -> tuple[float, str]:
    """Decode pagination cursor."""
    # Add padding if needed
    padding = 4 - len(cursor) % 4
    if padding != 4:
        cursor += '=' * padding
    
    json_bytes = base64.urlsafe_b64decode(cursor)
    data = json.loads(json_bytes)
    return data["s"], data["id"]
```

### Pagination Limits
- **Minimum limit**: 1
- **Default limit**: 20
- **Maximum limit**: 100 (configurable)
- **Maximum offset**: 10,000 (offset-based only, to prevent abuse)
- **Recommended**: Use cursor-based for offsets > 1000

### Consistency Guarantees
- **Offset-based**: Results may shift if data changes during pagination
- **Cursor-based**: Each page starts where previous ended, stable across updates
- **Real-time**: New inserts with higher similarity may appear on subsequent pages

## Error Handling

| Error Code | Condition | Response |
|------------|-----------|----------|
| `400 Bad Request` | limit < 1 | `"limit must be at least 1"` |
| `400 Bad Request` | limit > max_limit | `"limit exceeds maximum (100)"` |
| `400 Bad Request` | offset < 0 | `"offset cannot be negative"` |
| `400 Bad Request` | offset > max_offset | `"offset too large; use cursor-based pagination"` |
| `400 Bad Request` | Invalid cursor format | `"Invalid cursor format"` |
| `400 Bad Request` | Expired cursor | `"Cursor references deleted data"` |
| `400 Bad Request` | Cursor with different query | `"Cursor is not valid for this search query"` |

### Cursor Validation
```python
def validate_cursor(cursor: str, query_hash: str) -> CursorData:
    """Validate cursor matches current query context."""
    data = decode_cursor(cursor)
    
    # Verify cursor was generated for same query parameters
    if data.get("q") != query_hash:
        raise InvalidCursorError("Cursor not valid for this query")
    
    # Check cursor age (optional: expire after N minutes)
    if time.time() - data.get("t", 0) > 300:  # 5 minutes
        logger.warning("expired_cursor_used")
    
    return data
```

## Performance Considerations

### Offset Performance Degradation
```
Offset    Approximate Time    Index Usage
------    ----------------    -----------
0         1ms                 Index scan
1,000     5ms                 Index scan + skip
10,000    50ms                Index scan + skip
100,000   500ms               Sequential scan likely
1,000,000 5s+                 Full table scan
```

### Cursor Performance
- Constant time regardless of page depth
- Uses index efficiently: `WHERE (similarity, id) < (cursor_s, cursor_id)`
- No need to count or skip rows

### Query Plan Optimization
```sql
-- Ensure composite index exists for cursor pagination
CREATE INDEX idx_document_chunks_similarity_id 
ON document_chunks(
    (1 - (embedding <=> query_vector)) DESC, 
    id ASC
);
-- Note: This is a conceptual index; actual implementation 
-- relies on HNSW index + secondary ordering
```

### Count Estimation
- Exact `COUNT(*)` is expensive on large filtered results
- Provide estimated count or omit total for cursor-based
- Use `pg_class` statistics for rough estimates:
  ```sql
  SELECT reltuples::bigint AS estimate 
  FROM pg_class 
  WHERE relname = 'document_chunks';
  ```

### Memory Considerations
- Offset-based loads entire preceding result set into memory
- Cursor-based only keeps current page
- For very large `top_k` with pagination, use server-side cursor:
  ```python
  async with db.stream(query) as stream:
      async for row in stream:
          # Process in streaming fashion
  ```

### Hybrid Approach
```python
async def search_paginated(self, ...):
    # Use offset for first few pages (simple)
    if params.offset and params.offset < 1000:
        return await self._offset_pagination(params)
    
    # Switch to cursor for deep pagination (efficient)
    if params.offset and params.offset >= 1000:
        # Auto-convert offset to approximate cursor
        cursor = await self._offset_to_cursor(params)
        return await self._cursor_pagination(params.with_cursor(cursor))
    
    # Default to cursor for new queries
    return await self._cursor_pagination(params)
```

### Streaming Large Results
For exporting or processing all results:
```python
async def search_streaming(
    self,
    query_vector: list[float],
    params: SearchParams,
) -> AsyncIterator[SearchResult]:
    """Stream all results without pagination limits."""
    cursor = None
    while True:
        page = await self.search_with_cursor(
            query_vector, cursor, limit=100, params
        )
        for result in page.results:
            yield result
        
        if not page.next_cursor:
            break
        cursor = page.next_cursor
```
