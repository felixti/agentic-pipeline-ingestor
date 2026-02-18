# Spec: Configurable Top-K Retrieval with Similarity Thresholds

## Purpose
Control the quantity and quality of semantic search results by configuring the maximum number of results (top-k) and minimum similarity threshold.

## Interface

### Configuration
```yaml
# config/vector_store.yaml
vector_store:
  search:
    default_top_k: 10
    max_top_k: 100
    default_similarity_threshold: 0.0  # No filtering by default
    min_similarity_threshold: 0.0
    max_similarity_threshold: 1.0
```

### Service Interface
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchParams:
    top_k: int = 10
    min_similarity: float = 0.0  # 0.0 = no threshold, 1.0 = exact match only
    max_distance: Optional[float] = None  # Alternative: filter by distance

class VectorSearchService:
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 100
    DEFAULT_MIN_SIMILARITY: float = 0.0
    
    async def search(
        self,
        query_vector: list[float],
        params: SearchParams | None = None,
    ) -> SearchResults:
        """
        Execute vector search with configurable limits and thresholds.
        
        Args:
            query_vector: The embedding vector to search for
            params: Search parameters including top_k and similarity threshold
        """
        pass
```

### API Interface
```http
POST /api/v1/search/semantic
Content-Type: application/json

{
  "query_vector": [0.023, -0.156, 0.892, ...],
  "top_k": 20,
  "min_similarity": 0.75
}

# Response
{
  "success": true,
  "data": {
    "results": [...],
    "total_found": 45,
    "returned": 20,
    "min_similarity_applied": 0.75,
    "threshold_filtered": 25
  }
}
```

### SQL Interface
```sql
-- Top-k with similarity threshold (cosine)
SELECT id, content, metadata, 
       1 - (embedding <=> :query_vector) AS similarity
FROM document_chunks
WHERE 1 - (embedding <=> :query_vector) >= :min_similarity
ORDER BY embedding <=> :query_vector ASC
LIMIT :top_k;

-- Using CTE for efficient filtering
WITH scored_chunks AS (
    SELECT id, content, metadata, embedding,
           1 - (embedding <=> :query_vector) AS similarity
    FROM document_chunks
)
SELECT * FROM scored_chunks
WHERE similarity >= :min_similarity
ORDER BY similarity DESC
LIMIT :top_k;
```

## Behavior

### Top-K Parameter
- **Default**: 10 results
- **Minimum**: 1
- **Maximum**: 100 (configurable via `max_top_k`)
- **Effect**: Limits the number of results returned to the client

### Similarity Threshold
- **Default**: 0.0 (no filtering, all results returned)
- **Range**: 0.0 to 1.0
- **Interpretation**:
  - 0.0: Return all results regardless of similarity
  - 0.7: Return only results with 70% or higher similarity
  - 0.9: Return only highly similar results (90%+)
  - 1.0: Exact matches only (rarely useful for semantic search)

### Query Execution Flow
1. Validate `top_k` is within allowed range (1 to `max_top_k`)
2. Validate `min_similarity` is within [0.0, 1.0]
3. Execute vector search retrieving up to `top_k` results
4. Apply similarity threshold filter
5. If fewer than `top_k` results meet threshold, return only matching results
6. Include metadata about filtering in response

### Threshold Application
```python
# Pseudo-code for threshold application
raw_results = await db.fetch(
    "SELECT ... ORDER BY distance LIMIT :top_k",
    top_k=params.top_k * 2  # Fetch extra for threshold filtering
)

filtered = [
    r for r in raw_results 
    if r.similarity >= params.min_similarity
][:params.top_k]
```

### Dynamic Adjustment
- When `min_similarity` filters out too many results, consider:
  - Returning fewer results (current behavior)
  - Automatically relaxing threshold (with warning)
  - Suggesting alternative queries

## Error Handling

| Error Code | Condition | Response |
|------------|-----------|----------|
| `400 Bad Request` | `top_k` < 1 | `"top_k must be at least 1"` |
| `400 Bad Request` | `top_k` > `max_top_k` | `"top_k exceeds maximum allowed (100)"` |
| `400 Bad Request` | `min_similarity` < 0.0 or > 1.0 | `"min_similarity must be between 0.0 and 1.0"` |
| `400 Bad Request` | `min_similarity` = 1.0 with large dataset | `"Similarity threshold of 1.0 may return no results"` |
| `200 OK` | No results meet threshold | Return empty `results` array with `total_found: 0` |

### Warning Headers
```http
HTTP/1.1 200 OK
X-Search-Warning: threshold_filtered_90_percent
X-Original-Result-Count: 100
X-Filtered-Result-Count: 10
```

## Performance Considerations

### Threshold Optimization
- Similarity threshold filtering at database level is more efficient
- Use `WHERE` clause when possible to reduce data transfer
- For high thresholds (>0.8), consider using index hints

### Pagination Interaction
- `top_k` is applied BEFORE pagination
- Pagination parameters (`limit`, `offset`) are applied AFTER threshold filtering
- Example: `top_k=100`, `limit=10`, `offset=0` → return first 10 of 100 best matches

### Memory Usage
- Fetching large `top_k` values increases memory usage
- Streaming results for `top_k > 1000` (if supported)
- Consider server-side cursor for very large result sets

### Recommended Thresholds by Use Case
| Use Case | Recommended `top_k` | Recommended `min_similarity` |
|----------|--------------------|------------------------------|
| RAG/LLM context | 5-10 | 0.70-0.80 |
| Document deduplication | 1 | 0.95-0.99 |
| Semantic exploration | 20-50 | 0.50-0.60 |
| Clustering | 100 | 0.00-0.30 |
| Strict matching | 1-5 | 0.85-0.90 |

### Query Planning
- High `min_similarity` (>0.8) may benefit from different index strategies
- Monitor execution plans for threshold queries
- Consider materialized similarity ranges for frequently queried vectors
