# Spec: Cosine Similarity Queries Using pgvector Operators

## Purpose
Enable semantic similarity search by computing cosine distance between query vectors and stored embeddings using pgvector's native operators.

## Interface

### SQL Interface
```sql
-- Cosine distance operator (<=>)
SELECT id, content, metadata, embedding <=> :query_vector AS distance
FROM document_chunks
ORDER BY embedding <=> :query_vector ASC
LIMIT :top_k;

-- Cosine similarity (1 - distance)
SELECT id, content, metadata, 1 - (embedding <=> :query_vector) AS similarity
FROM document_chunks
ORDER BY similarity DESC
LIMIT :top_k;
```

### Python Service Interface
```python
from typing import List, Tuple
import numpy as np

class VectorSearchService:
    async def search_by_cosine_similarity(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int = 10,
        job_id: str | None = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for chunks by cosine similarity.
        
        Returns list of (chunk, similarity_score) tuples ordered by 
        similarity descending (highest similarity first).
        """
        pass
```

### API Interface
```http
POST /api/v1/search/semantic
Content-Type: application/json

{
  "query_vector": [0.023, -0.156, 0.892, ...],
  "top_k": 10,
  "metric": "cosine"
}
```

## Behavior

### Distance Metric Selection
1. **Cosine Distance** (`<=>`): Default metric for semantic search
   - Measures angle between vectors, normalized by magnitude
   - Range: 0 (identical) to 2 (opposite)
   - Returns: `distance` where lower is more similar

2. **Negative Inner Product** (`<#>`): Alternative for normalized embeddings
   - Use when vectors are already L2-normalized
   - Slightly faster than cosine distance

3. **L2/Euclidean Distance** (`<->`): When magnitude matters
   - Use for models where vector magnitude carries semantic meaning
   - Range: 0 (identical) to unbounded

### Query Execution Flow
1. Validate query vector dimensions match configured embedding model
2. Convert query vector to pgvector-compatible format
3. Execute similarity query with HNSW index if available
4. Return results ordered by similarity (highest first)
5. Include similarity score (0.0 to 1.0) in response

### Similarity Score Normalization
- Raw distance values are converted to similarity scores:
  - Cosine: `similarity = 1 - distance` (range: -1 to 1, typically 0 to 1 for positive embeddings)
- Scores are clamped to [0.0, 1.0] range in API responses

### Result Ordering
- Results ordered by `distance ASC` (cosine) or `similarity DESC`
- Ties broken by `created_at DESC` (newest chunks first)

## Error Handling

| Error Code | Condition | Response |
|------------|-----------|----------|
| `400 Bad Request` | Dimension mismatch between query vector and stored embeddings | `"Query vector dimension 768 does not match expected 1536"` |
| `400 Bad Request` | Invalid vector format (NaN, Inf values) | `"Invalid vector: contains NaN or infinite values"` |
| `400 Bad Request` | Empty query vector | `"Query vector cannot be empty"` |
| `422 Unprocessable` | pgvector extension not enabled | `"Vector search requires pgvector extension"` |
| `500 Internal Error` | Database query failure | `"Vector search failed: {details}"` |

### Dimension Validation
- Store expected dimensions in `config.vector_store.dimensions`
- Validate at query time: `len(query_vector) == expected_dimensions`
- Log mismatch with sample of stored embedding dimensions for debugging

## Performance Considerations

### Index Usage
- **Required**: HNSW index on `embedding` column with `vector_cosine_ops`
- Query plan must show `Index Scan using idx_document_chunks_embedding`
- Without index: sequential scan with O(n) complexity - unacceptable for production

### Index Configuration
```sql
-- HNSW index for cosine similarity
CREATE INDEX idx_document_chunks_embedding_cosine
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Query Optimization
- Use `SET hnsw.ef_search = 32` for query-time accuracy vs speed tradeoff
- Higher `ef_search` = better recall, slower queries
- Default `ef_search = 32` provides ~95% recall for most datasets

### Performance Targets
| Dataset Size | Target Latency (p99) | Minimum Recall |
|--------------|---------------------|----------------|
| 10K chunks | < 10ms | 99% |
| 100K chunks | < 25ms | 97% |
| 1M chunks | < 100ms | 95% |
| 10M+ chunks | < 500ms | 90% |

### Monitoring
- Log query execution time and result count
- Track index hit ratio via `pg_stat_user_indexes`
- Alert on sequential scans exceeding threshold
