# Spec: Weighted Scoring Algorithm

## Purpose
Combine vector similarity scores and text relevance scores into a single unified ranking using configurable weighted sums.

## Interface

### API Parameters
```json
{
  "query": "search query text",
  "vector_weight": 0.7,
  "text_weight": 0.3,
  "fusion_method": "weighted_sum",
  "top_k": 10
}
```

### Scoring Formula
```
final_score = (vector_weight × normalized_vector_score) + (text_weight × normalized_text_score)

Where:
- normalized_vector_score = 1 - (cosine_distance / 2)  // Maps [0,2] to [1,0], then to [0,1]
- normalized_text_score = ts_rank_cd / MAX(ts_rank_cd)  // Normalized to [0,1]
```

### Configuration
```yaml
# config/vector_store.yaml
hybrid_search:
  vector_weight: 0.7      # Default weight for vector similarity
  text_weight: 0.3        # Default weight for text relevance
  fusion_method: "weighted_sum"
```

## Behavior

### Score Normalization
1. **Vector Score Normalization**
   - pgvector cosine distance returns values in range [0, 2]
   - Convert to similarity: `similarity = 1 - (distance / 2)`
   - Result is in range [0, 1] where 1 = identical vectors

2. **Text Score Normalization**
   - PostgreSQL `ts_rank_cd` returns unbounded values
   - Normalize against maximum rank in result set: `normalized = rank / max_rank`
   - If all ranks are 0, set normalized_text_score = 0 for all

3. **Weight Validation**
   - Weights must satisfy: `vector_weight + text_weight = 1.0` (±0.001 tolerance)
   - Each weight must be in range [0.0, 1.0]
   - Invalid weights return HTTP 400 with validation error

### Final Score Calculation
```sql
WITH vector_results AS (
    SELECT 
        id,
        1 - (embedding <=> query_embedding) / 2 AS vector_score
    FROM document_chunks
    ORDER BY embedding <=> query_embedding
    LIMIT 100
),
text_results AS (
    SELECT 
        id,
        ts_rank_cd(search_vector, query_tsquery) AS text_rank,
        MAX(ts_rank_cd(search_vector, query_tsquery)) OVER () AS max_rank
    FROM document_chunks
    WHERE search_vector @@ query_tsquery
),
combined AS (
    SELECT 
        COALESCE(v.id, t.id) AS id,
        COALESCE(v.vector_score, 0) AS vector_score,
        COALESCE(t.text_rank / NULLIF(t.max_rank, 0), 0) AS text_score
    FROM vector_results v
    FULL OUTER JOIN text_results t ON v.id = t.id
)
SELECT 
    id,
    (:vector_weight * vector_score) + (:text_weight * text_score) AS final_score,
    vector_score,
    text_score
FROM combined
ORDER BY final_score DESC
LIMIT :top_k;
```

### Result Format
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "uuid",
        "content": "chunk text",
        "final_score": 0.85,
        "vector_score": 0.92,
        "text_score": 0.68,
        "metadata": {}
      }
    ],
    "weights_applied": {
      "vector_weight": 0.7,
      "text_weight": 0.3
    }
  }
}
```

## Error Handling

| Error Code | Condition | Response |
|------------|-----------|----------|
| 400 | `vector_weight + text_weight != 1.0` | `{"error": "Weights must sum to 1.0"}` |
| 400 | Weight outside [0, 1] range | `{"error": "Weights must be between 0.0 and 1.0"}` |
| 400 | Negative top_k | `{"error": "top_k must be positive"}` |
| 404 | No results from either search | `{"data": {"results": []}, "message": "No matching documents found"}` |
| 500 | Database normalization error (max_rank = 0) | Return vector scores only, log warning |

## Performance Considerations

### Query Optimization
- Execute vector and text searches as **parallel async queries**
- Use CTEs to materialize intermediate results
- Limit each sub-query to `max(top_k * 3, 100)` to reduce join overhead

### Index Requirements
- HNSW index on `embedding` column for fast vector search
- GIN index on `to_tsvector('english', content)` for text search
- B-tree index on `job_id` for metadata filtering

### Caching Strategy
- Cache normalized scores for repeated queries
- Store pre-computed `max_rank` per job for consistent normalization
- Use connection pooling for concurrent sub-queries

### Resource Limits
- Query timeout: 5 seconds per sub-query
- Maximum intermediate results: 1000 per search type
- Memory: Stream results rather than loading full result sets
