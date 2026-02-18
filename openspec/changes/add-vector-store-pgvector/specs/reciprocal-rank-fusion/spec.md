# Spec: Reciprocal Rank Fusion

## Purpose
Provide an alternative fusion method that combines vector and text search rankings using Reciprocal Rank Fusion (RRF) to avoid score normalization issues and provide robust ranking combination.

## Interface

### API Parameters
```json
{
  "query": "search query text",
  "fusion_method": "rrf",
  "rrf_k": 60,
  "top_k": 10
}
```

### RRF Scoring Formula
```
RRF_score(d) = sum(1 / (k + rank_i(d)))

Where:
- k = 60 (configurable constant, default)
- rank_i(d) = position of document d in ranking i (1-indexed)
- If document not in ranking i, contribution = 0
```

### Configuration
```yaml
# config/vector_store.yaml
hybrid_search:
  fusion_method: "rrf"    # or "weighted_sum"
  rrf_k: 60               # RRF constant (typically 10-100)
```

## Behavior

### RRF Algorithm
1. Execute vector similarity search, retrieve top N results
2. Execute text search, retrieve top N results
3. For each document appearing in either result set:
   - Calculate contribution from vector ranking: `1 / (k + vector_rank)`
   - Calculate contribution from text ranking: `1 / (k + text_rank)`
   - Sum contributions to get final RRF score
4. Sort by RRF score descending
5. Return top_k results

### Rank Assignment
```python
def assign_ranks(results):
    """Assign 1-indexed ranks to results"""
    return {doc_id: rank + 1 for rank, doc_id in enumerate(results)}

# Example
vector_results = ["doc_A", "doc_B", "doc_C"]  # ranks: A=1, B=2, C=3
text_results = ["doc_B", "doc_D", "doc_A"]    # ranks: B=1, D=2, A=3

# RRF calculation with k=60
# doc_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
# doc_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325  ← Winner
# doc_C: 1/(60+3) + 0          = 0.0159 + 0      = 0.0159
# doc_D: 0          + 1/(60+2) = 0      + 0.0161 = 0.0161
```

### SQL Implementation
```sql
WITH vector_ranked AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> query_embedding) AS rank
    FROM document_chunks
    ORDER BY embedding <=> query_embedding
    LIMIT 100
),
text_ranked AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(search_vector, query_tsquery) DESC) AS rank
    FROM document_chunks
    WHERE search_vector @@ query_tsquery
    ORDER BY ts_rank_cd(search_vector, query_tsquery) DESC
    LIMIT 100
),
rrf_scores AS (
    SELECT 
        COALESCE(v.id, t.id) AS id,
        COALESCE(1.0 / (:rrf_k + v.rank), 0) + 
        COALESCE(1.0 / (:rrf_k + t.rank), 0) AS rrf_score,
        v.rank AS vector_rank,
        t.rank AS text_rank
    FROM vector_ranked v
    FULL OUTER JOIN text_ranked t ON v.id = t.id
)
SELECT 
    id,
    rrf_score,
    vector_rank,
    text_rank
FROM rrf_scores
ORDER BY rrf_score DESC
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
        "rrf_score": 0.0325,
        "vector_rank": 2,
        "text_rank": 1,
        "metadata": {}
      }
    ],
    "fusion_method": "rrf",
    "rrf_k": 60
  }
}
```

### Comparison: Weighted Sum vs RRF

| Aspect | Weighted Sum | RRF |
|--------|--------------|-----|
| Score normalization | Required (can be tricky) | Not required |
| Parameter tuning | Weight values needed | Single k parameter |
| Handling missing docs | Needs zero-fill | Natural handling (no contribution) |
| Score distribution | Preserves magnitude differences | Rank-based only |
| Use case | When score quality matters | When rank stability matters |

## Error Handling

| Error Code | Condition | Response |
|------------|-----------|----------|
| 400 | `rrf_k < 1` | `{"error": "rrf_k must be at least 1"}` |
| 400 | `rrf_k > 1000` | `{"error": "rrf_k must not exceed 1000"}` |
| 400 | Invalid fusion_method | `{"error": "fusion_method must be 'weighted_sum' or 'rrf'"}` |
| 404 | Empty results from both searches | Return empty results array |
| 500 | Rank calculation error | Log error, return partial results if available |

## Performance Considerations

### RRF k Parameter Selection
- **k = 10-20**: Emphasizes top ranks heavily, sensitive to ranking changes
- **k = 60** (default): Balanced, commonly used in literature
- **k = 100+**: More lenient, documents lower in rankings contribute more

### Optimization Strategies
1. **Early Termination**: Stop rank calculation when score contribution < 0.001
2. **Parallel Execution**: Run vector and text queries concurrently
3. **Result Deduplication**: Use hash set for document IDs before ranking
4. **Limit Input Size**: Cap each search to `top_k * 5` results maximum

### Memory Management
- Store only document IDs and ranks in memory, not full content
- Fetch full chunk content only for final top_k results
- Use streaming for large intermediate result sets

### Benchmark Targets
- RRF fusion overhead: < 5ms for 200 combined results
- Total hybrid query: < 100ms p99
- Memory usage: < 10MB for 1000 intermediate results
