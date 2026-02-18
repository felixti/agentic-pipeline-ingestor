# Spec: Fallback Strategies When One Search Type Returns No Results

## Purpose
Handle edge cases gracefully when either vector similarity search or text search returns empty results, ensuring users always receive meaningful search results.

## Interface

### API Parameters
```json
{
  "query": "search query",
  "fallback_mode": "auto",
  "min_vector_results": 3,
  "min_text_results": 3,
  "top_k": 10
}
```

### Fallback Modes
| Mode | Description |
|------|-------------|
| `auto` | Automatically detect and apply appropriate fallback (default) |
| `strict` | Return empty if either search type returns insufficient results |
| `vector_only` | Always use vector search, ignore text results |
| `text_only` | Always use text search, ignore vector results |
| `require_both` | Require results from both search types, no fallback |

### Configuration
```yaml
# config/vector_store.yaml
hybrid_search:
  fallback:
    mode: "auto"                    # Default fallback behavior
    min_results_threshold: 3        # Minimum results to consider search "successful"
    expand_query_on_empty: true     # Enable query expansion for empty results
    expansion_limit: 50             # Max additional results from expansion
  
  # Thresholds for auto fallback decisions
  thresholds:
    vector_similarity_min: 0.5      # Minimum cosine similarity to include
    text_rank_min: 0.01             # Minimum ts_rank to consider relevant
```

## Behavior

### Auto Fallback Logic
```python
async def hybrid_search_with_fallback(query, fallback_mode="auto"):
    # Execute both searches in parallel
    vector_results = await vector_search(query)
    text_results = await text_search(query)
    
    # Count valid results
    vector_count = count_valid_results(vector_results, min_similarity=0.5)
    text_count = count_valid_results(text_results, min_rank=0.01)
    
    if fallback_mode == "auto":
        return apply_auto_fallback(vector_results, text_results, vector_count, text_count)
    elif fallback_mode == "strict":
        if vector_count < MIN_THRESHOLD or text_count < MIN_THRESHOLD:
            return {"results": [], "fallback_applied": None, "reason": "insufficient_results"}
        return merge_results(vector_results, text_results)
    # ... other modes

def apply_auto_fallback(vector_results, text_results, vector_count, text_count):
    """
    Auto fallback decision tree:
    1. Both have results → normal hybrid merge
    2. Only vector has results → return vector results with warning
    3. Only text has results → return text results with warning  
    4. Neither has results → try query expansion or return empty
    """
    
    # Case 1: Both searches returned sufficient results
    if vector_count >= MIN_THRESHOLD and text_count >= MIN_THRESHOLD:
        return {
            "results": merge_results(vector_results, text_results),
            "fallback_applied": None,
            "vector_count": vector_count,
            "text_count": text_count
        }
    
    # Case 2: Only vector has results
    if vector_count >= MIN_THRESHOLD and text_count < MIN_THRESHOLD:
        return {
            "results": vector_results[:top_k],
            "fallback_applied": "vector_only",
            "reason": f"Text search returned only {text_count} results (min: {MIN_THRESHOLD})",
            "warning": "Results based on semantic similarity only"
        }
    
    # Case 3: Only text has results
    if vector_count < MIN_THRESHOLD and text_count >= MIN_THRESHOLD:
        return {
            "results": text_results[:top_k],
            "fallback_applied": "text_only",
            "reason": f"Vector search returned only {vector_count} results (min: {MIN_THRESHOLD})",
            "warning": "Results based on text matching only"
        }
    
    # Case 4: Neither has sufficient results
    return handle_empty_results(query)
```

### Query Expansion Fallback
When both searches return empty results:
1. Try fuzzy matching with `pg_trgm` similarity
2. Remove stop words and retry
3. Expand with synonyms (if configured)
4. Return "no results" with suggestions

```sql
-- Fuzzy fallback using trigram similarity
SELECT id, content, similarity(content, query) AS sml
FROM document_chunks
WHERE content % query  -- trigram similarity operator
ORDER BY sml DESC
LIMIT 10;
```

### Result Format with Fallback
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "uuid",
        "content": "chunk text",
        "score": 0.92,
        "source": "vector"
      }
    ],
    "fallback_applied": "vector_only",
    "fallback_reason": "Text search returned only 1 results (min: 3)",
    "warning": "Results based on semantic similarity only. Try adding more specific keywords for better text matching.",
    "search_metadata": {
      "vector_results_found": 25,
      "text_results_found": 1,
      "original_query": "xyzabc obscure term",
      "query_expanded": false
    }
  }
}
```

### Empty Results Response
```json
{
  "success": true,
  "data": {
    "results": [],
    "fallback_applied": null,
    "message": "No matching documents found",
    "suggestions": [
      "Try using more general terms",
      "Check your spelling",
      "Remove filters to broaden search"
    ]
  }
}
```

## Error Handling

| Error Code | Condition | Response |
|------------|-----------|----------|
| 400 | Invalid fallback_mode | `{"error": "Invalid fallback_mode 'hybrid'", "valid_modes": ["auto", "strict", "vector_only", "text_only", "require_both"]}` |
| 400 | Negative min_results | `{"error": "min_vector_results must be non-negative"}` |
| 200 | Strict mode with insufficient results | `{"data": {"results": []}, "reason": "strict_mode_insufficient_results"}` |
| 200 | Query expansion failed | Return empty results with suggestions |

## Performance Considerations

### Fallback Execution Flow
1. **Parallel Phase**: Execute vector + text searches concurrently
2. **Decision Phase**: Evaluate results (< 1ms)
3. **Fallback Phase** (if needed): 
   - No additional queries for `vector_only`/`text_only` modes
   - One additional fuzzy query for expansion case
4. **Total Budget**: 3 queries maximum (hybrid + optional expansion)

### Timeout Handling
```python
# Per-search timeouts
VECTOR_SEARCH_TIMEOUT = 3  # seconds
TEXT_SEARCH_TIMEOUT = 2    # seconds
FALLBACK_TIMEOUT = 2       # seconds

# If vector search times out, fall back to text-only
# If text search times out, fall back to vector-only
# If both timeout, return service unavailable
```

### Monitoring & Alerting
Track fallback frequency to identify issues:
```
metric: hybrid_search_fallback_total{mode="vector_only"}
metric: hybrid_search_fallback_total{mode="text_only"}
metric: hybrid_search_fallback_total{mode="empty_expanded"}
metric: hybrid_search_fallback_total{mode="empty_final"}

alert: fallback_rate > 10% for 5 minutes
```

### Resource Limits
- Maximum expansion attempts: 2
- Maximum additional results from expansion: 50
- Query expansion timeout: 1 second

## Common Scenarios

### Scenario 1: Niche Technical Term
```
Query: "pgvector hnsw ef_construction parameter"
Vector: 15 results (good semantic matches)
Text: 2 results (exact term rare)
Fallback: vector_only applied
```

### Scenario 2: Typo in Query
```
Query: "machine learnning" (typo)
Vector: 20 results (semantic similarity catches meaning)
Text: 0 results (typo prevents match)
Fallback: vector_only applied, suggest correction
```

### Scenario 3: Exact Match Requirement
```
Query: "RFC 7231 section 6.5.3"
Vector: 5 results (generic HTTP content)
Text: 1 result (exact RFC match)
Fallback: text_only applied
```

### Scenario 4: Both Searches Empty
```
Query: "xyz123 nonexistent term"
Vector: 0 results
Text: 0 results
Fallback: fuzzy search attempted → still empty
Result: Empty with suggestions
```
