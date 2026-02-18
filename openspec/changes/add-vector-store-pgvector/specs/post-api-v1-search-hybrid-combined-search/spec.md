# Spec: Hybrid Search (Combined Vector + Text)

## Purpose
Combine semantic vector similarity with full-text relevance scoring for optimal search results.

## Interface
- **Method**: POST
- **Path**: `/api/v1/search/hybrid`
- **Content-Type**: `application/json`

## Request Schema
```json
{
  "query_text": "string (required, max 4096 chars)",
  "top_k": "integer (optional, default: 10, max: 100, min: 1)",
  "vector_weight": "number (optional, default: 0.7, min: 0.0, max: 1.0)",
  "text_weight": "number (optional, default: 0.3, min: 0.0, max: 1.0)",
  "fusion_method": "string (optional, default: 'weighted_sum', values: 'weighted_sum', 'rrf')",
  "rrf_k": "integer (optional, default: 60, min: 1, for RRF fusion)",
  "similarity_threshold": "number (optional, default: 0.5, min: 0.0, max: 1.0)",
  "language": "string (optional, default: 'english')",
  "highlight": "boolean (optional, default: true)",
  "metadata_filter": {
    "job_id": "uuid (optional)",
    "source_file": "string (optional)",
    "date_from": "ISO 8601 datetime (optional)",
    "date_to": "ISO 8601 datetime (optional)",
    "custom_fields": "object (optional)"
  }
}
```

## Response Schema
**200 OK**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "chunk_id": "uuid",
        "job_id": "uuid",
        "chunk_index": 0,
        "content": "string (first 500 chars)",
        "content_highlighted": "string with <mark>highlights</mark>",
        "combined_score": 0.8234,
        "vector_score": 0.9123,
        "text_score": 0.6543,
        "vector_rank": 2,
        "text_rank": 5,
        "metadata": {
          "source_file": "document.pdf",
          "page_number": 5
        },
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "total_results": 25,
    "fusion_method": "weighted_sum",
    "weights_applied": {
      "vector": 0.7,
      "text": 0.3
    },
    "query_embedding_time_ms": 245,
    "vector_search_time_ms": 12,
    "text_search_time_ms": 8,
    "fusion_time_ms": 2,
    "total_time_ms": 267
  },
  "error": null
}
```

## Behavior
1. Validate request body using Pydantic validators
2. Execute vector similarity search (see semantic search spec)
3. Execute full-text search (see text search spec)
4. Combine results using specified fusion method
5. Apply weights and return fused ranking
6. Include individual scores for transparency
7. Return results with both vector and text rank information

### Fusion Methods

#### Weighted Sum (default)
```
combined_score = (vector_weight × vector_score) + (text_weight × text_score)
```
- Normalizes scores to [0, 1] range before weighting
- Weights must sum to 1.0 (auto-normalized if not)
- Best for balanced relevance scenarios

#### Reciprocal Rank Fusion (RRF)
```
rrf_score = Σ(1 / (k + rank))
combined_score = rrf_vector_score + rrf_text_score
```
- `rrf_k` parameter prevents dominant top results (default: 60)
- No weight parameters needed
- Best when vector and text rankings differ significantly

### Fallback Strategy
If one search type returns no results:
- If `vector_weight > 0.5`, fallback to text-only
- If `text_weight > 0.5`, fallback to vector-only
- Log warning for monitoring

## Error Handling
- **400 Bad Request**: Invalid request body or parameters
  ```json
  {
    "success": false,
    "data": null,
    "error": {
      "code": "VALIDATION_ERROR",
      "message": "Invalid search parameters",
      "details": [
        { "field": "vector_weight", "error": "Must be between 0.0 and 1.0" },
        { "field": "fusion_method", "error": "Invalid value, expected 'weighted_sum' or 'rrf'" }
      ]
    }
  }
  ```
- **422 Unprocessable Entity**: Embedding generation failed
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Database or unexpected error

## Rate Limiting
- **Limit**: 20 requests per minute per user/IP
- **Window**: 60 seconds
- **Key**: `search_hybrid:{user_id_or_ip}`
- **Burst**: Allow 3 requests immediately
- **Note**: Hybrid search consumes both vector and text search quotas
