# Spec: Semantic Vector Search

## Purpose
Perform semantic similarity search using vector embeddings to find chunks most similar to a query text.

## Interface
- **Method**: POST
- **Path**: `/api/v1/search/semantic`
- **Content-Type**: `application/json`

## Request Schema
```json
{
  "query_text": "string (required, max 4096 chars)",
  "top_k": "integer (optional, default: 10, max: 100, min: 1)",
  "similarity_threshold": "number (optional, default: 0.7, min: 0.0, max: 1.0)",
  "metadata_filter": {
    "job_id": "uuid (optional)",
    "source_file": "string (optional)",
    "date_from": "ISO 8601 datetime (optional)",
    "date_to": "ISO 8601 datetime (optional)",
    "custom_fields": "object (optional, key-value pairs for JSONB filtering)"
  },
  "embedding_model": "string (optional, uses default if not specified)"
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
        "content": "string (first 1000 chars with ellipsis if truncated)",
        "similarity_score": 0.9234,
        "metadata": {
          "source_file": "document.pdf",
          "page_number": 5
        },
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "total_results": 25,
    "query_embedding_time_ms": 245,
    "search_time_ms": 12,
    "total_time_ms": 257
  },
  "error": null
}
```

## Behavior
1. Validate request body using Pydantic validators
2. Generate embedding for `query_text` using configured embedding model
3. Execute cosine similarity search using pgvector `<=>` operator
4. Filter results by optional `metadata_filter` conditions
5. Apply `similarity_threshold` to exclude low-similarity matches
6. Return top `top_k` results sorted by similarity score (descending)
7. Include timing metrics for performance monitoring
8. Content truncated to 1000 characters for preview

### Metadata Filter Support
- `job_id`: Exact match on job UUID
- `source_file`: Exact match on source filename
- `date_from`/`date_to`: Range filter on `created_at`
- `custom_fields`: JSONB containment queries for custom metadata

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
        { "field": "query_text", "error": "Field required" },
        { "field": "top_k", "error": "Must be between 1 and 100" }
      ]
    }
  }
  ```
- **422 Unprocessable Entity**: Embedding generation failed
  ```json
  {
    "success": false,
    "data": null,
    "error": {
      "code": "EMBEDDING_ERROR",
      "message": "Failed to generate embedding for query"
    }
  }
  ```
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Database or unexpected error

## Rate Limiting
- **Limit**: 30 requests per minute per user/IP
- **Window**: 60 seconds
- **Key**: `search_semantic:{user_id_or_ip}`
- **Burst**: Allow 5 requests immediately, then apply limit
