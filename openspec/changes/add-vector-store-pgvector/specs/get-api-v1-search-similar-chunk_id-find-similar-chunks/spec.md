# Spec: Find Similar Chunks

## Purpose
Find chunks that are semantically similar to a given reference chunk, useful for finding related content.

## Interface
- **Method**: GET
- **Path**: `/api/v1/search/similar/{chunk_id}`
- **Query Parameters**:
  - `top_k` (integer, optional): Number of similar chunks to return (default: 10, max: 50, min: 1)
  - `exclude_self` (boolean, optional): Exclude the reference chunk from results (default: true)
  - `same_job_only` (boolean, optional): Only return chunks from the same job (default: false)
  - `similarity_threshold` (number, optional): Minimum similarity score (default: 0.7, min: 0.0, max: 1.0)

## Request Schema
No request body required.

## Response Schema
**200 OK**
```json
{
  "success": true,
  "data": {
    "reference_chunk": {
      "id": "uuid",
      "job_id": "uuid",
      "chunk_index": 0,
      "content_preview": "string (first 200 chars)"
    },
    "similar_chunks": [
      {
        "chunk_id": "uuid",
        "job_id": "uuid",
        "chunk_index": 5,
        "content": "string (first 500 chars)",
        "similarity_score": 0.8912,
        "similarity_percentage": 89.12,
        "metadata": {
          "source_file": "document.pdf",
          "page_number": 10
        },
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "total_similar": 15,
    "search_time_ms": 15,
    "embedding_dimensions": 1536
  },
  "error": null
}
```

## Behavior
1. Validate `chunk_id` is a valid UUID format
2. Retrieve the reference chunk and its embedding vector
3. Execute vector similarity search using the reference embedding as query
4. Filter by `same_job_only` constraint if specified
5. Exclude self if `exclude_self=true` (default)
6. Apply `similarity_threshold` filter
7. Return top `top_k` most similar chunks
8. Include similarity as both raw score and percentage

### Similarity Calculation
- Uses cosine similarity via pgvector `<=>` operator
- Score range: 0.0 to 1.0 (1.0 = identical)
- Converted to percentage for user-friendly display

### Performance Optimization
- Uses HNSW index for fast approximate nearest neighbor search
- Query-time `ef_search` parameter configurable (default: 64)
- Caches reference chunk embedding for repeated calls

## Error Handling
- **400 Bad Request**: Invalid UUID format for `chunk_id` or invalid query parameters
  ```json
  {
    "success": false,
    "data": null,
    "error": {
      "code": "VALIDATION_ERROR",
      "message": "Invalid chunk_id format",
      "details": { "field": "chunk_id" }
    }
  }
  ```
- **404 Not Found**: Reference chunk does not exist
  ```json
  {
    "success": false,
    "data": null,
    "error": {
      "code": "CHUNK_NOT_FOUND",
      "message": "Reference chunk with ID {chunk_id} not found"
    }
  }
  ```
- **422 Unprocessable Entity**: Reference chunk has no embedding (not vectorized)
  ```json
  {
    "success": false,
    "data": null,
    "error": {
      "code": "NO_EMBEDDING",
      "message": "Reference chunk does not have an embedding vector"
    }
  }
  ```
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Database or unexpected error

## Rate Limiting
- **Limit**: 40 requests per minute per user/IP
- **Window**: 60 seconds
- **Key**: `search_similar:{user_id_or_ip}`
- **Burst**: Allow 5 requests immediately
