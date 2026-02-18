# Spec: List Chunks by Job ID with Pagination

## Purpose
List all document chunks belonging to a specific job with pagination support, ordered by chunk index for consistent traversal.

## Interface

**Endpoint:** `GET /api/v1/jobs/{job_id}/chunks`

**Path Parameters:**
- `job_id` (UUID, required): The job identifier to list chunks for

**Query Parameters:**
- `limit` (integer, optional): Number of chunks to return per page. Default: `20`, Max: `100`, Min: `1`
- `offset` (integer, optional): Number of chunks to skip for pagination. Default: `0`, Min: `0`
- `include_embeddings` (boolean, optional): Whether to include embeddings. Default: `false`

## Request/Response

### Request Example
```http
GET /api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/chunks?limit=10&offset=20
```

### Response Schema (200 OK)
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "chunk_index": 2,
        "content": "First chunk content here...",
        "content_hash": "a3f5c8e9d2b1...",
        "metadata": {
          "source_page": 1,
          "char_offset": 0,
          "token_count": 128
        },
        "created_at": "2024-01-15T10:30:00Z"
      },
      {
        "id": "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "chunk_index": 3,
        "content": "Second chunk content here...",
        "content_hash": "b4g6d9f0e3c2...",
        "metadata": {
          "source_page": 1,
          "char_offset": 512,
          "token_count": 128
        },
        "created_at": "2024-01-15T10:30:01Z"
      }
    ],
    "pagination": {
      "total": 156,
      "limit": 10,
      "offset": 20,
      "has_more": true,
      "next_offset": 30
    }
  }
}
```

## Behavior

1. **Parameter Validation:**
   - `limit` clamped to valid range [1, 100]
   - `offset` must be non-negative integer
   - `job_id` validated as proper UUID v4 format

2. **Database Query:**
   - Filter by `job_id` with index on `document_chunks.job_id`
   - Order results by `chunk_index ASC` for document sequence preservation
   - Use `LIMIT` and `OFFSET` for pagination
   - Execute `COUNT(*)` query for total count (cached for 30 seconds)

3. **Ordering Guarantee:**
   - Results always ordered by `chunk_index` ascending
   - This preserves document reading order for reconstruction
   - Gaps in chunk_index are preserved (deleted chunks leave gaps)

4. **Pagination Strategy:**
   - Offset-based pagination for simplicity
   - Include `total` count for progress indicators
   - `has_more` flag indicates if more results exist
   - `next_offset` calculated as `offset + limit`

5. **Performance Considerations:**
   - Composite index on `(job_id, chunk_index)` for efficient filtering and sorting
   - COUNT query cached to reduce overhead
   - Query timeout: 10 seconds
   - Large offset (>10000) triggers warning in logs

## Error Handling

| Status | Error Code | Message | Condition |
|--------|------------|---------|-----------|
| 400 | `INVALID_UUID` | "Invalid UUID format for job_id" | Malformed job_id parameter |
| 400 | `INVALID_PARAMETER` | "Limit must be between 1 and 100" | Limit out of valid range |
| 400 | `INVALID_PARAMETER` | "Offset must be non-negative" | Negative offset provided |
| 404 | `JOB_NOT_FOUND` | "Job not found" | Job ID does not exist |
| 500 | `INTERNAL_ERROR` | "Failed to list chunks" | Database or unexpected error |

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Limit must be between 1 and 100",
    "details": {
      "parameter": "limit",
      "provided": 500,
      "max_allowed": 100
    }
  }
}
```
