# Spec: Get Chunk by UUID Endpoint

## Purpose
Retrieve a specific document chunk by its unique UUID identifier, returning complete chunk data including content, metadata, and embedding.

## Interface

**Endpoint:** `GET /api/v1/jobs/{job_id}/chunks/{chunk_id}`

**Path Parameters:**
- `job_id` (UUID, required): The parent job identifier
- `chunk_id` (UUID, required): The unique chunk identifier

**Query Parameters:**
- `include_embedding` (boolean, optional): Whether to include the vector embedding in the response. Default: `false`

## Request/Response

### Request Example
```http
GET /api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/chunks/6ba7b810-9dad-11d1-80b4-00c04fd430c8?include_embedding=true
```

### Response Schema (200 OK)
```json
{
  "success": true,
  "data": {
    "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "chunk_index": 3,
    "content": "The quick brown fox jumps over the lazy dog. This is a sample chunk of processed document content.",
    "content_hash": "a3f5c8e9d2b1...",
    "embedding": [0.023, -0.045, 0.178, ...],
    "metadata": {
      "source_page": 1,
      "char_offset": 256,
      "token_count": 128
    },
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### Response Schema (without embedding)
```json
{
  "success": true,
  "data": {
    "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "chunk_index": 3,
    "content": "The quick brown fox jumps over the lazy dog...",
    "content_hash": "a3f5c8e9d2b1...",
    "metadata": {
      "source_page": 1,
      "char_offset": 256,
      "token_count": 128
    },
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

## Behavior

1. **Validation:**
   - Validate `job_id` and `chunk_id` are valid UUID v4 format
   - Verify the chunk belongs to the specified job (security check)

2. **Database Query:**
   - Query the `document_chunks` table by primary key
   - Join with `jobs` table to verify ownership
   - Conditionally select embedding column based on `include_embedding` parameter

3. **Response Construction:**
   - Return full chunk data with all metadata fields
   - Omit `embedding` array by default to reduce payload size
   - Include `content_hash` for deduplication verification

4. **Performance Considerations:**
   - Primary key lookup is O(1) with index
   - Embedding retrieval adds ~6KB-12KB per chunk (1536 dims × 4 bytes)
   - Set response timeout to 5 seconds for embedding inclusion

## Error Handling

| Status | Error Code | Message | Condition |
|--------|------------|---------|-----------|
| 400 | `INVALID_UUID` | "Invalid UUID format for chunk_id" | Malformed UUID parameter |
| 400 | `INVALID_UUID` | "Invalid UUID format for job_id" | Malformed job_id parameter |
| 404 | `CHUNK_NOT_FOUND` | "Chunk not found" | Chunk ID does not exist |
| 404 | `JOB_NOT_FOUND` | "Job not found" | Job ID does not exist |
| 403 | `ACCESS_DENIED` | "Chunk does not belong to specified job" | Chunk exists but under different job |
| 500 | `INTERNAL_ERROR` | "Failed to retrieve chunk" | Database or unexpected error |

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "CHUNK_NOT_FOUND",
    "message": "Chunk not found",
    "details": {
      "chunk_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    }
  }
}
```
