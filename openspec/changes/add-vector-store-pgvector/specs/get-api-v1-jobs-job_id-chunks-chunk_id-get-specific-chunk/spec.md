# Spec: Get Specific Chunk

## Purpose
Retrieve detailed information for a single document chunk including full content and embedding vector.

## Interface
- **Method**: GET
- **Path**: `/api/v1/jobs/{job_id}/chunks/{chunk_id}`
- **Query Parameters**:
  - `include_embedding` (boolean, optional): Include the full embedding vector in response (default: false)

## Request Schema
No request body required.

## Response Schema
**200 OK**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "job_id": "uuid",
    "chunk_index": 0,
    "content": "string (full content)",
    "embedding": [0.123, -0.456, ...],  // Only if include_embedding=true
    "metadata": {
      "source_file": "document.pdf",
      "page_number": 5,
      "total_pages": 20,
      "section": "Introduction",
      "tokens": 512,
      "char_count": 2048,
      "custom_tags": ["important", "summary"]
    },
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "error": null
}
```

## Behavior
1. Validate `job_id` and `chunk_id` are valid UUID formats
2. Verify the job exists in the database
3. Query `document_chunks` table for the specific chunk filtered by both `job_id` and `chunk_id`
4. Return full chunk content (not truncated)
5. Include embedding vector only if `include_embedding=true` (default false for performance)
6. Return complete metadata JSONB object
7. Include chunk index for ordering context

## Error Handling
- **400 Bad Request**: Invalid UUID format for `job_id` or `chunk_id`
  ```json
  {
    "success": false,
    "data": null,
    "error": {
      "code": "VALIDATION_ERROR",
      "message": "Invalid UUID format",
      "details": { "field": "chunk_id" }
    }
  }
  ```
- **404 Not Found**: Job or chunk does not exist
  ```json
  {
    "success": false,
    "data": null,
    "error": {
      "code": "CHUNK_NOT_FOUND",
      "message": "Chunk with ID {chunk_id} not found for job {job_id}"
    }
  }
  ```
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Database or unexpected error

## Rate Limiting
- **Limit**: 200 requests per minute per user/IP
- **Window**: 60 seconds
- **Key**: `get_chunk:{user_id_or_ip}`
