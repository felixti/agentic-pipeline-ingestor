# Spec: List Job Chunks

## Purpose
List all document chunks associated with a specific job with pagination and sorting support.

## Interface
- **Method**: GET
- **Path**: `/api/v1/jobs/{job_id}/chunks`
- **Query Parameters**:
  - `limit` (integer, optional): Number of results per page (default: 20, max: 100)
  - `offset` (integer, optional): Number of results to skip (default: 0)
  - `sort_by` (string, optional): Sort field (values: `created_at`, `chunk_index`, `updated_at`; default: `chunk_index`)
  - `sort_order` (string, optional): Sort direction (values: `asc`, `desc`; default: `asc`)

## Request Schema
No request body required.

## Response Schema
**200 OK**
```json
{
  "success": true,
  "data": {
    "chunks": [
      {
        "id": "uuid",
        "job_id": "uuid",
        "chunk_index": 0,
        "content": "string (truncated to first 500 chars)",
        "metadata": {
          "source_file": "string",
          "page_number": 1,
          "total_pages": 10
        },
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
      }
    ],
    "pagination": {
      "total": 100,
      "page": 1,
      "per_page": 20,
      "total_pages": 5,
      "has_next": true,
      "has_prev": false
    }
  },
  "error": null
}
```

## Behavior
1. Validate `job_id` is a valid UUID format
2. Verify the job exists in the database
3. Query `document_chunks` table filtered by `job_id`
4. Apply sorting based on `sort_by` and `sort_order` parameters
5. Apply pagination using `limit` and `offset`
6. Return chunk content truncated to 500 characters for preview
7. Exclude full embedding vector from list response (use GET /chunks/{chunk_id} for full data)
8. Return total count for pagination metadata

## Error Handling
- **400 Bad Request**: Invalid UUID format for `job_id`, or invalid query parameters
  ```json
  {
    "success": false,
    "data": null,
    "error": {
      "code": "VALIDATION_ERROR",
      "message": "Invalid job_id format",
      "details": { "field": "job_id", "value": "invalid-uuid" }
    }
  }
  ```
- **404 Not Found**: Job does not exist
  ```json
  {
    "success": false,
    "data": null,
    "error": {
      "code": "JOB_NOT_FOUND",
      "message": "Job with ID {job_id} not found"
    }
  }
  ```
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Database or unexpected error

## Rate Limiting
- **Limit**: 100 requests per minute per user/IP
- **Window**: 60 seconds
- **Key**: `list_chunks:{user_id_or_ip}`
