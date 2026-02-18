# Spec: Include Source Document Context in Responses

## Purpose
Enrich chunk responses with parent job (source document) metadata via optional query parameter, enabling clients to understand document provenance without additional API calls.

## Interface

**Applicable Endpoints:**
- `GET /api/v1/jobs/{job_id}/chunks/{chunk_id}`
- `GET /api/v1/jobs/{job_id}/chunks`
- `POST /api/v1/chunks/bulk`
- `GET /api/v1/chunks/{chunk_id}` (if global endpoint exists)

**Query Parameter:**
- `include_source` (boolean, optional): Include parent job metadata in response. Default: `false`

## Request/Response

### Request Example
```http
GET /api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/chunks/6ba7b810-9dad-11d1-80b4-00c04fd430c8?include_source=true
```

### Response Schema (200 OK) - Single Chunk
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
    "source": {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "job_name": "quarterly_report_2024.pdf",
      "job_type": "document_processing",
      "status": "completed",
      "file_name": "quarterly_report_2024.pdf",
      "file_type": "application/pdf",
      "file_size": 2457600,
      "created_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T10:30:00Z",
      "total_chunks": 42
    },
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### Response Schema (200 OK) - List Endpoint
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "chunk_index": 3,
        "content": "The quick brown fox jumps over the lazy dog...",
        "content_hash": "a3f5c8e9d2b1...",
        "metadata": { ... },
        "source": {
          "job_id": "550e8400-e29b-41d4-a716-446655440000",
          "job_name": "quarterly_report_2024.pdf",
          "job_type": "document_processing",
          "status": "completed",
          "file_name": "quarterly_report_2024.pdf",
          "file_type": "application/pdf",
          "file_size": 2457600,
          "created_at": "2024-01-15T10:00:00Z",
          "completed_at": "2024-01-15T10:30:00Z",
          "total_chunks": 42
        },
        "created_at": "2024-01-15T10:30:00Z"
      }
    ],
    "pagination": {
      "total": 156,
      "limit": 20,
      "offset": 0,
      "has_more": true
    }
  }
}
```

### Source Object Schema
```json
{
  "source": {
    "job_id": "UUID of the parent job",
    "job_name": "Human-readable job name",
    "job_type": "Type of processing job (e.g., document_processing)",
    "status": "Job status (pending, processing, completed, failed)",
    "file_name": "Original file name",
    "file_type": "MIME type of source file",
    "file_size": "Size in bytes (integer)",
    "created_at": "ISO 8601 timestamp when job was created",
    "completed_at": "ISO 8601 timestamp when job completed (null if not completed)",
    "total_chunks": "Total number of chunks for this job (integer)"
  }
}
```

## Behavior

1. **Parameter Processing:**
   - Parse `include_source` as boolean (accept "true", "1", "yes")
   - Default to `false` for backward compatibility
   - Reject non-boolean values with 400 error

2. **Database Query:**
   - Perform JOIN with `jobs` table when `include_source=true`
   - Select specific job columns (avoid sensitive data)
   - Use efficient query plan with proper indexing

3. **Source Metadata Assembly:**
   - Query `jobs` table for parent job information
   - Include computed `total_chunks` count via subquery
   - Omit internal fields (config, error_details, etc.)

4. **Performance Optimization:**
   - Cache `total_chunks` per job for 60 seconds
   - Use `SELECT` with specific columns (avoid `SELECT *`)
   - Foreign key index on `document_chunks.job_id` ensures fast JOIN

5. **Response Structure:**
   - Add `source` object at root of each chunk object
   - Maintain consistent schema across all endpoints
   - `source` object identical structure regardless of endpoint

6. **Null Handling:**
   - If job is deleted (cascade), chunk won't exist
   - `completed_at` is null for pending/processing jobs
   - `total_chunks` may be 0 for failed jobs

## Error Handling

| Status | Error Code | Message | Condition |
|--------|------------|---------|-----------|
| 400 | `INVALID_PARAMETER` | "include_source must be a boolean" | Non-boolean value provided |
| 404 | `JOB_NOT_FOUND` | "Source job not found" | Job deleted after chunk query (race condition) |
| 500 | `INTERNAL_ERROR` | "Failed to include source context" | Database or unexpected error |

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "include_source must be a boolean",
    "details": {
      "parameter": "include_source",
      "provided": "maybe",
      "expected": "true, false, 1, or 0"
    }
  }
}
```

## Performance Considerations

| Scenario | Impact | Optimization |
|----------|--------|--------------|
| Single chunk lookup | +1 JOIN, minimal impact | Primary key lookup on jobs |
| List chunks (20 items) | +1 JOIN, +20 row lookups | Batch fetch job data |
| Bulk retrieval (100 items) | +1 JOIN, shared job data | Cache job data per request |
| All chunks same job | Single job lookup | Deduplicate job IDs before query |

## Use Cases

1. **RAG Applications:** Show source document name alongside retrieved chunks
2. **Audit Trails:** Display job information for compliance reporting
3. **Search Results:** Include document metadata in search result listings
4. **Chunk Verification:** Confirm chunk belongs to expected source document
5. **UI Display:** Show file name and type in chunk browsing interfaces

## Backward Compatibility

- Default `include_source=false` maintains existing response format
- Existing clients unaffected by this feature
- New clients opt-in by setting query parameter
- No breaking changes to response structure when disabled
