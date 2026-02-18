# Spec: Get Chunks by Content Hash for Deduplication

## Purpose
Find document chunks matching a specific content hash to enable content deduplication across jobs and detect duplicate document processing.

## Interface

**Endpoint:** `GET /api/v1/chunks/by-hash/{content_hash}`

**Alternative Endpoint:** `GET /api/v1/chunks?content_hash={hash}`

**Path/Query Parameters:**
- `content_hash` (string, required): SHA-256 hash of chunk content (hex-encoded, 64 characters)

**Query Parameters:**
- `job_id` (UUID, optional): Filter results to specific job
- `limit` (integer, optional): Max results to return. Default: `50`, Max: `200`
- `include_duplicates_only` (boolean, optional): Only return if count > 1. Default: `false`

## Request/Response

### Request Example
```http
GET /api/v1/chunks/by-hash/a3f5c8e9d2b1f6e4a7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6
```

### Response Schema (200 OK)
```json
{
  "success": true,
  "data": {
    "content_hash": "a3f5c8e9d2b1f6e4a7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6",
    "match_count": 3,
    "is_duplicate": true,
    "chunks": [
      {
        "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "job_name": "quarterly_report_2024.pdf",
        "chunk_index": 5,
        "content_preview": "The quick brown fox jumps over the lazy dog...",
        "metadata": {
          "source_page": 2,
          "token_count": 128
        },
        "created_at": "2024-01-15T10:30:00Z"
      },
      {
        "id": "7cb8c921-aead-22e2-91c5-11d15fd541d9",
        "job_id": "660f9511-f3ac-52e5-b827-557766551111",
        "job_name": "duplicate_report_copy.pdf",
        "chunk_index": 5,
        "content_preview": "The quick brown fox jumps over the lazy dog...",
        "metadata": {
          "source_page": 2,
          "token_count": 128
        },
        "created_at": "2024-01-16T14:22:00Z"
      }
    ]
  }
}
```

### Empty Result Response (200 OK)
```json
{
  "success": true,
  "data": {
    "content_hash": "a3f5c8e9d2b1f6e4a7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6",
    "match_count": 0,
    "is_duplicate": false,
    "chunks": []
  }
}
```

## Behavior

1. **Hash Validation:**
   - Validate content_hash is 64-character hexadecimal string (SHA-256)
   - Convert to lowercase for case-insensitive matching
   - Reject malformed hashes with 400 error

2. **Database Query:**
   - Query `document_chunks` table with index on `content_hash` column
   - Join with `jobs` table to include job metadata
   - Apply optional `job_id` filter if specified
   - Return matching chunks ordered by `created_at DESC`

3. **Deduplication Detection:**
   - `is_duplicate` flag set to `true` when `match_count > 1`
   - `include_duplicates_only` filters to only show results when duplicates exist
   - Include `job_name` from parent job for duplicate identification

4. **Content Preview:**
   - Return first 200 characters of content as `content_preview`
   - Full content available via individual chunk retrieval endpoint
   - Preview truncated with ellipsis if content exceeds limit

5. **Performance Considerations:**
   - Hash column indexed with B-tree index for O(log n) lookups
   - Query timeout: 5 seconds
   - Results limited to prevent memory issues with highly duplicated content

## Error Handling

| Status | Error Code | Message | Condition |
|--------|------------|---------|-----------|
| 400 | `INVALID_HASH` | "Invalid content hash format" | Hash not 64 hex characters |
| 400 | `INVALID_UUID` | "Invalid UUID format for job_id" | Malformed job_id filter |
| 400 | `INVALID_PARAMETER` | "Limit must be between 1 and 200" | Limit out of valid range |
| 500 | `INTERNAL_ERROR` | "Failed to query chunks by hash" | Database or unexpected error |

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "INVALID_HASH",
    "message": "Invalid content hash format",
    "details": {
      "expected": "64-character hexadecimal SHA-256 hash",
      "provided": "a3f5c8e9d2b1",
      "length": 12
    }
  }
}
```

## Use Cases

1. **Pre-processing Deduplication:** Check if content already processed before creating new job
2. **Storage Optimization:** Identify duplicate chunks across job results
3. **Integrity Verification:** Verify chunk integrity by comparing content hashes
4. **Duplicate Job Detection:** Find jobs that processed identical or near-identical documents
