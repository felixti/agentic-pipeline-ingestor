# Spec: Bulk Chunk Retrieval for Batch Operations

## Purpose
Retrieve multiple document chunks by their UUIDs in a single efficient batch query, reducing network overhead for bulk operations and RAG context assembly.

## Interface

**Endpoint:** `POST /api/v1/chunks/bulk`

**Request Body:**
```json
{
  "chunk_ids": ["uuid1", "uuid2", ...],
  "include_embeddings": false,
  "include_source": false
}
```

**Query Parameters:**
- None (all parameters in request body)

## Request/Response

### Request Schema
```json
{
  "chunk_ids": {
    "type": "array",
    "items": {
      "type": "string",
      "format": "uuid"
    },
    "minItems": 1,
    "maxItems": 100,
    "description": "Array of chunk UUIDs to retrieve"
  },
  "include_embeddings": {
    "type": "boolean",
    "default": false,
    "description": "Include vector embeddings in response"
  },
  "include_source": {
    "type": "boolean",
    "default": false,
    "description": "Include parent job source information"
  }
}
```

### Request Example
```http
POST /api/v1/chunks/bulk
Content-Type: application/json

{
  "chunk_ids": [
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "7cb8c921-aead-22e2-91c5-11d15fd541d9",
    "8dc9da32-bfbe-33f3-02d6-22e26fe652ea"
  ],
  "include_embeddings": false,
  "include_source": true
}
```

### Response Schema (200 OK)
```json
{
  "success": true,
  "data": {
    "chunks": [
      {
        "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "chunk_index": 3,
        "content": "First chunk content here...",
        "content_hash": "a3f5c8e9d2b1...",
        "metadata": {
          "source_page": 1,
          "token_count": 128
        },
        "source": {
          "job_name": "quarterly_report_2024.pdf",
          "job_type": "document_processing",
          "created_at": "2024-01-15T10:00:00Z"
        },
        "created_at": "2024-01-15T10:30:00Z"
      },
      {
        "id": "7cb8c921-aead-22e2-91c5-11d15fd541d9",
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "chunk_index": 4,
        "content": "Second chunk content here...",
        "content_hash": "b4g6d9f0e3c2...",
        "metadata": {
          "source_page": 1,
          "token_count": 128
        },
        "source": {
          "job_name": "quarterly_report_2024.pdf",
          "job_type": "document_processing",
          "created_at": "2024-01-15T10:00:00Z"
        },
        "created_at": "2024-01-15T10:30:01Z"
      }
    ],
    "found_count": 2,
    "requested_count": 3,
    "not_found": ["8dc9da32-bfbe-33f3-02d6-22e26fe652ea"]
  }
}
```

## Behavior

1. **Request Validation:**
   - Validate all chunk_ids are valid UUID v4 format
   - Enforce maximum of 100 chunk IDs per request
   - Reject empty chunk_ids array
   - Deduplicate chunk_ids internally (process unique set)

2. **Database Query:**
   - Use single `WHERE id IN (...)` query for efficiency
   - Leverage primary key index for O(n log n) retrieval
   - Join with `jobs` table if `include_source=true`
   - Conditionally select embedding column based on parameter

3. **Partial Success Handling:**
   - Return successfully found chunks in `chunks` array
   - List missing chunk IDs in `not_found` array
   - HTTP 200 returned even with partial results
   - `found_count` and `requested_count` indicate completeness

4. **Ordering:**
   - Results ordered by chunk_id UUID (database default)
   - Not guaranteed to match input order
   - Clients should map results by ID for ordering

5. **Performance Considerations:**
   - Single round-trip reduces network latency vs N individual requests
   - Query timeout: 15 seconds for large batches
   - Embedding inclusion increases response size significantly
   - Recommended batch size: 10-50 for optimal performance

6. **Use Case Optimizations:**
   - Designed for RAG context assembly (multiple relevant chunks)
   - Efficient for bulk export operations
   - Supports reconstituting full documents from chunk IDs

## Error Handling

| Status | Error Code | Message | Condition |
|--------|------------|---------|-----------|
| 400 | `INVALID_REQUEST` | "chunk_ids array is required" | Missing chunk_ids field |
| 400 | `INVALID_REQUEST` | "chunk_ids must contain 1-100 items" | Empty array or >100 items |
| 400 | `INVALID_UUID` | "Invalid UUID format at index N" | Malformed UUID in array |
| 413 | `PAYLOAD_TOO_LARGE` | "Request payload too large" | Request body exceeds 1MB |
| 500 | `INTERNAL_ERROR` | "Failed to retrieve chunks" | Database or unexpected error |

### Error Response Format (Complete Failure)
```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "chunk_ids must contain 1-100 items",
    "details": {
      "provided": 150,
      "max_allowed": 100
    }
  }
}
```

### Partial Success Response (200 OK with not_found)
When some chunks are found and others are not:
- HTTP 200 status (partial success is still success)
- `found_count < requested_count` indicates partial result
- Client should check `not_found` array for missing IDs

## Use Cases

1. **RAG Context Assembly:** Retrieve top-K relevant chunks from semantic search
2. **Document Reconstruction:** Fetch all chunks for document reassembly
3. **Bulk Export:** Export specific chunks for external analysis
4. **Cache Warming:** Pre-load chunks into application cache
5. **Batch Verification:** Verify existence and integrity of multiple chunks
