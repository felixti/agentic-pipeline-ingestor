# Spec: Full-Text Search

## Purpose
Perform traditional full-text search using PostgreSQL's tsvector/tsquery with BM25-like ranking.

## Interface
- **Method**: POST
- **Path**: `/api/v1/search/text`
- **Content-Type**: `application/json`

## Request Schema
```json
{
  "query": "string (required, max 1024 chars)",
  "top_k": "integer (optional, default: 10, max: 100, min: 1)",
  "language": "string (optional, default: 'english', values: 'english', 'spanish', 'french', 'german', 'portuguese', 'simple')",
  "highlight": "boolean (optional, default: true)",
  "highlight_start": "string (optional, default: '<mark>')",
  "highlight_end": "string (optional, default: '</mark>')",
  "metadata_filter": {
    "job_id": "uuid (optional)",
    "source_file": "string (optional)",
    "date_from": "ISO 8601 datetime (optional)",
    "date_to": "ISO 8601 datetime (optional)",
    "custom_fields": "object (optional)"
  },
  "fuzzy_match": "boolean (optional, default: false, enable pg_trgm fuzzy matching)"
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
        "content": "string (first 500 chars with highlights)",
        "content_highlighted": "string with <mark>highlighted</mark> terms",
        "relevance_score": 0.7563,
        "rank": 1,
        "metadata": {
          "source_file": "document.pdf",
          "page_number": 5
        },
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "total_results": 42,
    "search_time_ms": 8,
    "query_parsed": "'search' & 'term'"
  },
  "error": null
}
```

## Behavior
1. Validate request body using Pydantic validators
2. Parse query using `plainto_tsquery()` for natural language processing
3. Support quoted phrases with `phraseto_tsquery()`
4. Execute full-text search using GIN index on `to_tsvector(content)`
5. Calculate relevance using `ts_rank_cd()` with BM25-like weighting
6. Optionally enable pg_trgm similarity for fuzzy matching
7. Apply highlighting using `ts_headline()` if `highlight=true`
8. Filter by metadata conditions if provided
9. Return results sorted by relevance score (descending)

### Ranking Weights
Default weights follow BM25-like scoring:
- Title/header matches: 1.0 (A weight)
- Body content: 0.4 (D weight)
- Adjusted via normalization

### Fuzzy Matching
When `fuzzy_match=true`:
- Use `pg_trgm` similarity operator `%` for trigram matching
- Minimum similarity threshold: 0.3
- Combines with tsvector search using OR logic

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
        { "field": "query", "error": "Query too long (max 1024 chars)" }
      ]
    }
  }
  ```
- **400 Bad Request**: Invalid language specified
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Database or unexpected error

## Rate Limiting
- **Limit**: 60 requests per minute per user/IP
- **Window**: 60 seconds
- **Key**: `search_text:{user_id_or_ip}`
- **Burst**: Allow 10 requests immediately
