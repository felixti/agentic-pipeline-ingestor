# Spec: Metadata Filtering

## Purpose
Enable precise filtering of vector search results by metadata attributes such as job_id, date ranges, and custom metadata fields for targeted retrieval.

## Interface

### Service Interface
```python
from typing import Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class FilterOperator(Enum):
    EQ = "eq"           # Equal
    NE = "ne"           # Not equal
    GT = "gt"           # Greater than
    GTE = "gte"         # Greater than or equal
    LT = "lt"           # Less than
    LTE = "lte"         # Less than or equal
    IN = "in"           # In list
    NIN = "nin"         # Not in list
    CONTAINS = "contains"  # JSON contains
    EXISTS = "exists"   # Field exists

@dataclass
class MetadataFilter:
    field: str
    operator: FilterOperator
    value: Any

class VectorSearchService:
    async def search_with_metadata_filter(
        self,
        query_vector: list[float],
        metadata_filters: list[MetadataFilter],
        top_k: int = 10,
        combine_operator: str = "AND",  # "AND" or "OR"
    ) -> List[SearchResult]:
        """
        Search with metadata filtering.
        
        Args:
            query_vector: The embedding vector to search for
            metadata_filters: List of filter conditions
            top_k: Maximum results to return
            combine_operator: How to combine multiple filters (AND/OR)
        """
        pass
```

### API Interface
```http
POST /api/v1/search/semantic
Content-Type: application/json

{
  "query_vector": [0.023, -0.156, 0.892, ...],
  "top_k": 10,
  "metadata_filter": {
    "operator": "AND",
    "conditions": [
      {"field": "job_id", "op": "eq", "value": "uuid-here"},
      {"field": "created_at", "op": "gte", "value": "2024-01-01T00:00:00Z"},
      {"field": "metadata.source", "op": "eq", "value": "pdf"},
      {"field": "metadata.tags", "op": "contains", "value": "important"}
    ]
  }
}
```

### SQL Interface
```sql
-- Single job filter with vector search
SELECT id, content, metadata, 
       1 - (embedding <=> :query_vector) AS similarity
FROM document_chunks
WHERE job_id = :job_id
  AND 1 - (embedding <=> :query_vector) >= :min_similarity
ORDER BY embedding <=> :query_vector ASC
LIMIT :top_k;

-- Date range filter with JSON metadata
SELECT id, content, metadata,
       1 - (embedding <=> :query_vector) AS similarity
FROM document_chunks
WHERE job_id = :job_id
  AND created_at BETWEEN :start_date AND :end_date
  AND metadata @> '{"source": "pdf"}'
  AND metadata @> '{"tags": ["important"]}'
ORDER BY embedding <=> :query_vector ASC
LIMIT :top_k;

-- Complex filter with OR logic
SELECT id, content, metadata,
       1 - (embedding <=> :query_vector) AS similarity
FROM document_chunks
WHERE (
    job_id = :job_id_1 OR job_id = :job_id_2
  )
  AND metadata @> '{"department": "engineering"}'
ORDER BY embedding <=> :query_vector ASC
LIMIT :top_k;
```

## Behavior

### Supported Filter Fields

| Field | Type | Operators | Description |
|-------|------|-----------|-------------|
| `job_id` | UUID | eq, ne, in, nin | Filter by parent job |
| `chunk_index` | integer | eq, ne, gt, gte, lt, lte, in | Filter by position in document |
| `created_at` | timestamp | gt, gte, lt, lte | Filter by creation time |
| `content` | text | contains | Full-text search within content |
| `metadata.*` | jsonb | eq, contains, exists | Filter by custom metadata fields |

### Metadata JSON Structure
```json
{
  "job_id": "uuid",
  "chunk_index": 5,
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "source": "pdf",
    "filename": "document.pdf",
    "page_number": 12,
    "department": "engineering",
    "tags": ["api", "documentation"],
    "confidence": 0.95,
    "custom_fields": {
      "project": "alpha",
      "priority": "high"
    }
  }
}
```

### Filter Evaluation Order
1. **Database-level filters** (job_id, created_at) - Applied in SQL WHERE clause
2. **Vector similarity** - Computed and ordered by pgvector
3. **JSON metadata filters** - Applied via `@>` operator or JSON path
4. **Post-filtering** - Any complex logic not expressible in SQL

### Nested Metadata Access
- Use dot notation for nested fields: `metadata.custom_fields.project`
- Array containment: `metadata.tags` contains `"api"`
- JSON path queries for deep nesting: `$.custom_fields.priority`

### Combine Operators
- **AND** (default): All conditions must match
- **OR**: Any condition can match
- **Nested**: Support for grouped conditions `(A AND B) OR (C AND D)`

```python
# Complex filter structure
metadata_filter = {
    "operator": "OR",
    "conditions": [
        {
            "operator": "AND",
            "conditions": [
                {"field": "job_id", "op": "eq", "value": "job-1"},
                {"field": "metadata.priority", "op": "eq", "value": "high"}
            ]
        },
        {
            "operator": "AND",
            "conditions": [
                {"field": "job_id", "op": "eq", "value": "job-2"},
                {"field": "metadata.department", "op": "eq", "value": "sales"}
            ]
        }
    ]
}
```

## Error Handling

| Error Code | Condition | Response |
|------------|-----------|----------|
| `400 Bad Request` | Invalid filter field | `"Unknown filter field: 'invalid_field'"` |
| `400 Bad Request` | Invalid operator for field type | `"Operator 'gt' not supported for text field"` |
| `400 Bad Request` | Invalid date format | `"Invalid ISO 8601 date: '2024-13-45'"` |
| `400 Bad Request` | Invalid UUID format | `"Invalid UUID format for job_id"` |
| `400 Bad Request` | Nested metadata path not found | `"Metadata path 'metadata.invalid.key' does not exist"` |

### Validation Rules
- job_id must be valid UUID format
- Date fields must be ISO 8601 format
- Array operators (`in`, `nin`, `contains`) require array values
- JSON path fields must start with `metadata.`

## Performance Considerations

### Index Strategy
```sql
-- B-tree index for job_id (highly selective)
CREATE INDEX idx_document_chunks_job_id 
ON document_chunks(job_id);

-- BRIN index for created_at (time-series data)
CREATE INDEX idx_document_chunks_created_at 
ON document_chunks USING BRIN(created_at);

-- GIN index for metadata JSONB
CREATE INDEX idx_document_chunks_metadata 
ON document_chunks USING GIN(metadata jsonb_path_ops);

-- Composite index for common filter combinations
CREATE INDEX idx_document_chunks_job_created 
ON document_chunks(job_id, created_at DESC);
```

### Query Optimization
- **Most selective filters first**: Place `job_id` before date ranges
- **Avoid filtering on low-cardinality fields** alone
- **Use partial indexes** for frequently filtered values:
  ```sql
  CREATE INDEX idx_high_priority 
  ON document_chunks(job_id) 
  WHERE metadata @> '{"priority": "high"}';
  ```

### Performance Impact
| Filter Type | Impact | Notes |
|-------------|--------|-------|
| `job_id` | Low | Uses B-tree index, very fast |
| `created_at` range | Low-Medium | BRIN index efficient for time series |
| `metadata.*` JSON | Medium-High | GIN index helps but can be slower |
| Multiple AND filters | Medium | Combined selectivity helps |
| OR conditions | High | Often requires sequential scans |

### Filter Pre-check
```python
# Validate filters before executing expensive vector search
async def validate_filters(self, filters: list[MetadataFilter]) -> None:
    for f in filters:
        if f.field == "job_id":
            # Verify job exists
            exists = await self.db.fetch_one(
                "SELECT 1 FROM jobs WHERE id = :job_id",
                {"job_id": f.value}
            )
            if not exists:
                raise JobNotFoundError(f.value)
```

### Cursor-Based Pagination with Filters
When using filters with pagination:
1. Apply filters to get filtered result set
2. Apply vector ordering within filtered set
3. Use cursor based on `(similarity, id)` for consistent pagination

```sql
-- Cursor-based with filter
SELECT id, content, metadata, similarity
FROM (
    SELECT id, content, metadata,
           1 - (embedding <=> :query_vector) AS similarity
    FROM document_chunks
    WHERE job_id = :job_id
      AND metadata @> '{"source": "pdf"}'
) filtered
WHERE (similarity, id) < (:cursor_similarity, :cursor_id)
ORDER BY similarity DESC, id DESC
LIMIT :page_size;
```
