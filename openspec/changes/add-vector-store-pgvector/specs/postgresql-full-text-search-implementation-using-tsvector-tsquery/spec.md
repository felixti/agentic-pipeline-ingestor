# Spec: PostgreSQL Full-Text Search Implementation Using tsvector/tsquery

## Purpose
Enable full-text search on chunk content using PostgreSQL's native text search capabilities with tsvector and tsquery.

## Interface

### SQL Schema
```sql
-- GIN index for full-text search on document_chunks content
CREATE INDEX idx_document_chunks_content_search 
ON document_chunks 
USING gin (to_tsvector('english', content));

-- Query using tsquery
SELECT id, job_id, content, chunk_index
FROM document_chunks
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'search query');
```

### API Parameters
```json
{
  "query": "machine learning algorithms",
  "language": "english",
  "top_k": 10,
  "job_id": "uuid-optional-filter"
}
```

### SQLAlchemy Model Integration
```python
from sqlalchemy import Index, text
from sqlalchemy.dialects.postgresql import TSVECTOR

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    # ... existing columns ...
    
    # Functional index for full-text search
    __table_args__ = (
        Index(
            'idx_document_chunks_content_search',
            text("to_tsvector('english', content)"),
            postgresql_using='gin'
        ),
    )
```

## Behavior

### Text Search Processing
1. **Tokenization**: Convert raw content into lexemes (normalized word forms)
2. **Stop Word Removal**: Filter out common words (the, and, is, etc.) based on language
3. **Stemming**: Reduce words to root forms (running → run, algorithms → algorithm)
4. **Query Matching**: Match search terms against indexed tsvector

### Supported Query Types
- **Simple terms**: `machine learning` matches documents containing both words
- **Phrase search**: `"neural networks"` matches exact phrase (with proximity)
- **OR logic**: `machine | learning` matches either word
- **AND logic**: `machine & learning` matches both words
- **NOT logic**: `machine !learning` excludes documents with "learning"
- **Prefix matching**: `machin:*` matches machine, machines, machining

### Search Result Format
```json
{
  "results": [
    {
      "id": "chunk-uuid",
      "job_id": "job-uuid",
      "chunk_index": 5,
      "content": "original content...",
      "rank": 0.345,
      "matched_terms": ["machine", "learning"]
    }
  ],
  "total": 42,
  "query": "machine learning"
}
```

## Error Handling

| Error Case | Error Type | Handling |
|------------|------------|----------|
| Empty query | ValidationError | Return 400 with "Query cannot be empty" |
| Query too long (>1000 chars) | ValidationError | Return 400 with "Query exceeds maximum length" |
| Invalid language config | RuntimeError | Fallback to 'english' with warning log |
| No matching documents | EmptyResult | Return empty array with 200 status |
| GIN index missing | PerformanceWarning | Log warning, query executes with sequential scan |
| Special character injection | Sanitization | Strip or escape tsquery special characters |

## Performance Considerations

### Index Optimization
- **GIN (Generalized Inverted Index)**: Default choice for full-text search
  - Build time: O(n log n) where n = number of lexemes
  - Query time: O(log n) for typical queries
  - Size: ~50-100% of text data size
- **GiST Alternative**: Available but generally slower for full-text

### Query Optimization
```sql
-- Fast: Uses GIN index
SELECT * FROM document_chunks
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'query');

-- Slower: Function on column prevents index use
SELECT * FROM document_chunks
WHERE content ILIKE '%query%';
```

### Tuning Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| gin_fuzzy_search_limit | 0 (off) | Limit results for fuzzy matching |
| gin_pending_list_limit | 4MB | Size of pending list for fast updates |

### Benchmark Targets
- Search latency p50 < 20ms for 100K chunks
- Search latency p99 < 100ms for 1M chunks
- Index build time < 5 minutes for 1M documents
