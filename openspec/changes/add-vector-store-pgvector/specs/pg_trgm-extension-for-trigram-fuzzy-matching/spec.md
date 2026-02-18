# Spec: pg_trgm Extension for Trigram Fuzzy Matching

## Purpose
Enable fuzzy string matching to find similar words and handle typos, spelling variations, and approximate matches using trigram similarity.

## Interface

### SQL Setup
```sql
-- Enable the pg_trgm extension
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- GIN index for trigram search (supports similarity and LIKE/ILIKE)
CREATE INDEX idx_document_chunks_content_trgm 
ON document_chunks 
USING gin (content gin_trgm_ops);

-- GiST index alternative (better for similarity thresholds)
CREATE INDEX idx_document_chunks_content_trgm_gist 
ON document_chunks 
USING gist (content gist_trgm_ops);
```

### Similarity Functions
```sql
-- Basic similarity (0.0 to 1.0, where 1.0 is exact match)
SELECT similarity('word', 'words');  -- Returns ~0.5

-- Show trigram overlap
SELECT show_trgm('word');  -- Returns {"  w"," wo",wor,ord,"rd "}

-- Word similarity (best matching word in string)
SELECT word_similarity('PostgreSQL database', 'postresql');

-- Strict word similarity
SELECT strict_word_similarity('PostgreSQL database', 'postresql');
```

### API Parameters
```json
{
  "query": "accomodation",
  "similarity_threshold": 0.3,
  "operator": "similarity",  // or "word_similarity"
  "top_k": 10
}
```

### SQLAlchemy Integration
```python
from sqlalchemy import Index, func
from sqlalchemy.dialects.postgresql import ARRAY

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    # ... existing columns ...
    
    __table_args__ = (
        # GIN index for trigram operations
        Index(
            'idx_document_chunks_content_trgm',
            'content',
            postgresql_using='gin',
            postgresql_ops={'content': 'gin_trgm_ops'}
        ),
    )
```

## Behavior

### Trigram Generation
A trigram is a contiguous sequence of 3 characters:
- **Input**: "hello"
- **Trigrams**: `  h`, ` he`, `hel`, `ell`, `llo`, `lo `

### Similarity Calculation
```
similarity(A, B) = |trigrams(A) ∩ trigrams(B)| / |trigrams(A) ∪ trigrams(B)|

Example: similarity("word", "words")
  trigrams("word")  = {  w,  wo, wor, ord, rd }
  trigrams("words") = {  w,  wo, wor, ord, rds, ds }
  Intersection = {  w,  wo, wor, ord } = 4
  Union = {  w,  wo, wor, ord, rd, rds, ds } = 7
  Similarity = 4/7 ≈ 0.57
```

### Fuzzy Search Modes

#### 1. Similarity Search (Default)
```sql
SELECT id, content, similarity(content, 'search_term') AS sim
FROM document_chunks
WHERE content % 'search_term'  -- % = similarity exceeds threshold
ORDER BY sim DESC
LIMIT 10;
```

#### 2. Word Similarity
```sql
SELECT id, content, word_similarity(content, 'search_term') AS sim
FROM document_chunks
WHERE content <% 'search_term'  -- <% = word similarity
ORDER BY sim DESC;
```

#### 3. Trigram LIKE/ILIKE (Index-accelerated)
```sql
-- Uses trigram index for ILIKE patterns with >=3 chars
SELECT * FROM document_chunks
WHERE content ILIKE '%machine%';
```

### Threshold Configuration
```sql
-- Set session-level threshold (default 0.3)
SET pg_trgm.similarity_threshold = 0.4;
SET pg_trgm.word_similarity_threshold = 0.6;
```

### Search Result Format
```json
{
  "results": [
    {
      "id": "chunk-uuid",
      "content": "accommodation facilities...",
      "similarity": 0.714,
      "match_type": "trigram_similarity"
    }
  ],
  "threshold": 0.3,
  "total_matches": 15
}
```

## Error Handling

| Error Case | Error Type | Handling |
|------------|------------|----------|
| Extension not installed | RuntimeError | Return 500 with "pg_trgm extension required" |
| Query too short (<3 chars) | ValidationError | Return 400 with "Query must be at least 3 characters" |
| Similarity threshold out of range | ValidationError | Clamp to [0.0, 1.0] with warning |
| Empty result set | EmptyResult | Return empty array, suggest lowering threshold |
| Invalid UTF-8 in query | EncodingError | Return 400 with "Invalid character encoding" |

## Performance Considerations

### Index Selection
| Index Type | Best For | Tradeoffs |
|------------|----------|-----------|
| GIN + gin_trgm_ops | Similarity operators (`%`, `<%`) | Larger index, faster queries |
| GiST + gist_trgm_ops | Similarity with ORDER BY | Smaller index, slower queries |
| btree | Equality only | Not recommended for fuzzy search |

### Query Performance
```sql
-- Fast: Uses trigram index
SELECT * FROM document_chunks WHERE content % 'machine';

-- Fast: ILIKE with 3+ char prefix uses index
SELECT * FROM document_chunks WHERE content ILIKE 'mac%';

-- Slow: Leading wildcard without trigrams
SELECT * FROM document_chunks WHERE content ILIKE '%ine';
```

### Optimization Guidelines
1. **Minimum query length**: Require 3+ characters for indexed queries
2. **Threshold tuning**: Higher thresholds (0.5+) = fewer results, faster queries
3. **Combine with text search**: Use trigrams for fuzzy, tsvector for relevance
4. **Limit early**: Apply LIMIT before sorting when possible

### Resource Usage
| Metric | Estimate |
|--------|----------|
| Index size | ~2-3x text column size |
| Build time | ~30s per 1M rows |
| Query overhead | +5-10ms vs exact match |

### Benchmark Targets
- Fuzzy search p50 < 50ms for 100K chunks
- Fuzzy search p99 < 200ms for 1M chunks
- Similarity calculation: <1ms per comparison
