# Spec: Document Chunks Table with Vector Column

## Purpose
Define the database table schema for storing document chunks with their associated vector embeddings, supporting semantic search and retrieval operations.

## Interface

### SQL DDL
```sql
-- Enable pgvector extension (prerequisite)
CREATE EXTENSION IF NOT EXISTS vector;

-- Document chunks table with vector embedding support
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Constraints
ALTER TABLE document_chunks 
    ADD CONSTRAINT unique_job_chunk 
    UNIQUE (job_id, chunk_index);

-- Validation constraint for positive chunk index
ALTER TABLE document_chunks 
    ADD CONSTRAINT positive_chunk_index 
    CHECK (chunk_index >= 0);
```

### Column Definitions
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique identifier for the chunk |
| job_id | UUID | NOT NULL, FK → jobs(id), ON DELETE CASCADE | Reference to parent job |
| chunk_index | INTEGER | NOT NULL, CHECK >= 0 | Position of chunk within document (0-indexed) |
| content | TEXT | NOT NULL | The actual text content of the chunk |
| embedding | VECTOR(1536) | NULLABLE | Vector embedding (dimensions configurable) |
| metadata | JSONB | DEFAULT '{}'::jsonb | Flexible metadata storage (source, page, etc.) |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Record creation timestamp |

## Behavior

### Table Creation
1. Table must be created after `jobs` table exists
2. `pgvector` extension must be enabled before creating table with VECTOR type
3. Default dimension size is 1536 (OpenAI text-embedding-3-small)
4. Dimension size is configurable via migration parameters

### Data Insertion
1. `id` is auto-generated if not provided
2. `job_id` must reference an existing job; cascade delete removes all chunks when job is deleted
3. `chunk_index` must be unique per job (ordered sequence from 0)
4. `content` must not be empty or null
5. `embedding` may be null during initial chunk creation, populated later by embedding service
6. `metadata` accepts any valid JSONB object; keys should use snake_case convention

### Query Patterns
```sql
-- Get chunks for a specific job, ordered by index
SELECT * FROM document_chunks 
WHERE job_id = $1 
ORDER BY chunk_index ASC;

-- Get chunk count per job
SELECT job_id, COUNT(*) as chunk_count 
FROM document_chunks 
GROUP BY job_id;

-- Find chunks without embeddings (for backfill operations)
SELECT * FROM document_chunks 
WHERE embedding IS NULL;

-- Filter by metadata
SELECT * FROM document_chunks 
WHERE metadata @> '{"source": "api"}'::jsonb;
```

## Error Handling

| Error Code | Condition | Handling |
|------------|-----------|----------|
| `23503` (FK violation) | job_id references non-existent job | Reject insertion with clear error message |
| `23505` (Unique violation) | Duplicate (job_id, chunk_index) | Reject insertion; client should use upsert if needed |
| `23514` (Check violation) | chunk_index < 0 | Reject with validation error |
| `42704` (Undefined object) | pgvector not installed | Fail migration with prerequisite error |
| `22P02` (Invalid text representation) | Invalid vector format | Reject with format validation error |

## Performance Considerations

### Storage
- VECTOR type stores as fixed-size array (dimension count * 4 bytes per float)
- 1536 dimensions = ~6KB per embedding (uncompressed)
- JSONB metadata adds overhead; recommend keeping under 10KB per row
- TOAST storage used automatically for large content/text fields

### Query Optimization
- Primary key lookups on `id` are indexed by default
- Foreign key on `job_id` benefits from index (added in relationship spec)
- Queries filtering by `chunk_index` alone should include `job_id` for efficient index usage
- JSONB containment queries (`@>`, `?`) benefit from GIN indexes (see related specs)

### Bulk Operations
- Batch inserts recommended for embedding population (100-1000 rows per batch)
- Use COPY command for initial data loading of large datasets
- Disable indexes temporarily during bulk load, rebuild after

### Dimension Flexibility
- Migration supports parameterized dimensions: 384, 768, 1536, 3072
- All rows in table must use same dimension (VECTOR is fixed-size)
- Changing dimensions requires table recreation or separate tables per dimension
