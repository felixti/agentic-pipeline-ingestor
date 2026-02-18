# Spec: Relationship Linking to Jobs and Job Results Tables

## Purpose
Define foreign key relationships between `document_chunks` and existing `jobs`/`job_results` tables, enabling cascade deletes, efficient joins, and referential integrity for chunk lifecycle management.

## Interface

### SQL DDL - Foreign Key Constraints
```sql
-- Primary relationship: chunks belong to jobs
ALTER TABLE document_chunks 
ADD CONSTRAINT fk_document_chunks_job_id 
FOREIGN KEY (job_id) 
REFERENCES jobs(id) 
ON DELETE CASCADE;

-- Optional relationship: link to specific job result (for result-associated chunks)
ALTER TABLE document_chunks 
ADD COLUMN job_result_id UUID;

ALTER TABLE document_chunks 
ADD CONSTRAINT fk_document_chunks_job_result_id 
FOREIGN KEY (job_result_id) 
REFERENCES job_results(id) 
ON DELETE SET NULL;

-- Index on foreign keys for efficient joins
CREATE INDEX idx_document_chunks_job_id 
ON document_chunks(job_id);

CREATE INDEX idx_document_chunks_job_result_id 
ON document_chunks(job_result_id);

-- Composite index for common query pattern
CREATE INDEX idx_document_chunks_job_id_chunk_index 
ON document_chunks(job_id, chunk_index);
```

### Bidirectional Relationship SQL
```sql
-- View for job with chunk summary
CREATE VIEW job_chunk_summary AS
SELECT 
    j.id AS job_id,
    j.status AS job_status,
    j.created_at AS job_created_at,
    COUNT(dc.id) AS chunk_count,
    COUNT(dc.embedding) AS embedded_chunk_count,
    MIN(dc.chunk_index) AS first_chunk_index,
    MAX(dc.chunk_index) AS last_chunk_index
FROM jobs j
LEFT JOIN document_chunks dc ON dc.job_id = j.id
GROUP BY j.id, j.status, j.created_at;

-- View for complete job result with chunks
CREATE VIEW job_result_with_chunks AS
SELECT 
    jr.id AS result_id,
    jr.job_id,
    jr.output_path,
    jr.status AS result_status,
    dc.id AS chunk_id,
    dc.chunk_index,
    dc.content,
    dc.metadata AS chunk_metadata,
    dc.created_at AS chunk_created_at
FROM job_results jr
LEFT JOIN document_chunks dc ON dc.job_id = jr.job_id
ORDER BY jr.id, dc.chunk_index;
```

### SQLAlchemy Relationship Definitions
```python
"""Relationship definitions for document chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING, List
from uuid import UUID

from sqlalchemy import ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PGUUID

if TYPE_CHECKING:
    from .job import Job
    from .job_result import JobResult


class DocumentChunk(Base):
    """Document chunk with relationships to Job and JobResult."""
    
    __tablename__ = "document_chunks"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Foreign key to jobs table with CASCADE delete
    job_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Optional foreign key to job_results with SET NULL
    job_result_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("job_results.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    chunk_index: Mapped[int]
    content: Mapped[str]
    # ... other columns ...
    
    # Relationships
    job: Mapped["Job"] = relationship(
        "Job",
        back_populates="chunks",
        lazy="selectin",  # Eager load with selectin for N+1 prevention
    )
    
    job_result: Mapped["JobResult | None"] = relationship(
        "JobResult",
        back_populates="chunks",
        lazy="selectin",
    )
    
    __table_args__ = (
        # Unique constraint: one chunk_index per job
        UniqueConstraint('job_id', 'chunk_index', name='uq_document_chunks_job_chunk'),
        # Composite index for job_id + chunk_index queries
        Index('idx_document_chunks_job_chunk', 'job_id', 'chunk_index'),
    )


# Job model extension
class Job(Base):
    """Job model with chunks relationship."""
    
    __tablename__ = "jobs"
    
    # ... existing columns ...
    
    # Relationship to chunks
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="job",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="DocumentChunk.chunk_index",
    )
    
    @property
    def chunk_count(self) -> int:
        """Return number of chunks for this job."""
        return len(self.chunks) if self.chunks else 0
    
    @property
    def has_embeddings(self) -> bool:
        """Check if all chunks have embeddings."""
        if not self.chunks:
            return False
        return all(chunk.embedding is not None for chunk in self.chunks)


# JobResult model extension
class JobResult(Base):
    """JobResult model with optional chunks relationship."""
    
    __tablename__ = "job_results"
    
    # ... existing columns ...
    
    # Relationship to chunks (optional, may be null)
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="job_result",
        lazy="selectin",
    )
```

## Behavior

### Cascade Delete Rules

| Parent Table | Child Table | Action | Rationale |
|--------------|-------------|--------|-----------|
| `jobs` | `document_chunks` | `CASCADE` | Chunks are job-specific, delete with job |
| `job_results` | `document_chunks.job_result_id` | `SET NULL` | Keep chunks if result deleted, unlink only |

### Referential Integrity
1. **Insert**: `job_id` must reference existing job; rejects orphan chunks
2. **Update**: Changing `job_id` validates new job exists
3. **Delete Job**: All associated chunks automatically deleted
4. **Delete JobResult**: `job_result_id` in chunks set to NULL

### Query Patterns
```sql
-- Get job with all chunks (ordered)
SELECT j.*, dc.*
FROM jobs j
LEFT JOIN document_chunks dc ON dc.job_id = j.id
WHERE j.id = $1
ORDER BY dc.chunk_index;

-- Get chunks for job result
SELECT dc.*
FROM document_chunks dc
WHERE dc.job_result_id = $1
ORDER BY dc.chunk_index;

-- Count chunks per job (fast with index)
SELECT job_id, COUNT(*) 
FROM document_chunks 
GROUP BY job_id;

-- Find jobs without chunks (for cleanup)
SELECT j.*
FROM jobs j
LEFT JOIN document_chunks dc ON dc.job_id = j.id
WHERE dc.id IS NULL;
```

### Lifecycle Events

#### Job Creation
- Job created first
- Chunks inserted with valid `job_id`
- Optional: Link to `job_result_id` when result created

#### Job Deletion
- All chunks automatically deleted (CASCADE)
- No orphan chunks remain
- Cleanup efficient (single DELETE cascade)

#### JobResult Deletion
- Chunks remain but lose `job_result_id` reference
- Chunks still accessible via `job_id`
- Allows result regeneration without re-chunking

## Error Handling

| Error Code | Condition | Handling |
|------------|-----------|----------|
| `23503` (FK violation) | Insert chunk with invalid `job_id` | Validate job exists before chunk creation |
| `23503` (FK violation) | Update `job_id` to invalid value | Validate new job exists |
| `23505` (Unique violation) | Duplicate `(job_id, chunk_index)` | Ensure chunk_index generation is sequential |
| `23514` (Check violation) | `chunk_index < 0` | Validate non-negative before insert |

### Integrity Check Queries
```sql
-- Find orphaned chunks (should be empty)
SELECT dc.*
FROM document_chunks dc
LEFT JOIN jobs j ON j.id = dc.job_id
WHERE j.id IS NULL;

-- Find chunks with invalid job_result_id
SELECT dc.*
FROM document_chunks dc
LEFT JOIN job_results jr ON jr.id = dc.job_result_id
WHERE dc.job_result_id IS NOT NULL 
  AND jr.id IS NULL;
```

## Performance Considerations

### Index Strategy
| Index | Columns | Purpose |
|-------|---------|---------|
| `idx_document_chunks_job_id` | `job_id` | FK lookups, cascade delete performance |
| `idx_document_chunks_job_result_id` | `job_result_id` | Optional FK lookups |
| `idx_document_chunks_job_chunk` | `job_id, chunk_index` | Ordered chunk retrieval per job |
| `idx_document_chunks_job_id_chunk_index` | `job_id, chunk_index` | Covering index for common queries |

### Join Optimization
```sql
-- Efficient: Uses composite index
SELECT * FROM document_chunks 
WHERE job_id = $1 
ORDER BY chunk_index;

-- Efficient: FK index + limit
SELECT * FROM document_chunks 
WHERE job_id = $1 
LIMIT 10;

-- Consider covering index for:
SELECT id, chunk_index, content 
FROM document_chunks 
WHERE job_id = $1;
```

### Cascade Performance
- Cascading delete from `jobs` uses index on `document_chunks.job_id`
- Time complexity: O(n) where n = chunks for that job
- For jobs with 100K+ chunks, consider batching deletion

### Batch Operations
```python
# Efficient bulk delete by job_id
await session.execute(
    delete(DocumentChunk).where(DocumentChunk.job_id == job_id)
)

# Efficient bulk insert with FK validation
chunks = [DocumentChunk(job_id=job_id, ...) for _ in range(n)]
await session.add_all(chunks)
```

### Partitioning Considerations
For very large deployments, consider partitioning `document_chunks` by `job_id` hash:
```sql
-- Hash partitioning for parallel chunk operations
CREATE TABLE document_chunks (
    ...
) PARTITION BY HASH (job_id);

CREATE TABLE document_chunks_p0 PARTITION OF document_chunks FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE document_chunks_p1 PARTITION OF document_chunks FOR VALUES WITH (MODULUS 4, REMAINDER 1);
-- ... etc
```

### Connection Pool Impact
- Relationship queries use `selectin` loading to prevent N+1
- Large job chunk lists load in batches
- Consider pagination for jobs with > 10K chunks
