# Spec: SQLAlchemy Model with pgvector Integration

## Purpose
Define a SQLAlchemy ORM model for the `document_chunks` table with proper type annotations, pgvector integration, and relationship mappings to enable type-safe database operations.

## Interface

### Python Model Definition
```python
"""Document chunk model with vector embedding support."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, TYPE_CHECKING

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UserDefinedType

if TYPE_CHECKING:
    from .job import Job


class Vector(UserDefinedType):
    """SQLAlchemy type for pgvector VECTOR."""
    
    cache_ok = True
    
    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions
    
    def get_col_spec(self, **kw: Any) -> str:
        return f"VECTOR({self.dimensions})"
    
    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            if isinstance(value, list):
                return f"[{','.join(str(x) for x in value)}]"
            return value
        return process
    
    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            # Parse vector string representation into list of floats
            if isinstance(value, str):
                value = value.strip('[]')
                return [float(x) for x in value.split(',')]
            return value
        return process


class DocumentChunk(Base):
    """Document chunk with vector embedding for semantic search.
    
    Attributes:
        id: Unique identifier (UUID)
        job_id: Reference to parent job
        chunk_index: Position of chunk within document (0-indexed)
        content: Text content of the chunk
        embedding: Vector embedding for semantic search
        metadata: JSONB metadata (source, page numbers, etc.)
        created_at: Timestamp of record creation
    """
    
    __tablename__ = "document_chunks"
    __table_args__ = (
        # Unique constraint: one chunk_index per job
        {"UniqueConstraint": ("job_id", "chunk_index"), "name": "unique_job_chunk"},
    )
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    
    # Foreign key to jobs table
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Chunk ordering within document
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    
    # Content storage
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    
    # Vector embedding (nullable during initial creation)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(dimensions=1536),
        nullable=True,
    )
    
    # Flexible metadata storage
    metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        server_default=func.now(),
    )
    
    # Relationships
    job: Mapped["Job"] = relationship("Job", back_populates="chunks")
    
    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id}, "
            f"job_id={self.job_id}, "
            f"chunk_index={self.chunk_index}, "
            f"has_embedding={self.embedding is not None})>"
        )
    
    @property
    def has_embedding(self) -> bool:
        """Check if chunk has an embedding vector."""
        return self.embedding is not None
    
    def set_embedding(self, embedding: list[float]) -> None:
        """Set the embedding vector with dimension validation.
        
        Args:
            embedding: List of float values
            
        Raises:
            ValueError: If embedding dimensions don't match expected size
        """
        expected_dims = 1536  # From Vector type definition
        if len(embedding) != expected_dims:
            raise ValueError(
                f"Embedding must have {expected_dims} dimensions, "
                f"got {len(embedding)}"
            )
        self.embedding = embedding
```

### Alternative: Using pgvector-python Library
```python
"""Alternative implementation using pgvector-python library."""

from pgvector.sqlalchemy import Vector as PgVector
from sqlalchemy.orm import Mapped, mapped_column

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    # ... other columns ...
    
    # Using pgvector-python library (simpler, recommended)
    embedding: Mapped[list[float] | None] = mapped_column(
        PgVector(1536),  # Dimensions specified here
        nullable=True,
    )
```

### Repository/Service Layer
```python
"""Data access layer for document chunks."""

from typing import Sequence
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .models import DocumentChunk


class DocumentChunkRepository:
    """Repository for document chunk database operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
    
    async def get_by_id(self, chunk_id: UUID) -> DocumentChunk | None:
        """Get a chunk by its UUID."""
        result = await self.session.execute(
            select(DocumentChunk).where(DocumentChunk.id == chunk_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_job(
        self, 
        job_id: UUID, 
        *, 
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[DocumentChunk]:
        """Get all chunks for a job, ordered by chunk_index."""
        result = await self.session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.job_id == job_id)
            .order_by(DocumentChunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()
    
    async def semantic_search(
        self,
        query_vector: list[float],
        job_id: UUID | None = None,
        top_k: int = 10,
        ef_search: int = 32,
    ) -> list[tuple[DocumentChunk, float]]:
        """Perform semantic similarity search using HNSW index.
        
        Returns:
            List of (chunk, distance) tuples, ordered by distance (ascending)
        """
        # Set ef_search for this query
        await self.session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
        
        # Build query with optional job filter
        query = select(
            DocumentChunk,
            DocumentChunk.embedding.op("<=>")(query_vector).label("distance"),
        )
        
        if job_id:
            query = query.where(DocumentChunk.job_id == job_id)
        
        query = (
            query.order_by(text("distance"))
            .limit(top_k)
        )
        
        result = await self.session.execute(query)
        return [(row.DocumentChunk, row.distance) for row in result]
    
    async def create(self, chunk: DocumentChunk) -> DocumentChunk:
        """Create a new document chunk."""
        self.session.add(chunk)
        await self.session.flush()
        await self.session.refresh(chunk)
        return chunk
```

## Behavior

### Model Initialization
1. `id` auto-generated on insert (UUID4)
2. `created_at` auto-populated with current timestamp
3. `metadata` defaults to empty JSON object `{}`
4. `embedding` is optional (nullable) to support two-phase creation

### Type Conversion
1. Python `list[float]` ↔ PostgreSQL `VECTOR`
2. Python `dict` ↔ PostgreSQL `JSONB`
3. Python `datetime` (UTC) ↔ PostgreSQL `TIMESTAMPTZ`
4. Python `uuid.UUID` ↔ PostgreSQL `UUID`

### Validation
1. Embedding dimension validation on assignment
2. Chunk index non-negative validation
3. Content non-empty validation
4. Metadata must be JSON-serializable dict

## Error Handling

| Exception | Condition | Handling |
|-----------|-----------|----------|
| `IntegrityError` | FK violation (job_id) | Catch and raise `JobNotFoundError` |
| `IntegrityError` | Unique constraint violation | Catch and raise `DuplicateChunkError` |
| `ValueError` | Dimension mismatch in embedding | Validate before assignment |
| `StatementError` | Invalid JSON in metadata | Validate JSON serializability |
| `SQLAlchemyError` | Database connection issues | Log and propagate as `DatabaseError` |

## Performance Considerations

### Lazy vs Eager Loading
```python
# Lazy loading (default) - job loaded on access
chunk = await repo.get_by_id(chunk_id)
print(chunk.job.name)  # Triggers separate query

# Eager loading - job loaded in single query
from sqlalchemy.orm import joinedload

query = select(DocumentChunk).options(joinedload(DocumentChunk.job))
```

### Batched Operations
```python
# Efficient bulk insert
chunks = [DocumentChunk(...) for _ in range(1000)]
await session.add_all(chunks)
await session.flush()
```

### Index Usage
- Queries filtering by `job_id` use FK index
- Queries ordering by `chunk_index` efficient with composite index
- Similarity search uses HNSW index automatically for `ORDER BY embedding <=> query`

### Connection Pooling
- Use async session for non-blocking operations
- Connection pool size should accommodate concurrent vector searches
- Consider dedicated pool for heavy embedding operations
