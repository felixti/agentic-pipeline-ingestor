# Spec: Migration Scripts with Index Tuning Options

## Purpose
Provide Alembic migration scripts for creating the vector storage schema with configurable index parameters, supporting safe upgrades, downgrades, and environment-specific tuning.

## Interface

### Migration File Structure
```python
"""Add document_chunks table with vector embeddings and HNSW index.

Revision ID: 003_add_document_chunks
Revises: 002_add_job_results
Create Date: 2024-02-18 10:00:00.000000

Configuration:
    - VECTOR_DIMENSIONS: Embedding dimensions (384, 768, 1536, 3072)
    - HNSW_M: Number of connections per layer (2-100)
    - HNSW_EF_CONSTRUCTION: Build-time accuracy parameter (4-1000)
"""

import os
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '003_add_document_chunks'
down_revision = '002_add_job_results'
branch_labels = None
dends_on = None


# Configuration (override via environment variables)
VECTOR_DIMENSIONS = int(os.getenv('VECTOR_DIMENSIONS', '1536'))
HNSW_M = int(os.getenv('HNSW_M', '16'))
HNSW_EF_CONSTRUCTION = int(os.getenv('HNSW_EF_CONSTRUCTION', '64'))


def upgrade():
    """Create document_chunks table, indexes, and constraints."""
    
    # Validate configuration
    _validate_config()
    
    # Enable required extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create document_chunks table
    op.create_table(
        'document_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', sa.Text(), nullable=True),  # Will alter to VECTOR
        sa.Column('metadata', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_document_chunks')),
    )
    
    # Add check constraint for positive chunk_index
    op.create_check_constraint(
        'positive_chunk_index',
        'document_chunks',
        'chunk_index >= 0'
    )
    
    # Alter embedding column to VECTOR type
    op.execute(f"""
        ALTER TABLE document_chunks 
        ALTER COLUMN embedding 
        TYPE VECTOR({VECTOR_DIMENSIONS}) 
        USING embedding::VECTOR({VECTOR_DIMENSIONS})
    """)
    
    # Add foreign key constraint
    op.create_foreign_key(
        op.f('fk_document_chunks_job_id_jobs'),
        'document_chunks',
        'jobs',
        ['job_id'],
        ['id'],
        ondelete='CASCADE'
    )
    
    # Add unique constraint for job_id + chunk_index
    op.create_unique_constraint(
        op.f('uq_document_chunks_job_chunk'),
        'document_chunks',
        ['job_id', 'chunk_index']
    )
    
    # Create indexes
    _create_indexes()
    
    # Create GIN index for metadata JSONB queries
    op.create_index(
        'idx_document_chunks_metadata_gin',
        'document_chunks',
        ['metadata'],
        postgresql_using='gin'
    )


def downgrade():
    """Remove document_chunks table and related objects."""
    
    # Drop indexes
    op.drop_index('idx_document_chunks_metadata_gin', table_name='document_chunks')
    op.drop_index('idx_document_chunks_embedding_hnsw', table_name='document_chunks')
    op.drop_index('idx_document_chunks_job_id', table_name='document_chunks')
    op.drop_index('idx_document_chunks_created_at', table_name='document_chunks')
    
    # Drop constraints
    op.drop_constraint(op.f('uq_document_chunks_job_chunk'), 'document_chunks')
    op.drop_constraint(op.f('fk_document_chunks_job_id_jobs'), 'document_chunks')
    op.drop_constraint('positive_chunk_index', 'document_chunks')
    
    # Drop table
    op.drop_table('document_chunks')
    
    # Note: We don't drop pgvector extension as it may be used by other tables


def _validate_config():
    """Validate migration configuration parameters."""
    valid_dimensions = [384, 768, 1536, 3072]
    if VECTOR_DIMENSIONS not in valid_dimensions:
        raise ValueError(
            f"VECTOR_DIMENSIONS must be one of {valid_dimensions}, "
            f"got {VECTOR_DIMENSIONS}"
        )
    
    if not (2 <= HNSW_M <= 100):
        raise ValueError(f"HNSW_M must be between 2 and 100, got {HNSW_M}")
    
    if not (4 <= HNSW_EF_CONSTRUCTION <= 1000):
        raise ValueError(
            f"HNSW_EF_CONSTRUCTION must be between 4 and 1000, "
            f"got {HNSW_EF_CONSTRUCTION}"
        )


def _create_indexes():
    """Create indexes with configurable HNSW parameters."""
    
    # Foreign key index for job lookups
    op.create_index(
        'idx_document_chunks_job_id',
        'document_chunks',
        ['job_id']
    )
    
    # Timestamp index for time-based queries
    op.create_index(
        'idx_document_chunks_created_at',
        'document_chunks',
        ['created_at']
    )
    
    # HNSW index for vector similarity search
    op.execute(f"""
        CREATE INDEX idx_document_chunks_embedding_hnsw 
        ON document_chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})
    """)
```

### Environment-Specific Configuration
```yaml
# alembic/versions/config/vector_store.yaml
environments:
  development:
    vector_dimensions: 1536
    hnsw:
      m: 8
      ef_construction: 32
      ef_search: 16
  
  staging:
    vector_dimensions: 1536
    hnsw:
      m: 16
      ef_construction: 64
      ef_search: 32
  
  production:
    vector_dimensions: 1536
    hnsw:
      m: 16
      ef_construction: 64
      ef_search: 32
    
  production_high_accuracy:
    vector_dimensions: 1536
    hnsw:
      m: 32
      ef_construction: 128
      ef_search: 64
```

### Migration Helper Script
```python
"""CLI helper for running migrations with custom parameters."""

import click
import os

@click.command()
@click.option('--dimensions', default=1536, type=int, help='Vector dimensions')
@click.option('--hnsw-m', default=16, type=int, help='HNSW m parameter')
@click.option('--hnsw-ef', default=64, type=int, help='HNSW ef_construction parameter')
def migrate_vector_store(dimensions: int, hnsw_m: int, hnsw_ef: int) -> None:
    """Run vector store migration with custom parameters."""
    os.environ['VECTOR_DIMENSIONS'] = str(dimensions)
    os.environ['HNSW_M'] = str(hnsw_m)
    os.environ['HNSW_EF_CONSTRUCTION'] = str(hnsw_ef)
    
    # Run alembic upgrade
    import subprocess
    subprocess.run(['alembic', 'upgrade', '003_add_document_chunks'], check=True)


if __name__ == '__main__':
    migrate_vector_store()
```

## Behavior

### Upgrade Process
1. Validate configuration parameters
2. Enable `pgvector` extension (idempotent)
3. Create `document_chunks` table with all columns
4. Add constraints (FK, unique, check)
5. Create indexes in order: B-tree first, HNSW last
6. Log configuration used for audit trail

### Downgrade Process
1. Drop indexes (reverse order of creation)
2. Drop constraints
3. Drop table
4. **Preserve** `pgvector` extension (may be used by other features)
5. Log downgrade completion

### Parameter Validation
- Vector dimensions: 384, 768, 1536, or 3072 (common embedding sizes)
- HNSW `m`: 2-100 (builds fail outside this range)
- HNSW `ef_construction`: 4-1000 (builds fail outside this range)

### Idempotency
- Extension creation uses `IF NOT EXISTS`
- Index creation uses `IF NOT EXISTS` (when supported)
- Constraint names use deterministic naming convention
- Safe to run upgrade multiple times (no-op on subsequent runs)

## Error Handling

| Error | Condition | Resolution |
|-------|-----------|------------|
| `ValueError` | Invalid configuration | Check env vars, fix and retry |
| `ProgrammingError` | pgvector not installed | Install extension: `CREATE EXTENSION vector` |
| `IntegrityError` | Existing table with same name | Manual cleanup or use `--sql` to generate script |
| `OperationalError` | Insufficient privileges | Grant CREATE privileges to migration user |
| `TimeoutError` | Large table build timeout | Increase `statement_timeout` temporarily |

### Pre-Migration Checks
```python
def pre_migration_check(conn):
    """Verify prerequisites before running migration."""
    # Check pgvector availability
    result = conn.execute("SELECT * FROM pg_available_extensions WHERE name = 'vector'")
    if not result.fetchone():
        raise RuntimeError("pgvector extension not available. Install pgvector first.")
    
    # Check PostgreSQL version
    result = conn.execute("SHOW server_version")
    version = result.scalar()
    major = int(version.split('.')[0])
    if major < 14:
        raise RuntimeError(f"PostgreSQL 14+ required, found {version}")
    
    # Check available memory for HNSW build
    result = conn.execute("SHOW shared_buffers")
    shared_buffers = result.scalar()
    print(f"Note: Current shared_buffers = {shared_buffers}")
```

## Performance Considerations

### Migration Timing Estimates
| Records | Build Time | Recommended Approach |
|---------|------------|---------------------|
| 0 (empty) | <1s | Standard migration |
| 10K | 2-5s | Standard migration |
| 100K | 30-60s | Standard migration |
| 1M | 5-15min | Maintenance window |
| 10M+ | 1-3 hours | Background migration strategy |

### Large Dataset Strategy
```python
def upgrade_large_dataset():
    """Migration strategy for datasets > 1M records."""
    # 1. Create table without HNSW index
    op.create_table(...)
    
    # 2. Create B-tree indexes only
    _create_btree_indexes()
    
    # 3. Backfill data (if migrating existing data)
    # This happens outside migration in background job
    
    # 4. Create HNSW index CONCURRENTLY (doesn't lock table)
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_document_chunks_embedding_hnsw 
        ON document_chunks 
        USING hnsw (embedding vector_cosine_ops)
    """)
```

### Memory Configuration
```sql
-- Temporarily increase memory for index build
SET maintenance_work_mem = '2GB';
SET max_parallel_maintenance_workers = 4;
```

### Concurrent Index Creation
For zero-downtime deployments on large tables:
```sql
-- Use CONCURRENTLY to avoid table locks
CREATE INDEX CONCURRENTLY idx_document_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops);
```

**Note:** CONCURRENTLY cannot run in a transaction block, requires special handling in Alembic.
