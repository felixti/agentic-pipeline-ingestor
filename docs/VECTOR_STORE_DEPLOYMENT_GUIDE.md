# Vector Store Deployment Guide

This guide covers deploying the pgvector-based vector store feature to your environment.

## Prerequisites

- PostgreSQL 14+ with pgvector extension support
- Docker (for containerized deployment)
- Existing Agentic Pipeline Ingestor installation

## Quick Start

### 1. Update Docker Image

Pull the pgvector-enabled PostgreSQL image:

```bash
docker-compose pull postgres
docker-compose up -d postgres
```

### 2. Run Database Migrations

Apply the vector store migrations:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run migrations
alembic upgrade head
```

This will:
- Enable pgvector extension
- Enable pg_trgm extension
- Create document_chunks table
- Create HNSW and GIN indexes

### 3. Configure Vector Store

Edit `config/vector_store.yaml`:

```yaml
vector_store:
  enabled: true
  
  embedding:
    model: "text-embedding-3-small"  # or your preferred model
    dimensions: 1536
    batch_size: 100
    
  search:
    default_top_k: 10
    max_top_k: 100
    default_min_similarity: 0.7
    
  index:
    hnsw_m: 16
    hnsw_ef_construction: 64
    hnsw_ef_search: 32
    
  hybrid:
    default_vector_weight: 0.7
    default_text_weight: 0.3
```

### 4. Set Environment Variables

```bash
# Required for embedding generation
export EMBEDDING_MODEL="text-embedding-3-small"
export EMBEDDING_DIMENSIONS=1536
export EMBEDDING_API_KEY="your-api-key"

# Optional: Custom embedding endpoint
export EMBEDDING_API_BASE="https://your-endpoint.com"
```

### 5. Verify Installation

Check the health endpoint:

```bash
curl http://localhost:8000/health/vector
```

Expected response:
```json
{
  "healthy": true,
  "status": "healthy",
  "extensions": {
    "vector": "0.5.1",
    "pg_trgm": "1.6"
  }
}
```

## Cloud Provider Specific Instructions

### AWS RDS

AWS RDS doesn't support pgvector directly. Options:

1. **Use RDS PostgreSQL with pgvector compiled**
   - Launch RDS PostgreSQL 14+
   - Use custom parameter group
   - Contact AWS support for pgvector extension

2. **Use Aurora PostgreSQL**
   - Aurora supports pgvector in compatible versions
   - Check AWS documentation for availability

3. **Self-managed PostgreSQL on EC2**
   - Full control over extensions
   - Install pgvector manually

### Azure Database for PostgreSQL

Azure Database for PostgreSQL Flexible Server supports pgvector:

```sql
-- Connect to your database
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### Google Cloud SQL

Google Cloud SQL PostgreSQL supports pgvector:

1. Enable the extension in Cloud Console
2. Run migrations as normal

## Production Considerations

### 1. Index Tuning

For large datasets (>1M chunks), tune HNSW parameters:

```yaml
index:
  hnsw_m: 32              # More connections = better recall, slower build
  hnsw_ef_construction: 128  # Higher = better recall, slower build
  hnsw_ef_search: 64      # Higher = better recall, slower search
```

### 2. Connection Pooling

Ensure adequate database connections:

```yaml
database:
  pool_size: 20
  max_overflow: 40
```

### 3. Memory Requirements

pgvector HNSW indexes are memory-intensive:

- 1M chunks × 1536 dims ≈ 6GB index size
- Plan for 2-3x index size in RAM for best performance

### 4. Monitoring

Monitor these metrics:

```sql
-- Index size
SELECT pg_size_pretty(pg_relation_size('idx_document_chunks_embedding_hnsw'));

-- Query performance
EXPLAIN ANALYZE
SELECT * FROM document_chunks
ORDER BY embedding <=> 'query_vector'::vector
LIMIT 10;
```

### 5. Backup Strategy

Include vector data in backups:

```bash
# Standard PostgreSQL backup includes everything
pg_dump -h localhost -U postgres pipeline > backup.sql
```

## Migration from External Vector Stores

If migrating from Pinecone, Weaviate, etc.:

1. Export vectors from external store
2. Import to PostgreSQL:

```python
# Bulk import script
import asyncio
from src.db.repositories.document_chunk_repository import DocumentChunkRepository
from src.db.models import DocumentChunkModel

async def import_vectors(chunks_data):
    async with get_session() as session:
        repo = DocumentChunkRepository(session)
        chunks = [
            DocumentChunkModel(
                job_id=chunk['job_id'],
                chunk_index=chunk['chunk_index'],
                content=chunk['content'],
                embedding=chunk['embedding'],
                chunk_metadata=chunk.get('metadata', {})
            )
            for chunk in chunks_data
        ]
        await repo.bulk_create(chunks)
```

## Troubleshooting

### Extension Not Available

```
ERROR: could not open extension control file
```

**Solution**: PostgreSQL image doesn't have pgvector. Use `pgvector/pgvector:pg17` image.

### Out of Memory During Index Build

```
ERROR: out of memory
```

**Solution**: Reduce `hnsw_m` and `hnsw_ef_construction`, or increase PostgreSQL memory:

```yaml
# docker-compose.yml
postgres:
  shm_size: '2gb'
  environment:
    - PGOPTIONS=-c maintenance_work_mem=1GB
```

### Slow Search Queries

**Check**: 
1. HNSW index is being used:
```sql
EXPLAIN SELECT ... ORDER BY embedding <=> query;
-- Should show "Index Scan using idx_document_chunks_embedding_hnsw"
```

2. Increase `ef_search`:
```sql
SET hnsw.ef_search = 100;
```

### Dimension Mismatch

```
ERROR: expected 1536 dimensions, not 768
```

**Solution**: Ensure embedding model matches configured dimensions:

```yaml
embedding:
  dimensions: 768  # Match your model
```

## Rollback

If you need to rollback:

```bash
# Revert migrations
alembic downgrade -2

# Or remove vector store completely
alembic downgrade base
```

## Support

For issues:
1. Check logs: `docker-compose logs api`
2. Verify health: `curl /health/vector`
3. Review QA report: `shared/qa-reports/vector-store-qa-report.md`
