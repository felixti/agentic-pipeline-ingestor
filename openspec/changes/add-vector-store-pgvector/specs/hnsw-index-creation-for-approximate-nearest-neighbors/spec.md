# Spec: HNSW Index Creation for Approximate Nearest Neighbors

## Purpose
Create HNSW (Hierarchical Navigable Small World) indexes on the embedding column to enable fast approximate nearest neighbor (ANN) similarity search with configurable accuracy/speed tradeoffs.

## Interface

### SQL DDL - Basic HNSW Index
```sql
-- HNSW index with cosine similarity (recommended for normalized embeddings)
CREATE INDEX idx_document_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops);
```

### SQL DDL - Configured HNSW Index
```sql
-- HNSW index with custom parameters for production workloads
CREATE INDEX idx_document_chunks_embedding_hnsw_tuned 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,              -- Number of connections per layer (2-100)
    ef_construction = 64  -- Build-time accuracy vs speed tradeoff (4-1000)
);
```

### Alternative Distance Operators
```sql
-- For Euclidean/L2 distance (when embeddings not normalized)
CREATE INDEX idx_document_chunks_embedding_l2 
ON document_chunks 
USING hnsw (embedding vector_l2_ops);

-- For inner product (for specific embedding models)
CREATE INDEX idx_document_chunks_embedding_ip 
ON document_chunks 
USING hnsw (embedding vector_ip_ops);
```

### Configuration Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| m | 16 | 2-100 | Number of bi-directional links per node; higher = more accurate, slower build |
| ef_construction | 64 | 4-1000 | Size of dynamic candidate list during construction; higher = more accurate, slower build |

## Behavior

### Index Construction
1. HNSW index builds incrementally as rows are inserted
2. Index build time scales with `m` and `ef_construction` parameters
3. Index size scales with `m` parameter (more links = more storage)
4. Index maintains itself automatically on INSERT, UPDATE, DELETE

### Query-Time Behavior
1. Uses `vector_cosine_ops` for cosine similarity: `embedding <=> query_vector` (distance, lower is closer)
2. Uses `1 - (embedding <=> query_vector)` for cosine similarity score (higher is closer)
3. Query performance controlled by `ef_search` parameter (set per query)
4. Returns approximate nearest neighbors (may miss true nearest neighbor occasionally)

### Similarity Search Query Pattern
```sql
-- Get top-k most similar chunks using HNSW index
SET hnsw.ef_search = 32;  -- Query-time accuracy parameter

SELECT 
    id,
    job_id,
    chunk_index,
    content,
    embedding <=> $1::vector AS distance,  -- Cosine distance
    1 - (embedding <=> $1::vector) AS similarity_score
FROM document_chunks
WHERE job_id = $2  -- Optional: filter by job
ORDER BY embedding <=> $1::vector
LIMIT $3;  -- top-k
```

## Error Handling

| Error Code | Condition | Handling |
|------------|-----------|----------|
| `42704` | pgvector extension not installed | Prerequisite check required before migration |
| `22023` | Invalid parameter value (m < 2, ef_construction < 4) | Validate parameters before index creation |
| `54000` | Statement too complex | Reduce vector dimensions or batch operations |
| `53200` | Out of memory | Insufficient `maintenance_work_mem` for index build |
| Custom | Dimension mismatch between index and query | Validate embedding dimensions match at query time |

### Parameter Validation
```python
# Recommended parameter validation
def validate_hnsw_params(m: int, ef_construction: int) -> None:
    if not (2 <= m <= 100):
        raise ValueError(f"HNSW parameter 'm' must be between 2 and 100, got {m}")
    if not (4 <= ef_construction <= 1000):
        raise ValueError(f"HNSW parameter 'ef_construction' must be between 4 and 1000, got {ef_construction}")
```

## Performance Considerations

### Index Build Performance
| Records | m=16, ef=64 | m=32, ef=128 | Notes |
|---------|-------------|--------------|-------|
| 10K | ~2s | ~5s | Fast enough for dev/test |
| 100K | ~20s | ~60s | Acceptable for production deploy |
| 1M | ~5min | ~15min | Consider background migration |
| 10M | ~1hr | ~3hr | Requires maintenance window |

### Query Performance vs Accuracy Tradeoff
```sql
-- Set ef_search per query (default is 40)
SET LOCAL hnsw.ef_search = 16;   -- Fast, less accurate
SET LOCAL hnsw.ef_search = 64;   -- Balanced (recommended)
SET LOCAL hnsw.ef_search = 200;  -- Slower, more accurate
SET LOCAL hnsw.ef_search = 400;  -- Near-exhaustive search
```

| ef_search | Recall@10 | Query Time (1M vectors) | Use Case |
|-----------|-----------|------------------------|----------|
| 16 | ~85% | 2ms | Real-time, acceptable approx |
| 32 | ~92% | 3ms | Balanced default |
| 64 | ~97% | 5ms | High accuracy required |
| 100 | ~99% | 8ms | Critical retrieval |

### Memory Requirements
- HNSW index loads entirely into `shared_buffers`
- Approximate size: rows * (m * 8 bytes overhead + dimension * 4 bytes)
- For 1M records, 1536 dims, m=16: ~8GB RAM recommended

### Storage Tradeoffs
| m | Index Size | Build Time | Query Speed | Recall |
|---|------------|------------|-------------|--------|
| 8 | 0.5x data | Fast | Fast | Lower |
| 16 | 1x data | Medium | Medium | Good |
| 32 | 2x data | Slow | Slow | High |
| 64 | 4x data | Very Slow | Slower | Very High |

### Recommended Configurations
```yaml
# Development
hnsw:
  m: 8
  ef_construction: 32
  ef_search: 16

# Production (balanced)
hnsw:
  m: 16
  ef_construction: 64
  ef_search: 32

# High-accuracy (search-heavy)
hnsw:
  m: 32
  ef_construction: 128
  ef_search: 64
```

### Concurrent Considerations
- HNSW supports concurrent INSERT and SELECT
- Index is not locked during queries
- Heavy write load may degrade query performance temporarily
- Consider batching writes during peak read periods
