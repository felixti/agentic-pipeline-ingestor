# Spec: Performance Optimization with Combined Indexes

## Purpose
Ensure hybrid search performs efficiently at scale through strategic database indexing, query optimization, and caching strategies.

## Interface

### Index Configuration
```yaml
# config/vector_store.yaml
index_optimization:
  # HNSW index parameters for vector search
  hnsw:
    m: 16                       # Connections per layer (2-100)
    ef_construction: 64         # Build-time accuracy tradeoff (10-1000)
    ef_search: 32              # Query-time accuracy tradeoff (1-1000)
    distance_metric: "cosine"  # cosine, l2, inner_product
  
  # Full-text search indexes
  text_search:
    index_type: "gin"          # gin or gist
    languages: ["english"]     # Text search configurations
    include_trgm: true         # Enable trigram indexes for fuzzy matching
  
  # Combined/covering indexes
  composite:
    enabled: true
    include_metadata: true     # Include common filter columns
  
  # Partial indexes for common queries
  partial:
    enabled: true
    recent_data_days: 30       # Index only recent data for hot queries
```

### Query Hints API (Advanced)
```json
{
  "query": "machine learning",
  "optimization_hints": {
    "use_index_only_scan": true,
    "parallel_workers": 2,
    "work_mem_mb": 64
  }
}
```

## Behavior

### Required Indexes

#### 1. HNSW Vector Index
```sql
-- Primary vector similarity index
CREATE INDEX idx_document_chunks_embedding_hnsw
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat for memory-constrained environments
CREATE INDEX idx_document_chunks_embedding_ivf
ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

#### 2. Full-Text Search Index
```sql
-- GIN index for tsvector (fast containment queries)
CREATE INDEX idx_document_chunks_content_gin
ON document_chunks
USING gin (to_tsvector('english', content));

-- Trigram index for fuzzy matching
CREATE INDEX idx_document_chunks_content_trgm
ON document_chunks
USING gin (content gin_trgm_ops);
```

#### 3. Composite Indexes
```sql
-- Covering index for filtered vector queries
CREATE INDEX idx_document_chunks_job_embedding
ON document_chunks (job_id)
INCLUDE (embedding, content, metadata);

-- Partial index for recent data (hot queries)
CREATE INDEX idx_document_chunks_recent
ON document_chunks (created_at, job_id)
WHERE created_at > NOW() - INTERVAL '30 days';

-- Combined text + metadata index
CREATE INDEX idx_document_chunks_job_search
ON document_chunks (job_id, to_tsvector('english', content));
```

### Query Optimization Patterns

#### Optimized Hybrid Query
```sql
-- Use CTEs for materialization and parallel planning
EXPLAIN (ANALYZE, BUFFERS)
WITH 
-- Materialize vector results with covering index
vector_hits AS MATERIALIZED (
    SELECT 
        dc.id,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> $1) AS vector_similarity
    FROM document_chunks dc
    WHERE dc.job_id = $4  -- Filter early
    ORDER BY dc.embedding <=> $1
    LIMIT $2 * 3  -- Over-fetch for better merging
),
-- Materialize text results
text_hits AS MATERIALIZED (
    SELECT 
        dc.id,
        ts_rank_cd(
            to_tsvector('english', dc.content), 
            plainto_tsquery('english', $3),
            32  -- normalization option
        ) AS text_rank
    FROM document_chunks dc
    WHERE dc.job_id = $4
      AND to_tsvector('english', dc.content) @@ plainto_tsquery('english', $3)
    ORDER BY text_rank DESC
    LIMIT $2 * 3
),
-- Combine with fallback handling
combined AS (
    SELECT 
        COALESCE(v.id, t.id) AS id,
        v.content,
        v.metadata,
        COALESCE(v.vector_similarity, 0) AS vector_score,
        COALESCE(t.text_rank / NULLIF(MAX(t.text_rank) OVER (), 0), 0) AS text_score
    FROM vector_hits v
    FULL OUTER JOIN text_hits t ON v.id = t.id
)
SELECT 
    id,
    content,
    metadata,
    (0.7 * vector_score + 0.3 * text_score) AS final_score,
    vector_score,
    text_score
FROM combined
ORDER BY final_score DESC
LIMIT $2;
```

### Index Selection Strategy

| Query Pattern | Primary Index | Secondary | Notes |
|--------------|---------------|-----------|-------|
| Vector only | HNSW | - | Approximate nearest neighbor |
| Text only | GIN (tsvector) | - | Inverted index lookup |
| Hybrid (unfiltered) | HNSW + GIN | - | Parallel index scans |
| Hybrid (job-filtered) | Composite (job_id) | HNSW | Bitmap AND operation |
| Recent data only | Partial (created_at) | - | Smaller index footprint |
| Fuzzy matching | GIN (trgm) | - | Similarity threshold queries |

### Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Vector search p99 | < 50ms | 1M vectors, 1536 dimensions |
| Text search p99 | < 20ms | Full-text with GIN index |
| Hybrid search p99 | < 100ms | Combined with fallback |
| Index build time | < 30min | 1M vectors on standard hardware |
| Memory per 1M vectors | ~6GB | HNSW with m=16, 1536 dims |
| Index-only scan ratio | > 80% | Covering indexes effective |

### Caching Strategy

#### Application-Level Caching
```python
# Cache frequent vector searches
@cache(ttl=300, key_prefix="vector_search")
async def vector_search_cached(query_embedding, job_id, top_k):
    return await vector_search(query_embedding, job_id, top_k)

# Cache normalized text ranks per job
@cache(ttl=600, key_prefix="text_rank_stats")
async def get_text_rank_stats(job_id):
    """Cache max text rank for normalization"""
    return await calculate_max_rank(job_id)
```

#### Database Configuration
```sql
-- Enable parallel query execution
SET max_parallel_workers_per_gather = 4;
SET work_mem = '64MB';

-- Optimize for index-only scans
SET enable_indexonlyscan = on;
SET random_page_cost = 1.1;  -- For SSD storage
```

### Connection Pooling
```yaml
# Connection pool sizing for hybrid search
database:
  pool:
    min_size: 5
    max_size: 20
    max_overflow: 10
    
  # Separate pools for read-heavy search operations
  search_pool:
    max_size: 30
    command_timeout: 5  # seconds
```

## Error Handling

| Error Code | Condition | Response |
|------------|-----------|----------|
| 503 | HNSW index not available | Fall back to exact vector search with warning |
| 503 | Query timeout | Return partial results, log slow query |
| 500 | Index corruption | Rebuild index automatically, retry query |
| 429 | Too many concurrent searches | Queue request or return rate limit error |

## Performance Considerations

### HNSW Tuning Guide

**For High Recall (accuracy):**
- Increase `ef_construction` to 128-256 during index build
- Increase `ef_search` to 64-100 at query time
- Trade-off: Slower index build, more memory, slower queries

**For Low Latency:**
- Decrease `ef_search` to 16-32
- Use smaller `m` value (8-12)
- Trade-off: Lower recall, faster queries

**Memory Formula:**
```
HNSW_memory ≈ (dimensions × 4 bytes + overhead) × n_vectors × (m/2)
Example: 1536 dims × 4 × 1M × 8 ≈ 49GB (worst case, usually ~6GB)
```

### Index Maintenance

#### Automated Maintenance Tasks
```sql
-- Weekly: Update statistics
ANALYZE document_chunks;

-- Monthly: Reindex if fragmentation detected
REINDEX INDEX CONCURRENTLY idx_document_chunks_embedding_hnsw;

-- Monitor index bloat
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan,
    idx_tup_read
FROM pg_stat_user_indexes
WHERE tablename = 'document_chunks';
```

#### Partitioning Strategy (Future)
For >10M chunks:
```sql
-- Partition by job_id range or hash
CREATE TABLE document_chunks (
    id UUID,
    job_id UUID,
    content TEXT,
    embedding VECTOR(1536)
) PARTITION BY HASH (job_id);
```

### Monitoring Queries

```sql
-- Check index usage
SELECT 
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- Query performance
SELECT 
    query,
    mean_exec_time,
    calls,
    total_exec_time
FROM pg_stat_statements
WHERE query LIKE '%document_chunks%'
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### Benchmarking Checklist
- [ ] Vector search latency at 1K, 100K, 1M, 10M vectors
- [ ] Text search latency with various query complexities
- [ ] Hybrid search with different weight configurations
- [ ] Concurrent user load testing (10, 50, 100 users)
- [ ] Memory usage under load
- [ ] Index build time for various data sizes
- [ ] Query plan analysis for index usage
