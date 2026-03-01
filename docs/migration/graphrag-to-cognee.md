# Migration Guide: API GraphRAG to Local Cognee

## Overview

This guide covers migrating from the API-based GraphRAG destination to the new local Cognee destination with Neo4j. The new implementation provides:

- **10x faster** graph operations (< 100ms latency vs. 1-2s API calls)
- **Local processing** - no external API dependencies for graph operations
- **Hybrid search** - combines vector similarity (pgvector) with graph traversal (Neo4j)
- **Lower costs** - no per-request API charges for graph operations
- **Better privacy** - data stays within your infrastructure

## Prerequisites

- Neo4j Docker service running
- PostgreSQL with pgvector (existing)
- Python 3.11+
- Migration script: `scripts/migrate_to_cognee_local.py` (from Task 12)

### Required Environment Variables

```bash
# Neo4j (new)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cognee-graph-db

# LLM (existing - used by Cognee for entity extraction)
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-azure-key
OPENROUTER_API_KEY=your-openrouter-key  # fallback

# PostgreSQL (existing)
DB_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/pipeline
```

## Migration Steps

### 1. Pre-Migration Checklist

Before starting the migration:

- [ ] **Backup existing GraphRAG data**
  ```bash
  # Export existing graph data
  curl -X POST "https://your-graphrag-api.com/export" \
    -H "Authorization: Bearer $TOKEN" \
    -o graph_backup_$(date +%Y%m%d).json
  ```

- [ ] **Verify Neo4j is running**
  ```bash
  docker-compose ps neo4j
  # Should show "healthy"
  ```

- [ ] **Verify PostgreSQL is running**
  ```bash
  docker-compose exec postgres pg_isready -U postgres
  ```

- [ ] **Check disk space** (recommend 2x current data size)
  ```bash
  df -h
  # Ensure at least 10GB free for Neo4j + indexes
  ```

- [ ] **Verify Cognee dependencies installed**
  ```bash
  pip show cognee neo4j
  ```

### 2. Deploy Neo4j Service

Add to your `docker-compose.yml`:

```yaml
neo4j:
  image: neo4j:5.15-community
  container_name: pipeline-neo4j
  environment:
    - NEO4J_AUTH=neo4j/cognee-graph-db
    - NEO4J_dbms_memory_heap_max__size=2G
    - NEO4J_dbms_memory_pagecache_size=1G
  volumes:
    - neo4j-data:/data
    - neo4j-logs:/logs
  ports:
    - "7687:7687"  # Bolt protocol
    - "7474:7474"  # HTTP/Browser
  networks:
    - pipeline-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "cognee-graph-db", "RETURN 1"]
    interval: 30s
    timeout: 10s
    retries: 5

volumes:
  neo4j-data:
  neo4j-logs:
```

Start the service:

```bash
docker-compose up -d neo4j

# Wait for healthy status
docker-compose ps neo4j
```

Access Neo4j Browser at http://localhost:7474 (login: neo4j/cognee-graph-db)

### 3. Run Migration Script

The migration script (`scripts/migrate_to_cognee_local.py`) handles data migration from API GraphRAG to local Cognee.

#### Dry Run (Recommended First Step)

```bash
python scripts/migrate_to_cognee_local.py \
  --source-dataset "your-dataset-name" \
  --dry-run \
  --verbose
```

This will:
- Connect to source API GraphRAG
- Analyze data to be migrated
- Report estimated time and resources
- **Not modify any data**

#### Actual Migration

```bash
# Basic migration
python scripts/migrate_to_cognee_local.py \
  --source-dataset "your-dataset-name" \
  --batch-size 100

# With progress tracking
python scripts/migrate_to_cognee_local.py \
  --source-dataset "your-dataset-name" \
  --batch-size 100 \
  --progress-file migration_progress.json

# Parallel processing (if supported)
python scripts/migrate_to_cognee_local.py \
  --source-dataset "your-dataset-name" \
  --batch-size 100 \
  --workers 4
```

#### With Verification

```bash
# Migrate and verify data integrity
python scripts/migrate_to_cognee_local.py \
  --source-dataset "your-dataset-name" \
  --batch-size 100 \
  --verify

# Verify with custom sample size
python scripts/migrate_to_cognee_local.py \
  --source-dataset "your-dataset-name" \
  --verify \
  --verify-sample-size 50
```

### 4. Post-Migration Verification

After migration completes, verify the following:

- [ ] **Document counts match**
  ```bash
  # Check source count
  curl -s "https://your-graphrag-api.com/datasets/your-dataset/count" | jq '.count'
  
  # Check Neo4j count
  docker-compose exec neo4j cypher-shell -u neo4j -p cognee-graph-db \
    "MATCH (d:Document) RETURN count(d) as document_count"
  ```

- [ ] **Entity counts reasonable**
  ```bash
  docker-compose exec neo4j cypher-shell -u neo4j -p cognee-graph-db \
    "MATCH (e:Entity) RETURN count(e) as entity_count"
  ```

- [ ] **Relationships created**
  ```bash
  docker-compose exec neo4j cypher-shell -u neo4j -p cognee-graph-db \
    "MATCH ()-[r]->() RETURN count(r) as relationship_count"
  ```

- [ ] **Sample queries work**
  ```python
  # Test search
  from src.plugins.destinations import CogneeLocalDestination
  
  destination = CogneeLocalDestination()
  await destination.initialize({
      "dataset_id": "your-dataset-name",
      "neo4j_uri": "bolt://neo4j:7687",
      "neo4j_user": "neo4j",
      "neo4j_password": "cognee-graph-db"
  })
  
  results = await destination.search(
      query="machine learning",
      search_type="hybrid",
      top_k=10
  )
  print(f"Found {len(results)} results")
  ```

- [ ] **Vector embeddings stored**
  ```sql
  -- Connect to PostgreSQL
  SELECT COUNT(*) FROM cognee_vectors WHERE dataset_name = 'your-dataset-name';
  ```

### 5. Switch Destination

Update your pipeline configuration to use `CogneeLocalDestination` instead of `GraphRAGDestination`:

#### API Request

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "source": {
      "type": "upload",
      "uri": "/uploads/document.pdf"
    },
    "destination": {
      "type": "cognee_local",
      "config": {
        "dataset_id": "your-dataset-name",
        "graph_name": "knowledge-graph",
        "extract_entities": true,
        "extract_relationships": true,
        "neo4j_uri": "bolt://neo4j:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "cognee-graph-db"
      }
    }
  }'
```

#### Python SDK

```python
from pipeline_ingestor import PipelineClient

client = PipelineClient(api_key="your-api-key")

# Old: API GraphRAG
# job = client.submit_job(
#     source={"type": "upload", "uri": "/uploads/doc.pdf"},
#     destination={"type": "graphrag", ...}
# )

# New: Local Cognee
job = client.submit_job(
    source={"type": "upload", "uri": "/uploads/doc.pdf"},
    destination={
        "type": "cognee_local",
        "config": {
            "dataset_id": "your-dataset-name",
            "extract_entities": True,
            "extract_relationships": True
        }
    }
)
```

#### Configuration File

```yaml
# config/pipeline.yaml
destinations:
  default: cognee_local
  
  cognee_local:
    type: cognee_local
    config:
      dataset_id: "${DATASET_ID}"
      graph_name: "knowledge-graph"
      extract_entities: true
      extract_relationships: true
      # Neo4j settings from env vars
      neo4j_uri: "${NEO4J_URI}"
      neo4j_user: "${NEO4J_USER}"
      neo4j_password: "${NEO4J_PASSWORD}"
```

## Rollback

If issues occur, you can rollback to API GraphRAG:

### Rollback Specific Dataset

```bash
# Mark migration as failed and restore from backup
python scripts/migrate_to_cognee_local.py \
  --source-dataset "your-dataset-name" \
  --rollback

# Clean up Neo4j data for this dataset
docker-compose exec neo4j cypher-shell -u neo4j -p cognee-graph-db \
  "MATCH (n) WHERE n.dataset = 'your-dataset-name' DETACH DELETE n"
```

### Full Rollback

1. Stop using `CogneeLocalDestination` in pipeline config
2. Revert to `GraphRAGDestination` configuration
3. Clear Neo4j data (optional):
   ```bash
   docker-compose exec neo4j cypher-shell -u neo4j -p cognee-graph-db \
     "MATCH (n) DETACH DELETE n"
   ```

## Troubleshooting

### Issue: Neo4j connection failed

**Symptoms:**
```
neo4j.exceptions.ServiceUnavailable: Failed to establish connection
```

**Solution:**
1. Verify Neo4j is running:
   ```bash
   docker-compose ps neo4j
   docker-compose logs neo4j | tail -50
   ```

2. Check network connectivity:
   ```bash
   docker-compose exec api nc -zv neo4j 7687
   ```

3. Verify credentials:
   ```bash
   docker-compose exec neo4j cypher-shell -u neo4j -p cognee-graph-db "RETURN 1"
   ```

4. Check firewall/network policies

### Issue: Entity extraction timeout

**Symptoms:**
```
cognee.exceptions.ProcessingTimeout: Entity extraction took > 60s
```

**Solution:**
1. Reduce batch size:
   ```bash
   python scripts/migrate_to_cognee_local.py --batch-size 50
   ```

2. Increase timeout:
   ```bash
   export COGNEE_TIMEOUT=120
   ```

3. Check LLM rate limits

### Issue: Memory errors during migration

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solution:**
1. Reduce batch size
2. Increase Neo4j memory:
   ```yaml
   environment:
     - NEO4J_dbms_memory_heap_max__size=4G
   ```

3. Restart Neo4j after memory change:
   ```bash
   docker-compose up -d --force-recreate neo4j
   ```

### Issue: Vector dimension mismatch

**Symptoms:**
```
ValueError: Embedding dimension 1536 does not match expected 768
```

**Solution:**
1. Check embedding model configuration:
   ```bash
   echo $COGNEE_EMBEDDING_MODEL
   ```

2. Ensure consistency with existing data:
   ```python
   # Verify model settings
   import os
   assert os.getenv("COGNEE_EMBEDDING_MODEL") == "azure/text-embedding-3-small"
   ```

### Issue: Dataset already exists

**Symptoms:**
```
cognee.exceptions.DatasetExistsError: Dataset 'your-dataset' already exists
```

**Solution:**
1. Use force flag to overwrite:
   ```bash
   python scripts/migrate_to_cognee_local.py --force
   ```

2. Or delete existing dataset first:
   ```python
   await cognee.delete_dataset("your-dataset")
   ```

## Performance Expectations

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| Migration speed | 50-100 docs/minute | Depends on document size and LLM rate limits |
| Verification | 200 docs/minute | Sample-based verification |
| Graph query latency | < 100ms | For typical 10-entity queries |
| Entity extraction | 5-10 docs/minute | LLM-bound operation |
| Total time estimate | Dataset size / 50 docs/min | Add 20% buffer |

### Example Timelines

| Dataset Size | Estimated Time | With Verification |
|-------------|----------------|-------------------|
| 1,000 docs | 20-30 min | 25-35 min |
| 10,000 docs | 3-4 hours | 3.5-5 hours |
| 50,000 docs | 15-20 hours | 17-24 hours |

**Note:** Times are estimates. Actual performance depends on:
- Document complexity
- LLM response times
- Network latency
- Hardware resources

## Best Practices

1. **Always run dry-run first** to identify potential issues
2. **Migrate in batches** for large datasets (>10,000 docs)
3. **Monitor resource usage** during migration
4. **Keep API GraphRAG running** until verification complete
5. **Test search functionality** before decommissioning old system
6. **Document any custom entity types** for reference

## Monitoring Migration Progress

```bash
# Watch Neo4j node count
docker-compose exec neo4j cypher-shell -u neo4j -p cognee-graph-db \
  "MATCH (n) RETURN count(n) as nodes, timestamp() as time"

# Monitor PostgreSQL vector count
docker-compose exec postgres psql -U postgres -d pipeline -c \
  "SELECT dataset_name, COUNT(*) FROM cognee_vectors GROUP BY dataset_name;"

# Check migration log
tail -f migration_progress.json
```

## Support

For migration support:

- **Technical Issues:** Open a GitHub issue with migration logs
- **Performance Questions:** Check the benchmarking guide at `docs/performance/cognee-benchmarks.md`
- **Architecture Questions:** Review design doc at `openspec/changes/graphrag-implementation-cognee/design.md`

## Migration Checklist Summary

- [ ] Backup existing GraphRAG data
- [ ] Deploy Neo4j service
- [ ] Run dry-run migration
- [ ] Execute actual migration
- [ ] Verify document counts
- [ ] Verify entity relationships
- [ ] Test search queries
- [ ] Update pipeline configuration
- [ ] Monitor production usage
- [ ] Decommission API GraphRAG (after stable period)

---

**Last Updated:** 2026-02-28  
**Version:** 1.0  
**Related Docs:** [CogneeLocalDestination Usage](../usage/cognee-local.md), [Neo4j Infrastructure](../../openspec/changes/graphrag-implementation-cognee/specs/neo4j-infrastructure/spec.md)
