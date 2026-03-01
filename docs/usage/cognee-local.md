# Using CogneeLocalDestination

This guide covers the configuration and usage of `CogneeLocalDestination` for local GraphRAG with Neo4j and pgvector.

## Overview

`CogneeLocalDestination` is a pipeline destination plugin that:
- Extracts entities and relationships from documents using LLM
- Stores entities in Neo4j (graph database)
- Stores vector embeddings in PostgreSQL/pgvector
- Provides hybrid search (vector + graph) capabilities

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# ============================================
# Neo4j Configuration (Required)
# ============================================
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cognee-graph-db

# ============================================
# LLM Configuration (Required)
# Uses existing litellm provider
# ============================================
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-azure-key
OPENROUTER_API_KEY=your-openrouter-key  # Optional fallback

# ============================================
# Cognee Settings (Optional)
# ============================================
COGNEE_LLM_PROVIDER=litellm
COGNEE_LLM_MODEL=azure/gpt-4.1
COGNEE_LLM_FALLBACK_MODEL=openrouter/openai/gpt-4.1
COGNEE_EMBEDDING_MODEL=azure/text-embedding-3-small
COGNEE_GRAPH_PROVIDER=neo4j
COGNEE_VECTOR_PROVIDER=pgvector
COGNEE_TIMEOUT=60

# ============================================
# PostgreSQL (Existing - Required)
# ============================================
DB_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/pipeline
```

### Plugin Configuration

When submitting jobs via API:

```python
destination_config = {
    "dataset_id": "my-dataset",           # Required: Dataset name
    "graph_name": "knowledge-graph",      # Optional: Graph namespace
    "extract_entities": True,              # Enable entity extraction
    "extract_relationships": True,         # Enable relationship mapping
    "search_type": "hybrid",               # Default search type
    "top_k": 10,                           # Default results count
}
```

### YAML Configuration

```yaml
# config/destinations.yaml
destinations:
  cognee_local:
    type: cognee_local
    enabled: true
    config:
      # Connection settings (from env vars)
      neo4j_uri: "${NEO4J_URI}"
      neo4j_user: "${NEO4J_USER}"
      neo4j_password: "${NEO4J_PASSWORD}"
      
      # Processing settings
      default_dataset: "default-graph"
      extract_entities: true
      extract_relationships: true
      
      # Search settings
      default_search_type: "hybrid"
      default_top_k: 10
      
      # Performance settings
      batch_size: 100
      max_workers: 4
```

## Usage Examples

### Initialize Destination

```python
from src.plugins.destinations import CogneeLocalDestination

# Create instance
destination = CogneeLocalDestination()

# Initialize with configuration
await destination.initialize({
    "dataset_id": "my-dataset",
    "graph_name": "knowledge-graph",
    "extract_entities": True,
    "extract_relationships": True,
    # Connection details from env vars
})

# Verify initialization
print(f"Destination initialized: {destination.is_initialized}")
```

### Write Documents

```python
from src.pipelines.models import TransformedData, Chunk

# Prepare transformed data
transformed_data = TransformedData(
    job_id="job-123",
    chunks=[
        Chunk(
            content="Machine learning is a subset of artificial intelligence...",
            metadata={"page": 1, "source": "ml-overview.pdf"}
        ),
        Chunk(
            content="Neural networks are inspired by biological neurons...",
            metadata={"page": 2, "source": "ml-overview.pdf"}
        )
    ],
    metadata={"total_pages": 10}
)

# Connect to dataset
conn = await destination.connect({
    "dataset_id": "my-dataset"
})

# Write data
result = await destination.write(conn, transformed_data)

print(f"Success: {result.success}")
print(f"Records written: {result.records_written}")
print(f"Destination URI: {result.destination_uri}")
```

### Search

```python
# Connect first
conn = await destination.connect({"dataset_id": "my-dataset"})

# Hybrid search (recommended)
results = await destination.search(
    conn=conn,
    query="machine learning applications in healthcare",
    search_type="hybrid",
    top_k=10
)

# Process results
for result in results:
    print(f"Content: {result.content[:100]}...")
    print(f"Score: {result.score}")
    print(f"Entities: {result.entities}")
    print(f"Source: {result.metadata.get('source')}")
    print("---")
```

### Search Types

#### 1. Vector Search

Semantic similarity using pgvector:

```python
results = await destination.search(
    conn=conn,
    query="neural network architecture",
    search_type="vector",
    top_k=10
)
```

**Best for:** Finding conceptually similar content, semantic matching

#### 2. Graph Search

Graph traversal using Neo4j:

```python
results = await destination.search(
    conn=conn,
    query="What companies work on machine learning?",
    search_type="graph",
    top_k=10
)
```

**Best for:** Entity relationships, structured queries

#### 3. Hybrid Search (Recommended)

Combines vector + graph with fusion ranking:

```python
results = await destination.search(
    conn=conn,
    query="machine learning",
    search_type="hybrid",
    top_k=10
)
```

**Best for:** General queries, comprehensive results

## Advanced Usage

### Batch Processing

```python
# Process multiple documents in batch
from asyncio import gather

async def process_batch(documents):
    tasks = []
    for doc in documents:
        task = destination.write(conn, doc)
        tasks.append(task)
    
    results = await gather(*tasks, return_exceptions=True)
    
    success_count = sum(1 for r in results if getattr(r, 'success', False))
    print(f"Processed {success_count}/{len(documents)} documents")
    return results

# Usage
docs = [doc1, doc2, doc3, ...]
results = await process_batch(docs)
```

### Custom Entity Extraction

```python
# Configure entity types
custom_config = {
    "dataset_id": "my-dataset",
    "extract_entities": True,
    "entity_types": ["PERSON", "ORGANIZATION", "TECHNOLOGY", "CONCEPT"],
    "relationship_types": ["WORKS_AT", "USES", "RELATED_TO"],
    "confidence_threshold": 0.7
}

await destination.initialize(custom_config)
```

### Query with Filters

```python
# Search with metadata filters
results = await destination.search(
    conn=conn,
    query="machine learning",
    filters={
        "source": "research-papers",
        "date_range": {"from": "2024-01-01", "to": "2024-12-31"}
    },
    top_k=10
)
```

### Multi-Dataset Search

```python
# Search across multiple datasets
results = await destination.search(
    conn=conn,
    query="artificial intelligence",
    datasets=["dataset-1", "dataset-2", "dataset-3"],
    top_k=10
)
```

## Monitoring

### Neo4j Browser

Access Neo4j Browser at http://localhost:7474

**Useful Queries:**

```cypher
-- Count nodes by type
MATCH (n) RETURN labels(n)[0] as type, count(n) as count

-- Find entities related to a concept
MATCH (e:Entity)-[r]-(related)
WHERE e.name CONTAINS 'machine learning'
RETURN e.name, type(r), related.name
LIMIT 10

-- Document-entity relationships
MATCH (d:Document)<-[:APPEARS_IN]-(e:Entity)
RETURN d.title, count(e) as entity_count
ORDER BY entity_count DESC
```

### Health Check

```python
# Check destination health
health = await destination.health_check()
print(f"Healthy: {health.healthy}")
print(f"Neo4j: {health.components['neo4j']}")
print(f"PostgreSQL: {health.components['postgresql']}")
```

### Metrics

```python
# Get processing metrics
metrics = await destination.get_metrics()
print(f"Documents processed: {metrics.documents_processed}")
print(f"Entities extracted: {metrics.entities_extracted}")
print(f"Average processing time: {metrics.avg_processing_time_ms}ms")
```

## Error Handling

### Common Errors

```python
from src.plugins.destinations.exceptions import (
    DestinationError,
    ConnectionError,
    WriteError,
    SearchError
)

try:
    conn = await destination.connect(config)
    result = await destination.write(conn, data)
except ConnectionError as e:
    print(f"Connection failed: {e}")
    # Retry logic or fallback
except WriteError as e:
    print(f"Write failed: {e}")
    # Log and continue with next batch
except SearchError as e:
    print(f"Search failed: {e}")
    # Return empty results
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def write_with_retry(destination, conn, data):
    return await destination.write(conn, data)
```

## Performance Tuning

### Batch Size Optimization

```python
# Larger batches = better throughput, higher memory
# Smaller batches = lower latency, less memory

# For small documents (< 1000 chars)
config = {"batch_size": 200}

# For medium documents (1000-5000 chars)
config = {"batch_size": 100}

# For large documents (> 5000 chars)
config = {"batch_size": 50}
```

### Connection Pooling

```python
# Reuse connections for multiple operations
conn = await destination.connect({"dataset_id": "my-dataset"})

try:
    for batch in batches:
        await destination.write(conn, batch)
        
    results = await destination.search(conn, query)
finally:
    await destination.disconnect(conn)
```

### Async Processing

```python
import asyncio

# Process multiple datasets concurrently
async def process_datasets(datasets):
    async with asyncio.TaskGroup() as tg:
        for dataset in datasets:
            tg.create_task(process_dataset(dataset))
```

## API Integration

### REST API

```bash
# Submit job with Cognee destination
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
        "dataset_id": "research-papers",
        "extract_entities": true,
        "extract_relationships": true
      }
    }
  }'

# Search via API
curl -X POST http://localhost:8000/api/v1/search/graph \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "machine learning",
    "dataset_id": "research-papers",
    "search_type": "hybrid",
    "top_k": 10
  }'
```

### WebSocket (Real-time)

```python
# Real-time document processing
async with websockets.connect("ws://localhost:8000/ws") as ws:
    await ws.send(json.dumps({
        "action": "process",
        "destination": {
            "type": "cognee_local",
            "config": {"dataset_id": "real-time-docs"}
        }
    }))
    
    response = await ws.recv()
    print(f"Progress: {response}")
```

## Migration from Other Destinations

### From GraphRAGDestination

```python
# Old
from src.plugins.destinations import GraphRAGDestination
destination = GraphRAGDestination()

# New
from src.plugins.destinations import CogneeLocalDestination
destination = CogneeLocalDestination()

# Same interface, different config
await destination.initialize({
    "dataset_id": "my-dataset",
    # No API endpoint needed - runs locally
})
```

See [Migration Guide](../migration/graphrag-to-cognee.md) for detailed migration steps.

## Troubleshooting

### Debug Mode

```python
import logging
logging.getLogger("cognee").setLevel(logging.DEBUG)
logging.getLogger("neo4j").setLevel(logging.DEBUG)
```

### Connection Issues

```python
# Test Neo4j connection
from neo4j import AsyncGraphDatabase

driver = AsyncGraphDatabase.driver(
    "bolt://neo4j:7687",
    auth=("neo4j", "cognee-graph-db")
)

await driver.verify_connectivity()
print("Neo4j connection OK")
```

### Performance Debugging

```python
import time

start = time.time()
result = await destination.search(conn, query)
print(f"Search took {time.time() - start:.2f}s")
```

## Best Practices

1. **Use hybrid search** for best results
2. **Batch writes** for better throughput
3. **Reuse connections** when processing multiple documents
4. **Monitor Neo4j memory** - adjust heap size if needed
5. **Set appropriate confidence thresholds** for entity extraction
6. **Use filters** to narrow search scope
7. **Implement retry logic** for transient failures

## See Also

- [Migration Guide](../migration/graphrag-to-cognee.md) - Migrating from API GraphRAG
- [Neo4j Infrastructure Spec](../../openspec/changes/graphrag-implementation-cognee/specs/neo4j-infrastructure/spec.md) - Infrastructure details
- [Cognee Integration Spec](../../openspec/changes/graphrag-implementation-cognee/specs/cognee-integration/spec.md) - Technical specification

---

**Last Updated:** 2026-02-28  
**Version:** 1.0
