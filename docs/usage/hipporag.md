# Using HippoRAG for Multi-Hop Reasoning

## Overview

HippoRAG provides advanced multi-hop reasoning using a neurobiological memory model. It's ideal for complex questions that require connecting information across multiple documents.

## When to Use HippoRAG

**Best for:**
- Complex questions spanning multiple documents
- Research paper synthesis
- Legal case analysis
- Medical diagnosis from multiple records
- "Connecting the dots" scenarios

**Not ideal for:**
- Simple factual lookup
- Single-document questions
- Real-time chat (indexing takes time)

## Quick Start

### 1. Configure Environment

```bash
# Required
HIPPO_SAVE_DIR=/data/hipporag
HIPPO_LLM_MODEL=azure/gpt-4.1
HIPPO_EMBEDDING_MODEL=azure/text-embedding-3-small

# Optional
HIPPO_RETRIEVAL_K=10
```

### 2. Initialize Destination

```python
from src.plugins.destinations import HippoRAGDestination

destination = HippoRAGDestination()
await destination.initialize({
    "save_dir": "/data/hipporag",
    "llm_model": "azure/gpt-4.1",
    "embedding_model": "azure/text-embedding-3-small",
})
```

### 3. Index Documents

```python
conn = await destination.connect({})

# Write documents (automatically indexes)
result = await destination.write(conn, transformed_data)
```

### 4. Multi-Hop Retrieval

```python
# Retrieve passages for a question
results = await destination.retrieve(
    queries=["What company did Steve Jobs found after Apple?"],
    num_to_retrieve=10
)

for result in results:
    print(f"Query: {result.query}")
    for passage, score in zip(result.passages, result.scores):
        print(f"  [{score:.3f}] {passage[:100]}...")
```

### 5. RAG QA

```python
# Full RAG pipeline
qa_results = await destination.rag_qa(
    queries=["What county is Erik Hort's birthplace a part of?"],
    num_to_retrieve=10
)

for qa in qa_results:
    print(f"Q: {qa.query}")
    print(f"A: {qa.answer}")
    print(f"Confidence: {qa.confidence:.2f}")
    print(f"Sources: {len(qa.sources)}")
```

## How It Works

### 1. OpenIE Triple Extraction

Documents are processed to extract subject-predicate-object triples:

```
"Steve Jobs founded Apple" → (Steve Jobs, founded, Apple)
```

### 2. Knowledge Graph Construction

Triples form a knowledge graph stored in file-based storage.

### 3. Personalized PageRank (PPR)

For retrieval:
1. Extract query entities
2. Run PPR on knowledge graph
3. Score passages based on entity importance
4. Return top-k passages

### 4. Single-Step Multi-Hop

Unlike iterative RAG, HippoRAG traverses multiple hops in a single step:

```
Query: "What county is Erik Hort's birthplace a part of?"

Traditional RAG:
  1. Search "Erik Hort birthplace" → Montebello
  2. Search "Montebello county" → Rockland County
  (2 LLM calls, 2 searches)

HippoRAG:
  1. Query nodes: [Erik Hort]
  2. PPR traverses: Erik Hort → birthplace → Montebello → part_of → Rockland County
  3. Single retrieval returns answer
  (1 retrieval, no iterative LLM calls)
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HIPPO_SAVE_DIR` | `/data/hipporag` | Storage directory |
| `HIPPO_LLM_MODEL` | `azure/gpt-4.1` | LLM for OpenIE/QA |
| `HIPPO_EMBEDDING_MODEL` | `azure/text-embedding-3-small` | Embedding model |
| `HIPPO_RETRIEVAL_K` | `10` | Default retrieval count |

### Plugin Configuration

```python
config = {
    "save_dir": "/data/hipporag",
    "llm_model": "azure/gpt-4.1",
    "embedding_model": "azure/text-embedding-3-small",
    "retrieval_k": 10,
}
```

### YAML Configuration

```yaml
# config/destinations.yaml
destinations:
  hipporag:
    type: hipporag
    enabled: true
    config:
      save_dir: "/data/hipporag"
      llm_model: "azure/gpt-4.1"
      embedding_model: "azure/text-embedding-3-small"
      retrieval_k: 10
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/hipporag/retrieve` | POST | Multi-hop retrieval |
| `/api/v1/hipporag/qa` | POST | RAG QA |
| `/api/v1/hipporag/stats` | GET | Graph statistics |

### REST API Examples

#### Retrieve Passages

```bash
curl -X POST http://localhost:8000/api/v1/hipporag/retrieve \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "queries": ["What company did Steve Jobs found?"],
    "num_to_retrieve": 10
  }'
```

**Response:**
```json
{
  "results": [
    {
      "query": "What company did Steve Jobs found?",
      "passages": ["Steve Jobs founded Apple in 1976...", ...],
      "scores": [0.95, 0.87, ...],
      "entities": ["Steve Jobs"],
      "source_documents": ["doc-123", "doc-456"]
    }
  ]
}
```

#### RAG QA

```bash
curl -X POST http://localhost:8000/api/v1/hipporag/qa \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "queries": ["What county is Erik Hort'"'"'s birthplace a part of?"],
    "num_to_retrieve": 10
  }'
```

**Response:**
```json
{
  "results": [
    {
      "query": "What county is Erik Hort's birthplace a part of?",
      "answer": "Rockland County",
      "confidence": 0.92,
      "sources": ["Erik Hort was born in Montebello...", "Montebello is part of Rockland County..."]
    }
  ]
}
```

## Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **Query Latency** | < 500ms | PPR computation on graph |
| **Indexing Throughput** | > 50 docs/min | Depends on LLM speed |
| **Storage** | ~500MB per 1000 docs | File-based |

### Benchmarks

Multi-hop QA accuracy compared to baseline:

| Dataset | Baseline | HippoRAG | Improvement |
|---------|----------|----------|-------------|
| HotpotQA | 45% | 65% | +20% |
| MuSiQue | 38% | 55% | +17% |

## Comparison with Cognee

| Feature | HippoRAG | Cognee |
|---------|----------|--------|
| Multi-hop QA | **+20% better** | Good |
| Speed | **Single-step** | Multi-step |
| Storage | File-based | Neo4j + PostgreSQL |
| Multi-modal | No | **Yes** |
| Enterprise | Research | **Production** |
| Use case | Complex reasoning | General purpose |

**Recommendation:** Use **HippoRAG** for complex multi-hop reasoning, **Cognee** for general production workloads.

## Advanced Usage

### Batch Processing

```python
# Process multiple queries at once
queries = [
    "What company did Steve Jobs found?",
    "Who founded Microsoft?",
    "What is the capital of France?"
]

results = await destination.retrieve(queries, num_to_retrieve=10)
for result in results:
    print(f"Query: {result.query}")
    print(f"Entities found: {result.entities}")
    print(f"Passages: {len(result.passages)}")
```

### Health Check

```python
# Check destination health
health = await destination.health_check()
print(f"Healthy: {health == HealthStatus.HEALTHY}")
```

### Graph Statistics

```python
# Get graph statistics
stats = {
    "entities": len(destination._graph.entities),
    "triples": len(destination._graph.triples),
    "passages": len(destination._graph.passages),
}
print(f"Graph stats: {stats}")
```

## Monitoring

### Logging

HippoRAG provides structured logging for observability:

```python
# Key log events:
# - hipporag_initialized
# - hipporag_triples_extracted
# - hipporag_retrieval_completed
# - hipporag_qa_completed
# - hipporag_write_completed
```

### Metrics

Track these metrics:

| Metric | Description |
|--------|-------------|
| `entities_total` | Total entities in graph |
| `triples_total` | Total triples in graph |
| `passages_total` | Total passages indexed |
| `query_latency_ms` | Retrieval query latency |
| `indexing_rate` | Documents indexed per minute |

## Troubleshooting

### Issue: Slow Indexing

**Symptoms:** Documents take a long time to index

**Causes & Solutions:**
- LLM rate limits → Use OpenRouter fallback
- Large documents → Reduce batch size
- Slow embeddings → Use smaller embedding model

```python
# Reduce batch processing
config = {
    "llm_model": "azure/gpt-4.1",  # Faster model
    "embedding_model": "azure/text-embedding-3-small",
}
```

### Issue: Low Retrieval Accuracy

**Symptoms:** Retrieved passages don't answer the query

**Causes & Solutions:**
- Poor OpenIE quality → Check LLM model
- Insufficient passages → Increase `retrieval_k`
- Missing entities → Verify document coverage

```python
# Increase retrieval count
results = await destination.retrieve(
    queries=[query],
    num_to_retrieve=20  # Increase from default 10
)
```

### Issue: Storage Growing Too Large

**Symptoms:** Disk usage increasing rapidly

**Solutions:**
- Implement graph pruning
- Archive old passages
- Use compression

```python
# Check storage usage
import os

total_size = sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, dirnames, filenames in os.walk("/data/hipporag")
    for filename in filenames
)
print(f"Total storage: {total_size / 1024 / 1024:.2f} MB")
```

### Issue: Connection Errors

**Symptoms:** `ConnectionError: HippoRAGDestination not initialized`

**Solutions:**
- Call `initialize()` before `connect()`
- Check save directory permissions
- Verify disk space available

```python
# Proper initialization sequence
destination = HippoRAGDestination()
await destination.initialize(config)  # Must call first
conn = await destination.connect({})  # Then connect
```

## Error Handling

```python
from src.plugins.destinations.hipporag import (
    RetrievalResult,
    QAResult,
)

try:
    results = await destination.retrieve(queries)
except RuntimeError as e:
    print(f"Retrieval error: {e}")
    # Handle error - return empty results or fallback
    results = [RetrievalResult(query=q) for q in queries]

try:
    qa_results = await destination.rag_qa(queries)
except Exception as e:
    print(f"QA error: {e}")
    # Handle error
    qa_results = [QAResult(query=q, answer="Error processing query") for q in queries]
```

## Best Practices

1. **Use appropriate LLM model** - GPT-4o-mini for OpenIE provides good balance of quality and speed
2. **Batch queries** - Process multiple queries together for better throughput
3. **Monitor graph size** - Implement pruning for long-running systems
4. **Check OpenIE quality** - Review extracted triples periodically
5. **Use caching** - Leverage file-based persistence for restarts
6. **Set appropriate retrieval_k** - Higher for complex queries, lower for simple ones

## See Also

- [HippoRAG Architecture](../architecture/hipporag.md) - Technical architecture details
- [Cognee Usage Guide](./cognee-local.md) - Alternative GraphRAG destination
- [Migration Guide](../migration/graphrag-to-cognee.md) - Migrating from API GraphRAG

---

**Last Updated:** 2026-02-28  
**Version:** 1.0
