# GraphRAG Overview

**Agentic Data Pipeline Ingestor**  
_Complete guide to GraphRAG capabilities: Cognee and HippoRAG_

---

## Overview

This pipeline now includes two powerful GraphRAG implementations for advanced knowledge retrieval:

| Feature          | Cognee                        | HippoRAG                       |
| ---------------- | ----------------------------- | ------------------------------ |
| **Best For**     | General production workloads  | Complex multi-hop reasoning    |
| **Storage**      | Neo4j + PostgreSQL/pgvector   | File-based (persistent volume) |
| **Multi-hop QA** | Good                          | **+20% better**                |
| **Speed**        | Fast                          | **Single-step retrieval**      |
| **Multi-modal**  | **Yes** (text, images, audio) | No                             |
| **Enterprise**   | **Production-ready**          | Research-grade                 |

---

## Quick Comparison

### When to Use Cognee

✅ **Use Cognee when you need:**

- Enterprise knowledge graph storage
- Multi-modal data support (documents + images)
- Hybrid search (vector + graph)
- Production-grade reliability
- Entity relationship visualization
- General-purpose GraphRAG

**Example queries:**

- "What are the relationships between these companies?"
- "Find documents mentioning machine learning and healthcare"
- "Show me the knowledge graph for this dataset"

### When to Use HippoRAG

✅ **Use HippoRAG when you need:**

- Complex multi-hop reasoning (2+ steps)
- Single-step multi-hop retrieval
- Research synthesis across documents
- Legal/medical case analysis
- +20% better multi-hop QA accuracy

**Example queries:**

- "What county is Erik Hort's birthplace a part of?" (3 hops)
- "What technologies do companies founded by Elon Musk use?" (2+ hops)
- "Connect the dots between these research papers"

---

## Architecture

### Cognee Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Worker / API                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│              CogneeLocalDestination                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Cognee     │  │   Neo4j     │  │   PostgreSQL/pgvector   │ │
│  │  Library    │◄─┤  (Graph)    │  │   (Vectors)             │ │
│  │             │  │             │  │                         │ │
│  └──────┬──────┘  └─────────────┘  └─────────────────────────┘ │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LLM via litellm Provider                   │   │
│  │         (Azure OpenAI → OpenRouter fallback)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### HippoRAG Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Worker / API                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│              HippoRAGDestination                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────────────────────────┐  │
│  │  HippoRAG   │  │   File-based Storage                    │  │
│  │  Library    │◄─┤   - Knowledge Graph (JSON/pickle)       │  │
│  │             │  │   - Embeddings cache                    │  │
│  │             │  │   - OpenIE results                      │  │
│  │             │  │   - PPR indices                         │  │
│  └──────┬──────┘  └─────────────────────────────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LLM via litellm Provider                   │   │
│  │         (Azure OpenAI → OpenRouter fallback)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Environment Variables

### Cognee Configuration

```bash
# Neo4j (required)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cognee-graph-db

# Cognee LLM settings
COGNEE_LLM_PROVIDER=litellm
COGNEE_LLM_MODEL=azure/gpt-4.1
COGNEE_EMBEDDING_MODEL=azure/text-embedding-3-small
```

### HippoRAG Configuration

```bash
# Storage (required)
HIPPO_SAVE_DIR=/data/hipporag

# HippoRAG LLM settings
HIPPO_LLM_MODEL=azure/gpt-4.1
HIPPO_EMBEDDING_MODEL=azure/text-embedding-3-small
HIPPO_RETRIEVAL_K=10
```

---

## API Quick Reference

### Cognee Endpoints

| Endpoint                          | Method | Description                     |
| --------------------------------- | ------ | ------------------------------- |
| `/api/v1/cognee/search`           | POST   | Search with vector/graph/hybrid |
| `/api/v1/cognee/extract-entities` | POST   | Extract entities from text      |
| `/api/v1/cognee/stats`            | GET    | Graph statistics                |

**Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/cognee/search" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "machine learning applications",
    "search_type": "hybrid",
    "top_k": 10
  }'
```

### HippoRAG Endpoints

| Endpoint                           | Method | Description              |
| ---------------------------------- | ------ | ------------------------ |
| `/api/v1/hipporag/retrieve`        | POST   | Multi-hop retrieval      |
| `/api/v1/hipporag/qa`              | POST   | Full RAG QA pipeline     |
| `/api/v1/hipporag/extract-triples` | POST   | OpenIE triple extraction |

**Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/hipporag/qa" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "queries": ["What county is Erik Hort's birthplace part of?"],
    "num_to_retrieve": 10
  }'
```

---

## Performance Characteristics

| Metric             | Cognee             | HippoRAG          | Standard Vector Store |
| ------------------ | ------------------ | ----------------- | --------------------- |
| **Query Latency**  | ~100ms             | ~350ms            | ~30ms                 |
| **Indexing Speed** | >100 docs/min      | >50 docs/min      | >100 docs/min         |
| **Multi-hop QA**   | Good               | **+20% better**   | N/A                   |
| **Storage**        | Neo4j + PostgreSQL | ~500MB/1000 docs  | PostgreSQL            |
| **Best For**       | Production         | Complex reasoning | Simple search         |

---

## Decision Tree

```
START: What type of query do you have?
│
├─► Simple similarity search?
│   └─► Use Standard Vector Store (pgvector)
│
├─► Need entity relationships & knowledge graph?
│   ├─► Production workload?
│   │   └─► Use Cognee GraphRAG
│   │
│   └─► Complex multi-hop questions?
│       └─► Use HippoRAG
│
└─► Research/complex reasoning?
    └─► Use HippoRAG (+20% multi-hop accuracy)
```

---

## Documentation

- **[Cognee Usage Guide](usage/cognee-local.md)** - Detailed Cognee documentation
- **[HippoRAG Usage Guide](usage/hipporag.md)** - Detailed HippoRAG documentation
- **[Cognee Architecture](architecture/hipporag.md)** - Technical architecture details
- **[Migration Guide](migration/graphrag-to-cognee.md)** - Migrating to Cognee
- **[API Guide](API_GUIDE.md)** - Complete API reference
- **[RAG Strategy Guide](RAG_STRATEGY_GUIDE.md)** - RAG strategy selection

---

## Migration from API-based GraphRAG

If you were using the old API-based Cognee (`COGNEE_API_URL`), migrate to local Cognee:

```bash
# Old (API-based) - REMOVED
COGNEE_API_URL=http://cognee:8001
COGNEE_API_KEY=your-key

# New (Local) - USE THIS
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cognee-graph-db
COGNEE_LLM_PROVIDER=litellm
COGNEE_LLM_MODEL=azure/gpt-4.1
```

See the [Migration Guide](migration/graphrag-to-cognee.md) for detailed instructions.

---

## Support

For more information:

- Cognee GitHub: https://github.com/topoteretes/cognee
- HippoRAG Paper: [Neurobiological Memory Models for RAG](https://arxiv.org/abs/2405.14831)
- Pipeline Docs: See `/docs` directory
