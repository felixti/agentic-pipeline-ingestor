# HippoRAG Architecture

## Overview

HippoRAG implements a neurobiological memory model for multi-hop reasoning. It mimics the human hippocampal memory system to enable efficient retrieval of information that requires connecting multiple pieces of knowledge.

## Memory Architecture

```
┌─────────────────┐         ┌─────────────────────────┐
│   Neocortex     │         │      Hippocampus        │
│   (LLM)         │◄───────►│      (Knowledge Graph   │
│                 │         │       + PPR)            │
│  - Entity       │         │                         │
│    Extraction   │         │  - Pattern Separation   │
│  - OpenIE       │         │  - Pattern Completion   │
│  - Reasoning    │         │  - Associative Retrieval│
└─────────────────┘         └───────────┬─────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │  Para-hippocampal       │
                           │  (Retrieval Encoders)   │
                           └─────────────────────────┘
```

### Components

| Component | Biological Analog | Technical Implementation |
|-----------|-------------------|-------------------------|
| **Neocortex** | Long-term storage, reasoning | LLM (Azure GPT-4o-mini) |
| **Hippocampus** | Rapid learning, pattern separation | Knowledge Graph + PPR |
| **Para-hippocampal** | Encoding, retrieval | Embedding models |

## Data Flow

### Offline Indexing (Document Ingestion)

```
Documents → OpenIE (LLM) → Triples → Knowledge Graph
        ↓                              ↓
        └──→ Embeddings ───────────────┘
```

**Step-by-step:**

1. **Document Parsing**
   - Extract text from document chunks
   - Preserve metadata (source, page, etc.)

2. **OpenIE Triple Extraction**
   - LLM extracts subject-predicate-object triples
   - Example: `"Steve Jobs founded Apple" → (Steve Jobs, founded, Apple)`

3. **Knowledge Graph Construction**
   - Entities become nodes
   - Triples become edges
   - Passages linked to entities

4. **Embedding Generation**
   - Generate vector embeddings for passages
   - Store for similarity search fallback

### Online Retrieval (Query Processing)

```
Query → NER (LLM) → Query Nodes → PPR → Ranked Passages
```

**Step-by-step:**

1. **Query Entity Extraction**
   - LLM extracts entities from query
   - Example: `"What did Steve Jobs found?" → ["Steve Jobs"]`

2. **Personalized PageRank**
   - Initialize with query entities
   - Propagate scores through graph
   - Converge to stationary distribution

3. **Passage Ranking**
   - Score passages by entity importance
   - Return top-k passages

## Storage Layout

```
/data/hipporag/
├── knowledge_graph.pkl     # Main graph storage (pickle)
├── config.json             # Configuration
└── metadata/
    └── stats.json          # Graph statistics
```

### Knowledge Graph Structure

```python
@dataclass
class KnowledgeGraph:
    entities: dict[str, dict]           # Entity name → metadata
    triples: list[tuple[str, str, str]] # (subject, predicate, object)
    passages: dict[str, str]            # Passage ID → text
    passage_embeddings: dict[str, np.ndarray]  # Passage ID → embedding
    entity_passages: dict[str, list[str]]      # Entity → passage IDs
```

### Storage Characteristics

| Component | Format | Size (per 1000 docs) |
|-----------|--------|---------------------|
| Knowledge Graph | Pickle | ~200MB |
| Embeddings | NumPy arrays | ~100MB |
| Passage text | JSON | ~100MB |
| Metadata | JSON | ~10MB |
| **Total** | - | **~410MB** |

## Key Algorithms

### OpenIE Triple Extraction

Uses LLM with structured prompting to extract triples:

```python
OPENIE_PROMPT = """Extract subject-predicate-object triples from the text.

Return a JSON object in this exact format:
{
    "triples": [
        ["subject", "predicate", "object"],
        ...
    ]
}

Text: {text}
"""
```

**Example Extraction:**

| Text | Triples |
|------|---------|
| "Steve Jobs founded Apple in California." | `[(Steve Jobs, founded, Apple), (Apple, located in, California)]` |
| "Einstein developed the theory of relativity." | `[(Einstein, developed, theory of relativity)]` |

### Personalized PageRank (PPR)

```python
# Power iteration with teleport probability α
r = (1 - α) * M * r + α * v

# Where:
# - M: Transition matrix from knowledge graph
# - r: PageRank vector (passage scores)
# - v: Personalization vector (query entities)
# - α: Teleport probability (default 0.15)
```

**Algorithm Steps:**

1. Build adjacency list from triples
2. Initialize scores: `scores[node] = 0` for all entities
3. Set personalization: `scores[query_entity] = 1/len(query_entities)`
4. Iterate until convergence:
   - For each node: `new_score = α * teleport + (1-α) * neighbor_sum`
   - Check `max_diff < threshold`
5. Score passages by summing entity scores

**Convergence:**
- Typically 20-50 iterations
- Threshold: 1e-6
- Guaranteed to converge (graph is strongly connected)

### Single-Step Multi-Hop

Unlike iterative RAG which requires multiple LLM calls:

```
Traditional Iterative RAG:
  Query: "What county is Erik Hort's birthplace a part of?"
  
  Step 1: LLM → "Erik Hort's birthplace is Montebello"
  Step 2: Search "Montebello county" 
  Step 3: LLM → "Montebello is in Rockland County"
  
  Total: 2 LLM calls, 2 searches

HippoRAG Single-Step:
  Query: "What county is Erik Hort's birthplace a part of?"
  
  Step 1: Extract entities: [Erik Hort]
  Step 2: PPR propagates scores:
          Erik Hort (1.0) 
          → birthplace (0.85)
          → Montebello (0.72)
          → part_of (0.61)
          → Rockland County (0.52)
  Step 3: Return passages about Rockland County
  
  Total: 1 entity extraction, 1 PPR computation
```

**Advantages:**
- Single retrieval operation
- No intermediate LLM calls
- Deterministic traversal
- Faster overall latency

## Implementation Details

### HippoRAGDestination Class

```python
class HippoRAGDestination(DestinationPlugin):
    """
    HippoRAG destination for multi-hop reasoning.
    
    Key methods:
    - initialize(): Set up storage and LLM provider
    - write(): Index documents into knowledge graph
    - retrieve(): Multi-hop retrieval using PPR
    - rag_qa(): Full RAG pipeline
    """
```

### Thread Safety

- Uses `ThreadPoolExecutor` for sync file I/O
- Async/await pattern for API methods
- Pickle serialization for persistence

### Error Handling

| Error Type | Handling |
|------------|----------|
| LLM failures | Return empty results, log warning |
| File I/O errors | Retry with exponential backoff |
| Invalid triples | Skip invalid, continue processing |
| Graph load failure | Start with empty graph |

## Performance Characteristics

### Query Performance

| Operation | Complexity | Typical Latency |
|-----------|------------|-----------------|
| Entity extraction | O(query_length) | ~100ms |
| PPR computation | O(iterations × edges) | ~200ms |
| Passage ranking | O(passages) | ~50ms |
| **Total retrieval** | - | **~350ms** |

### Indexing Performance

| Operation | Throughput | Bottleneck |
|-----------|------------|------------|
| OpenIE extraction | ~50 docs/min | LLM rate limits |
| Embedding generation | ~100 docs/min | Embedding API |
| Graph persistence | ~500 docs/sec | Disk I/O |

### Scalability

| Metric | Limit | Mitigation |
|--------|-------|------------|
| Graph size | Memory bound | Graph partitioning |
| Concurrent queries | CPU bound | Horizontal scaling |
| Storage growth | Linear | Archival strategy |

## Integration Points

### LLM Provider (hipporag_llm.py)

```python
class HippoRAGLLMProvider:
    """
    Provides LLM capabilities for HippoRAG:
    - extract_triples(): OpenIE extraction
    - extract_query_entities(): NER for queries
    - answer_question(): RAG answer generation
    - embed_text(): Passage embeddings
    """
```

### Existing Pipeline Integration

```
Pipeline Worker
    ↓
HippoRAGDestination
    ↓
├─→ HippoRAGLLMProvider (litellm)
│     ├─→ Azure OpenAI (primary)
│     └─→ OpenRouter (fallback)
│
└─→ File-based Storage
      └─→ Persistent Volume (/data/hipporag)
```

## Configuration Schema

```json
{
  "type": "object",
  "properties": {
    "save_dir": {
      "type": "string",
      "description": "Directory for file-based storage",
      "default": "/data/hipporag"
    },
    "llm_model": {
      "type": "string",
      "description": "LLM model for OpenIE and QA",
      "default": "azure/gpt-4.1"
    },
    "embedding_model": {
      "type": "string",
      "description": "Embedding model for passages",
      "default": "azure/text-embedding-3-small"
    },
    "retrieval_k": {
      "type": "integer",
      "description": "Default number of passages to retrieve",
      "default": 10
    }
  }
}
```

## Comparison with Other Architectures

### HippoRAG vs Traditional Vector Search

| Aspect | Vector Search | HippoRAG |
|--------|--------------|----------|
| Multi-hop | Requires iteration | Single-step |
| Entity awareness | No | Yes |
| Reasoning path | Not explicit | Traceable via graph |
| Latency | Low | Medium |
| Complexity | Simple | Complex |

### HippoRAG vs Cognee

| Aspect | HippoRAG | Cognee |
|--------|----------|--------|
| Storage | File-based | Neo4j + PostgreSQL |
| Multi-hop | PPR-based | Graph traversal |
| Multi-modal | No | Yes |
| Production ready | Research | Yes |
| Setup complexity | Low | Medium |

## Future Enhancements

### Potential Improvements

1. **Graph Pruning**
   - Remove low-importance edges
   - Compress entity representations
   - Archive old passages

2. **Incremental Updates**
   - Add documents without full reindex
   - Merge new triples efficiently
   - Handle updates and deletions

3. **Query Optimization**
   - Cache frequent queries
   - Pre-compute entity embeddings
   - Parallel PPR computation

4. **Hybrid Retrieval**
   - Combine PPR with vector similarity
   - Weight graph vs semantic relevance
   - Dynamic weight adjustment

## References

- HippoRAG Paper: [Neurobiological Memory Model for RAG](https://arxiv.org/abs/2405.14831)
- Personalized PageRank: [Page et al., 1999](https://www.cs.cornell.edu/home/kleinber/auth.pdf)
- OpenIE: [Stanford OpenIE](https://nlp.stanford.edu/software/openie.html)

---

**Last Updated:** 2026-02-28  
**Version:** 1.0
