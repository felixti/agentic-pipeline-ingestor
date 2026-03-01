# Capability: HippoRAG Multi-Hop Reasoning Integration

## Overview

HippoRAG destination plugin for advanced multi-hop reasoning using neurobiological memory model.

## Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| HIP-001 | HippoRAG must operate as local Python library | Must |
| HIP-002 | Use existing Azure OpenAI/OpenRouter via litellm | Must |
| HIP-003 | Support embedding models (NV-Embed-v2, text-embedding-3-small) | Must |
| HIP-004 | File-based storage in persistent volume | Must |
| HIP-005 | Document ingestion with OpenIE triple extraction | Must |
| HIP-006 | Single-step multi-hop retrieval | Must |
| HIP-007 | Support entity-based query answering | Must |
| HIP-008 | Provide retrieval + QA in one call | Should |
| HIP-009 | Separate retrieval and QA modes | Should |
| HIP-010 | Graph persistence across restarts | Must |

### Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| HIP-NF-001 | Multi-hop query latency | < 500ms |
| HIP-NF-002 | Document indexing throughput | > 50 docs/min |
| HIP-NF-003 | Storage per 1000 docs | < 500MB |
| HIP-NF-004 | Multi-hop QA accuracy | > 15% improvement |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Worker / API                        │
│                         (Your Code)                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Function Call
┌───────────────────────────▼─────────────────────────────────────┐
│              HippoRAGDestination Plugin                         │
│                   (src/plugins/destinations/)                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────────────────────────┐  │
│  │  HippoRAG   │  │   File-based Storage (Persistent)       │  │
│  │  Library    │◄─┤   - Knowledge Graph (JSON/pickle)       │  │
│  │  (pip)      │  │   - Embeddings cache                    │  │
│  │             │  │   - OpenIE results                      │  │
│  │             │  │   - PPR indices                         │  │
│  └──────┬──────┘  └─────────────────────────────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LLM Provider (litellm)                     │   │
│  │     Azure OpenAI (primary) → OpenRouter (fallback)      │   │
│  │              Embedding Models                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## How HippoRAG Works

### Neurobiological Memory Model

```
┌─────────────────────────────────────────────────────────────────┐
│                 HippoRAG Memory Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────┐         ┌─────────────────────────┐      │
│   │   Neocortex     │         │      Hippocampus        │      │
│   │   (LLM)         │◄───────►│      (Knowledge Graph   │      │
│   │                 │         │       + PPR)            │      │
│   │  - Entity       │         │                         │      │
│   │    Extraction   │         │  - Pattern Separation   │      │
│   │  - OpenIE       │         │  - Pattern Completion   │      │
│   │  - Reasoning    │         │  - Associative Retrieval│      │
│   └─────────────────┘         └───────────┬─────────────┘      │
│                                           │                     │
│                                           ▼                     │
│                              ┌─────────────────────────┐       │
│                              │  Para-hippocampal       │       │
│                              │  (Retrieval Encoders)   │       │
│                              └─────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Offline Indexing (Document Ingestion):**
```
Documents → OpenIE (LLM) → Triples → Knowledge Graph
        ↓                              ↓
        └──→ Embeddings ───────────────┘
```

**Online Retrieval (Query):**
```
Query → NER (LLM) → Query Nodes → PPR → Ranked Passages
```

## Configuration

### Environment Variables

```env
# LLM (via existing litellm)
AZURE_OPENAI_API_BASE=<your-azure-endpoint>
AZURE_OPENAI_API_KEY=<your-azure-key>
OPENROUTER_API_KEY=<your-openrouter-key>

# HippoRAG Settings
HIPPO_SAVE_DIR=/data/hipporag
HIPPO_LLM_MODEL=azure/gpt-4.1
HIPPO_EMBEDDING_MODEL=azure/text-embedding-3-small
# Alternative: nvidia/NV-Embed-v2 (requires HuggingFace)
HIPPO_RETRIEVAL_K=10
```

### Storage

| Component | Location | Size |
|-----------|----------|------|
| Knowledge Graph | `$HIPPO_SAVE_DIR/graph/` | ~200MB per 1000 docs |
| Embeddings | `$HIPPO_SAVE_DIR/embeddings/` | ~100MB per 1000 docs |
| OpenIE Results | `$HIPPO_SAVE_DIR/openie/` | ~100MB per 1000 docs |
| PPR Indices | `$HIPPO_SAVE_DIR/ppr/` | ~50MB per 1000 docs |

## API Interface

### HippoRAGDestination

```python
class HippoRAGDestination(DestinationPlugin):
    """
    HippoRAG destination for multi-hop reasoning.
    
    Uses neurobiological memory model with single-step
    multi-hop retrieval via Personalized PageRank.
    """
    
    async def initialize(self, config: dict) -> None:
        """
        Initialize HippoRAG with file-based storage.
        
        Args:
            config: Configuration dict with:
                - save_dir: Storage directory path
                - llm_model: LLM model name
                - embedding_model: Embedding model name
        """
    
    async def write(
        self, 
        conn: Connection, 
        data: TransformedData
    ) -> WriteResult:
        """
        Write documents to HippoRAG memory.
        
        Process:
        1. Extract text from chunks
        2. Run OpenIE to extract triples
        3. Build knowledge graph
        4. Generate embeddings
        5. Store in file-based storage
        """
    
    async def retrieve(
        self,
        queries: list[str],
        num_to_retrieve: int = 10
    ) -> list[RetrievalResult]:
        """
        Multi-hop retrieval using PPR.
        
        Single-step retrieval that can traverse
        multiple hops in the knowledge graph.
        """
    
    async def rag_qa(
        self,
        queries: list[str],
        retrieval_results: list[RetrievalResult] | None = None
    ) -> list[QAResult]:
        """
        Full RAG pipeline: retrieve + generate answer.
        """
```

## Dependencies

### Python Libraries

```toml
[project.dependencies]
hipporag = "^0.1.0"
```

### Infrastructure

- Persistent volume for `save_dir`
- No separate database required

## Key Features

### 1. OpenIE Triple Extraction

Extracts subject-predicate-object triples from text:
```python
"Steve Jobs founded Apple" → (Steve Jobs, founded, Apple)
```

### 2. Personalized PageRank (PPR)

For query "What company did Steve Jobs found?":
```
Query Nodes: [Steve Jobs]
PPR spreads probability through graph:
  Steve Jobs --founded--> Apple (high probability)
  Steve Jobs --worked_at--> Apple (high probability)
  Apple --competitor--> Microsoft (lower probability)
Return: passages about Apple
```

### 3. Single-Step Multi-Hop

Example query: "What county is Erik Hort's birthplace a part of?"

```
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

## Testing Strategy

| Test Type | Scope |
|-----------|-------|
| Unit | Plugin initialization, configuration |
| Integration | HippoRAG + LLM + storage |
| E2E | Document → graph → multi-hop query |
| Multi-hop QA | HotpotQA, MuSiQue benchmarks |
| Performance | Latency, throughput |

## Use Cases

### Best For
- Complex questions spanning multiple documents
- Research paper synthesis
- Legal case analysis
- Medical diagnosis from multiple records
- Any domain requiring "connecting the dots"

### Not Ideal For
- Simple factual lookup
- Single-document questions
- Real-time chat (indexing takes time)

## Comparison with Cognee

| Aspect | HippoRAG | Cognee |
|--------|----------|--------|
| **Multi-hop QA** | **+20% better** | Good |
| **Speed** | Faster (single-step) | Fast |
| **Storage** | File-based | Neo4j + PostgreSQL |
| **Multi-modal** | No | **Yes** |
| **Enterprise** | Research | **Production** |
| **Use case** | Complex reasoning | General purpose |

**Recommendation:** Use **HippoRAG** for complex multi-hop reasoning, **Cognee** for general production use.
