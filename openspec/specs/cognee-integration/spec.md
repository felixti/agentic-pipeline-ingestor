# Capability: Cognee GraphRAG Integration

## Overview

Local Cognee GraphRAG integration using Neo4j for graph storage and existing PostgreSQL/pgvector for vector storage.

## Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| COG-001 | Cognee must operate as local Python library (no API calls) | Must |
| COG-002 | Use existing Azure OpenAI/OpenRouter via litellm | Must |
| COG-003 | Use Neo4j for graph storage | Must |
| COG-004 | Use existing PostgreSQL/pgvector for vector storage | Must |
| COG-005 | Support document ingestion from pipeline | Must |
| COG-006 | Support graph search with hybrid (vector + graph) retrieval | Must |
| COG-007 | Support entity extraction and relationship mapping | Must |
| COG-008 | Provide migration path from API-based GraphRAG | Should |
| COG-009 | Support multi-modal content (images, audio) | Should |
| COG-010 | Graph visualization endpoint | Could |

### Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| COG-NF-001 | Graph operation latency | < 100ms |
| COG-NF-002 | Document indexing throughput | > 100 docs/min |
| COG-NF-003 | Neo4j memory usage | < 2GB |
| COG-NF-004 | Cognee library size | < 500MB |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Worker / API                        │
│                         (Your Code)                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Function Call
┌───────────────────────────▼─────────────────────────────────────┐
│              CogneeLocalDestination Plugin                      │
│                   (src/plugins/destinations/)                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Cognee     │  │   Neo4j     │  │   PostgreSQL/pgvector   │ │
│  │  Library    │◄─┤  (Graph)    │  │   (Vectors + Metadata)  │ │
│  │  (pip)      │  │  (Docker)   │  │   (Existing)            │ │
│  └──────┬──────┘  └─────────────┘  └─────────────────────────┘ │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LLM via litellm Provider                   │   │
│  │         (Azure OpenAI → OpenRouter fallback)            │   │
│  │              (src/llm/provider.py)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```env
# LLM (via existing litellm)
AZURE_OPENAI_API_BASE=<your-azure-endpoint>
AZURE_OPENAI_API_KEY=<your-azure-key>
OPENROUTER_API_KEY=<your-openrouter-key>

# Neo4j (new)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cognee-graph-db

# PostgreSQL (existing)
DB_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/pipeline

# Cognee Settings
COGNEE_LLM_PROVIDER=litellm
COGNEE_LLM_MODEL=azure/gpt-4.1
COGNEE_EMBEDDING_MODEL=azure/text-embedding-3-small
COGNEE_GRAPH_PROVIDER=neo4j
COGNEE_VECTOR_PROVIDER=pgvector
```

## API Interface

### CogneeLocalDestination

```python
class CogneeLocalDestination(DestinationPlugin):
    """Local Cognee destination using Neo4j and PostgreSQL."""
    
    async def initialize(self, config: dict) -> None:
        """Initialize Cognee with local storage backends."""
        
    async def write(self, conn: Connection, data: TransformedData) -> WriteResult:
        """Write documents to local Cognee graph."""
        
    async def search(
        self, 
        query: str, 
        search_type: str = "hybrid",
        top_k: int = 10
    ) -> list[SearchResult]:
        """Search the knowledge graph."""
```

## Data Flow

### Document Ingestion

```
Document Chunks → Entity Extraction → Relationship Mapping → Neo4j (Graph)
              ↓                                          ↓
              └──────→ Embeddings ───────────────────→ PostgreSQL/pgvector
```

### Query Flow

```
Query → LLM Entity Extraction → Vector Search (pgvector)
                              → Graph Traversal (Neo4j)
                              → Result Fusion → Response
```

## Dependencies

### Python Libraries

```toml
[project.dependencies]
cognee = "^0.3.0"
neo4j = "^5.15.0"
```

### Infrastructure

- Neo4j Community Edition (Docker)
- Existing PostgreSQL with pgvector

## Testing Strategy

| Test Type | Scope |
|-----------|-------|
| Unit | Plugin initialization, configuration |
| Integration | Cognee + Neo4j + PostgreSQL |
| E2E | Document ingestion → search → results |
| Performance | Latency, throughput benchmarks |

## Migration Path

1. Deploy Neo4j service
2. Install Cognee library
3. Create `CogneeLocalDestination` plugin
4. Run parallel with existing GraphRAG
5. Migrate existing graph data
6. Deprecate API-based GraphRAG
