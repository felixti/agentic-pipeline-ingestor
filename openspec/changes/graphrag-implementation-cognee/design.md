# Design: Local Cognee GraphRAG Integration

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Pipeline Application                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CogneeLocalDestination                        │   │
│  │                   (replaces API GraphRAG)                        │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  - initialize()                                                 │   │
│  │  - write() → Document → Entities → Relationships                │   │
│  │  - search() → Hybrid (Vector + Graph)                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│  ┌───────────────────────────┼───────────────────────────────────────┐ │
│  │                           ▼                                       │ │
│  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────────────────┐ │ │
│  │  │   Cognee     │  │   Neo4j     │  │   PostgreSQL/pgvector    │ │ │
│  │  │   Library    │◄─┤   Graph     │  │   Vectors + Metadata     │ │ │
│  │  │              │  │   (Docker)  │  │   (Existing)             │ │ │
│  │  └──────┬───────┘  └─────────────┘  └──────────────────────────┘ │ │
│  │         │                                                        │ │
│  │         ▼                                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │              LLM Provider (litellm)                         │ │ │
│  │  │     Azure OpenAI (primary) → OpenRouter (fallback)          │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. CogneeLocalDestination Plugin

```python
class CogneeLocalDestination(DestinationPlugin):
    """
    Local Cognee GraphRAG destination.
    
    Replaces API-based GraphRAG with local Cognee library.
    Uses Neo4j for graph, PostgreSQL/pgvector for vectors.
    """
    
    def __init__(self):
        self._cognee = None
        self._llm_provider = None
        self._config = {}
    
    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize Cognee with local storage backends.
        
        Args:
            config: Configuration dict with:
                - neo4j_uri: Neo4j connection string
                - neo4j_user: Neo4j username
                - neo4j_password: Neo4j password
                - db_url: PostgreSQL connection URL
        """
        # Set environment variables for Cognee
        os.environ["GRAPH_DATABASE_PROVIDER"] = "neo4j"
        os.environ["GRAPH_DATABASE_URL"] = config["neo4j_uri"]
        os.environ["GRAPH_DATABASE_NAME"] = "neo4j"
        os.environ["GRAPH_DATABASE_USERNAME"] = config["neo4j_user"]
        os.environ["GRAPH_DATABASE_PASSWORD"] = config["neo4j_password"]
        
        os.environ["VECTOR_DB_PROVIDER"] = "pgvector"
        os.environ["DB_PROVIDER"] = "postgres"
        os.environ["DB_URL"] = config["db_url"]
        
        # Configure LLM via litellm
        os.environ["LLM_PROVIDER"] = "litellm"
        os.environ["LLM_MODEL"] = config.get("llm_model", "azure/gpt-4.1")
        
        # Initialize Cognee
        import cognee
        self._cognee = cognee
        
        logger.info("Cognee local destination initialized")
    
    async def write(
        self, 
        conn: Connection, 
        data: TransformedData
    ) -> WriteResult:
        """
        Write transformed data to Cognee graph.
        
        Process:
        1. Extract text from chunks
        2. Add to Cognee dataset
        3. Run cognify() to build graph
        4. Extract entities to Neo4j
        5. Store embeddings to pgvector
        """
        dataset_name = f"job_{data.job_id}"
        
        try:
            # Add all chunks to Cognee
            for chunk in data.chunks:
                text = chunk.get("content", "")
                if text:
                    await self._cognee.add(text, dataset_name=dataset_name)
            
            # Build knowledge graph
            await self._cognee.cognify([dataset_name])
            
            return WriteResult(
                success=True,
                destination_id="cognee_local",
                destination_uri=f"cognee://{dataset_name}",
                records_written=len(data.chunks),
                metadata={
                    "dataset_name": dataset_name,
                    "job_id": str(data.job_id),
                }
            )
            
        except Exception as e:
            logger.error(f"Cognee write failed: {e}")
            return WriteResult(
                success=False,
                error=str(e)
            )
    
    async def search(
        self,
        query: str,
        dataset_name: str | None = None,
        top_k: int = 10
    ) -> list[dict[str, Any]]:
        """
        Search the knowledge graph.
        
        Uses Cognee's hybrid search combining vector similarity
        and graph traversal.
        """
        results = await self._cognee.search(
            query_text=query,
            datasets=[dataset_name] if dataset_name else None
        )
        return results
```

### 2. LLM Integration

Cognee will use the existing litellm provider:

```python
# In Cognee configuration
os.environ["LLM_PROVIDER"] = "litellm"
os.environ["LLM_MODEL"] = "azure/gpt-4.1"
os.environ["LLM_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_API_BASE")
```

Fallback to OpenRouter:
```python
# Cognee handles fallback internally or we wrap calls
async def _call_llm_with_fallback(self, prompt: str) -> str:
    try:
        return await self._llm_provider.complete(prompt)
    except Exception:
        # Switch to OpenRouter
        os.environ["LLM_MODEL"] = "openrouter/openai/gpt-4.1"
        return await self._llm_provider.complete(prompt)
```

### 3. Database Schema

#### Neo4j Graph Schema

```cypher
// Entity Node
(:Entity {
    id: string,
    name: string,
    type: string,
    description: string,
    source_document: string,
    confidence: float
})

// Document Node
(:Document {
    id: string,
    job_id: string,
    title: string,
    content_preview: string
})

// Relationships
(:Entity)-[:APPEARS_IN]->(:Document)
(:Entity)-[:RELATED_TO {relation_type: string}]->(:Entity)
```

#### PostgreSQL/pgvector Schema

```sql
-- Cognee vectors table
CREATE TABLE cognee_vectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_name VARCHAR(255),
    entity_id VARCHAR(255),
    embedding VECTOR(1536),  -- text-embedding-3-small
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for similarity search
CREATE INDEX ON cognee_vectors 
USING ivfflat (embedding vector_cosine_ops);

-- Index for dataset filtering
CREATE INDEX idx_cognee_dataset ON cognee_vectors(dataset_name);
```

## Data Flow

### Document Ingestion Flow

```
┌──────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Document │────►│ Chunking         │────►│ Text Extraction │
└──────────┘     └──────────────────┘     └────────┬────────┘
                                                   │
                          ┌────────────────────────┼────────────────────────┐
                          │                        │                        │
                          ▼                        ▼                        ▼
                   ┌──────────────┐      ┌─────────────────┐     ┌──────────────────┐
                   │ Entity       │      │ Embedding       │     │ Metadata         │
                   │ Extraction   │      │ Generation      │     │ Storage          │
                   │ (LLM)        │      │ (LLM)           │     │ (PostgreSQL)     │
                   └──────┬───────┘      └────────┬────────┘     └──────────────────┘
                          │                       │
                          ▼                       ▼
                   ┌──────────────┐      ┌─────────────────┐
                   │ Neo4j Graph  │      │ pgvector        │
                   │ (Entities +  │      │ (Embeddings)    │
                   │ Relations)   │      │                 │
                   └──────────────┘      └─────────────────┘
```

### Query Flow

```
┌──────────┐     ┌─────────────────────┐     ┌─────────────────────────┐
│ Query    │────►│ Entity Extraction   │────►│ Vector Search (pgvector)│
└──────────┘     │ (LLM)               │     └───────────┬─────────────┘
                 └─────────────────────┘                 │
                                                          ▼
                 ┌─────────────────────┐     ┌─────────────────────────┐
                 │ Result Fusion       │◄────│ Graph Traversal (Neo4j) │
                 │ (Ranking)           │     └─────────────────────────┘
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Response            │
                 │ (Context + Answer)  │
                 └─────────────────────┘
```

## Configuration

### Environment Variables

```bash
# Neo4j Configuration
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cognee-graph-db

# Cognee Configuration
COGNEE_LLM_PROVIDER=litellm
COGNEE_LLM_MODEL=azure/gpt-4.1
COGNEE_LLM_FALLBACK_MODEL=openrouter/openai/gpt-4.1
COGNEE_EMBEDDING_MODEL=azure/text-embedding-3-small
COGNEE_GRAPH_PROVIDER=neo4j
COGNEE_VECTOR_PROVIDER=pgvector
```

### Docker Compose

See `specs/neo4j-infrastructure/spec.md` for Neo4j service definition.

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| Neo4j unavailable | Retry with exponential backoff, fail after 3 attempts |
| LLM rate limit | Switch to fallback (OpenRouter), queue for retry |
| Embedding failure | Log error, skip document, continue with others |
| Graph build failure | Mark dataset as failed, notify admin |

## Performance Considerations

| Aspect | Strategy |
|--------|----------|
| Batch processing | Process documents in batches of 100 |
| Async operations | All I/O operations are async |
| Connection pooling | Reuse Neo4j and PostgreSQL connections |
| Caching | Cache entity extraction results |
| Indexing | Background indexing for large documents |

## Testing Strategy

| Test Type | Coverage |
|-----------|----------|
| Unit | Plugin methods, configuration |
| Integration | Cognee + Neo4j + PostgreSQL |
| E2E | Full pipeline: document → graph → search |
| Performance | 1000+ documents, query latency < 100ms |
| Migration | Data from API GraphRAG to Cognee |

## Deployment Plan

1. **Phase 1**: Add Neo4j service to Docker Compose
2. **Phase 2**: Install Cognee library, create plugin
3. **Phase 3**: Integrate with existing LLM provider
4. **Phase 4**: Testing and optimization
5. **Phase 5**: Migration from API GraphRAG
6. **Phase 6**: Deprecate old plugin
