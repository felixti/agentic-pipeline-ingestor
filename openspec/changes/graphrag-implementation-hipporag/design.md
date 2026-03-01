# Design: HippoRAG Multi-Hop Reasoning Integration

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Pipeline Application                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    HippoRAGDestination                           │   │
│  │              (for multi-hop reasoning use cases)                 │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  - initialize()                                                 │   │
│  │  - write() → Document → OpenIE → Triples → Knowledge Graph      │   │
│  │  - retrieve() → PPR-based multi-hop retrieval                   │   │
│  │  - rag_qa() → Retrieve + Generate                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│  ┌───────────────────────────┼───────────────────────────────────────┐ │
│  │                           ▼                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │              File-based Storage (Persistent Volume)         │ │ │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │ │ │
│  │  │  │  Knowledge   │  │  Embeddings  │  │  OpenIE Results  │  │ │ │
│  │  │  │  Graph       │  │  Cache       │  │                  │  │ │ │
│  │  │  │  (JSON)      │  │  (numpy)     │  │  (pickle)        │  │ │ │
│  │  │  └──────────────┘  └──────────────┘  └──────────────────┘  │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │                                                                   │ │
│  │                           │                                       │ │
│  │                           ▼                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │              LLM Provider (litellm)                         │ │ │
│  │  │     Azure OpenAI (primary) → OpenRouter (fallback)          │ │ │
│  │  │                                                             │ │ │
│  │  │  Entity Extraction (OpenIE)                                 │ │ │
│  │  │  Embedding Generation                                       │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. HippoRAGDestination Plugin

```python
class HippoRAGDestination(DestinationPlugin):
    """
    HippoRAG destination for multi-hop reasoning.
    
    Uses neurobiological memory model with:
    - OpenIE for triple extraction
    - Knowledge graph for entity-relationship storage
    - Personalized PageRank for single-step multi-hop retrieval
    """
    
    def __init__(self):
        self._hipporag: HippoRAG | None = None
        self._save_dir: str = "/data/hipporag"
        self._llm_provider = None
    
    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize HippoRAG with file-based storage.
        
        Args:
            config: Configuration dict with:
                - save_dir: Storage directory path
                - llm_model: LLM model (e.g., "azure/gpt-4.1")
                - embedding_model: Embedding model
                - llm_base_url: Optional custom LLM endpoint
                - embedding_base_url: Optional custom embedding endpoint
        """
        import os
        
        # Configure storage
        self._save_dir = config.get("save_dir", "/data/hipporag")
        os.makedirs(self._save_dir, exist_ok=True)
        
        # Get LLM config from environment (existing litellm setup)
        llm_model = config.get("llm_model", "azure/gpt-4.1")
        embedding_model = config.get(
            "embedding_model", 
            "azure/text-embedding-3-small"
        )
        
        # Build base URL for litellm
        llm_base_url = config.get("llm_base_url")
        if not llm_base_url:
            # Use existing litellm proxy or direct Azure
            llm_base_url = os.getenv("LITELLM_PROXY_URL") or \
                          os.getenv("AZURE_OPENAI_API_BASE")
        
        # Initialize HippoRAG
        from hipporag import HippoRAG
        
        self._hipporag = HippoRAG(
            save_dir=self._save_dir,
            llm_model_name=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            embedding_model_name=embedding_model,
            embedding_base_url=config.get("embedding_base_url"),
            embedding_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        
        logger.info(f"HippoRAG initialized at {self._save_dir}")
    
    async def write(
        self, 
        conn: Connection, 
        data: TransformedData
    ) -> WriteResult:
        """
        Write documents to HippoRAG memory.
        
        Process:
        1. Extract text from all chunks
        2. Index documents (runs OpenIE, builds graph, generates embeddings)
        3. Store in file-based storage
        """
        # Extract all text from chunks
        docs = []
        for chunk in data.chunks:
            text = chunk.get("content", "").strip()
            if text:
                docs.append(text)
        
        if not docs:
            return WriteResult(
                success=False,
                error="No text content in chunks"
            )
        
        try:
            # Index documents (this runs OpenIE, builds KG, generates embeddings)
            # HippoRAG.index() is synchronous, run in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self._hipporag.index, 
                docs
            )
            
            return WriteResult(
                success=True,
                destination_id="hipporag",
                destination_uri=f"hipporag://{self._save_dir}",
                records_written=len(docs),
                metadata={
                    "save_dir": self._save_dir,
                    "job_id": str(data.job_id),
                    "chunks_processed": len(docs),
                }
            )
            
        except Exception as e:
            logger.error(f"HippoRAG indexing failed: {e}")
            return WriteResult(
                success=False,
                error=str(e)
            )
    
    async def retrieve(
        self,
        queries: list[str],
        num_to_retrieve: int = 10
    ) -> list[RetrievalResult]:
        """
        Multi-hop retrieval using Personalized PageRank.
        
        This is the core HippoRAG capability - single-step
        multi-hop retrieval that can traverse the knowledge graph.
        
        Args:
            queries: List of query strings
            num_to_retrieve: Number of passages to retrieve per query
            
        Returns:
            List of retrieval results with passages and scores
        """
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._hipporag.retrieve,
                queries,
                num_to_retrieve
            )
            
            return [
                RetrievalResult(
                    query=query,
                    passages=result,
                    source="hipporag"
                )
                for query, result in zip(queries, results)
            ]
            
        except Exception as e:
            logger.error(f"HippoRAG retrieval failed: {e}")
            return []
    
    async def rag_qa(
        self,
        queries: list[str],
        retrieval_results: list[RetrievalResult] | None = None
    ) -> list[QAResult]:
        """
        Full RAG pipeline: retrieve + generate answer.
        
        If retrieval_results not provided, performs retrieval first.
        """
        try:
            loop = asyncio.get_event_loop()
            
            if retrieval_results:
                # Use provided retrieval results
                results = await loop.run_in_executor(
                    None,
                    self._hipporag.rag_qa,
                    retrieval_results
                )
            else:
                # Combined retrieval + QA
                results = await loop.run_in_executor(
                    None,
                    self._hipporag.rag_qa,
                    queries
                )
            
            return [
                QAResult(
                    query=query,
                    answer=result.get("answer", ""),
                    sources=result.get("sources", []),
                    confidence=result.get("confidence", 0.0)
                )
                for query, result in zip(queries, results)
            ]
            
        except Exception as e:
            logger.error(f"HippoRAG QA failed: {e}")
            return []
```

### 2. LLM Integration

HippoRAG will use the existing litellm provider:

```python
# Configuration for litellm integration
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_API_BASE")

# HippoRAG configuration
HippoRAG(
    save_dir=save_dir,
    llm_model_name="azure/gpt-4.1",
    llm_base_url=os.getenv("AZURE_OPENAI_API_BASE"),
    llm_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    embedding_model_name="azure/text-embedding-3-small",
    embedding_base_url=os.getenv("AZURE_OPENAI_API_BASE"),
    embedding_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
```

For OpenRouter fallback:
```python
# If Azure fails, retry with OpenRouter
try:
    result = await self.retrieve(queries)
except Exception:
    # Switch to OpenRouter
    self._hipporag = HippoRAG(
        ...,
        llm_model_name="openrouter/openai/gpt-4.1",
        llm_base_url="https://openrouter.ai/api/v1",
        llm_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    result = await self.retrieve(queries)
```

### 3. Storage Structure

```
/data/hipporag/
├── graph/
│   └── knowledge_graph.pkl       # NetworkX graph object
├── embeddings/
│   ├── passage_embeddings.npy    # Passage embedding vectors
│   ├── entity_embeddings.npy     # Entity embedding vectors
│   └── embedding_index.faiss     # FAISS index (optional)
├── openie/
│   └── openie_results.json       # Extracted triples
├── ppr/
│   └── ppr_indices.pkl           # Precomputed PPR indices
└── config.json                   # HippoRAG configuration
```

### 4. Docker Volume Configuration

```yaml
# docker-compose.yml addition
services:
  worker:
    volumes:
      - hipporag-data:/data/hipporag
  
  api:
    volumes:
      - hipporag-data:/data/hipporag

volumes:
  hipporag-data:
    driver: local
```

## Data Flow

### Document Ingestion Flow

```
┌────────────┐     ┌──────────────────────┐
│ Transformed│────►│ Extract Text Chunks  │
│ Data       │     │ (from chunks field)  │
└────────────┘     └──────────┬───────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ HippoRAG.index() │
                    │ (runs in thread) │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌────────────┐ ┌──────────┐ ┌──────────────┐
       │   OpenIE   │ │   KG     │ │ Embeddings   │
       │  Extraction│ │  Build   │ │ Generation   │
       │  (LLM)     │ │          │ │ (LLM/Model)  │
       └────────────┘ └──────────┘ └──────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌──────────────────┐
                    │  Save to Disk    │
                    │  (persistent vol)│
                    └──────────────────┘
```

### Query Flow

```
┌──────────┐     ┌─────────────────────────────┐
│  Query   │────►│ HippoRAG.retrieve()         │
│  String  │     │ (single-step multi-hop)     │
└──────────┘     └──────────────┬──────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
           ┌────────────────┐    ┌──────────────────┐
           │  NER (LLM)     │    │  Load Graph      │
           │  Extract       │    │  from Disk       │
           │  Query Entities│    │                  │
           └───────┬────────┘    └──────────────────┘
                   │                       │
                   └───────────┬───────────┘
                               ▼
                    ┌──────────────────┐
                    │ Personalized     │
                    │ PageRank (PPR)   │
                    │ on Knowledge     │
                    │ Graph            │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Ranked Passages  │
                    │ (Multi-hop       │
                    │  results)        │
                    └──────────────────┘
```

## Configuration

### Environment Variables

```bash
# LLM Configuration (existing)
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
OPENROUTER_API_KEY=your-openrouter-key

# HippoRAG Configuration
HIPPO_SAVE_DIR=/data/hipporag
HIPPO_LLM_MODEL=azure/gpt-4.1
HIPPO_EMBEDDING_MODEL=azure/text-embedding-3-small
HIPPO_RETRIEVAL_K=10

# Storage
HIPPO_PERSISTENT_VOLUME=/data/hipporag
```

### Plugin Configuration

```python
# config/destinations.yaml
hipporag:
  plugin: HippoRAGDestination
  config:
    save_dir: /data/hipporag
    llm_model: azure/gpt-4.1
    embedding_model: azure/text-embedding-3-small
    fallback_model: openrouter/openai/gpt-4.1
```

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| LLM rate limit | Retry with exponential backoff, fallback to OpenRouter |
| Embedding model unavailable | Use text-embedding-3-small fallback |
| Storage full | Log error, alert admin, stop ingestion |
| Graph too large | Implement graph pruning/archival |
| Index corruption | Rebuild from OpenIE results |

## Performance Considerations

| Aspect | Strategy |
|--------|----------|
| Async I/O | Run HippoRAG.sync operations in thread pool |
| Batch processing | Index documents in batches |
| Caching | Cache embeddings, reuse for similar docs |
| Lazy loading | Load graph components on demand |
| Graph pruning | Remove low-confidence edges periodically |

## Testing Strategy

| Test Type | Coverage |
|-----------|----------|
| Unit | Plugin methods, configuration |
| Integration | HippoRAG + LLM + storage |
| E2E | Document → graph → multi-hop query → answer |
| Multi-hop QA | HotpotQA, 2WikiMultiHopQA benchmarks |
| Performance | Latency, throughput, accuracy |

## Deployment

### Prerequisites
1. Persistent volume mounted at `/data/hipporag`
2. LLM provider configured (Azure/OpenRouter)
3. Sufficient disk space (~500MB per 1000 docs)

### Steps
1. Add volume to Docker Compose
2. Install `hipporag` library
3. Configure environment variables
4. Create destination plugin
5. Test with sample documents
6. Deploy to production

## Monitoring

| Metric | Method |
|--------|--------|
| Graph size | File system size of save_dir |
| Query latency | Timing around retrieve() calls |
| Indexing throughput | Docs/min during write() |
| Multi-hop accuracy | Benchmark datasets |
| Storage usage | Volume metrics |
