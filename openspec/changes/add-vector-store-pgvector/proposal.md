# Proposal: add-vector-store-pgvector

## Why

The current pipeline architecture relies exclusively on **external destination plugins** (Cognee, Neo4j, Pinecone, Weaviate) for vector storage and semantic search capabilities. This creates several critical limitations:

### Problems Being Solved

1. **No Direct Chunk Retrieval**: Users cannot query and retrieve individual chunks of processed documents without setting up external vector database infrastructure. The system stores job results but doesn't expose granular chunk-level access.

2. **Missing Semantic Search**: There's no built-in capability to perform semantic similarity searches across processed content. Users must integrate with external services to enable vector-based retrieval.

3. **Infrastructure Complexity**: Requiring external vector stores (Pinecone, Weaviate, etc.) adds operational overhead and costs. Many use cases need lightweight, embedded vector search without managing additional services.

4. **No Hybrid Search**: The system lacks the ability to combine traditional text search (BM25/fuzzy matching) with vector similarity for optimal retrieval quality—a key feature in modern RAG (Retrieval-Augmented Generation) applications.

5. **OpenSearch Feature Gap**: The current architecture uses OpenSearch only for audit logs, missing the opportunity to provide OpenSearch-like search features (full-text + vector hybrid) directly within the primary PostgreSQL datastore.

### Use Cases Enabled

- **Direct Chunk Access**: Retrieve specific document chunks by ID or source document for debugging, verification, or downstream processing
- **Semantic Document Search**: Find semantically similar content across the document corpus without exact keyword matching
- **RAG Pipeline Support**: Enable retrieval-augmented generation workflows by providing vector search + chunk context directly from the API
- **Hybrid Search Queries**: Combine keyword relevance (BM25) with semantic similarity for superior search results
- **Reduced Infrastructure**: Run complete document processing + search pipelines with only PostgreSQL as the data store

## What Changes

Implement a **native vector storage and search system** using PostgreSQL with the pgvector extension. This enables the pipeline to store chunk embeddings and perform vector similarity search without external dependencies.

### Technical Approach

#### 1. pgvector Extension Integration

- **Extension Setup**: Enable pgvector extension in PostgreSQL for vector storage and similarity operations
- **Index Types**: Support HNSW (Hierarchical Navigable Small World) indexes for approximate nearest neighbor search with configurable `m` and `ef_construction` parameters
- **Distance Metrics**: Support multiple distance functions (cosine similarity, L2/Euclidean, inner product) based on embedding model requirements

#### 2. Fuzzy Text Search Implementation

Research and implement PostgreSQL-native full-text search capabilities to emulate OpenSearch features:

- **BM25 Ranking**: Implement PostgreSQL's `ts_rank_cd` with custom weights for BM25-like scoring
- **Fuzzy Matching**: Use `pg_trgm` extension for trigram-based fuzzy string matching
- **Combined Scoring**: Develop weighted hybrid scoring that combines text relevance with vector similarity

#### 3. Schema Design

Create new database tables for vector storage:

```sql
-- Document chunks with embeddings
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),  -- Configurable dimensions
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast similarity search
CREATE INDEX idx_document_chunks_embedding 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops);

-- GIN index for full-text search
CREATE INDEX idx_document_chunks_content_search 
ON document_chunks 
USING gin (to_tsvector('english', content));
```

#### 4. Embedding Generation Integration

- Hook into the existing pipeline transformation stage to generate embeddings using the LLM adapter
- Support configurable embedding models via `config/llm.yaml`
- Batch embedding generation for efficiency with large documents
- Store embeddings alongside chunks for unified retrieval

#### 5. Search API Endpoints

Implement RESTful endpoints for chunk retrieval and search:

```
GET    /api/v1/jobs/{job_id}/chunks           # List chunks for a job
GET    /api/v1/jobs/{job_id}/chunks/{id}     # Get specific chunk
POST   /api/v1/search/semantic               # Semantic similarity search
POST   /api/v1/search/text                   # Full-text search (BM25-like)
POST   /api/v1/search/hybrid                 # Combined vector + text search
GET    /api/v1/search/similar/{chunk_id}     # Find similar chunks
```

#### 6. Configuration & Tuning

Add configuration options for:
- Embedding model selection and dimensions
- Similarity thresholds and top-k results
- HNSW index parameters (m, ef_construction, ef_search)
- Hybrid search weighting (vector vs. text relevance)

## Capabilities

- [ ] **pgvector extension setup and configuration**
  - [ ] Database migration to enable pgvector extension
  - [ ] Docker compose updates for pgvector-enabled PostgreSQL image
  - [ ] Connection validation and health checks
  - [ ] Dimension configuration (384, 768, 1536, etc.)

- [ ] **Vector embedding storage schema**
  - [ ] `document_chunks` table with vector column
  - [ ] HNSW index creation for approximate nearest neighbors
  - [ ] SQLAlchemy model with pgvector integration
  - [ ] Migration scripts with index tuning options
  - [ ] Relationship linking to jobs and job_results tables

- [ ] **Semantic similarity search**
  - [ ] Cosine similarity queries using pgvector operators (`<=>`, `<#>`, `<->`)
  - [ ] Configurable top-k retrieval with similarity thresholds
  - [ ] Vector search service layer with async support
  - [ ] Metadata filtering (filter by job_id, date ranges, etc.)
  - [ ] Pagination and cursor-based results

- [ ] **Chunk retrieval by ID/source**
  - [ ] Get chunk by UUID endpoint
  - [ ] List chunks by job_id with pagination
  - [ ] Get chunks by content hash for deduplication
  - [ ] Bulk chunk retrieval for batch operations
  - [ ] Include source document context in responses

- [ ] **Fuzzy text search (BM25 or similar)**
  - [ ] PostgreSQL full-text search implementation using `tsvector`/`tsquery`
  - [ ] `pg_trgm` extension for trigram fuzzy matching
  - [ ] BM25-like ranking using `ts_rank_cd` with custom weights
  - [ ] Highlighting of search terms in results
  - [ ] Support for multiple languages and custom dictionaries

- [ ] **Hybrid search combining vector + text**
  - [ ] Weighted scoring algorithm (e.g., 0.7 * vector_score + 0.3 * text_score)
  - [ ] Reciprocal Rank Fusion (RRF) for combining rankings
  - [ ] Configurable weight parameters per query
  - [ ] Fallback strategies when one search type returns no results
  - [ ] Performance optimization with combined indexes

- [ ] **Search API endpoints**
  - [ ] `GET /api/v1/jobs/{job_id}/chunks` - List job chunks
  - [ ] `GET /api/v1/jobs/{job_id}/chunks/{chunk_id}` - Get specific chunk
  - [ ] `POST /api/v1/search/semantic` - Vector similarity search
  - [ ] `POST /api/v1/search/text` - Full-text search
  - [ ] `POST /api/v1/search/hybrid` - Combined search
  - [ ] `GET /api/v1/search/similar/{chunk_id}` - Find similar chunks
  - [ ] Query parameter validation and sanitization
  - [ ] Rate limiting integration

## Impact

### Benefits

| Benefit | Description | Impact Level |
|---------|-------------|--------------|
| **Semantic Search** | Enable vector-based similarity search across all processed documents without external dependencies | High |
| **Better Retrieval** | Provide granular chunk-level access for debugging, verification, and RAG applications | High |
| **OpenSearch-like Features** | Implement full-text + vector hybrid search within PostgreSQL, reducing need for separate search infrastructure | Medium |
| **Reduced Infrastructure** | Run complete pipelines with PostgreSQL only—no external vector store required for many use cases | High |
| **Unified Data Store** | Keep embeddings, chunks, and metadata in one database for simpler backup, replication, and querying | Medium |
| **Cost Savings** | Avoid external vector database costs (Pinecone, Weaviate) for smaller deployments | Medium |
| **Lower Latency** | In-database search eliminates network hops to external services | Medium |

### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Performance degradation** | Medium | High | HNSW indexes for approximate search; configurable ef_search parameter; query timeout limits |
| **pgvector extension availability** | Low | High | Document PostgreSQL 14+ requirement; provide migration guide for cloud providers (AWS RDS, Azure, GCP) |
| **Memory usage** | Medium | Medium | Configurable index parameters; monitoring alerts; connection pooling |
| **Embedding model compatibility** | Low | Medium | Support multiple distance metrics; validate dimensions at startup; clear error messages |
| **Migration complexity** | Medium | Low | Alembic migrations with rollback support; gradual rollout; feature flags |

### Dependencies

#### Required
- **PostgreSQL 14+**: pgvector requires PostgreSQL 14 or later for full feature support
- **pgvector extension**: Must be installed and enabled (`CREATE EXTENSION vector;`)
- **pg_trgm extension**: For fuzzy text matching (`CREATE EXTENSION pg_trgm;`)

#### Optional
- **Embedding model endpoint**: Azure OpenAI, OpenRouter, or local model for generating embeddings
- **Increased PostgreSQL memory**: HNSW indexes perform best with sufficient shared_buffers

#### Configuration Changes
```yaml
# config/vector_store.yaml
vector_store:
  enabled: true
  embedding_model: "text-embedding-3-small"  # or local model
  dimensions: 1536
  
  search:
    default_top_k: 10
    max_top_k: 100
    similarity_threshold: 0.7
    
  index:
    type: "hnsw"
    m: 16                    # Number of connections per layer
    ef_construction: 64      # Build-time accuracy vs speed tradeoff
    ef_search: 32           # Query-time accuracy vs speed tradeoff
    
  hybrid_search:
    vector_weight: 0.7
    text_weight: 0.3
    fusion_method: "weighted_sum"  # or "rrf" for Reciprocal Rank Fusion
```

### Breaking Changes

**None** - This is an additive feature that:
- Adds new tables without modifying existing schema
- Introduces new API endpoints without changing existing ones
- Is opt-in via configuration (vector store can remain disabled)

### Migration Path

1. **Database Migration**: Run Alembic migration to create `document_chunks` table and indexes
2. **Extension Setup**: Enable pgvector and pg_trgm extensions
3. **Configuration**: Add vector store configuration to environment
4. **Pipeline Integration**: Enable embedding generation in pipeline config
5. **Existing Data**: Optionally backfill embeddings for historical jobs (background job)

### Success Metrics

- Vector search latency p99 < 100ms for 1M chunks
- Text search latency p99 < 50ms for indexed content
- Hybrid search relevance improvement > 20% over text-only search
- Zero breaking changes to existing API
- pgvector availability verified on target deployment platforms (AWS RDS, Azure Database, GCP Cloud SQL)
