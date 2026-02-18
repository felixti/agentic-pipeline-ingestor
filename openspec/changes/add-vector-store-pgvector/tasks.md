# Tasks: add-vector-store-pgvector

## Overview

This document outlines the implementation tasks for adding native vector storage and search capabilities using PostgreSQL with the pgvector extension. This enables semantic search, text search (BM25-like), and hybrid search without external dependencies.

**Goal**: Implement a complete vector search system including database schema, services, API endpoints, and pipeline integration.

---

## Phase 1: Foundation (Database & Infrastructure)

*Prerequisite: PostgreSQL 14+ with pgvector extension availability*

| # | Task | Effort | Deps | Owner |
|---|------|--------|------|-------|
| 1.1 | [ ] **Enable pgvector extension** - Create Alembic migration to enable pgvector and pg_trgm extensions with version validation | M | none | db-agent |
| 1.2 | [ ] **Docker Compose updates** - Switch postgres image to pgvector-enabled image (`pgvector/pgvector:pg17` or `ankane/pgvector`) | S | none | backend-dev |
| 1.3 | [ ] **PostgreSQL version check** - Add migration guard to verify PostgreSQL 14+ before enabling extensions | S | 1.1 | db-agent |
| 1.4 | [ ] **Extension availability validation** - Add health check endpoint `/health/vector_store` to verify pgvector availability | M | 1.1 | backend-dev |
| 1.5 | [ ] **Connection pool configuration** - Update database settings for vector search workload (pool_size, max_overflow) | S | 1.2 | backend-dev |

**Phase 1 Exit Criteria:**
- [ ] pgvector extension can be enabled via Alembic migration
- [ ] Docker Compose uses pgvector-enabled PostgreSQL image
- [ ] Health check validates extension availability
- [ ] All existing tests pass with new database image

---

## Phase 2: Data Layer (Schema & Models)

*Core database structures for document chunk storage*

| # | Task | Effort | Deps | Owner |
|---|------|--------|------|-------|
| 2.1 | [ ] **Create document_chunks table** - Alembic migration for table with UUID PK, job_id FK, chunk_index, content, embedding vector, metadata JSONB, created_at | M | 1.1 | db-agent |
| 2.2 | [ ] **Add table constraints** - Unique (job_id, chunk_index), CHECK (chunk_index >= 0), CASCADE delete on job removal | S | 2.1 | db-agent |
| 2.3 | [ ] **Create HNSW index** - Migration for `idx_document_chunks_embedding_hnsw` with configurable m, ef_construction parameters | M | 2.1 | db-agent |
| 2.4 | [ ] **Create GIN full-text index** - Migration for `idx_document_chunks_content_search` on to_tsvector('english', content) | S | 2.1 | db-agent |
| 2.5 | [ ] **Create trigram index** - Migration for `idx_document_chunks_content_trgm` with gin_trgm_ops for fuzzy matching | S | 2.1 | db-agent |
| 2.6 | [ ] **Create supporting indexes** - B-tree on job_id, BRIN on created_at, GIN on metadata jsonb_path_ops | S | 2.1 | db-agent |
| 2.7 | [ ] **SQLAlchemy DocumentChunk model** - Model class with pgvector-sqlalchemy Vector type, relationships to Job model | M | 2.1, 2.3 | backend-dev |
| 2.8 | [ ] **Dimension configuration** - Support configurable dimensions (384, 768, 1536, 3072) via migration parameter | S | 2.1 | db-agent |
| 2.9 | [ ] **DocumentChunkRepository** - Repository class with get_by_id, get_by_job, create, batch_create, count methods | M | 2.7 | backend-dev |
| 2.10 | [ ] **Job relationship** - Add chunks relationship to JobModel with back_populates | S | 2.7 | backend-dev |

**Phase 2 Exit Criteria:**
- [ ] All migrations run successfully and are reversible
- [ ] DocumentChunk SQLAlchemy model works with async session
- [ ] Repository layer has full CRUD operations
- [ ] All indexes are created and queryable

---

## Phase 3: Service Layer (Search Logic)

*Core search algorithms and business logic*

| # | Task | Effort | Deps | Owner |
|---|------|--------|------|-------|
| 3.1 | [ ] **VectorSearchService core** - Service class with async search_by_vector() using HNSW index and cosine similarity | M | 2.7 | backend-dev |
| 3.2 | [ ] **Vector dimension validation** - Validate query vector dimensions match configured dimension on search | S | 3.1 | backend-dev |
| 3.3 | [ ] **ef_search configuration** - Set hnsw.ef_search per query based on accuracy requirements | S | 3.1 | backend-dev |
| 3.4 | [ ] **Metadata filtering** - Add metadata filter support (equals, range, contains) to vector search | M | 3.1 | backend-dev |
| 3.5 | [ ] **TextSearchService BM25** - Implement search_bm25() using ts_rank_cd with custom weights | M | 2.4 | backend-dev |
| 3.6 | [ ] **Text highlighting** - Add ts_headline highlighting for search terms in results | M | 3.5 | backend-dev |
| 3.7 | [ ] **Fuzzy search (pg_trgm)** - Implement fuzzy_search() using similarity() and % operator | M | 2.5 | backend-dev |
| 3.8 | [ ] **HybridSearchService core** - Service class that combines vector and text search services | M | 3.1, 3.5 | backend-dev |
| 3.9 | [ ] **Weighted sum fusion** - Implement weighted scoring: combined = (v_weight × v_score) + (t_weight × t_score) | M | 3.8 | backend-dev |
| 3.10 | [ ] **RRF fusion** - Implement Reciprocal Rank Fusion: score = Σ(1 / (k + rank)) | M | 3.8 | backend-dev |
| 3.11 | [ ] **Fallback strategies** - Handle cases where one search returns no results | S | 3.8 | backend-dev |
| 3.12 | [ ] **ChunkService** - Service for chunk retrieval by ID, listing by job, bulk operations | M | 2.9 | backend-dev |
| 3.13 | [ ] **Similar chunks search** - Implement find_similar_chunks() by chunk_id using reference embedding | M | 3.1 | backend-dev |

**Phase 3 Exit Criteria:**
- [ ] Vector search returns results ordered by similarity
- [ ] Text search returns BM25-ranked results
- [ ] Hybrid search combines both with configurable fusion
- [ ] All services have comprehensive error handling

---

## Phase 4: API Layer (REST Endpoints)

*HTTP interface for chunk retrieval and search operations*

| # | Task | Effort | Deps | Owner |
|---|------|--------|------|-------|
| 4.1 | [ ] **Chunk list endpoint** - `GET /api/v1/jobs/{job_id}/chunks` with pagination (limit/offset), sorting | M | 3.12 | backend-dev |
| 4.2 | [ ] **Chunk get endpoint** - `GET /api/v1/jobs/{job_id}/chunks/{chunk_id}` with 404 handling | S | 3.12 | backend-dev |
| 4.3 | [ ] **Semantic search endpoint** - `POST /api/v1/search/semantic` with vector or text query input | M | 3.1 | backend-dev |
| 4.4 | [ ] **Text search endpoint** - `POST /api/v1/search/text` with BM25 search and highlighting | M | 3.5 | backend-dev |
| 4.5 | [ ] **Hybrid search endpoint** - `POST /api/v1/search/hybrid` with fusion method selection | M | 3.8 | backend-dev |
| 4.6 | [ ] **Similar chunks endpoint** - `GET /api/v1/search/similar/{chunk_id}` with exclude_same_job option | M | 3.13 | backend-dev |
| 4.7 | [ ] **Request validation** - Pydantic models for all search requests with query sanitization | M | 4.1-4.6 | backend-dev |
| 4.8 | [ ] **Response models** - Pydantic models for search results with scores, highlights, metadata | M | 4.1-4.6 | backend-dev |
| 4.9 | [ ] **Error handling** - Consistent error responses (400, 404, 429, 500, 503) with detailed messages | S | 4.1-4.6 | backend-dev |
| 4.10 | [ ] **Rate limiting** - Redis-based rate limiting for all search endpoints | M | 4.1-4.6 | backend-dev |
| 4.11 | [ ] **Route registration** - Add routes to FastAPI app with proper prefixes and tags | S | 4.1-4.6 | backend-dev |
| 4.12 | [ ] **API documentation** - OpenAPI/Swagger annotations for all endpoints | S | 4.1-4.11 | backend-dev |

**Phase 4 Exit Criteria:**
- [ ] All endpoints return correct HTTP status codes
- [ ] OpenAPI documentation is complete and accurate
- [ ] Rate limiting prevents abuse
- [ ] Input validation prevents SQL injection and XSS

---

## Phase 5: Integration & Configuration

*Pipeline integration and configuration management*

| # | Task | Effort | Deps | Owner |
|---|------|--------|------|-------|
| 5.1 | [ ] **VectorStoreConfig settings** - Pydantic settings class for vector store configuration | M | none | backend-dev |
| 5.2 | [ ] **vector_store.yaml config** - YAML configuration file with embedding, search, HNSW, hybrid search sections | M | 5.1 | backend-dev |
| 5.3 | [ ] **Configuration loading** - Integrate vector store config into main Settings class | S | 5.2 | backend-dev |
| 5.4 | [ ] **Embedding service integration** - Create EmbeddingService that uses LLM adapter for batch embedding generation | M | 5.1 | backend-dev |
| 5.5 | [ ] **Pipeline stage integration** - Add EmbeddingTransformStage to pipeline for automatic chunk embedding | M | 5.4 | backend-dev |
| 5.6 | [ ] **Chunking integration** - Hook chunk generation to store chunks in document_chunks table | M | 5.5 | backend-dev |
| 5.7 | [ ] **Batch embedding** - Implement batch embedding (100-1000 chunks) for efficiency | M | 5.4 | backend-dev |
| 5.8 | [ ] **Dimension validation** - Validate embedding model dimensions match vector_store config at startup | S | 5.1, 5.4 | backend-dev |
| 5.9 | [ ] **Feature flag** - Add `enabled` flag to disable vector store without code changes | S | 5.1 | backend-dev |
| 5.10 | [ ] **Service dependencies** - Implement FastAPI dependency injection for all search services | M | 3.1, 3.5, 3.8 | backend-dev |

**Phase 5 Exit Criteria:**
- [ ] Configuration loads from YAML and environment variables
- [ ] Pipeline automatically generates and stores embeddings
- [ ] Feature can be enabled/disabled via configuration
- [ ] All dependencies are injectable in FastAPI

---

## Phase 6: Testing & Documentation

*Quality assurance and documentation*

| # | Task | Effort | Deps | Owner |
|---|------|--------|------|-------|
| 6.1 | [ ] **Unit tests: VectorSearchService** - Test search_by_vector, dimension validation, metadata filtering | M | 3.1-3.4 | backend-dev |
| 6.2 | [ ] **Unit tests: TextSearchService** - Test search_bm25, fuzzy_search, highlighting | M | 3.5-3.7 | backend-dev |
| 6.3 | [ ] **Unit tests: HybridSearchService** - Test weighted fusion, RRF fusion, fallback strategies | M | 3.8-3.11 | backend-dev |
| 6.4 | [ ] **Unit tests: repositories** - Test DocumentChunkRepository CRUD and query operations | M | 2.9 | backend-dev |
| 6.5 | [ ] **Integration tests: API endpoints** - Test all search endpoints with test database | L | 4.1-4.12 | backend-dev |
| 6.6 | [ ] **Performance benchmarks** - Benchmark vector search latency at 10K, 100K, 1M chunk scales | M | 4.1-4.6 | backend-dev |
| 6.7 | [ ] **Load tests** - Test concurrent search requests and connection pooling | M | 4.1-4.6 | backend-dev |
| 6.8 | [ ] **Migration tests** - Test upgrade/downgrade paths with test data | S | 1.1-2.10 | db-agent |
| 6.9 | [ ] **API contract documentation** - Update `./shared/api-contracts.json` with all new endpoints | M | 4.1-4.12 | backend-dev |
| 6.10 | [ ] **User documentation** - Add vector search guide to docs/ with examples | M | 4.1-5.10 | backend-dev |
| 6.11 | [ ] **Configuration documentation** - Document all vector_store.yaml options | S | 5.2 | backend-dev |
| 6.12 | [ ] **Troubleshooting guide** - Document common issues (pgvector not installed, dimension mismatch, etc.) | S | 1.1-5.10 | backend-dev |

**Phase 6 Exit Criteria:**
- [ ] >80% code coverage for new services
- [ ] All integration tests pass
- [ ] Performance meets targets (p99 < 100ms for 1M chunks)
- [ ] API contracts documented in shared/api-contracts.json

---

## Summary

| Phase | Tasks | Effort | Status |
|-------|-------|--------|--------|
| Phase 1: Foundation | 5 | 2-3 days | Not Started |
| Phase 2: Data Layer | 10 | 4-5 days | Not Started |
| Phase 3: Service Layer | 13 | 6-7 days | Not Started |
| Phase 4: API Layer | 12 | 5-6 days | Not Started |
| Phase 5: Integration | 10 | 4-5 days | Not Started |
| Phase 6: Testing | 12 | 5-6 days | Not Started |
| **Total** | **62 tasks** | **~26-32 days** | **Not Started** |

---

## Dependencies Graph

```
Phase 1: Foundation
├── 1.1 Enable pgvector ──┬──► 2.1 Create table
│                         ├──► 2.3 HNSW index
│                         └──► 2.4 GIN index
├── 1.2 Docker Compose ───► 1.5 Connection pool
└── 1.4 Health check

Phase 2: Data Layer
├── 2.1-2.6 Migrations ───► 2.7 SQLAlchemy model
├── 2.7 Model ────────────► 2.9 Repository
│                         └──► 3.1 VectorSearchService
├── 2.3 HNSW index ───────► 3.1 VectorSearchService
├── 2.4 GIN index ────────► 3.5 TextSearchService
└── 2.5 Trigram index ────► 3.7 Fuzzy search

Phase 3: Service Layer
├── 3.1 Vector service ───► 3.8 HybridSearchService
├── 3.5 Text service ─────► 3.8 HybridSearchService
├── 3.8 Hybrid service ───► 4.5 Hybrid endpoint
├── 3.12 Chunk service ───► 4.1-4.2 Chunk endpoints
└── 3.13 Similar chunks ──► 4.6 Similar endpoint

Phase 4: API Layer
├── All endpoints ────────► 5.10 Service dependencies
└── 4.7-4.12 Validation ──► 6.5 Integration tests

Phase 5: Integration
├── 5.1-5.3 Config ───────► All dependent tasks
├── 5.4 Embedding svc ────► 5.5 Pipeline stage
└── 5.5 Pipeline stage ───► End-to-end functionality
```

---

## Risk Mitigation

| Risk | Mitigation Task |
|------|-----------------|
| pgvector not available on managed PostgreSQL | 1.4 Health check with clear error message |
| Performance degradation at scale | 2.3 HNSW tuning, 6.6 Performance benchmarks |
| Dimension mismatch | 5.8 Startup validation |
| Migration failure | 6.8 Migration tests, rollback procedures |
| Concurrent search overload | 1.5 Connection pooling, 4.10 Rate limiting |

---

## Notes for Implementers

### Critical Path
1. Enable pgvector (1.1) → Create table (2.1) → SQLAlchemy model (2.7) → Vector service (3.1) → API endpoints (4.x)

### Parallel Work Opportunities
- Docker Compose (1.2) can be done in parallel with migrations
- TextSearchService (3.5-3.7) can be developed in parallel with VectorSearchService
- API documentation (4.12) can be drafted during endpoint development

### Cloud Provider Considerations
- **AWS RDS**: pgvector available on RDS PostgreSQL 15+
- **Azure Database**: pgvector available on flexible server
- **GCP Cloud SQL**: pgvector available on Cloud SQL for PostgreSQL 15+
- **Neon/Supabase**: pgvector pre-installed

### Testing Requirements
- Use `pgvector/pgvector:pg17` Docker image for tests
- Create test fixtures with sample embeddings
- Mock embedding service for unit tests
- Use testcontainers for integration tests
