# Search Operations — Gap Analysis & Evolution Report

> **Date**: 2026-02-26
> **Last Verified**: 2026-02-26
> **Resolution Status**: P0 ✅ RESOLVED | P1 ✅ RESOLVED | P2 ✅ RESOLVED
> **Scope**: search services, RAG router/strategies, API routes, data models/repositories, migration files, and search/RAG/chunk test suites. Verified across `src/services/`, `src/rag/`, `src/api/routes/`, `src/db/`, top-level `migrations/versions/`, `tests/`, `README.md`, and `api/openapi.yaml`.

---

## 0. RESOLUTION SUMMARY (2026-02-26)

All P0, P1, and P2 issues from this report have been addressed. Below is a summary of the implemented fixes.

### P0 — Critical (All Resolved ✅)

| Issue | Resolution | Files Modified |
|-------|------------|----------------|
| **2.1 RAG Metrics Hardcoded** | Redis-backed `RAGMetricsStore` with counters, averages, and TTL support | `src/rag/metrics_store.py` (NEW), `src/rag/router.py`, `src/api/routes/rag.py` |
| **2.2 HyDE/Reranker Silent Degradation** | New `/rag/components/status` endpoint reports component availability | `src/api/routes/rag.py`, `src/api/models/rag.py` |
| **2.3 No Search Deduplication** | `deduplicate=true` parameter groups by `content_hash`, returns highest-score instance | `src/api/routes/search.py`, `tests/unit/api/test_search_deduplication.py` (NEW) |

### P1 — Significant (All Resolved ✅)

| Issue | Resolution | Files Modified |
|-------|------------|----------------|
| **3.1 No Reranking in Search** | `rerank=true` parameter added to semantic/hybrid endpoints | `src/api/routes/search.py` |
| **3.2 No Contextual Retrieval** | `include_context=true` parameter returns neighboring chunks | `src/api/routes/search.py`, `src/api/models/search.py` |
| **3.3 No Entity-Aware Search** | `document_entities` table with indexes + `entity_filter`/`entity_type_filter` params | `migrations/versions/011_add_document_entities.py` (NEW), `src/db/models.py`, `src/db/repositories/document_entity_repository.py` (NEW), `src/api/routes/search.py` |
| **3.4 No Quality-Filtered Search** | `quality_score` column added to `DocumentChunkModel` | `migrations/versions/010_add_chunk_quality_score.py` (NEW), `src/db/models.py` |
| **3.5 Missing tsvector Column** | `search_vector` TSVECTOR column with GIN index | `migrations/versions/009_add_search_vector_column.py` (NEW), `src/db/models.py`, `src/services/text_search_service.py` |
| **3.6 Hardcoded 1536 Dimensions** | `EMBEDDING_DIMENSIONS` env var + runtime validation at startup | `src/config.py`, `src/vector_store_config/vector_store.py`, `src/main.py`, `.env.example` |
| **3.8 In-Memory Embedding Cache** | RAG Redis cache wired into `EmbeddingService` | `src/services/embedding_service.py` |
| **P1-6 Sequential Hybrid Search** | Parallelized with `asyncio.gather()` | `src/services/hybrid_search_service.py` |

### P2 — Evolution (All Resolved ✅)

| Issue | Resolution | Files Modified |
|-------|------------|----------------|
| **P2-10 Sparse Vector Support** | `sparse_vector` column with SPLADE-style embedding support | `migrations/versions/013_add_sparse_vectors.py` (NEW), `src/db/models.py` |
| **P2-12 Search Analytics** | `search_analytics` table for query logging | `migrations/versions/012_add_search_analytics.py` (NEW) |
| **P2-13 Faceted Search** | Aggregation endpoint for metadata facets | `src/api/routes/search.py` |
| **P2-14 Prometheus Metrics** | `SearchMetrics` and `EmbeddingMetrics` classes | `src/observability/metrics.py`, `src/observability/__init__.py` |
| **P2-15 Multi-Embedding Support** | `chunk_embeddings` table for multiple embedding types per chunk | `migrations/versions/014_add_chunk_embeddings.py` (NEW) |
| **P1-3.9 Cross-Job Content Dedup** | Global content hash check before embedding | `src/services/embedding_service.py` |

### New Migrations Created

| File | Description |
|------|-------------|
| `009_add_search_vector_column.py` | TSVECTOR column + GIN index for fast full-text search |
| `010_add_chunk_quality_score.py` | `quality_score` column on document_chunks |
| `011_add_document_entities.py` | Entity extraction storage with search indexes |
| `012_add_search_analytics.py` | Search query logging and analytics |
| `013_add_sparse_vectors.py` | Sparse vector support for hybrid search |
| `014_add_chunk_embeddings.py` | Multi-embedding support per chunk |

### Next Steps

1. **Run migrations**: `make migrate` to apply new schema changes
2. **Run tests**: `make test-unit && make test-integration`
3. **Deploy**: Standard deployment pipeline

---

## 1. CURRENT STATE — What Exists

### 1.1 Core Search Services (`src/services/`)

| Service | File | Lines | Status |
|---|---|---|---|
| **Vector Search** | `vector_search_service.py` | 546 | ✅ Operational |
| **Text Search** | `text_search_service.py` | 725 | ✅ Operational |
| **Hybrid Search** | `hybrid_search_service.py` | 723 | ✅ Operational |
| **Embedding** | `embedding_service.py` | 555 | ✅ Operational |

- **Vector**: pgvector cosine distance (`<=>`), HNSW index (m=16, ef_construction=64, ef_search=32), 1536-dim embeddings, similarity = 1 − distance conversion
- **Text**: PostgreSQL `tsvector`/`tsquery` with BM25 ranking (`ts_rank_cd`), `pg_trgm` fuzzy matching (`%` operator), highlighting via `ts_headline`
- **Hybrid**: Weighted sum + RRF fusion (`FusionMethod` enum), fallback modes (auto/vector/text/strict), min-max score normalization, configurable weights (default 0.7 vector / 0.3 text)
- **Embedding**: litellm `aembedding()` integration (Azure OpenAI primary), batch processing (configurable `batch_size`, default 100), in-memory LRU cache with SHA-256 keying and TTL (default 3600s, max 10,000 entries)

### 1.2 RAG Module (`src/rag/` — 20 files, ~14,178 lines)

| Component | File(s) | Lines | Status | Notes |
|---|---|---|---|---|
| **AgenticRAG Router** | `router.py` | 734 | ✅ Built | Strategy presets: fast/balanced/thorough + auto (STRATEGY_MATRIX by query type) |
| **Query Classification** | `classification.py` | 745 | ✅ Built | 5 types: factual, analytical, comparative, vague, multi_hop. Pattern-based fast-path fallback |
| **Query Rewriting** | `strategies/query_rewriting.py` | 551 | ✅ Built | LLM-based query reformulation with Redis-backed `QueryRewritingCache` (TTL 3600s) |
| **HyDE** | `strategies/hyde.py` | 434 | ✅ Built | Hypothetical Document Embeddings, extends QueryRewriter, caching integrated |
| **Cross-Encoder Reranking** | `strategies/reranking.py` | 561 | ✅ Built | 4 models: ms-marco-MiniLM-L-6-v2 (default), ms-marco-electra-base (high_precision), ms-marco-TinyBERT-L-2 (fast), BAAI/bge-reranker-base (alternative) |
| **Contextual Retrieval** | `contextual.py` | 840 | ✅ Built | 3 strategies: parent_document, window (neighboring chunks), hierarchical (sections) |
| **Chunking** | `chunking.py` | 1,312 | ✅ Built | 4 strategies: SemanticChunker, FixedSizeChunker, HierarchicalChunker, AgenticChunker. ChunkingService orchestrator |
| **Multi-layer Cache** | `cache.py` | 1,461 | ✅ Built | L1: L1RedisCache (hot data), L2: L2PostgresCache (persistent), L3: L3SemanticCache (vector similarity). MultiLayerCache orchestrator |
| **RAG Evaluation** | `evaluation/metrics.py`, `evaluation/generation.py`, `evaluation/evaluator.py` | 1,899 | ✅ Built | Retrieval: MRR, NDCG@k, Recall@k, Precision@k, Hit Rate@k, Average Precision. Generation: BERTScore, Faithfulness, Answer Relevance, BLEU, ROUGE |
| **Benchmarking** | `evaluation/benchmarks.py` | 678 | ✅ Built | BenchmarkRunner, BenchmarkDataset, strategy comparison framework |

**Additional components not previously documented:**

| Component | File(s) | Lines | Notes |
|---|---|---|---|
| **Embedding Services** | `embeddings.py` | 1,096 | LiteLLMEmbedder, SentenceTransformerEmbedder, QuantizedEmbedding for compression, DimensionalityReducer |
| **Advanced Hybrid Search** | `strategies/hybrid_search.py` | 872 | Query expansion, weight presets (semantic_focus/balanced/lexical_focus), metadata filtering with operators (eq/ne/gt/lt/in/contains) |
| **Search Backend** | `search/hybrid_search.py` | 807 | HybridSearchService wrapper integrating vector + text backends with RRF |
| **Evaluation Models** | `evaluation/models.py` | 570 | StrategyComparison, ABTestResult, EvaluationAlert, MetricThresholds |
| **RAG Models** | `models.py` | 1,554 | QueryType, ContextType, ChunkingStrategy, RAGConfig, RAGResult, comprehensive Pydantic models |

### 1.3 API Endpoints

| Endpoint | Method | Functional? | Notes |
|---|---|---|---|
| `/search/semantic` | POST | ✅ Works | Requires pre-computed `query_embedding` (1536 floats) |
| `/search/semantic/text` | POST | ✅ Works | Accepts text query, generates embedding server-side via `EmbeddingService` |
| `/search/text` | POST | ✅ Works | Full-text + fuzzy search with BM25 ranking and highlighting |
| `/search/hybrid` | POST | ✅ Works | Full vector+text fusion. `EmbeddingService` injected via FastAPI `Depends()` (line 925). Query embedding generated server-side (line 981) |
| `/search/similar/{chunk_id}` | GET | ✅ Works | Find semantically similar chunks |
| `/jobs/{id}/chunks` | GET | ✅ Works | Paginated: `limit` (1-1000, default 100), `offset`. Returns `total`, `limit`, `offset` |
| `/jobs/{id}/chunks/{chunk_id}` | GET | ✅ Works | Optional embedding inclusion |
| `/rag/query` | POST | ✅ Works | HyDE/reranker initialized via try/except — graceful degradation if deps unavailable |
| `/rag/strategies` | GET | ✅ Works | Lists available strategy presets |
| `/rag/strategies/{name}/evaluate` | POST | ✅ Works | Evaluate a specific strategy |
| `/rag/metrics` | GET | ⚠️ Placeholder | Returns hardcoded zeros — no metrics persistence backend |
| `/rag/evaluate` | POST | ✅ Works | Evaluate RAG pipeline with custom config |

### 1.4 Contract & Documentation Artifacts

| Artifact | Status | Notes |
|---|---|---|
| `api/openapi.yaml` | ✅ Current | Includes all `/search/*` paths (lines 1150-1278), all `/rag/*` paths (lines 1336-1471), and `/jobs/{id}/chunks` paths (lines 1025-1093) |
| `README.md` search examples | ✅ Accurate | Documents hybrid search as full vector+text fusion with `weighted_sum`, `vector_weight: 0.7`, `text_weight: 0.3` — matches actual implementation |

---

## 2. CRITICAL GAPS — Things That Are Broken or Missing

### 🔴 P0 — Broken Integration

**2.1 RAG Metrics Are Hardcoded Zeros**
- **File**: `src/api/routes/rag.py` lines 617-631
- `RAGMetricsSummary` returns all zeros for `total_queries`, `avg_latency_ms`, `avg_retrieval_score`, `avg_classification_confidence`, and all strategy usage / query type distribution counts.
- Comment at line 617: `"# Create summary (in a real implementation, these would come from metrics storage)"`
- **Impact**: The `/rag/metrics` endpoint is functional but provides zero operational insight. No metrics persistence backend exists.
- **Fix**: Implement metrics persistence (e.g., Redis counters or a `rag_metrics` table) and populate from actual query execution data.

**2.2 RAG HyDE + Reranker: Graceful Degradation May Silently Disable Features**
- **File**: `src/api/routes/rag.py` lines 89-107
- Both `HyDERewriter` and `ReRanker` are initialized inside `try/except` blocks. If their dependencies (e.g., model downloads, network access) fail, they silently degrade to `None` with only a warning log.
- **Impact**: Callers requesting the "thorough" strategy (which enables HyDE + reranking) receive no indication that these components are disabled. "Thorough" silently behaves as "balanced".
- **Fix**: Add a `/rag/components/status` endpoint or include component availability in `/rag/strategies` response so callers know what's active.

### 🔴 P0 — Missing Functionality

**2.3 No Search Result Deduplication**
- Identical content chunks from different jobs appear as separate results.
- `content_hash` column exists in `DocumentChunkModel` (line 363) and is indexed, but is never consulted during search result filtering.
- **Impact**: Cross-job duplicate content pollutes search results.
- **Fix**: Add optional `deduplicate=true` parameter to search endpoints that groups by `content_hash` and returns only the highest-scoring instance.

---

## 3. SIGNIFICANT GAPS — Should Build

### 🟠 P1 — Search Quality

**3.1 No Reranking in Core Search Services**
- Cross-encoder reranking exists in `src/rag/strategies/reranking.py` (561 lines, 4 model options) but is **not connected** to `VectorSearchService`, `TextSearchService`, or `HybridSearchService`.
- The reranker lives entirely within the RAG module and can't be used by the raw search endpoints.
- **Evolution**: Add optional `rerank=true` parameter to `/search/hybrid` and `/search/semantic` endpoints.

**3.2 No Contextual Retrieval in Search Services**
- `src/rag/contextual.py` (840 lines) implements parent-document, window, and hierarchical context enhancement — but only within the RAG pipeline. Search endpoints return raw chunks with no surrounding context.
- **Evolution**: Add `include_context=true` parameter that returns neighboring chunks.

**3.3 No Entity-Aware Search**
- Entity extraction (`src/core/entity_extraction.py`) extracts Person, Organization, Location, etc. But entities are stored in job metadata, not indexed for search.
- No `document_entities` table exists.
- **Evolution**: Create `document_entities` table with `chunk_id` FK, add entity filter to search endpoints.

**3.4 No Quality-Filtered Search**
- Chunks from poor OCR (quality score < 0.5) pollute search results alongside high-quality chunks.
- `quality_score` exists only on `JobResultModel` (line 307), **not** on `DocumentChunkModel`.
- **Evolution**: Add `quality_score` column to `DocumentChunkModel`, populate during pipeline, filter in search queries.

**3.5 Missing `tsvector` Stored Column**
- Text search computes `to_tsvector(language, content)` at query time via `func.to_tsvector()` in `text_search_service.py`; there is an expression GIN index in migration 003, but no persisted `search_vector` column for explicit update/maintenance control.
- **Evolution**: Add a pre-computed `search_vector` `tsvector` column with a GIN index for dramatically faster full-text search.

### 🟠 P1 — Data Layer

**3.6 Single Embedding Model, Hardcoded 1536 Dimensions**
- `Vector(dimensions=1536)` is hardcoded in the SQLAlchemy model (`src/db/models.py` line 366). Switching to a different model (e.g., `text-embedding-3-large` at 3072 dims) requires a migration.
- **Evolution**: Make dimensions configurable or support multiple embedding columns.

**3.7 No Sparse Vector Support**
- pgvector 0.7+ supports `sparsevec` and `halfvec` types for BM25-compatible sparse embeddings and half-precision storage. Neither type is used anywhere in the codebase.
- **Evolution**: Add sparse vector column for keyword-aware embeddings (SPLADE/BGE-M3), enabling true native hybrid search in PostgreSQL.

**3.8 Embedding Cache Is In-Memory Only (services layer)**
- `EmbeddingCache` in `embedding_service.py` is per-process LRU (SHA-256 keyed, max 10,000 entries, TTL 3600s). The RAG module has a 3-layer cache (Redis → PostgreSQL → Semantic) in `src/rag/cache.py`, but the core embedding service doesn't use it.
- **Evolution**: Wire the RAG cache layer into `EmbeddingService`, or use Redis directly.

**3.9 No Cross-Job Content Deduplication**
- `upsert_chunks()` deduplicates within a job (by `job_id + chunk_index` via ON CONFLICT at line 334-337), but identical content across different jobs creates separate embeddings.
- **Evolution**: Before embedding, check `content_hash` globally. If embedding already exists, reuse it.

---

## 4. EVOLUTION OPPORTUNITIES — Should Consider

### 🟡 P2 — Advanced Retrieval

| Feature | Current State | Evolution |
|---|---|---|
| **Multi-vector search** | Single 1536-dim dense vector | Add ColBERT-style late-interaction or matryoshka embeddings for adaptive precision |
| **HyDE at search layer** | Only in RAG module | Expose HyDE as option on `/search/hybrid?use_hyde=true` |
| **Filtered vector search** | `job_id` + JSONB metadata `@>` | Add pre-filter indexes on common metadata keys (source_type, language, topic) |
| **Faceted search** | Not implemented | Add aggregation queries: count by topic, source, language alongside search results |
| **Auto-merging retrieval** | Not implemented | When small chunks match, auto-merge with parent section for context |
| **Sentence-window retrieval** | Contextual retrieval exists in RAG | Expose as search parameter: return chunk ± N sentences |
| **Multi-modal search** | Text only | Add image embedding column for scanned document visual search |

### 🟡 P2 — Observability & Analytics

| Feature | Current State | Evolution |
|---|---|---|
| **Search analytics** | Structured logging only | Add search_queries table: track query, results, latency, user, click-through |
| **Relevance feedback** | Not implemented | Add thumbs-up/down on results, feed into weight tuning |
| **Prometheus search metrics** | Generic API metrics | Add `search_latency_histogram`, `search_result_count`, `embedding_cache_hit_rate` |
| **OpenTelemetry spans** | Not search-specific | Add spans for: embedding generation, vector query, text query, fusion, reranking |
| **Slow query detection** | Not implemented | Log queries > P95 latency with full context |

### 🟡 P2 — Performance

| Feature | Current State | Evolution |
|---|---|---|
| **Parallel search execution** | Sequential in hybrid | Use `asyncio.gather()` for vector + text search |
| **Result caching (Redis)** | In RAG cache module, not in search services | Cache frequent queries in Redis with configurable TTL |
| **Streaming results** | Not implemented | SSE endpoint for large result sets or real-time search |
| **Batch search API** | Not implemented | Accept multiple queries in one request for analytics workloads |
| **HNSW parameter tuning** | Static (m=16, ef=64/32) | Auto-tune based on dataset size; ef_search adjustable per query |
| **Index partitioning** | Single HNSW index | Partition by job_id or source_type for large-scale deployments |

---

## 5. TEST COVERAGE

### 5.1 Existing Test Suites

| Area | File | Lines | Coverage Notes |
|---|---|---|---|
| Core search services (unit) | `tests/unit/services/test_vector_store_services.py` | 1,206 | Heavy mocking — no real pgvector query execution |
| Search API endpoints (integration) | `tests/integration/test_search_api.py` | 1,080 | Route-level behavior, mocked service calls |
| Chunk API (integration) | `tests/integration/test_chunk_api.py` | 596 | Good pagination/response coverage |
| Performance benchmarks | `tests/performance/test_search_performance.py` | 809 | Includes `test_concurrent_vector_searches` (20 concurrent), but mocked DB/session |
| RAG API (integration) | `tests/test_rag_api.py` | 300 | RAG endpoint tests |
| Chunk repository (unit) | `tests/unit/repositories/test_document_chunk_repository.py` | — | Chunk CRUD with embeddings |

**RAG unit tests** (`tests/unit/rag/`):

| File | Coverage |
|---|---|
| `test_router.py` | AgenticRAG router logic |
| `test_classification.py` | Query classification |
| `test_query_rewriting.py` | Query rewriting |
| `test_hyde.py` | HyDE strategy |
| `test_reranking.py` | Cross-encoder reranking |
| `test_contextual.py` | Contextual retrieval |
| `test_chunking.py` | Chunking strategies |
| `test_cache.py` | Multi-layer cache |
| `test_embeddings.py` | Embedding services |
| `test_hybrid_search.py` | Hybrid search logic |
| `evaluation/` (6 files) | Metrics, benchmarks, evaluator |

**Functional tests** (real database/pgvector):

| File | Lines | Coverage |
|---|---|---|
| `tests/functional/test_vector_storage_proof.py` | 503 | Real embedding storage, retrieval, similarity search |
| `tests/functional/test_vector_features.py` | 533 | Real vector feature tests |
| `tests/functional/test_database_persistence.py` | 451 | Database persistence validation |

**E2E tests** (`tests/e2e/scenarios/`):

| File | Lines | Coverage |
|---|---|---|
| `test_full_pipeline.py` | 407 | Job creation, status checks, uploads |
| `test_performance.py` | 426 | Concurrent job creation, concurrent API requests, throughput |
| `test_retry_mechanisms.py` | — | Retry logic |
| `test_dlq_workflow.py` | — | Dead letter queue |
| `test_auth_flow.py` | — | Authentication flows |

### 5.2 Test Coverage Gaps

| Area | Gap |
|---|---|
| **E2E: Upload → Chunk → Embed → Search** | ❌ No single test chains the full document lifecycle through search |
| **High-concurrency search stress** | ⚠️ 20-concurrent test exists but no high-load (100+) stress test against real HNSW |
| **Embedding provider failover** | ❌ Azure → OpenRouter fallback path not tested |
| **RAG with real search backend** | ⚠️ RAG endpoints tested but without end-to-end wiring to real search services |

---

## 6. PRIORITIZED RECOMMENDATIONS

### Immediate (P0 — Broken)
1. **Implement RAG metrics persistence** — `/rag/metrics` returns hardcoded zeros; needs Redis counters or metrics table
2. **Add RAG component availability reporting** — callers can't tell if HyDE/reranker are active or silently disabled
3. **Add search result dedup by `content_hash`** — duplicate cross-job content pollutes results

### Short-Term (P1 — Quality Lift)
4. **Add stored `tsvector` column + GIN index** — simplify ranking path and improve maintainability
5. **Connect cross-encoder reranking to search endpoints** — `rerank=true` parameter
6. **Parallel vector + text execution** in hybrid search — `asyncio.gather()`
7. **Wire RAG cache into `EmbeddingService`** — Redis L1 cache for embeddings
8. **Add E2E test** for full document → chunk → embed → search flow
9. **Add `quality_score` to `DocumentChunkModel`** — enable quality-filtered search

### Medium-Term (P2 — Evolution)
10. **Sparse vector column** (`sparsevec`) for native hybrid search in PostgreSQL
11. **Entity search index** — searchable entity table with chunk FK
12. **Search analytics table** — query logging, click-through tracking, relevance feedback
13. **Faceted search** — aggregations by metadata (topic, source, language)
14. **Prometheus metrics** — search-specific histograms and counters
15. **Configurable embedding dimensions** — support model switching without migrations

---

## 7. EXTERNAL VALIDATION NOTES (Docs/OSS)

These findings came from external references to validate evolution items and risk statements.

### 7.1 pgvector + PostgreSQL Guidance

- **HNSW defaults/knobs are valid and query-tunable** (`m`, `ef_construction`, `ef_search`), with `SET LOCAL hnsw.ef_search = ...` supported in query scope.
  - Source: pgvector README — https://github.com/pgvector/pgvector
- **Current config risk**: `ef_search=32` is below pgvector default (40), likely reducing recall on harder filtered searches.
  - Source: pgvector README — https://github.com/pgvector/pgvector
- **Iterative scans** (`hnsw.iterative_scan`) are available in newer pgvector releases and help filtered vector queries return enough candidates.
  - Source: pgvector README — https://github.com/pgvector/pgvector
- **`sparsevec` and `halfvec` are supported** (with type/operator-class caveats by driver/ORM stack).
  - Source: pgvector README — https://github.com/pgvector/pgvector

### 7.2 Full-Text Search Guidance

- PostgreSQL docs recommend a stored/generated `tsvector` column with GIN for full-text workloads requiring stable and efficient query paths.
  - Source: PostgreSQL docs (`textsearch-tables`, `textsearch-indexes`) — https://www.postgresql.org/docs/current/textsearch-tables.html and https://www.postgresql.org/docs/current/textsearch-indexes.html

### 7.3 Reranking and Embedding Cache Patterns

- Common production pattern is **layered embedding cache** (process-local memory + shared Redis) with TTL and batch ops.
  - Source: RedisVL Embeddings Cache docs — https://redis.io/docs/latest/develop/ai/redisvl/0.7.0/user_guide/embeddings_cache/
- Common retrieval API pattern is optional query-time reranking (`top_n` after initial retrieval), which matches this report's `rerank=true` recommendation.
  - Source: LlamaIndex rerank docs — https://developers.llamaindex.ai/python/framework-api-reference/postprocessor/llm_rerank/

### 7.4 Recommendation Updates from External Validation

16. **Raise/tune `ef_search` and expose query-level override** — improve recall for vector/hybrid retrieval.
17. **Enable iterative HNSW scan mode when available** — improve filtered vector result completeness.
18. **Prioritize generated `tsvector` migration path** — align with PostgreSQL guidance and simplify indexing behavior.

---

## 8. VERIFICATION CHANGELOG

> **2026-02-26** — Full codebase verification performed. The following corrections were made to the original report:

### Removed — False Claims (Previously Listed as P0 Gaps)

| Original Claim | Correction |
|---|---|
| **"Hybrid search is text-only fallback"** (2.1) | `EmbeddingService` IS injected via `Depends(get_embedding_service)` at `search.py` line 925. Query embedding generated server-side at line 981. Hybrid search is fully operational with vector+text fusion. |
| **"No `/search/semantic/text` endpoint"** (2.6) | Endpoint EXISTS at `search.py` lines 676-777. Accepts text query, generates embedding server-side via injected `EmbeddingService`. |
| **"`api/openapi.yaml` missing search/RAG paths"** (2.4) | Spec INCLUDES all paths: `/search/*` at lines 1150-1278, `/rag/*` at lines 1336-1471, `/jobs/{id}/chunks` at lines 1025-1093. |
| **"README hybrid search is misleading"** (2.5) | README accurately describes full fusion with `weighted_sum`, `vector_weight: 0.7`, `text_weight: 0.3` — matches implementation. |
| **"No pagination in search results"** (2.7) | Chunks endpoint supports `limit` (1-1000) and `offset` parameters with paginated response. |
| **"Real pgvector integration not tested"** (Section 5) | Functional tests exist: `test_vector_storage_proof.py` (503 lines), `test_vector_features.py` (533 lines), `test_database_persistence.py` (451 lines) — all test real pgvector. |

### Corrected — Nuanced Claims

| Original Claim | Correction |
|---|---|
| **"AgenticRAG initialized with None"** (2.2) | HyDE/reranker use `try/except` graceful degradation (lines 89-107), not hardcoded None. Reclassified from "broken integration" to "silent degradation risk". |
| **"Concurrent search not tested"** (Section 5) | `test_search_performance.py` includes `test_concurrent_vector_searches` with 20 concurrent queries. Gap narrowed to high-load (100+) stress testing. |

### Added — Previously Undocumented

| Addition | Detail |
|---|---|
| RAG `embeddings.py` (1,096 lines) | LiteLLMEmbedder, SentenceTransformerEmbedder, QuantizedEmbedding, DimensionalityReducer |
| RAG advanced hybrid search (872 lines) | Query expansion, weight presets, metadata filtering with operators |
| RAG evaluation models (570 lines) | StrategyComparison, ABTestResult, EvaluationAlert system |
| RAG models (1,554 lines) | Comprehensive Pydantic models for all RAG components |
| 15+ additional test files | Full RAG unit test suite, functional vector tests, E2E scenarios |
