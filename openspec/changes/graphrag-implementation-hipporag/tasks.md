# OpenSpec Tasks: HippoRAG Implementation

## Tasks Overview

| Task ID | Title | Owner | Status | Dependencies |
|---------|-------|-------|--------|--------------|
| HIP-INFRA-1 | Add HippoRAG persistent volume | db-agent | pending | None |
| HIP-PLUGIN-1 | Install HippoRAG library | backend-developer | pending | None |
| HIP-PLUGIN-2 | Create HippoRAGDestination plugin | backend-developer | pending | HIP-PLUGIN-1, HIP-INFRA-1 |
| HIP-PLUGIN-3 | Implement document indexing | backend-developer | pending | HIP-PLUGIN-2 |
| HIP-PLUGIN-4 | Implement multi-hop retrieval | backend-developer | pending | HIP-PLUGIN-2 |
| HIP-PLUGIN-5 | Implement RAG QA | backend-developer | pending | HIP-PLUGIN-4 |
| HIP-LLM-1 | Integrate litellm provider | backend-developer | pending | HIP-PLUGIN-2 |
| HIP-TEST-1 | Write unit tests | tester-agent | pending | HIP-PLUGIN-3 |
| HIP-TEST-2 | Write integration tests | tester-agent | pending | HIP-PLUGIN-5 |
| HIP-TEST-3 | Run multi-hop QA benchmarks | tester-agent | pending | HIP-TEST-2 |
| HIP-DOC-1 | Document use cases | qa-agent | pending | HIP-TEST-3 |

## Task Details

### HIP-INFRA-1: Add HippoRAG persistent volume

**Description:** Add persistent volume for HippoRAG file-based storage

**Acceptance Criteria:**
- [ ] Volume `hipporag-data` added to docker-compose.yml
- [ ] Mounted at `/data/hipporag` in worker and api services
- [ ] Volume persists across container restarts
- [ ] Size monitoring configured (optional)

**Files:**
- `docker/docker-compose.yml`

---

### HIP-PLUGIN-1: Install HippoRAG library

**Description:** Add HippoRAG to project dependencies

**Acceptance Criteria:**
- [ ] `hipporag = "^0.1.0"` added to pyproject.toml
- [ ] Dependencies installed successfully
- [ ] No version conflicts with existing packages

**Files:**
- `pyproject.toml`

---

### HIP-PLUGIN-2: Create HippoRAGDestination plugin

**Description:** Create new destination plugin for HippoRAG

**Acceptance Criteria:**
- [ ] Class implements DestinationPlugin interface
- [ ] initialize() method configures HippoRAG
- [ ] Uses file-based storage (save_dir)
- [ ] Configuration from environment variables
- [ ] Async support via thread pool

**Files:**
- `src/plugins/destinations/hipporag_local.py`

---

### HIP-PLUGIN-3: Implement document indexing

**Description:** Implement write() method for document indexing

**Acceptance Criteria:**
- [ ] Extract text from TransformedData chunks
- [ ] Call HippoRAG.index() in thread pool
- [ ] Handle OpenIE, KG building, embeddings
- [ ] Progress logging for long indexing operations
- [ ] Return proper WriteResult

**Files:**
- `src/plugins/destinations/hipporag_local.py`

---

### HIP-PLUGIN-4: Implement multi-hop retrieval

**Description:** Implement retrieve() method for multi-hop queries

**Acceptance Criteria:**
- [ ] Accept list of query strings
- [ ] Call HippoRAG.retrieve() with PPR
- [ ] Configurable num_to_retrieve parameter
- [ ] Return structured RetrievalResult
- [ ] Handle errors gracefully

**Files:**
- `src/plugins/destinations/hipporag_local.py`

---

### HIP-PLUGIN-5: Implement RAG QA

**Description:** Implement rag_qa() method for full pipeline

**Acceptance Criteria:**
- [ ] Combined retrieve + generate
- [ ] Accept pre-computed retrieval results
- [ ] Return structured QAResult with answer and sources
- [ ] Confidence scoring

**Files:**
- `src/plugins/destinations/hipporag_local.py`

---

### HIP-LLM-1: Integrate litellm provider

**Description:** Configure HippoRAG to use existing litellm provider

**Acceptance Criteria:**
- [ ] Azure OpenAI configured as primary LLM
- [ ] Azure text-embedding-3-small as embedding model
- [ ] Fallback to OpenRouter on failure
- [ ] API keys from environment variables
- [ ] Base URL configuration for litellm proxy

**Files:**
- `src/plugins/destinations/hipporag_local.py`
- `src/llm/hipporag_adapter.py` (if needed)

---

### HIP-TEST-1: Write unit tests

**Description:** Unit tests for HippoRAGDestination

**Acceptance Criteria:**
- [ ] Test initialize() with valid config
- [ ] Test initialize() with invalid config
- [ ] Test write() with sample data
- [ ] Test retrieve() with sample queries
- [ ] Mock HippoRAG library

**Files:**
- `tests/unit/test_hipporag_destination.py`

---

### HIP-TEST-2: Write integration tests

**Description:** Integration tests with real storage

**Acceptance Criteria:**
- [ ] Test document → index → retrieve flow
- [ ] Test multi-hop retrieval
- [ ] Test RAG QA pipeline
- [ ] Docker Compose test setup
- [ ] Persistent volume test

**Files:**
- `tests/integration/test_hipporag_integration.py`

---

### HIP-TEST-3: Run multi-hop QA benchmarks

**Description:** Benchmark multi-hop reasoning accuracy

**Acceptance Criteria:**
- [ ] Test on HotpotQA sample
- [ ] Test on 2WikiMultiHopQA sample
- [ ] Measure accuracy vs baseline
- [ ] Measure query latency
- [ ] Document results

**Files:**
- `tests/benchmarks/test_hipporag_multihop.py`
- `docs/benchmarks/hipporag_results.md`

---

### HIP-DOC-1: Document use cases

**Description:** Document when to use HippoRAG vs Cognee

**Acceptance Criteria:**
- [ ] Use case comparison document
- [ ] Decision matrix
- [ ] Example multi-hop queries
- [ ] Performance guidelines

**Files:**
- `docs/hipporag/use-cases.md`
- `docs/hipporag/decision-matrix.md`

## Task Dependencies Graph

```
HIP-INFRA-1 (Persistent volume)
    └── HIP-PLUGIN-2 (Create plugin)

HIP-PLUGIN-1 (Install library)
    └── HIP-PLUGIN-2 (Create plugin)
            ├── HIP-PLUGIN-3 (Document indexing)
            │       └── HIP-TEST-1 (Unit tests)
            │               └── HIP-TEST-2 (Integration tests)
            │                       └── HIP-TEST-3 (Benchmarks)
            │                               └── HIP-DOC-1 (Documentation)
            ├── HIP-PLUGIN-4 (Multi-hop retrieval)
            │       └── HIP-TEST-2 (Integration tests)
            └── HIP-PLUGIN-5 (RAG QA)
                    └── HIP-TEST-2 (Integration tests)

HIP-LLM-1 (LLM integration)
    └── HIP-PLUGIN-2 (Create plugin)
```

## Estimates

| Task | Estimate |
|------|----------|
| HIP-INFRA-1 | 1h |
| HIP-PLUGIN-1 | 1h |
| HIP-PLUGIN-2 | 4h |
| HIP-PLUGIN-3 | 6h |
| HIP-PLUGIN-4 | 4h |
| HIP-PLUGIN-5 | 4h |
| HIP-LLM-1 | 4h |
| HIP-TEST-1 | 3h |
| HIP-TEST-2 | 4h |
| HIP-TEST-3 | 4h |
| HIP-DOC-1 | 3h |
| **Total** | **~38 hours** |
