# OpenSpec Tasks: Cognee GraphRAG Implementation

## Tasks Overview

| Task ID | Title | Owner | Status | Dependencies |
|---------|-------|-------|--------|--------------|
| COG-INFRA-1 | Add Neo4j Docker Compose service | db-agent | pending | None |
| COG-INFRA-2 | Create Neo4j connection utility | backend-developer | pending | COG-INFRA-1 |
| COG-PLUGIN-1 | Install Cognee library dependency | backend-developer | pending | None |
| COG-PLUGIN-2 | Create CogneeLocalDestination plugin | backend-developer | pending | COG-PLUGIN-1, COG-INFRA-2 |
| COG-PLUGIN-3 | Implement document ingestion | backend-developer | pending | COG-PLUGIN-2 |
| COG-PLUGIN-4 | Implement graph search | backend-developer | pending | COG-PLUGIN-2 |
| COG-LLM-1 | Integrate litellm provider with Cognee | backend-developer | pending | COG-PLUGIN-2 |
| COG-DB-1 | Create pgvector schema for Cognee | db-agent | pending | None |
| COG-TEST-1 | Write unit tests for plugin | tester-agent | pending | COG-PLUGIN-3 |
| COG-TEST-2 | Write integration tests | tester-agent | pending | COG-PLUGIN-4, COG-DB-1 |
| COG-TEST-3 | Run performance benchmarks | tester-agent | pending | COG-TEST-2 |
| COG-MIGRATE-1 | Create migration script | backend-developer | pending | COG-TEST-3 |
| COG-MIGRATE-2 | Document migration procedure | qa-agent | pending | COG-MIGRATE-1 |

## Task Details

### COG-INFRA-1: Add Neo4j Docker Compose service

**Description:** Add Neo4j Community Edition service to docker-compose.yml

**Acceptance Criteria:**
- [ ] Neo4j service added to docker-compose.yml
- [ ] Uses official neo4j:5.15-community image
- [ ] Configured with 2GB max memory
- [ ] Persistent volume for data
- [ ] Health check configured
- [ ] Ports 7687 (Bolt) and 7474 (Browser) exposed

**Files:**
- `docker/docker-compose.yml`

---

### COG-INFRA-2: Create Neo4j connection utility

**Description:** Create utility class for Neo4j connection management

**Acceptance Criteria:**
- [ ] Neo4jConnection class created
- [ ] Async support via neo4j asyncio driver
- [ ] Connection pooling configured
- [ ] Retry logic for transient failures
- [ ] Health check method

**Files:**
- `src/infrastructure/neo4j/connection.py`
- `src/infrastructure/neo4j/__init__.py`

---

### COG-PLUGIN-1: Install Cognee library dependency

**Description:** Add Cognee to project dependencies

**Acceptance Criteria:**
- [ ] `cognee = "^0.3.0"` added to pyproject.toml
- [ ] `neo4j = "^5.15.0"` added to pyproject.toml
- [ ] Dependencies installed successfully
- [ ] No version conflicts

**Files:**
- `pyproject.toml`

---

### COG-PLUGIN-2: Create CogneeLocalDestination plugin

**Description:** Create new destination plugin for local Cognee

**Acceptance Criteria:**
- [ ] Class implements DestinationPlugin interface
- [ ] initialize() method configures Cognee
- [ ] Uses Neo4j for graph storage
- [ ] Uses PostgreSQL/pgvector for vectors
- [ ] Configuration from environment variables

**Files:**
- `src/plugins/destinations/cognee_local.py`

---

### COG-PLUGIN-3: Implement document ingestion

**Description:** Implement write() method for document ingestion

**Acceptance Criteria:**
- [ ] Extract text from TransformedData chunks
- [ ] Add documents to Cognee dataset
- [ ] Call cognify() to build graph
- [ ] Handle errors gracefully
- [ ] Return proper WriteResult

**Files:**
- `src/plugins/destinations/cognee_local.py`

---

### COG-PLUGIN-4: Implement graph search

**Description:** Implement search() method for graph queries

**Acceptance Criteria:**
- [ ] Search method accepts query string
- [ ] Supports hybrid search (vector + graph)
- [ ] Returns structured results
- [ ] Configurable top_k parameter
- [ ] Filter by dataset name

**Files:**
- `src/plugins/destinations/cognee_local.py`

---

### COG-LLM-1: Integrate litellm provider with Cognee

**Description:** Configure Cognee to use existing litellm provider

**Acceptance Criteria:**
- [ ] Cognee uses Azure OpenAI via litellm
- [ ] Fallback to OpenRouter configured
- [ ] Embedding calls use text-embedding-3-small
- [ ] API keys from environment variables

**Files:**
- `src/plugins/destinations/cognee_local.py`
- `src/llm/cognee_adapter.py` (if needed)

---

### COG-DB-1: Create pgvector schema for Cognee

**Description:** Create database tables and indexes for Cognee vectors

**Acceptance Criteria:**
- [ ] cognee_vectors table created
- [ ] VECTOR(1536) column for embeddings
- [ ] IVFFlat index on embedding column
- [ ] Index on dataset_name column
- [ ] Alembic migration created

**Files:**
- `src/db/migrations/versions/XXX_add_cognee_vectors.py`

---

### COG-TEST-1: Write unit tests for plugin

**Description:** Unit tests for CogneeLocalDestination

**Acceptance Criteria:**
- [ ] Test initialize() with valid config
- [ ] Test initialize() with invalid config
- [ ] Test write() with sample data
- [ ] Test search() with sample query
- [ ] Mock external dependencies

**Files:**
- `tests/unit/test_cognee_local_destination.py`

---

### COG-TEST-2: Write integration tests

**Description:** Integration tests with real Neo4j and PostgreSQL

**Acceptance Criteria:**
- [ ] Test document → graph flow
- [ ] Test query → results flow
- [ ] Test entity extraction
- [ ] Test relationship mapping
- [ ] Docker Compose test setup

**Files:**
- `tests/integration/test_cognee_integration.py`

---

### COG-TEST-3: Run performance benchmarks

**Description:** Benchmark Cognee vs API GraphRAG

**Acceptance Criteria:**
- [ ] Document ingestion throughput measured
- [ ] Query latency measured
- [ ] Memory usage profiled
- [ ] Comparison with baseline (API GraphRAG)
- [ ] Results documented

**Files:**
- `tests/performance/test_cognee_performance.py`
- `docs/performance/cognee_benchmarks.md`

---

### COG-MIGRATE-1: Create migration script

**Description:** Script to migrate data from API GraphRAG to Cognee

**Acceptance Criteria:**
- [ ] Export data from existing GraphRAG
- [ ] Transform to Cognee format
- [ ] Import to Cognee/Neo4j
- [ ] Verify data integrity
- [ ] Rollback capability

**Files:**
- `scripts/migrate_graphrag_to_cognee.py`

---

### COG-MIGRATE-2: Document migration procedure

**Description:** Document the migration process

**Acceptance Criteria:**
- [ ] Step-by-step migration guide
- [ ] Prerequisites documented
- [ ] Rollback procedure documented
- [ ] Troubleshooting guide

**Files:**
- `docs/migration/graphrag-to-cognee.md`

## Task Dependencies Graph

```
COG-INFRA-1 (Neo4j Docker)
    └── COG-INFRA-2 (Neo4j connection)
            └── COG-PLUGIN-2 (Create plugin)

COG-PLUGIN-1 (Install Cognee)
    └── COG-PLUGIN-2 (Create plugin)
            ├── COG-PLUGIN-3 (Document ingestion)
            │       └── COG-TEST-1 (Unit tests)
            │               └── COG-TEST-2 (Integration tests)
            │                       └── COG-TEST-3 (Performance)
            │                               └── COG-MIGRATE-1 (Migration script)
            │                                       └── COG-MIGRATE-2 (Docs)
            ├── COG-PLUGIN-4 (Graph search)
            │       └── COG-TEST-2 (Integration tests)
            └── COG-LLM-1 (LLM integration)

COG-DB-1 (DB schema)
    └── COG-TEST-2 (Integration tests)
```

## Estimates

| Task | Estimate |
|------|----------|
| COG-INFRA-1 | 2h |
| COG-INFRA-2 | 4h |
| COG-PLUGIN-1 | 1h |
| COG-PLUGIN-2 | 6h |
| COG-PLUGIN-3 | 6h |
| COG-PLUGIN-4 | 6h |
| COG-LLM-1 | 4h |
| COG-DB-1 | 3h |
| COG-TEST-1 | 4h |
| COG-TEST-2 | 6h |
| COG-TEST-3 | 4h |
| COG-MIGRATE-1 | 6h |
| COG-MIGRATE-2 | 3h |
| **Total** | **~55 hours** |
