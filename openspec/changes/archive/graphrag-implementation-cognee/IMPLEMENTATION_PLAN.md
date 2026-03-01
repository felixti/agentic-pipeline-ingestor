# Implementation Plan: Cognee GraphRAG

## OpenSpec Context
- **Change:** graphrag-implementation-cognee
- **Proposal:** proposal.md
- **Design:** design.md
- **Specs:** specs/cognee-integration/spec.md, specs/neo4j-infrastructure/spec.md

## Overview

Replace API-based GraphRAG with local Cognee GraphRAG using:
- **LLM:** Azure OpenAI (primary) + OpenRouter (fallback) via existing litellm
- **Graph DB:** Neo4j (Docker)
- **Vector DB:** Existing PostgreSQL/pgvector

## Task List

| # | Task | Owner | Dependencies | Status | Est. Time |
|---|------|-------|--------------|--------|-----------|
| 1 | Add Neo4j Docker service | db-agent | None | ✅ Done | 2h |
| 2 | Create Neo4j connection utility | backend-developer | Task 1 | ✅ Done | 4h |
| 3 | Install Cognee dependencies | backend-developer | None | ✅ Done | 1h |
| 4 | Create CogneeLocalDestination plugin | backend-developer | Task 2, 3 | ✅ Done | 6h |
| 5 | Implement document ingestion | backend-developer | Task 4 | ✅ Done | 6h |
| 6 | Implement graph search | backend-developer | Task 4 | ✅ Done | 6h |
| 7 | Integrate litellm LLM provider | backend-developer | Task 4 | ✅ Done | 4h |
| 8 | Create pgvector schema | db-agent | None | ✅ Done | 3h |
| 9 | Unit tests | tester-agent | Task 5 | ✅ Done | 4h |
| 10 | Integration tests | tester-agent | Task 6, 8 | ✅ Done | 6h |
| 11 | Performance benchmarks | tester-agent | Task 10 | ✅ Done | 4h |
| 12 | Migration script | backend-developer | Task 11 | ✅ Done | 6h |
| 13 | Migration docs | qa-agent | Task 12 | ✅ Done | 3h |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Worker / API                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Function Call
┌───────────────────────────▼─────────────────────────────────────┐
│              CogneeLocalDestination Plugin                      │
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
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Infrastructure Requirements

### Neo4j Docker Service
```yaml
neo4j:
  image: neo4j:5.15-community
  environment:
    - NEO4J_AUTH=neo4j/cognee-graph-db
    - NEO4J_dbms_memory_heap_max__size=2G
  ports:
    - "7687:7687"
    - "7474:7474"
  volumes:
    - neo4j-data:/data
```

### Python Dependencies
```toml
cognee = "^0.3.0"
neo4j = "^5.15.0"
```

## Implementation Notes

### Key Design Decisions

1. **LLM Integration:** Use existing litellm provider via Cognee's litellm adapter
2. **Storage:** Neo4j (graph) + existing PostgreSQL/pgvector (vectors)
3. **Configuration:** Environment variables for all settings
4. **Async:** All operations async via neo4j asyncio driver

### Critical Paths

1. Tasks 1-2-4: Infrastructure setup (blocking)
2. Tasks 4-5-6: Core plugin functionality (blocking)
3. Tasks 5-9-10: Testing (blocking release)

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Cognee library issues | Pin version, maintain API fallback |
| Neo4j performance | Monitor memory, tune queries |
| LLM rate limits | Implement retry with OpenRouter fallback |

## Validation Criteria

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Performance: 10x faster than API GraphRAG
- [ ] Zero data loss in migration
- [ ] Documentation complete

## Iteration Log

| Iteration | Date | Agent | Task | Result |
|-----------|------|-------|------|--------|
| 1 | 2026-02-28 | db-agent | Add Neo4j Docker service (Task 1) | ✅ Completed - Neo4j 5.15-community added with healthcheck, volumes, and memory limits |
| 2 | 2026-02-28 | backend-developer | Neo4j connection utility (Task 2) | ✅ Completed - Neo4jClient with async driver, health checks |
| 3 | 2026-02-28 | backend-developer | Install Cognee deps (Task 3) | ✅ Completed - cognee>=0.3.0, neo4j>=5.15.0 added |
| 4 | 2026-02-28 | backend-developer | CogneeLocalDestination plugin (Task 4) | ✅ Completed - Full plugin with Neo4j integration |
| 5-7 | 2026-02-28 | backend-developer | Doc ingestion, search, LLM (Tasks 5-7) | ✅ Completed - Entity extraction, graph search, litellm integration |
| 8 | 2026-02-28 | db-agent | pgvector schema (Task 8) | ✅ Completed - Migration 015 with cognee_vectors, documents, entities tables |
| 9-11 | 2026-02-28 | tester-agent | Tests and benchmarks (Tasks 9-11) | ✅ Completed - Unit, integration, performance tests |
| 12 | 2026-02-28 | backend-developer | Migration script (Task 12) | ✅ Completed - Full migration with verification and rollback |
| 13 | 2026-02-28 | qa-agent | Migration docs (Task 13) | ✅ Completed - Migration guide, usage docs, README updates |

## Phase 1 COMPLETE ✅

All 13 tasks completed for Cognee GraphRAG implementation.

### Summary

| Component | Status | Files |
|-----------|--------|-------|
| Neo4j Infrastructure | ✅ | docker-compose.yml, Neo4jClient |
| CogneeLocalDestination | ✅ | cognee_local.py, cognee_llm.py |
| pgvector Schema | ✅ | Migration 015 |
| Tests | ✅ | 2,900+ lines of test code |
| Migration | ✅ | Script + docs |

### Next Phase

**Phase 2: HippoRAG Implementation**
- See: `openspec/changes/graphrag-implementation-hipporag/`
- Estimated: 2 sprints, ~38 hours
