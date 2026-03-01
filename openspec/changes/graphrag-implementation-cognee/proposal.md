# Proposal: Local Cognee GraphRAG Integration

## Executive Summary

Replace the current **API-based GraphRAG** implementation with a **local/embedded Cognee** solution that provides:
- True local operation (no external API dependencies)
- Multi-modal support (text, images, audio)
- Multiple storage backends (Neo4j for graphs, PostgreSQL/pgvector for vectors)
- Integration with existing Azure OpenAI/OpenRouter LLM infrastructure

## Problem Statement

The current GraphRAG implementation (`src/plugins/destinations/graphrag.py`) relies on external HTTP API calls:
- Network latency for every operation
- External dependency (single point of failure)
- Data leaves the infrastructure
- API costs at scale

## Proposed Solution

Implement **Cognee** as a local Python library with:
- **LLM**: Azure OpenAI (primary) + OpenRouter (fallback) via existing litellm infrastructure
- **Graph Database**: Neo4j (via Docker Compose)
- **Vector Database**: Existing PostgreSQL with pgvector
- **Relational**: Existing PostgreSQL

## Benefits

| Benefit | Description |
|---------|-------------|
| **Performance** | No network latency, direct function calls |
| **Reliability** | No external dependencies |
| **Privacy** | Data never leaves your infrastructure |
| **Multi-modal** | Support for images, audio, PDFs |
| **Cost** | No API usage fees for graph operations |
| **Flexibility** | Modular backends, swap components as needed |

## Scope

### In Scope
- Cognee destination plugin (local, non-API)
- Neo4j Docker Compose service
- PostgreSQL/pgvector integration for Cognee vectors
- LLM integration via litellm (Azure OpenAI + OpenRouter)
- Migration from existing GraphRAG API plugin

### Out of Scope
- HippoRAG implementation (separate change)
- Graph visualization UI
- Multi-tenant graph isolation

## Success Criteria

1. Cognee destination plugin operates without external API calls
2. Neo4j service running in Docker Compose
3. Vector storage uses existing PostgreSQL/pgvector
4. LLM calls routed through existing litellm provider
5. Performance: 10x faster than current API-based GraphRAG
6. All existing GraphRAG tests pass with Cognee

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 | Week 1-2 | Neo4j infrastructure, Cognee dependency |
| Phase 2 | Week 3-4 | Cognee destination plugin |
| Phase 3 | Week 5-6 | LLM integration, testing, migration |

## Dependencies

- Neo4j Docker image
- Cognee Python library (`pip install cognee`)
- Existing PostgreSQL/pgvector
- Existing litellm provider

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Cognee library compatibility | Pin version, test thoroughly |
| Neo4j resource usage | Configure memory limits in Docker |
| Migration complexity | Maintain both plugins during transition |
| Performance with large graphs | Implement incremental indexing |

## Alternatives Considered

| Alternative | Decision | Reason |
|-------------|----------|--------|
| LightRAG | Rejected | Less mature, no multi-modal |
| HippoRAG | Separate change | Different use case (research) |
| Keep API GraphRAG | Rejected | Doesn't solve core problems |

## Recommendation

**Approve** this change to implement local Cognee GraphRAG with Neo4j and PostgreSQL/pgvector storage.
