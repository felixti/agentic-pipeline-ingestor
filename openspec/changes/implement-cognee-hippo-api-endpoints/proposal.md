# Proposal: Implement Cognee and HippoRAG API Endpoints

## Problem Statement

The project has two OpenAPI spec files:
1. `api/openapi.yaml` - Main API specification (41 paths, 3,078 lines)
2. `api/openapi.graphrag.yaml` - GraphRAG extension spec (6 paths, 409 lines)

The GraphRAG spec defines 6 endpoints for Cognee and HippoRAG:
- 3 Cognee endpoints (search, extract-entities, stats)
- 3 HippoRAG endpoints (retrieve, qa, extract-triples)

**Current State**: These endpoints are defined in the spec but NOT implemented in the FastAPI application.

## Proposed Solution

Implement FastAPI routes for all 6 GraphRAG endpoints:

### Cognee Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/cognee/search` | POST | Search Cognee knowledge graph |
| `/api/v1/cognee/extract-entities` | POST | Extract entities from text |
| `/api/v1/cognee/stats` | GET | Get Cognee graph statistics |

### HippoRAG Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/hipporag/retrieve` | POST | Multi-hop retrieval |
| `/api/v1/hipporag/qa` | POST | Full RAG QA with multi-hop |
| `/api/v1/hipporag/extract-triples` | POST | Extract OpenIE triples |

### Additional Tasks
- Add `/api/v1/openapi.graphrag.yaml` endpoint to serve the extension spec
- Ensure all endpoints have proper request/response models
- Add comprehensive tests for all endpoints

## Success Criteria

1. All 6 GraphRAG endpoints return proper responses (200 OK)
2. Request/response schemas match openapi.graphrag.yaml
3. `/api/v1/openapi.graphrag.yaml` endpoint serves the spec file
4. All contract tests pass
5. Code follows existing patterns in src/api/routes/

## Dependencies

- Existing cognee and hippo destination plugins in `src/plugins/destinations/`
- Neo4j infrastructure for Cognee
- Existing RAG routes in `src/api/routes/rag.py` (for pattern reference)
