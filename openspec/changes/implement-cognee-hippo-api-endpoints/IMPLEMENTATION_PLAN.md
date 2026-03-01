# Implementation Plan: Cognee and HippoRAG API Endpoints

## OpenSpec Context
- **Change**: implement-cognee-hippo-api-endpoints
- **Proposal**: proposal.md
- **Design**: design.md
- **Tasks**: tasks.md
- **Specs**: 
  - specs/cognee-endpoints/spec.md
  - specs/hipporag-endpoints/spec.md

## Task List

| # | Task | Owner | Dependencies | Status |
|---|------|-------|--------------|--------|
| 1 | Create Cognee Pydantic models | backend-developer | None | pending |
| 2 | Create HippoRAG Pydantic models | backend-developer | None | pending |
| 3 | Implement Cognee routes (3 endpoints) | backend-developer | Task 1 | pending |
| 4 | Implement HippoRAG routes (3 endpoints) | backend-developer | Task 2 | pending |
| 5 | Register routes in __init__.py and main.py | backend-developer | Tasks 3,4 | pending |
| 6 | Add /api/v1/openapi.graphrag.yaml endpoint | backend-developer | Task 5 | pending |
| 7 | Create unit tests | tester-agent | Task 6 | pending |
| 8 | Run contract tests | tester-agent | Task 7 | pending |
| 9 | QA validation | qa-agent | Task 8 | pending |

## Architecture Notes

### File Locations
- Models: `src/api/models/cognee.py`, `src/api/models/hipporag.py`
- Routes: `src/api/routes/cognee.py`, `src/api/routes/hipporag.py`
- Integration: `src/api/routes/__init__.py`, `src/main.py`

### Existing Plugins to Use
- `src/plugins/destinations/cognee_local.py` → `CogneeLocalGraph`
- `src/plugins/destinations/hipporag.py` → `HippoRAGDestination`

### Pattern Reference
- Routes: Follow `src/api/routes/rag.py` structure
- Models: Follow `src/api/models/rag.py` structure
- Error handling: Use existing patterns from rag.py

## Endpoints to Implement

### Cognee (3 endpoints)
1. `POST /api/v1/cognee/search` - Search knowledge graph
2. `POST /api/v1/cognee/extract-entities` - Extract entities
3. `GET /api/v1/cognee/stats` - Get graph stats

### HippoRAG (3 endpoints)
1. `POST /api/v1/hipporag/retrieve` - Multi-hop retrieval
2. `POST /api/v1/hipporag/qa` - QA with retrieval
3. `POST /api/v1/hipporag/extract-triples` - Extract triples

### System (1 endpoint)
1. `GET /api/v1/openapi.graphrag.yaml` - Serve GraphRAG spec

## Validation Criteria

1. All 6 GraphRAG endpoints return 200 OK
2. Request/response schemas match openapi.graphrag.yaml
3. `/api/v1/openapi.graphrag.yaml` endpoint serves spec file
4. All existing tests still pass
5. New unit tests cover all endpoints
6. Contract tests validate OpenAPI compliance

## Iteration Log

| Iteration | Date | Agent | Task | Result |
|-----------|------|-------|------|--------|
| 1 | 2026-03-01 | backend-developer | Task 1: Cognee models | ✅ COMPLETE |
| 2 | 2026-03-01 | backend-developer | Task 2: HippoRAG models | ✅ COMPLETE |
| 3 | 2026-03-01 | backend-developer | Task 3: Cognee routes | ✅ COMPLETE |
| 4 | 2026-03-01 | backend-developer | Task 4: HippoRAG routes | ✅ COMPLETE |
| 5 | 2026-03-01 | backend-developer | Task 5: Integration | ✅ COMPLETE |
| 6 | 2026-03-01 | backend-developer | Task 6: Spec endpoint | ✅ COMPLETE |
| 7 | 2026-03-01 | tester-agent | Task 7: Unit tests | ✅ COMPLETE (82 tests pass) |
| 8 | 2026-03-01 | tester-agent | Task 8: Contract tests | ✅ COMPLETE |
| 9 | 2026-03-01 | qa-agent | Task 9: QA validation | ✅ ALL PATHS VALIDATED |
