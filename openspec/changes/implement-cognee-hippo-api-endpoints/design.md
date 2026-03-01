# Design: Cognee and HippoRAG API Endpoints

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
├─────────────────────────────────────────────────────────────┤
│  src/api/routes/cognee.py       │  Cognee endpoints         │
│  src/api/routes/hipporag.py     │  HippoRAG endpoints       │
│  src/api/routes/system.py       │  openapi.graphrag.yaml    │
├─────────────────────────────────────────────────────────────┤
│  src/api/models/cognee.py       │  Cognee Pydantic models   │
│  src/api/models/hipporag.py     │  HippoRAG Pydantic models │
├─────────────────────────────────────────────────────────────┤
│  src/plugins/destinations/     │  Existing plugins         │
│    ├── cognee_local.py         │  Cognee implementation    │
│    ├── hipporag.py             │  HippoRAG implementation  │
│    └── hipporag_llm.py         │  HippoRAG LLM provider    │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

### New Files
1. `src/api/routes/cognee.py` - Cognee API routes
2. `src/api/routes/hipporag.py` - HippoRAG API routes
3. `src/api/models/cognee.py` - Cognee Pydantic models
4. `src/api/models/hipporag.py` - HippoRAG Pydantic models

### Modified Files
1. `src/api/routes/__init__.py` - Register new routers
2. `src/main.py` - Add new routes and openapi.graphrag.yaml endpoint

## Design Decisions

### 1. Route Organization
- Create separate route files for cognee and hipporag to keep code modular
- Follow existing pattern: one file per functional area (like rag.py, search.py)

### 2. Model Organization
- Create separate model files for cognee and hipporag
- Keep models close to their routes for maintainability

### 3. Plugin Integration
- Routes will call existing destination plugins directly
- No new database schemas needed (uses existing Neo4j/storage)

### 4. Error Handling
- Use existing error handling patterns from rag.py
- Return 422 for validation errors
- Return 503 for plugin/service unavailable
- Return 500 for unexpected errors

### 5. Authentication
- Follow existing auth patterns from other routes
- Use `get_current_user` dependency if required

## API Pattern Reference

Looking at `src/api/routes/rag.py`:
- Use `@router.post()` decorators with response models
- Include `tags` for OpenAPI documentation grouping
- Use dependency injection for services
- Include timing metrics in responses

## Testing Strategy

1. Unit tests for each endpoint
2. Integration tests with mocked plugins
3. Contract tests to verify OpenAPI spec compliance
