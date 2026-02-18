## Context

The `GET /api/v1/jobs/{job_id}` endpoint exists but returns hardcoded mock data. The database models are already defined in `src/db/models.py` with a `Job` model containing all necessary fields. The application uses FastAPI with SQLAlchemy async ORM.

## Goals / Non-Goals

**Goals:**
- Implement proper database lookup for job retrieval
- Return complete job details matching the existing API response structure
- Handle missing jobs with proper HTTP 404 responses
- Use FastAPI's dependency injection for database sessions

**Non-Goals:**
- Adding new job fields or changing the data model
- Implementing caching or performance optimizations
- Adding authentication/authorization checks (endpoint already exists)
- Supporting bulk job retrieval

## Decisions

### Decision 1: Database session dependency location
**Decision:** Use existing pattern in `src/main.py` - import and use database session directly within the endpoint function rather than creating a separate dependency module.

**Rationale:** The codebase currently handles database setup in `src/main.py`. Adding a separate `get_db()` dependency in `src/api/dependencies.py` would be cleaner long-term, but keeping changes minimal and consistent with existing patterns reduces risk for this small fix.

**Alternative considered:** Creating `src/api/dependencies.py` with a reusable `get_db()` dependency. This is the FastAPI best practice but adds unnecessary complexity for a single endpoint fix.

### Decision 2: 404 handling approach
**Decision:** Use FastAPI's `HTTPException` with status code 404 when job is not found.

**Rationale:** FastAPI automatically converts `HTTPException` to proper JSON error responses. This is idiomatic and requires no additional error handlers.

**Alternative considered:** Return a custom error response manually. Rejected as it adds boilerplate without benefit.

### Decision 3: Query method
**Decision:** Use `select(Job).where(Job.id == job_id)` with `session.execute()` and `scalars().first()`.

**Rationale:** This is the SQLAlchemy 2.0 recommended pattern for async queries. `first()` returns `None` if not found, making the 404 check straightforward.

**Alternative considered:** Using `session.get(Job, job_id)` which is simpler but less flexible if we need to add joins or filters later.

### Decision 4: Response serialization
**Decision:** Convert SQLAlchemy model to dict manually, mapping specific fields to match existing API contract.

**Rationale:** Explicit field mapping ensures the API contract remains stable even if the database model changes. The existing endpoint already uses `ApiResponse.create()` wrapper.

**Alternative considered:** Using Pydantic models for serialization. The project has API models in `src/api/models.py` but they don't have a direct Job response model yet. Creating one is out of scope for this fix.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Database connection failures | FastAPI dependency handles session lifecycle; exceptions bubble up as 500 errors |
| UUID format validation | FastAPI's `str` type hint validates it's a string; database will reject invalid UUIDs |
| N+1 query if we add relationships later | Current implementation uses simple SELECT; document this if extending |

## Migration Plan

No migration needed. This is a pure code change with no database schema modifications.

## Open Questions

- None. The implementation approach is straightforward.
