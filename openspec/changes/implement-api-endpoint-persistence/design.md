# Design: Implement API Endpoint Persistence

## Architecture

### Overview
Connect the FastAPI route handlers in `src/main.py` to the existing data persistence layers:
1. **OrchestrationEngine** - For job lifecycle management
2. **DetectionResultRepository** - For content detection results
3. **Database Session** - Via FastAPI dependency injection

### Component Interaction
```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client    │────▶│  FastAPI Routes  │────▶│   Validation    │
└─────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Orchestration  │
                    │     Engine       │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Repository/DB   │
                    │     Layer        │
                    └──────────────────┘
```

## Approach

### 1. Route Refactoring
Update route handlers in `src/main.py` to:
- Accept proper Pydantic request models
- Use `Depends(get_db)` for database sessions
- Call `OrchestrationEngine` methods for job operations
- Return actual database records instead of mocks

### 2. Job Creation Flow
```
POST /api/v1/jobs
  ├── Parse JobCreateRequest (Pydantic model)
  ├── Validate required fields
  ├── Call engine.create_job(request)
  ├── Persist to database
  └── Return JobResponse with real job ID
```

### 3. File Upload Flow
```
POST /api/v1/upload
  ├── Receive multipart form data
  ├── Validate file (size, type)
  ├── Save to /tmp/pipeline/{uuid}/
  ├── Run content detection (optional)
  ├── Create job via engine
  └── Return UploadResponse with job ID(s)
```

### 4. Job Listing Flow
```
GET /api/v1/jobs
  ├── Parse query params (page, limit, filters)
  ├── Build SQL query with filters
  ├── Execute via repository
  └── Return paginated JobListResponse
```

## Decisions

### Decision 1: Use OrchestrationEngine as Primary Interface
**Rationale**: The OrchestrationEngine already has `create_job()`, `update_job_status()`, etc. methods. Using it maintains separation of concerns and ensures business logic is centralized.

### Decision 2: Keep File Storage in /tmp/pipeline/
**Rationale**: The existing configuration uses `/tmp/pipeline/` for staging. We'll maintain this but ensure proper cleanup.

### Decision 3: Async Database Operations
**Rationale**: All DB operations must remain async to avoid blocking the event loop. Use `await session.commit()` etc.

### Decision 4: Minimal Response Changes
**Rationale**: Keep existing API response structure to maintain compatibility. Only change the data source (mock → real DB).

### Decision 5: Job Model Extension
**Rationale**: The current `JobModel` only has `id`. We need to extend it to match the `Job` API model with fields like:
- `source_type`, `source_uri`
- `file_name`, `file_size`, `mime_type`
- `status` (enum)
- `created_at`, `updated_at`
- `priority`, `mode`

## Implementation Notes

### Required Model Updates
```python
# src/db/models.py - JobModel needs:
- status: JobStatus enum
- source_type: str
- source_uri: str  
- file_name: str
- file_size: int
- mime_type: str
- priority: str
- mode: str
- created_at: datetime
- updated_at: datetime
```

### Required Route Updates
1. `POST /api/v1/jobs` - Create job via engine
2. `POST /api/v1/upload` - Save files + create jobs
3. `GET /api/v1/jobs` - Query with pagination
4. `GET /api/v1/jobs/{job_id}` - Get by ID
5. `DELETE /api/v1/jobs/{job_id}` - Cancel job
6. `POST /api/v1/jobs/{job_id}/retry` - Retry job

### Error Handling
- Use existing HTTPException patterns
- Return consistent error response format
- Log errors with structlog
