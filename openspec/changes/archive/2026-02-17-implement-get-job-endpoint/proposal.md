## Why

The `GET /api/v1/jobs/{job_id}` endpoint currently returns hardcoded mock data and cannot retrieve actual job information from the database. This makes it impossible for API consumers to track job status or view job details after creation. Implementing this endpoint enables proper job lifecycle management.

## What Changes

- Implement actual database query in `GET /api/v1/jobs/{job_id}` endpoint
- Add 404 Not Found response when job ID doesn't exist
- Return full job details including source info, file metadata, and status
- Add proper FastAPI dependency injection for async database session

## Capabilities

### New Capabilities
- `get-job-endpoint`: Retrieve a single job by ID with full details from database

### Modified Capabilities
- None (no existing spec-level behavior changes)

## Impact

- **src/main.py**: Modify `get_job()` function to query database
- **src/api/dependencies.py**: May need to add database session dependency
- **API Response**: Changes from mock data to real database response (same structure)
- **Dependencies**: Requires PostgreSQL connection (already configured)
