## Why

The `GET /api/v1/jobs` endpoint currently returns an empty hardcoded array `[]` and cannot retrieve actual job information from the database. This prevents API consumers from listing and monitoring their ingestion jobs. Implementing this endpoint enables users to view all jobs with pagination and filtering capabilities.

## What Changes

- Implement actual database query in `GET /api/v1/jobs` endpoint
- Support pagination with `page` and `limit` query parameters
- Support filtering by `status` query parameter
- Return full job details in a list response
- Include pagination metadata (total count, current page, etc.)
- Use proper FastAPI dependency injection for async database session

## Capabilities

### New Capabilities
- `list-jobs-endpoint`: List all jobs with pagination and status filtering from database

### Modified Capabilities
- None (no existing spec-level behavior changes)

## Impact

- **src/main.py**: Modify `list_jobs()` function to query database
- **src/api/models.py**: May need to add list response model with pagination metadata
- **API Response**: Changes from empty array to real database results
- **Dependencies**: Requires PostgreSQL connection (already configured)
