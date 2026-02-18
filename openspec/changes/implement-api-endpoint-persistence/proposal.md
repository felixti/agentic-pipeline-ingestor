# Proposal: Implement API Endpoint Persistence

## Why
The current API endpoints (`POST /api/v1/jobs`, `POST /api/v1/upload`, etc.) are stub implementations that return mock responses with generated job IDs but do not persist any data to the database. This was confirmed through testing where:
- Multiple API calls returned HTTP 202 with job IDs
- The database remained empty (0 rows in jobs table)
- The repository layer is working correctly (verified via direct test)

This gap prevents the system from functioning as a proper document processing pipeline since jobs cannot be tracked, retrieved, or processed.

## What Changes
Connect the API endpoint handlers in `src/main.py` to the existing repository and orchestration layers to enable actual data persistence:

1. Update `POST /api/v1/jobs` to create jobs via `OrchestrationEngine.create_job()`
2. Update `POST /api/v1/upload` to save uploaded files and create job records
3. Update `GET /api/v1/jobs` to query the database via repositories
4. Update `GET /api/v1/jobs/{job_id}` to retrieve specific jobs
5. Update `DELETE /api/v1/jobs/{job_id}` to cancel/delete jobs
6. Update `POST /api/v1/jobs/{job_id}/retry` to retry failed jobs

## Capabilities
- [ ] Job Creation Persistence
- [ ] File Upload Processing
- [ ] Job Listing and Retrieval
- [ ] Job Cancellation
- [ ] Job Retry

## Impact
- **High**: Core functionality - without this, the API is non-functional for actual use
- **Medium**: Affects all job-related endpoints
- **Low**: No breaking changes to API contract (responses remain compatible)
