## 1. Database Session Setup

- [x] 1.1 Add database session import to `src/main.py` (import `get_db_session` or create inline session)
- [x] 1.2 Verify SQLAlchemy async session is properly configured and accessible

## 2. Endpoint Implementation

- [x] 2.1 Modify `get_job()` function signature to accept database session via dependency injection
- [x] 2.2 Implement database query using `select(Job).where(Job.id == job_id)` pattern
- [x] 2.3 Add 404 Not Found handling when job doesn't exist in database
- [x] 2.4 Map Job model fields to response dictionary (id, status, source_type, source_uri, file_name, file_size, mime_type, mode, priority, created_at, updated_at)
- [x] 2.5 Update return statement to use real database data instead of hardcoded mock

## 3. Verification

- [x] 3.1 Run linters (`ruff check src/main.py`)
- [x] 3.2 Run type checker (`mypy src/main.py`)
- [x] 3.3 Verify endpoint is accessible and returns proper structure
- [x] 3.4 Test 404 response for non-existent job ID

## 4. Documentation

- [x] 4.1 Remove the `TODO: Implement actual job retrieval` comment
- [x] 4.2 Update docstring if needed to reflect actual behavior
