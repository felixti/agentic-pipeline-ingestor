## ADDED Requirements

### Requirement: Retrieve job by ID
The system SHALL retrieve a job from the database when given a valid job ID via `GET /api/v1/jobs/{job_id}`.

#### Scenario: Job exists
- **WHEN** a GET request is made to `/api/v1/jobs/{job_id}` with a valid UUID that exists in the database
- **THEN** the system SHALL return HTTP 200 with the job details including id, status, source_type, source_uri, file_name, file_size, mime_type, mode, priority, created_at, and updated_at

#### Scenario: Job not found
- **WHEN** a GET request is made to `/api/v1/jobs/{job_id}` with a UUID that does not exist in the database
- **THEN** the system SHALL return HTTP 404 with an error message indicating the job was not found

### Requirement: Database session management
The system SHALL use FastAPI dependency injection to provide an async database session for querying job data.

#### Scenario: Database query execution
- **WHEN** the get job endpoint is invoked
- **THEN** the system SHALL use SQLAlchemy async session to execute a SELECT query for the job by ID
- **AND** the session SHALL be properly managed (opened and closed) via FastAPI dependency
