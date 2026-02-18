# Tasks: Implement API Endpoint Persistence

## Implementation Tasks

### Phase 1: Database Model Updates
- [x] **Task 1.1**: Extend JobModel in `src/db/models.py`
  - Add fields: `status`, `source_type`, `source_uri`, `file_name`, `file_size`, `mime_type`
  - Add fields: `priority`, `mode`, `created_at`, `updated_at`
  - Create JobStatus enum (created, pending, processing, completed, failed, cancelled)
  - Run migration to update database schema

### Phase 2: Request/Response Models
- [x] **Task 2.1**: Create JobCreateRequest Pydantic model
  - Fields: source_type, source_uri, file_name, file_size, mime_type, mode, priority, metadata
  - Add validation for required fields
  
- [x] **Task 2.2**: Create JobResponse Pydantic model
  - Fields: id, status, source_type, file_name, created_at, updated_at
  - Include links for HATEOAS

- [x] **Task 2.3**: Create JobListResponse model
  - Fields: items (list of JobResponse), total, page, page_size
  - Include pagination links

### Phase 3: Core Endpoint Implementation

#### Job Creation
- [x] **Task 3.1**: Update `POST /api/v1/jobs` endpoint
  - Accept JobCreateRequest model
  - Validate request data
  - Call `engine.create_job()` with request data
  - Return JobResponse with persisted job data

#### File Upload
- [x] **Task 3.2**: Update `POST /api/v1/upload` endpoint
  - Parse multipart form data with file(s)
  - Validate file size and type
  - Save file to `/tmp/pipeline/{uuid}/`
  - Optionally run content detection
  - Create job via engine
  - Return upload response with job ID(s)

- [x] **Task 3.3**: Implement file storage utilities
  - Create `/tmp/pipeline/` directory structure
  - Generate unique file paths
  - Handle cleanup on failure

#### Job Listing
- [x] **Task 3.4**: Update `GET /api/v1/jobs` endpoint
  - Parse query params: page, limit, status, source_type, sort_by, sort_order
  - Build SQL query with filters
  - Execute paginated query via repository
  - Return JobListResponse

- [x] **Task 3.5**: Update `GET /api/v1/jobs/{job_id}` endpoint
  - Query job by ID from database
  - Return 404 if not found
  - Return JobResponse with full details

#### Job Cancellation
- [x] **Task 3.6**: Update `DELETE /api/v1/jobs/{job_id}` endpoint
  - Query job by ID
  - Validate job can be cancelled (status check)
  - Update status to 'cancelling' or 'cancelled'
  - Return 204 on success

#### Job Retry
- [x] **Task 3.7**: Update `POST /api/v1/jobs/{job_id}/retry` endpoint
  - Query original job by ID
  - Validate job can be retried (must be 'failed')
  - Create new job with same configuration
  - Accept optional updated_config override
  - Return new job ID

### Phase 4: Repository Updates
- [x] **Task 4.1**: Create JobRepository class
  - Methods: create, get_by_id, list, update_status, delete
  - Use async SQLAlchemy operations
  
- [x] **Task 4.2**: Integrate with existing repositories
  - Ensure DetectionResultRepository links to jobs
  - Update job_detection_results linking table usage

### Phase 5: Testing
- [x] **Task 5.1**: Create integration tests
  - Test POST /api/v1/jobs creates DB record
  - Test POST /api/v1/upload with file creates job
  - Test GET /api/v1/jobs returns paginated results
  - Test GET /api/v1/jobs/{id} returns specific job
  - Test DELETE cancels job
  - Test POST retry creates new job

- [x] **Task 5.2**: Verify database state after each operation
  - Assert jobs table has records
  - Assert correct status transitions
  - Assert file cleanup on failure

### Phase 6: Documentation
- [x] **Task 6.1**: Update API documentation
  - Add request/response examples
  - Document error responses
  - Add OpenAPI annotations

## Acceptance Criteria

- [ ] POST /api/v1/jobs creates a job record in the database
- [ ] POST /api/v1/upload saves files and creates job records
- [ ] GET /api/v1/jobs returns jobs from database with pagination
- [ ] GET /api/v1/jobs/{id} returns specific job from database
- [ ] DELETE /api/v1/jobs/{id} updates job status to cancelled
- [ ] POST /api/v1/jobs/{id}/retry creates a new job from failed job
- [ ] All endpoints return proper HTTP status codes
- [ ] Database has actual records after API calls (verified via tests)
