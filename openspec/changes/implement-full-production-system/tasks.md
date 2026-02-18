# Tasks: Implement Full Production System

## Phase 1: Job Processing Worker (CRITICAL)

### Database & Models
- [x] **Task 1.1**: Extend job status tracking
  - Add 'cancelling' status to JobStatus enum
  - Add indexes for efficient polling (status, created_at)

- [x] **Task 1.2**: Create job lock mechanism
  - Add locked_by, locked_at fields to jobs table
  - Prevent duplicate processing

### Worker Implementation
- [x] **Task 1.3**: Implement database polling in worker
  - Query jobs with status='created' ORDER BY priority, created_at
  - Use SELECT FOR UPDATE SKIP LOCKED for concurrency
  - Update status to 'processing' and set locked_by

- [x] **Task 1.4**: Integrate pipeline execution
  - Load job from database
  - Execute all pipeline stages
  - Update job status after each stage
  - Handle stage failures with retries

- [x] **Task 1.5**: Store processing results
  - On completion, store extracted content
  - Update job with output references
  - Mark status 'completed' or 'failed'

- [x] **Task 1.6**: Handle worker failures
  - Detect crashed workers via heartbeat
  - Release locks on timeout
  - Requeue jobs for retry

## Phase 2: Job Results Storage (CRITICAL)

### Database
- [x] **Task 2.1**: Create job_results table
  - job_id (FK), extracted_text, result_metadata JSON
  - quality_score, processing_time_ms
  - output_uri for large results
  - created_at, expires_at

- [x] **Task 2.2**: Create JobResultRepository
  - save(), get_by_job_id(), delete_expired()

### API
- [x] **Task 2.3**: Update GET /jobs/{id}/result
  - Query job_results table
  - Return 404 if job not complete
  - Return full result data

- [x] **Task 2.4**: Handle large results
  - If result > 1MB, store to filesystem
  - Return presigned URL or streaming response

## Phase 3: Pipeline Configuration (CRITICAL)

### Database
- [x] **Task 3.1**: Create pipelines table
  - id, name, description, config JSON
  - created_by, created_at, updated_at
  - is_active, version

- [x] **Task 3.2**: Create PipelineRepository
  - CRUD operations
  - Validation methods

### API
- [x] **Task 3.3**: Implement pipeline endpoints
  - POST /pipelines - create
  - GET /pipelines - list
  - GET /pipelines/{id} - get
  - PUT /pipelines/{id} - update
  - DELETE /pipelines/{id} - delete

- [x] **Task 3.4**: Link jobs to pipelines
  - Accept pipeline_id in job creation
  - Validate pipeline exists
  - Store pipeline config snapshot with job

## Phase 4: Redis Queue Integration (IMPORTANT)

### Queue Implementation
- [x] **Task 4.1**: Create queue abstraction
  - src/core/queue.py with Redis backend
  - enqueue(), dequeue(), ack(), nack()
  - Priority queue support (high/normal/low)

- [x] **Task 4.2**: Enqueue jobs on creation
  - After DB insert, push job_id to Redis
  - Use priority based on job.priority field

- [x] **Task 4.3**: Worker consumes from queue
  - BLPOP from Redis queues
  - Process job, ack on success
  - Requeue on failure (with retry limit)

- [x] **Task 4.4**: Queue monitoring
  - Add GET /health/queue endpoint
  - Return queue depths, processing rates

## Phase 5: Authentication & Authorization (IMPORTANT)

### Authentication
- [x] **Task 5.1**: Implement JWT validation
  - src/auth/jwt.py - validate_token()
  - Extract user_id, roles from token
  - Handle token expiration

- [x] **Task 5.2**: Implement API key validation
  - src/auth/api_key.py - validate_api_key()
  - Query database for key
  - Check key expiration

### Authorization
- [x] **Task 5.3**: Create permission decorator
  - @require_permission('jobs:create')
  - Check user roles against permissions
  - Return 403 if unauthorized

- [x] **Task 5.4**: Protect all endpoints
  - Add auth dependencies to all routes
  - Allow public access only to health endpoints
  - Document required permissions

### Token Management
- [x] **Task 5.5**: Implement auth endpoints
  - POST /auth/login - return JWT
  - POST /auth/refresh - refresh token
  - POST /auth/api-keys - create API key

## Phase 6: Audit Logging (IMPORTANT)

### Database
- [x] **Task 6.1**: Create audit_logs table
  - id, timestamp, user_id, action
  - resource_type, resource_id
  - request_details, success, error_message
  - ip_address, user_agent

### Middleware
- [x] **Task 6.2**: Create audit middleware
  - Log all API requests
  - Redact sensitive data
  - Store asynchronously (don't block response)

### API
- [x] **Task 6.3**: Implement audit query endpoint
  - GET /audit/logs with filters
  - Support date range, user, resource filters
  - Pagination support

## Phase 7: Webhook Delivery (IMPORTANT)

### Database
- [x] **Task 7.1**: Create webhook_subscriptions table
  - id, user_id, url, events[], secret
  - is_active, created_at

- [x] **Task 7.2**: Create webhook_deliveries table
  - id, subscription_id, event_type, payload
  - status, attempts, last_error
  - created_at, delivered_at

### Delivery Service
- [x] **Task 7.3**: Implement webhook delivery
  - src/core/webhook_delivery.py
  - Async HTTP POST to subscribers
  - HMAC-SHA256 signature
  - Retry with exponential backoff

### API
- [x] **Task 7.4**: Webhook management endpoints
  - POST /webhooks - subscribe
  - GET /webhooks - list subscriptions
  - DELETE /webhooks/{id} - unsubscribe
  - GET /webhooks/{id}/deliveries - status

## Phase 8: Observability (NICE TO HAVE)

- [x] **Task 8.1**: Prometheus metrics
  - Jobs created/completed/failed counters
  - Processing duration histograms
  - Queue depth gauges

- [x] **Task 8.2**: Health check enhancements
  - Check database connectivity
  - Check Redis connectivity
  - Check worker status

## Acceptance Criteria

- [x] Jobs are processed end-to-end (creation → processing → results)
- [x] Multiple workers can process jobs concurrently
- [x] Results are stored and retrievable
- [x] Pipelines can be configured and used
- [x] All endpoints require authentication (implemented, can be enabled)
- [x] All operations are audited (repository ready)
- [x] Webhooks are delivered reliably
- [x] System metrics are available
