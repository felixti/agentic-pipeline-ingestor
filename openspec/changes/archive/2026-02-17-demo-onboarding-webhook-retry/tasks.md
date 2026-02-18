# Tasks: Webhook Retry Mechanism

## Implementation Tasks

### Phase 1: Configuration Layer
- [ ] **TASK-1**: Create `RetryConfig` Pydantic model with validation
  - Location: `src/api/models.py`
  - Validation: max_retries 0-10, base_delay >= 1s, max_delay >= base_delay
  - Acceptance: Unit tests pass for all validation rules

- [ ] **TASK-2**: Add retry_config column to WebhookEndpoint model
  - Location: `src/db/models.py`
  - Type: JSONB nullable column
  - Acceptance: Migration runs successfully, column exists

- [ ] **TASK-3**: Create database migration
  - Location: `migrations/versions/xxx_add_webhook_retry.py`
  - Using: Alembic autogenerate + manual review
  - Acceptance: `alembic upgrade head` and `alembic downgrade` work

- [ ] **TASK-4**: Add retry config API endpoints
  - Location: `src/api/routes/webhooks.py`
  - Endpoints: GET/PUT /webhooks/{id}/retry-config
  - Acceptance: API contract tests pass

### Phase 2: Backoff Engine
- [ ] **TASK-5**: Implement `ExponentialBackoffEngine` class
  - Location: `src/core/retry_engine.py` (new file)
  - Methods: `calculate_delay()`, `is_retryable()`, `apply_jitter()`
  - Acceptance: Unit tests with edge cases (max cap, jitter range)

- [ ] **TASK-6**: Create error classifier
  - Location: `src/core/retry_engine.py`
  - Logic: 5xx, 408, 429 = retryable; 4xx (except 408/429) = non-retryable
  - Acceptance: Test matrix of all status codes

- [ ] **TASK-7**: Add retry engine unit tests
  - Location: `tests/unit/test_retry_engine.py`
  - Coverage: 100% of engine logic
  - Acceptance: `pytest tests/unit/test_retry_engine.py -v` passes

### Phase 3: Webhook Integration
- [ ] **TASK-8**: Modify WebhookDeliveryService for retry
  - Location: `src/core/webhooks.py`
  - Changes: Add attempt counter, integrate backoff engine
  - Acceptance: Existing webhook tests still pass

- [ ] **TASK-9**: Implement Redis delayed queue
  - Location: `src/core/retry_queue.py` (new file)
  - Methods: `schedule_retry()`, `get_due_retries()`, `complete_retry()`
  - Acceptance: Integration test with Redis

- [ ] **TASK-10**: Create RetryWorker processor
  - Location: `src/worker/retry_processor.py` (new file)
  - Logic: Poll Redis, process due retries, update job state
  - Acceptance: Worker runs continuously, processes retries correctly

- [ ] **TASK-11**: Add job state management for retrying
  - Location: `src/db/models.py`, `src/core/pipeline.py`
  - States: `retrying` (new), track attempt_number
  - Acceptance: State transitions work correctly

### Phase 4: Observability
- [ ] **TASK-12**: Add Prometheus metrics
  - Location: `src/observability/metrics.py`
  - Metrics: `webhook_delivery_total`, `webhook_retry_total`, histograms
  - Acceptance: Metrics visible at `/metrics` endpoint

- [ ] **TASK-13**: Emit retry attempt events
  - Location: `src/core/webhooks.py`
  - Format: Structured JSON with job_id, attempt, delay, error
  - Acceptance: Events logged to stdout (JSON)

- [ ] **TASK-14**: Add OpenTelemetry spans for retries
  - Location: `src/observability/tracing.py`
  - Attributes: `retry.attempt_number`, `retry.delay_ms`
  - Acceptance: Spans visible in Jaeger

- [ ] **TASK-15**: Create Grafana dashboard
  - Location: `config/grafana/dashboards/webhook-retries.json`
  - Panels: Success rate, retry distribution, top failing endpoints
  - Acceptance: Dashboard imports successfully

### Phase 5: Integration & Testing
- [ ] **TASK-16**: Write integration tests
  - Location: `tests/integration/test_webhook_retry.py`
  - Scenarios: Full retry flow, DLQ routing, manual retry
  - Acceptance: `pytest tests/integration/test_webhook_retry.py -v` passes

- [ ] **TASK-17**: Update API documentation
  - Location: `api/openapi.yaml`
  - Add: Retry config schema, new endpoints, error codes
  - Acceptance: Documentation renders correctly in Swagger UI

- [ ] **TASK-18**: Add end-to-end test
  - Location: `tests/e2e/test_webhook_retry_flow.py`
  - Flow: Create webhook → trigger failure → verify retries → verify metrics
  - Acceptance: E2E test passes in CI

### Phase 6: Deployment
- [ ] **TASK-19**: Update docker-compose
  - Location: `docker/docker-compose.yml`
  - Add: Retry worker service
  - Acceptance: `docker-compose up` starts all services

- [ ] **TASK-20**: Add environment configuration
  - Location: `.env.example`
  - Add: `RETRY_WORKER_POLL_INTERVAL`, `RETRY_QUEUE_MAX_SIZE`
  - Acceptance: Configuration documented

---

## Verification Checklist

Before archiving this change:
- [ ] All unit tests pass (>90% coverage on new code)
- [ ] All integration tests pass
- [ ] API contract tests pass
- [ ] Manual testing: Verified retry flow with real HTTP endpoint
- [ ] Metrics verified in Prometheus
- [ ] Traces visible in Jaeger
- [ ] Documentation updated
