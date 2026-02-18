# Design: Webhook Retry Mechanism

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Webhook Delivery Service                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           RetryConfiguration (Pydantic Model)             │   │
│  │  - max_retries, base_delay, max_delay, strategy          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         ExponentialBackoffEngine                          │   │
│  │  - calculate_delay(attempt) → timedelta                  │   │
│  │  - is_retryable(error) → bool                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         WebhookRetryHandler                               │   │
│  │  - deliver_with_retry(webhook, payload)                  │   │
│  │  - schedule_retry(job_id, delay)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Redis (Delayed Queue)                        │   │
│  │         zadd retry_queue score=timestamp                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Webhook Trigger
   │
   ▼
2. Delivery Attempt ──► HTTP POST to endpoint
   │                           │
   │                   ┌───────┴───────┐
   │                   ▼               ▼
   │              Success (2xx)    Failure (4xx/5xx)
   │                   │               │
   │                   ▼               ▼
   │            Mark Complete    is_retryable()?
   │                                   │
   │                           ┌───────┴───────┐
   │                           ▼               ▼
   │                      No (4xx)         Yes (5xx/timeout)
   │                           │               │
   │                           ▼               ▼
   │                   Mark Failed      Calculate delay
   │                   Route to DLQ     Schedule retry
   │                                      (Redis ZADD)
   │
3. Retry Worker (polling Redis ZRANGEBYSCORE)
   │
   └──► Dequeue due retries ──► Repeat from step 2
```

## Approach

### Phase 1: Configuration Layer
1. Add `RetryConfig` Pydantic model with validation
2. Extend WebhookEndpoint model with optional retry_config field
3. Create API endpoints for CRUD operations on retry config
4. Add migration for new database columns

### Phase 2: Backoff Engine
1. Implement `ExponentialBackoffEngine` class
2. Add jitter using `random.uniform(0.5, 1.5)`
3. Create error classifier for retryable vs non-retryable
4. Unit test with various edge cases (max delay cap, etc.)

### Phase 3: Retry Handler Integration
1. Modify `WebhookDeliveryService` to use retry logic
2. Add job state: `retrying` with attempt counter
3. Implement Redis-based delayed queue for retries
4. Create `RetryWorker` async processor

### Phase 4: Observability
1. Add Prometheus counters and histograms
2. Emit structured events for each retry attempt
3. Extend OpenTelemetry spans with retry attributes
4. Create Grafana dashboard JSON

## Decisions

### DEC-1: Redis Sorted Set for Delay Queue
**Choice:** Use Redis ZADD with timestamp as score  
**Rationale:** O(log N) insertion, O(log N + M) retrieval of due items. Native support in existing Redis infrastructure.  
**Alternative Considered:** RabbitMQ delayed messages - adds operational complexity  
**Trade-off:** Simplicity vs feature richness - Redis is sufficient

### DEC-2: Separate RetryWorker Process
**Choice:** Dedicated async worker for retry processing  
**Rationale:** Decouples retry timing from main API request handling. Prevents blocking during backoff periods.  
**Alternative Considered:** APScheduler in main process - couples concerns  
**Trade-off:** Additional process to monitor vs cleaner architecture

### DEC-3: Per-Endpoint Configuration
**Choice:** Allow retry config per webhook endpoint  
**Rationale:** Different endpoints have different reliability characteristics. Payment webhooks need more retries than analytics.  
**Alternative Considered:** Global only - too rigid  
**Trade-off:** Complexity vs flexibility - worth it for enterprise use case

### DEC-4: Circuit Breaker NOT Included
**Choice:** Implement retry without circuit breaker pattern  
**Rationale:** Circuit breaker adds significant complexity. Retry alone addresses 80% of the problem.  
**Alternative Considered:** Full circuit breaker with half-open state  
**Trade-off:** Future work - can add circuit breaker as separate change if needed

### DEC-5: Idempotency Key Propagation
**Choice:** Preserve idempotency-key header across retries  
**Rationale:** Receivers need idempotency for exactly-once processing. Same payload + same key = safe to retry.  
**Trade-off:** Assumes receivers implement idempotency - documented requirement

## Files to Modify

| File | Change |
|------|--------|
| `src/core/webhooks.py` | Add retry logic to delivery service |
| `src/core/retry_engine.py` | New: Exponential backoff engine |
| `src/api/models.py` | Add RetryConfig Pydantic model |
| `src/db/models.py` | Add retry columns to WebhookEndpoint |
| `src/worker/retry_processor.py` | New: Retry worker implementation |
| `src/observability/metrics.py` | Add retry metrics |
| `migrations/xxx_add_webhook_retry.py` | Database migration |
