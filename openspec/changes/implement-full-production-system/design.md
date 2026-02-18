# Design: Implement Full Production System

## Architecture

### System Overview
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client     │────▶│   API        │────▶│   Database   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                     │
                            ▼                     ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   Redis      │◄────│   Worker     │
                     │   Queue      │     │   (xN)       │
                     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │  Destinations│
                                           │  (Cognee,S3) │
                                           └──────────────┘
```

## Implementation Approach

### Phase 1: Job Processing (CRITICAL)
**Files to Modify:**
- `src/worker/main.py` - Add database polling
- `src/worker/processor.py` - Integrate with database
- `src/core/engine.py` - Add DB persistence

**Key Changes:**
1. Worker queries DB for jobs with status='created'
2. Uses SELECT FOR UPDATE to claim jobs
3. Updates status throughout processing
4. Stores results on completion

### Phase 2: Job Results (CRITICAL)
**Files to Create/Modify:**
- `src/db/models.py` - Add JobResultModel
- `src/db/repositories/job_result.py` - Result repository
- `src/main.py` - Update result endpoint

### Phase 3: Pipeline Configuration (CRITICAL)
**Files:**
- `src/db/models.py` - Add PipelineModel
- `src/db/repositories/pipeline.py` - Pipeline repository
- `src/main.py` - Implement pipeline endpoints

### Phase 4: Redis Queue (IMPORTANT)
**Files:**
- `src/core/queue.py` - Queue abstraction
- `src/main.py` - Enqueue on job creation
- `src/worker/main.py` - Consume from queue

### Phase 5: Authentication (IMPORTANT)
**Files:**
- `src/auth/dependencies.py` - Auth checks
- `src/main.py` - Add auth to all routes
- `src/api/routes/auth.py` - Token endpoints

### Phase 6: Audit Logging (IMPORTANT)
**Files:**
- `src/audit/middleware.py` - Audit middleware
- `src/db/models.py` - AuditLogModel
- `src/main.py` - Integrate middleware

### Phase 7: Webhooks (IMPORTANT)
**Files:**
- `src/db/models.py` - WebhookSubscriptionModel
- `src/core/webhook_delivery.py` - Delivery service
- `src/main.py` - Webhook management endpoints

## Decisions

1. **Database Polling vs Queue**: Use both - DB polling as fallback, Redis for performance
2. **Result Storage**: Store small results in DB, large results in S3/filesystem
3. **Auth Strategy**: Support both JWT (users) and API keys (services)
4. **Webhook Delivery**: Async background task with retry logic

## Implementation Order
1. Job Processing (makes system functional)
2. Job Results (completes user flow)
3. Pipeline Config (enables customization)
4. Redis Queue (scalability)
5. Authentication (security)
6. Audit Logging (compliance)
7. Webhooks (integrations)
