# Proposal: Fix Engine Job Cache Race Condition

## Problem Statement

In production, jobs fail with error:
```
"Job not found: afe49def-bd30-4068-9d58-54d385c4a90f"
```

This happens when:
1. API creates job in PostgreSQL via `JobRepository.create()`
2. API enqueues job ID to Redis
3. Worker polls PostgreSQL (not Redis) via `poll_pending_job()`
4. Worker locks job and calls `engine.process_job(job_id)`
5. Engine looks in in-memory cache `_active_jobs`
6. Job is **not found** because it was never added to the cache

## Root Cause

The `OrchestrationEngine` maintains an in-memory dictionary `_active_jobs` that is only populated when jobs are created via `engine.create_job()`. However, the API bypasses this method and creates jobs directly via the repository.

### Code Flow Analysis

```
API Layer (src/main.py:503-527)
├── JobRepository.create() → DB commit
└── queue.enqueue() → Redis

Worker Layer (src/worker/main.py:204-235)
├── poll_pending_job() → DB query with lock
└── Calls engine.process_job(job_id)

Engine Layer (src/core/engine.py:146-148)
├── Looks in _active_jobs cache
└── ❌ Not found → ValueError
```

## Proposed Solution

Modify `engine.process_job()` to:
1. First check the `_active_jobs` cache (fast path)
2. If not found, load from PostgreSQL via `JobRepository`
3. Convert `JobModel` → `Job` and add to cache
4. Continue with normal processing

This makes the engine resilient to jobs created outside its cache.

## Benefits

- **Minimal changes**: Only modify the engine, no API or worker changes needed
- **Backward compatible**: Existing cached jobs continue to work
- **Resilient**: Handles jobs created via any path (API, CLI, migrations)

## Acceptance Criteria

- [ ] Jobs created via API can be processed by workers
- [ ] Jobs created via `engine.create_job()` continue to work
- [ ] All existing tests pass
- [ ] New test covers the DB loading path
