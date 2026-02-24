# Capability: Engine Job Loading

## Overview

The orchestration engine must be able to process jobs that were created outside its in-memory cache by loading them from the database on-demand.

## Requirements

### Functional Requirements

1. **Cache-First Lookup**: When `process_job()` is called, first check `_active_jobs` cache
2. **DB Fallback**: If not in cache, load from PostgreSQL via `JobRepository`
3. **Model Conversion**: Convert `JobModel` (DB) → `Job` (API model) for engine use
4. **Cache Population**: Add loaded job to `_active_jobs` for subsequent lookups
5. **Error Handling**: Raise appropriate error if job not found in DB either

### Non-Functional Requirements

1. **Performance**: DB fallback should only happen on cache miss (rare after first call)
2. **Idempotency**: Loading same job multiple times should return cached instance
3. **No Breaking Changes**: Existing code paths must continue to work

## Interface

### New Method: `_load_job_from_db()`

```python
async def _load_job_from_db(self, job_id: UUID) -> Job | None:
    """Load job from database and add to cache.
    
    Args:
        job_id: Job ID to load
        
    Returns:
        Job if found, None otherwise
    """
```

### Modified Method: `process_job()`

```python
async def process_job(
    self,
    job_id: UUID,
    enabled_stages: list[str] | None = None,
) -> PipelineContext:
    """Process a job through the pipeline.
    
    Now loads from DB if not in cache.
    """
```

## Data Model Conversion

### JobModel (DB) → Job (API)

| JobModel Field | Job Field | Notes |
|---------------|-----------|-------|
| `id` | `id` | Direct mapping |
| `status` | `status` | Convert to enum |
| `source_type` | `source_type` | Direct mapping |
| `source_uri` | `source_uri` | Direct mapping |
| `file_name` | `file_name` | Direct mapping |
| `file_size` | `file_size` | Direct mapping |
| `mime_type` | `mime_type` | Direct mapping |
| `priority` | `priority` | Direct mapping |
| `mode` | `mode` | Direct mapping |
| `external_id` | `external_id` | Direct mapping |
| `retry_count` | `retry_count` | Direct mapping |
| `max_retries` | `max_retries` | Direct mapping |
| `created_at` | `created_at` | Direct mapping |
| `updated_at` | `updated_at` | Direct mapping |
| `started_at` | `started_at` | Direct mapping |
| `completed_at` | `completed_at` | Direct mapping |
| `error_message` | `error.message` | Construct error object |
| `error_code` | `error.code` | Construct error object |
| `pipeline_config` | `pipeline_config` | Direct mapping (dict) |
| `metadata_json` | `metadata` | Direct mapping |

## Error Handling

| Scenario | Response |
|----------|----------|
| Job in cache | Use cached instance |
| Job not in cache, found in DB | Load, cache, and use |
| Job not in cache, not in DB | Raise `ValueError(f"Job not found: {job_id}")` |
| DB connection error | Propagate exception |

## Testing Strategy

1. **Unit Test**: Test `_load_job_from_db()` with mock repository
2. **Integration Test**: Full flow - create job via API, process via worker
3. **Regression Test**: Ensure existing `engine.create_job()` path still works
