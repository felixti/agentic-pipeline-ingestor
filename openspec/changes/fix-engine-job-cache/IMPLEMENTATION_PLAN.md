# Implementation Plan: Fix Engine Job Cache Race Condition

## OpenSpec Context

- **Change**: fix-engine-job-cache
- **Proposal**: proposal.md
- **Design**: design.md
- **Specs**: specs/engine/spec.md
- **Tasks**: tasks.md

## Problem Summary

The `OrchestrationEngine` fails to process jobs created via the API because it only looks in its in-memory cache (`_active_jobs`), which is only populated when jobs are created via `engine.create_job()`. The API creates jobs directly via `JobRepository`, bypassing the engine.

## Solution

Modify `engine.process_job()` to implement a cache-first lookup with DB fallback:
1. Check `_active_jobs` cache first
2. If not found, load from PostgreSQL via `JobRepository`
3. Convert `JobModel` → `Job` and cache it
4. Continue with normal processing

## Task List

| # | Task | Owner | Dependencies | Acceptance Criteria |
|---|------|-------|--------------|---------------------|
| 1 | Add `_convert_job_model_to_job()` helper | backend-developer | none | Converts JobModel to Job API model |
| 2 | Add `_load_job_from_db()` method | backend-developer | Task 1 | Loads job from DB, adds to cache |
| 3 | Modify `process_job()` for cache+DB fallback | backend-developer | Task 2 | Tries cache first, falls back to DB |
| 4 | Run existing tests | tester-agent | Task 3 | No regressions |
| 5 | Validate fix | qa-agent | Task 4 | QA approval |

## Architecture Notes

### File to Modify
- `src/core/engine.py`

### Key Changes

1. **New imports needed**:
   ```python
   from src.db.repositories.job import JobRepository
   from src.db.models import JobModel, get_async_engine
   from sqlalchemy.ext.asyncio import AsyncSession
   ```

2. **New helper method**:
   ```python
   async def _load_job_from_db(self, job_id: UUID) -> Job | None:
       """Load job from database if not in cache."""
       from sqlalchemy.ext.asyncio import AsyncSession
       from src.db.models import get_async_engine
       
       engine = get_async_engine()
       async with AsyncSession(engine) as session:
           repo = JobRepository(session)
           job_model = await repo.get_by_id(job_id)
           
           if job_model:
               job = self._convert_job_model_to_job(job_model)
               self._active_jobs[job_id] = job
               return job
           return None
   ```

3. **Modified `process_job()`**:
   ```python
   async def process_job(self, job_id: UUID, enabled_stages=None):
       # Try cache first
       job = self._active_jobs.get(job_id)
       
       # Fall back to DB if not in cache
       if not job:
           job = await self._load_job_from_db(job_id)
       
       if not job:
           raise ValueError(f"Job not found: {job_id}")
       
       # Continue with existing logic...
   ```

## Validation Criteria

1. Jobs created via API can be processed successfully
2. Jobs created via `engine.create_job()` still work
3. All existing tests pass
4. No performance degradation

## Iteration Log

| Iteration | Date | Agent | Task | Result |
|-----------|------|-------|------|--------|
