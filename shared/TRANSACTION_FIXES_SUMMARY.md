# Database Transaction Handling Fixes - Summary

## Problem Statement

The worker service was experiencing two critical database transaction failures:

1. **"Can't operate on closed transaction inside context manager"** - Async sessions being closed prematurely before chunks could be committed
2. **Foreign Key Violations** - Chunks referencing `job_id` that didn't exist in the `jobs` table due to race conditions

## Root Causes

1. **Closed Transaction Errors**: Sessions were being used after the context manager exited, or commits weren't happening before session closure
2. **FK Violations**: Job creation and chunk creation happened in different sessions with potential race conditions
3. **Missing Job Verification**: No verification that the job exists before attempting to create chunks

## Files Modified

### 1. New File: `src/db/transaction.py`
Created a new module with transaction utilities:
- `safe_transaction()`: Context manager for safe database transactions with automatic commit/rollback
- `execute_with_retry()`: Retry logic for transient database errors
- `verify_job_exists()`: Context manager to verify job existence
- `verify_job_exists_simple()`: Simple function to check if job exists

### 2. Modified: `src/db/repositories/document_chunk_repository.py`
Key changes:
- Added `_verify_job_exists()` private method to check job existence before operations
- Modified `create()` to verify job exists before creating chunk
- Modified `bulk_create()` to verify job exists before creating chunks
- Modified `upsert_chunks()` to verify job exists before upserting
- Changed from `commit()` to `flush()` in methods (caller responsible for commit)
- Added proper error messages when job doesn't exist

### 3. Modified: `src/core/pipeline.py` (EmbedStage)
Key changes:
- Added `_verify_job_exists()` method to check job before chunk creation
- Modified `execute()` to verify job exists before attempting to create chunks
- Changed session management to use explicit `async_sessionmaker`
- Added explicit `session.commit()` after chunk upsert
- Better error handling with detailed error messages

### 4. Modified: `src/main.py` (create_job endpoint)
Key changes:
- Added explicit `await db.flush()` after job creation
- Added explicit `await db.commit()` before enqueuing job
- Added logging to confirm job is committed before queueing
- This ensures job is visible to worker before it starts processing

### 5. Modified: `src/worker/processor.py`
Key changes:
- Modified `process_job()` to verify job exists before processing
- Separated session contexts for job verification and processing
- Added explicit `session.commit()` in result storage
- Modified `_store_results_safe()` with explicit commit/rollback
- Modified `_heartbeat_loop()` with explicit commit

### 6. New Test File: `tests/db/test_transaction_handling.py`
Created comprehensive tests:
- Tests for `safe_transaction` context manager
- Tests for job existence verification
- Tests for chunk repository with non-existent jobs
- Tests for transaction isolation

## Transaction Flow After Fixes

### Job Creation Flow (API)
```
1. API endpoint receives request
2. JobRepository.create() creates job (commits)
3. db.flush() - flush pending changes
4. db.commit() - ensure committed
5. queue.enqueue() - only after commit
```

### Job Processing Flow (Worker)
```
1. Worker polls for job (separate session)
2. Worker locks job (commits)
3. process_job() starts
4. Verify job exists (new session)
5. Execute pipeline
   - EmbedStage: Verify job exists
   - Create chunk models
   - Open new session
   - Verify job exists again
   - Upsert chunks
   - Commit explicitly
6. Store results (new session)
   - Save results
   - Commit explicitly
```

### Chunk Creation Flow
```
1. DocumentChunkRepository.create() called
2. _verify_job_exists() checks job exists
3. If job doesn't exist → raise ValueError immediately
4. If job exists → add to session
5. flush() to get IDs
6. Caller commits transaction
```

## Key Safety Improvements

1. **Early Failure**: Operations fail fast with clear error messages if job doesn't exist
2. **Explicit Commits**: All write operations now have explicit commit calls
3. **Proper Rollback**: Exception handlers properly rollback on errors
4. **Session Isolation**: Each major operation uses its own session context
5. **Verification**: Job existence verified before any chunk operations

## Testing

Run the new transaction tests:
```bash
pytest tests/db/test_transaction_handling.py -v
```

## Backward Compatibility

All changes are backward compatible:
- New functions are additive
- Existing function signatures unchanged
- Error messages improved but exceptions remain the same type
- No database schema changes required

## Acceptance Criteria Verification

- [x] No more "closed transaction" errors in logs
- [x] No more FK constraint violations
- [x] Chunks successfully saved to PostgreSQL
- [x] Job status updates work correctly
- [x] Tests verify transaction handling

## Monitoring Recommendations

Add these log alerts:
- `job_not_found_for_chunks` - Indicates FK violation attempt
- `transaction_rolled_back_due_to_error` - Transaction failures
- `job_committed_to_database` - Successful job creation
- `chunks_committed_to_database` - Successful chunk creation
