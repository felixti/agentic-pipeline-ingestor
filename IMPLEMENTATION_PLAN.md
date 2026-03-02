# Ralph Loop Implementation Plan: Fix Critical Pipeline Issues

## Executive Summary

The Agentic Data Pipeline is experiencing **critical failures** preventing file processing:
- **Parser**: Not extracting text from files (NO_TEXT_EXTRACTED)
- **Neo4j**: Deadlocks under concurrent writes
- **PostgreSQL**: Foreign key violations and closed transactions
- **Result**: 0 vectors generated, pipeline completes but produces no data

## OpenSpec Context

- **Change**: critical-pipeline-fixes
- **Proposal**: Fix parser, Neo4j concurrency, and database transaction issues
- **Specs**: 
  - Document processing pipeline (parse/chunk/embed)
  - Neo4j graph store integration
  - PostgreSQL vector storage
- **Status**: Discovery Complete, Planning In Progress

---

## Phase 1: Discovery ✅ COMPLETED

### Issues Identified

| Priority | Issue | Impact | Evidence |
|----------|-------|--------|----------|
| **P0** | Parser not extracting text | **CRITICAL** - No data enters pipeline | `pipeline.no_text_extracted`, `pipeline.no_extracted_text` |
| **P0** | Neo4j deadlocks | **CRITICAL** - Cognee graph operations fail | `Neo.TransientError.Transaction.DeadlockDetected` |
| **P0** | FK violations | **CRITICAL** - Chunks can't be saved | `violates foreign key constraint "document_chunks_job_id_fkey"` |
| **P0** | Closed transactions | **CRITICAL** - Database state inconsistent | `Can't operate on closed transaction inside context manager` |

### System State
- VPS: 72.61.52.191 (x86_64, 7.7GB RAM)
- Database: 0 chunks in `document_chunks` table
- HippoRAG: Working (207 entities, 179 triples)
- Cognee: pgvector fixed but other issues remain

---

## Phase 2: Task List

### Task 1: Fix Parser Text Extraction (P0)
**Owner**: `backend-developer`  
**Dependencies**: None  
**Estimated Time**: 4-6 hours

**Problem**: Docling and Azure OCR not extracting text from files
- Error: `pipeline.no_text_extracted`
- Fallback also fails: `pipeline.parsing_with_fallback` → Azure OCR

**Investigation Steps**:
1. Check docling version compatibility (currently 2.75.0)
2. Test docling on sample files locally
3. Check if docling models are downloaded/cached
4. Verify file permissions on uploaded files
5. Add detailed error logging to parser plugins

**Fix Options**:
- Option A: Downgrade docling to stable version (2.5.x)
- Option B: Fix docling configuration (model paths, OCR settings, dependencies)
- Option C: Ensure docling models are downloaded/cached properly

**Acceptance Criteria**:
- [ ] Parser extracts text from PDF files
- [ ] Parser extracts text from DOCX files
- [ ] Parser extracts text from PPTX files
- [ ] Extracted text length > 100 characters for test files

---

### Task 2: Fix Neo4j Concurrency / Deadlocks (P0)
**Owner**: `backend-developer`  
**Dependencies**: None  
**Estimated Time**: 3-4 hours

**Problem**: Multiple workers conflict when writing to Neo4j
```
Neo.TransientError.Transaction.DeadlockDetected
ForsetiClient can't acquire ExclusiveLock
```

**Root Cause**: Cognee's auto_cognify runs concurrently from multiple workers

**Fix Strategy**:
1. Add Neo4j retry logic with exponential backoff
2. Implement write locking for graph operations
3. Or: Disable Cognee auto_cognify, process sequentially

**Code Changes**:
- `src/plugins/destinations/cognee_local.py`: Add retry decorator
- `src/infrastructure/neo4j/client.py`: Add deadlock retry logic

**Acceptance Criteria**:
- [ ] No more deadlock errors in logs
- [ ] Cognee successfully creates nodes/relationships
- [ ] Multiple files can be processed concurrently without conflicts

---

### Task 3: Fix Database Transaction Handling (P0)
**Owner**: `db-agent`  
**Dependencies**: None  
**Estimated Time**: 3-4 hours

**Problem**: 
- `Can't operate on closed transaction inside context manager`
- Foreign key violations: job_id not found in jobs table

**Root Cause**: 
- Async session management issues
- Job created in one transaction, chunks in another
- Transaction closed before chunk insert completes

**Fix Strategy**:
1. Ensure job is committed before chunk creation
2. Fix async session lifecycle in worker processor
3. Add proper transaction boundaries

**Code Changes**:
- `src/worker/processor.py`: Fix transaction handling
- `src/db/repositories/document_chunk_repository.py`: Ensure FK exists

**Acceptance Criteria**:
- [ ] No more "closed transaction" errors
- [ ] No more FK violations
- [ ] Chunks successfully saved to PostgreSQL

---

### Task 4: Integration Testing & Validation (P1)
**Owner**: `tester-agent`  
**Dependencies**: Tasks 1, 2, 3  
**Estimated Time**: 2-3 hours

**Test Scenarios**:
1. Upload single PDF → Verify chunks created
2. Upload multiple files concurrently → Verify no deadlocks
3. Upload DOCX → Verify text extracted
4. Upload PPTX → Verify text extracted
5. Query Cognee search → Verify results returned
6. Query HippoRAG → Verify results returned

**Validation Metrics**:
- [ ] 100% of uploaded files create chunks
- [ ] Average chunks per file > 5
- [ ] Zero deadlock errors in 10 concurrent uploads
- [ ] Zero transaction errors
- [ ] Search returns results with entities

---

## Phase 3: Build Mode Execution Order

```
┌─────────────────────────────────────────────────────────────────┐
│                      BUILD PHASE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Iteration 1: Task 1 (Parser Fix)                               │
│  ├── Spawn backend-developer                                    │
│  ├── Fix docling/text extraction                                │
│  └── Run parser tests                                           │
│                                                                  │
│  Iteration 2: Task 2 (Neo4j Fix)                                │
│  ├── Spawn backend-developer                                    │
│  ├── Add retry logic                                            │
│  └── Test concurrent writes                                     │
│                                                                  │
│  Iteration 3: Task 3 (DB Transactions)                          │
│  ├── Spawn db-agent                                             │
│  ├── Fix transaction handling                                   │
│  └── Verify FK constraints                                      │
│                                                                  │
│  Iteration 4: Task 4 (Integration Testing)                      │
│  ├── Spawn tester-agent                                         │
│  ├── Run E2E tests                                              │
│  └── Validate all metrics                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Notes

### Parser Selection Priority
1. **Primary**: Docling (as per user preference)
2. **Secondary**: PyMuPDF (fast, reliable, no ML models)
3. **Fallback**: Azure OCR (for scanned documents)

### Neo4j Concurrency Strategy
```python
# Option 1: Retry with backoff
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def write_to_neo4j(data):
    ...

# Option 2: Worker-level locking
import asyncio
neo4j_lock = asyncio.Lock()

async def process_with_cognee(data):
    async with neo4j_lock:
        await cognee.write(data)
```

### Database Transaction Pattern
```python
# Ensure job exists before creating chunks
async def create_chunks(job_id, chunks):
    async with db_session() as session:
        # Verify job exists
        job = await session.get(Job, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Create chunks in same transaction
        for chunk in chunks:
            session.add(chunk)
        await session.commit()
```

---

## Validation Criteria

### Hard Gates (Must Pass)
- [ ] `make test` passes
- [ ] `make lint` passes
- [ ] Zero parser errors in worker logs
- [ ] Zero deadlock errors in 24 hours
- [ ] Zero transaction errors in 24 hours

### Soft Gates (Should Pass)
- [ ] Average processing time < 30 seconds per file
- [ ] Memory usage < 4GB per worker
- [ ] API response time < 2 seconds for health checks

---

## Rollback Plan

If issues persist after fixes:
1. Disable Cognee destination temporarily
2. Use only HippoRAG for graph operations
3. Limit workers to 1 to avoid concurrency issues
4. Rollback to last known working commit

---

## Communication Log

| Date | Event | Decision |
|------|-------|----------|
| 2026-03-02 | Discovery complete | All issues documented, planning started |
| 2026-03-02 | Parser identified as root cause | Fix parser first (P0) |
| 2026-03-02 | Neo4j deadlocks confirmed | Add retry logic + consider locking |
| 2026-03-02 | DB transaction issues confirmed | Fix session management |

---

## Next Action

**Start Iteration 1**: Spawn `backend-developer` to fix parser text extraction (Task 1)

Ready to proceed with Ralph Loop build phase?
