# Spec: Job Processing Worker

## Overview
Implement a worker service that polls the database for pending jobs and processes them through the complete pipeline.

### Requirements

#### R1: Poll Database for Pending Jobs
**Given** the worker is running
**When** it polls for jobs
**Then** it should query jobs with status='created' or 'pending'

#### R2: Claim Job for Processing
**Given** a pending job is found
**When** the worker starts processing
**Then** update job status to 'processing' and set started_at timestamp

#### R3: Execute Pipeline Stages
**Given** a job is claimed
**When** processing begins
**Then** execute all pipeline stages in order:
  1. Ingest - validate and load file
  2. Detect - content type detection
  3. Parse - extract text/content
  4. Enrich - add metadata/entities
  5. Transform - chunk and format
  6. Output - store to destination

#### R4: Handle Success
**Given** all stages complete successfully
**When** processing finishes
**Then** update status to 'completed', store results, set completed_at

#### R5: Handle Failure
**Given** a stage fails
**When** error occurs
**Then** update status to 'failed', store error details, increment retry_count

#### R6: Support Multiple Workers
**Given** multiple workers are running
**When** they poll simultaneously
**Then** use row-level locking to prevent duplicate processing

### Scenarios

#### SC1: Process PDF Document
Worker finds job for PDF:
- Downloads file from source_uri
- Detects content type (text-based)
- Parses with docling
- Stores extracted text to Cognee
- Marks job complete

#### SC2: Handle Processing Failure
Worker encounters error during parsing:
- Catches exception
- Updates job with error_code and error_message
- Increments retry_count
- If retry_count < max_retries, job stays for retry
- If max retries exceeded, moves to DLQ

#### SC3: Concurrent Workers
Two workers poll simultaneously:
- Both query for pending jobs
- First worker acquires lock via SELECT FOR UPDATE
- Second worker skips locked rows
- No duplicate processing occurs
