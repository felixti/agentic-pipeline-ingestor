# Spec: Job Creation Persistence

## Overview
Enable the `POST /api/v1/jobs` endpoint to persist job data to the database using the existing OrchestrationEngine and repository layers.

### Requirements

#### R1: Accept Job Creation Request
**Given** a valid job creation request with source_type and configuration
**When** the request is received at `POST /api/v1/jobs`
**Then** the endpoint should parse and validate the request body

#### R2: Persist Job to Database
**Given** a validated job creation request
**When** the orchestration engine creates the job
**Then** a record should be inserted into the `jobs` table with status `created`

#### R3: Return Created Job Details
**Given** a successfully persisted job
**When** the creation is complete
**Then** return HTTP 202 with the job ID, status, and confirmation message

#### R4: Handle Validation Errors
**Given** an invalid job creation request (missing required fields)
**When** validation fails
**Then** return HTTP 400 with detailed error information

### Scenarios

#### SC1: Create Job with Upload Source
A client submits a job to process an uploaded file:
```json
{
  "source_type": "upload",
  "source_uri": "/tmp/uploads/document.pdf",
  "file_name": "document.pdf",
  "file_size": 1024567,
  "mime_type": "application/pdf",
  "mode": "async",
  "priority": "normal"
}
```
The system creates a job record and returns the job ID for tracking.

#### SC2: Create Job with S3 Source
A client submits a job to process a file from S3:
```json
{
  "source_type": "s3",
  "source_uri": "s3://bucket/path/file.pdf",
  "file_name": "file.pdf",
  "mode": "async"
}
```
The system validates S3 configuration and creates the job.

#### SC3: Validation Failure - Missing Source Type
A client submits a request without source_type:
```json
{
  "file_name": "test.pdf"
}
```
The system returns HTTP 400 with error: "source_type is required"
