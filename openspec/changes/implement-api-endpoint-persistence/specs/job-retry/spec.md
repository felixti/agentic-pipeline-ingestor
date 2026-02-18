# Spec: Job Retry

## Overview
Enable the `POST /api/v1/jobs/{job_id}/retry` endpoint to retry failed jobs by creating new job instances or resetting job state.

### Requirements

#### R1: Retry Failed Job
**Given** a job with status `failed`
**When** retry request received
**Then** create a new job based on the failed job's configuration or reset state

#### R2: Prevent Retry of Non-failed Jobs
**Given** a job with status other than `failed`
**When** retry request received
**Then** return HTTP 400 with error message

#### R3: Increment Retry Count
**Given** a job being retried
**When** retry is initiated
**Then** increment the job's retry count in the database

#### R4: Preserve Original Configuration
**Given** a job being retried
**When** creating the retry job
**Then** preserve all original configuration (source, destination, pipeline config)

#### R5: Allow Configuration Updates on Retry
**Given** a failed job and updated configuration
**When** retry request includes new configuration
**Then** use the updated configuration for the retry

### Scenarios

#### SC1: Simple Retry
A client retries a failed job:
```
POST /api/v1/jobs/failed-job-id/retry
```
Creates new job with same configuration, returns new job ID.

#### SC2: Retry with Updated Config
A client retries with different pipeline configuration:
```
POST /api/v1/jobs/failed-job-id/retry
Content-Type: application/json

{
  "updated_config": {
    "parser": "azure_ocr",
    "timeout": 600
  }
}
```
Creates new job with updated configuration.

#### SC3: Cannot Retry Running Job
A client tries to retry a running job:
```
POST /api/v1/jobs/running-job-id/retry
```
Returns HTTP 400: "Cannot retry job with status 'processing'"
