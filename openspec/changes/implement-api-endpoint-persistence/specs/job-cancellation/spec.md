# Spec: Job Cancellation

## Overview
Enable the `DELETE /api/v1/jobs/{job_id}` endpoint to cancel running jobs and update their status in the database.

### Requirements

#### R1: Cancel Pending Job
**Given** a job with status `created` or `pending`
**When** delete request received
**Then** mark job as `cancelled` and return success

#### R2: Cancel Running Job
**Given** a job with status `processing`
**When** delete request received
**Then** signal cancellation to worker and mark as `cancelling`

#### R3: Prevent Cancellation of Completed Jobs
**Given** a job with status `completed`, `failed`, or `cancelled`
**When** delete request received
**Then** return HTTP 400 with error indicating job cannot be cancelled

#### R4: Handle Non-existent Job
**Given** an invalid job ID
**When** delete request received
**Then** return HTTP 404

### Scenarios

#### SC1: Cancel Pending Job
A client cancels a job that hasn't started:
```
DELETE /api/v1/jobs/550e8400-e29b-41d4-a716-446655440000
```
Job status updated to `cancelled`, returns HTTP 204.

#### SC2: Cancel Running Job
A client cancels a job currently being processed:
```
DELETE /api/v1/jobs/550e8400-e29b-41d4-a716-446655440000
```
Worker receives cancellation signal, job marked as `cancelling` then `cancelled`.

#### SC3: Cannot Cancel Completed Job
A client tries to cancel a completed job:
```
DELETE /api/v1/jobs/completed-job-id
```
Returns HTTP 400: "Cannot cancel job with status 'completed'"
