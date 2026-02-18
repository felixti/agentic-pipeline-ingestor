# Spec: Job Listing and Retrieval

## Overview
Enable the `GET /api/v1/jobs` and `GET /api/v1/jobs/{job_id}` endpoints to query and retrieve job data from the database.

### Requirements

#### R1: List All Jobs with Pagination
**Given** a request to list jobs
**When** received at `GET /api/v1/jobs`
**Then** return paginated results from the database with metadata

#### R2: Filter Jobs by Status
**Given** a status filter parameter
**When** listing jobs
**Then** return only jobs matching the specified status(es)

#### R3: Filter Jobs by Source Type
**Given** a source_type filter parameter
**When** listing jobs
**Then** return only jobs from the specified source

#### R4: Sort Jobs
**Given** sort parameters (field and order)
**When** listing jobs
**Then** return results sorted accordingly

#### R5: Get Single Job by ID
**Given** a valid job ID
**When** received at `GET /api/v1/jobs/{job_id}`
**Then** return the complete job details

#### R6: Handle Non-existent Job
**Given** an invalid job ID
**When** requesting that job
**Then** return HTTP 404 with appropriate error message

### Scenarios

#### SC1: List Jobs with Pagination
A client requests page 2 with 20 items per page:
```
GET /api/v1/jobs?page=2&limit=20
```
Returns 20 jobs, with pagination metadata including total count and links.

#### SC2: Filter by Status
A client wants only failed jobs:
```
GET /api/v1/jobs?status=failed
```
Returns only jobs with status="failed".

#### SC3: Get Specific Job
A client requests job details:
```
GET /api/v1/jobs/550e8400-e29b-41d4-a716-446655440000
```
Returns complete job information including status, progress, and metadata.

#### SC4: Job Not Found
A client requests a non-existent job:
```
GET /api/v1/jobs/invalid-id
```
Returns HTTP 404: "Job with ID 'invalid-id' not found"
