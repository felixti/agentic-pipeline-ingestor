# Spec: File Upload Processing

## Overview
Enable the `POST /api/v1/upload` endpoint to accept file uploads, store them temporarily, and create job records for processing.

### Requirements

#### R1: Accept Multipart File Upload
**Given** a multipart/form-data request with a file
**When** the request is received at `POST /api/v1/upload`
**Then** the endpoint should extract and validate the uploaded file

#### R2: Store Uploaded File
**Given** a validated uploaded file
**When** processing begins
**Then** the file should be saved to a temporary staging directory

#### R3: Create Job for Uploaded File
**Given** a successfully stored file
**When** the file is ready for processing
**Then** a job record should be created linking to the stored file

#### R4: Support Multiple Files
**Given** multiple files in a single request
**When** uploaded
**Then** create separate jobs for each file and return all job IDs

#### R5: Handle Large Files
**Given** a file exceeding size limits
**When** uploaded
**Then** return HTTP 413 with appropriate error message

### Scenarios

#### SC1: Upload Single PDF
A client uploads a PDF document:
```
POST /api/v1/upload
Content-Type: multipart/form-data

file: document.pdf (binary)
source_type: upload
metadata: {"department": "finance"}
```
The system stores the file and creates a job.

#### SC2: Upload Multiple Files
A client uploads 3 documents at once:
```
POST /api/v1/upload
Content-Type: multipart/form-data

files: [doc1.pdf, doc2.pdf, doc3.pdf]
```
The system creates 3 separate jobs and returns all job IDs.

#### SC3: Upload with Content Detection
A client uploads a PDF and requests automatic parser selection:
```
POST /api/v1/upload
Content-Type: multipart/form-data

file: scanned-document.pdf
detect_content: true
```
The system runs content detection and stores the results before creating the job.
