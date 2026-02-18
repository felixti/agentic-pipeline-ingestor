# Spec: Job Results Storage

## Overview
Store and retrieve job processing results including extracted content, metadata, and quality metrics.

### Requirements

#### R1: Store Processing Results
**Given** a job completes processing
**When** results are available
**Then** store in job_results table with:
  - job_id (foreign key)
  - extracted_text or output_data
  - metadata (page count, word count, etc.)
  - quality_score
  - processing_time_ms
  - output_uri (if stored externally)

#### R2: Retrieve Job Results
**Given** a completed job exists
**When** GET /api/v1/jobs/{id}/result is called
**Then** return the stored results

#### R3: Handle Large Results
**Given** extracted content exceeds size limit
**When** storing results
**Then** store to file system/S3 and save reference URI

#### R4: Result Expiration
**Given** results are stored
**When** expiration period passes
**Then** automatically clean up old results (configurable)

### Scenarios

#### SC1: Store Text Extraction Results
Job processes PDF and extracts text:
```json
{
  "job_id": "...",
  "extracted_text": "Full document text...",
  "metadata": {
    "page_count": 10,
    "word_count": 5000,
    "language": "en"
  },
  "quality_score": 0.95,
  "processing_time_ms": 5000
}
```

#### SC2: Retrieve Results
Client requests results:
```
GET /api/v1/jobs/123/result
```
Returns stored extraction data.

#### SC3: Large Document Handling
Document has 100MB of text:
- Results stored to S3
- Database stores S3 URI
- API returns presigned URL
