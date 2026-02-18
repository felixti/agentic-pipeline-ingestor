# Spec: Detection Result Storage

## Overview

Persistence layer for storing content detection results, enabling caching, audit trails, and avoiding redundant re-analysis of identical files.

### Requirements

#### REQ-1: Detection Result Persistence
**Given** a content detection has been performed  
**When** the analysis completes  
**Then** the result is stored in the database with file hash as key

#### REQ-2: Result Caching by File Hash
**Given** a file that was previously analyzed  
**When** detection is requested again  
**Then** the cached result is returned if file hash matches (avoiding re-analysis)

#### REQ-3: Cache Expiration
**Given** a cached detection result  
**When** 30 days have passed  
**Then** the cache entry expires and new analysis is performed on next request

#### REQ-4: Cache Invalidation
**Given** a cached result with file hash  
**When** a different file with same hash is detected (hash collision)  
**Then** the system re-analyzes and updates the cache

#### REQ-5: Job Metadata Integration
**Given** a detection performed during job creation  
**When** the job is persisted  
**Then** the detection result is linked to the job record

#### REQ-6: Audit Trail
**Given** any detection operation  
**When** the operation completes  
**Then** an audit log entry is created with: timestamp, file_hash, result, user_id

### Data Model

```python
class ContentDetectionResult(BaseModel):
    id: UUID
    file_hash: str  # SHA-256 of file content
    file_size: int
    content_type: ContentType  # text_based | scanned | mixed
    confidence: float  # 0.0 - 1.0
    recommended_parser: str
    alternative_parsers: List[str]
    text_statistics: TextStatistics
    image_statistics: ImageStatistics
    page_results: List[PageResult]
    processing_time_ms: int
    created_at: datetime
    expires_at: datetime
    
class TextStatistics(BaseModel):
    total_pages: int
    total_characters: int
    has_text_layer: bool
    font_count: int
    encoding: str
    
class ImageStatistics(BaseModel):
    total_images: int
    image_area_ratio: float
    average_dpi: Optional[int]
    color_pages: int
    
class PageResult(BaseModel):
    page_number: int
    content_type: ContentType
    confidence: float
    text_ratio: float
    image_ratio: float
```

### Database Schema

```sql
CREATE TABLE content_detection_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA-256
    file_size BIGINT NOT NULL,
    content_type VARCHAR(20) NOT NULL CHECK (content_type IN ('text_based', 'scanned', 'mixed')),
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    recommended_parser VARCHAR(50) NOT NULL,
    alternative_parsers TEXT[],  -- PostgreSQL array
    text_statistics JSONB NOT NULL,
    image_statistics JSONB NOT NULL,
    page_results JSONB NOT NULL,  -- Array of page results
    processing_time_ms INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '30 days'),
    access_count INTEGER DEFAULT 1,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_detection_hash ON content_detection_results(file_hash);
CREATE INDEX idx_detection_type ON content_detection_results(content_type);
CREATE INDEX idx_detection_expires ON content_detection_results(expires_at);

-- Link to jobs
CREATE TABLE job_detection_results (
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    detection_result_id UUID REFERENCES content_detection_results(id) ON DELETE CASCADE,
    PRIMARY KEY (job_id, detection_result_id)
);
```

### Scenarios

#### SC-1: Cache Hit
User uploads "contract_v1.pdf" (hash: abc123). System analyzes and stores result. Same file uploaded again 5 minutes later. System returns cached result in < 10ms, increments access_count.

#### SC-2: Cache Miss (New File)
User uploads "contract_v2.pdf" (hash: def456). No cache entry found. System performs full analysis (245ms), stores result for future use.

#### SC-3: Cache Expiration
Detection result for "old_report.pdf" was stored 31 days ago. User uploads same file. Cache expired, system re-analyzes and updates record with new timestamp.

#### SC-4: Job Association
Job #12345 is created with file "invoice.pdf". Detection is performed, result stored, and linked to job. Parser selection stage reads from this linked record.

#### SC-5: Audit Query
Compliance officer queries: "Show all scanned documents processed last month". System joins detection results with jobs and audit logs to generate report.
