# Design: Content Detection for PDFs

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Content Detection System                         │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    ContentDetectionService                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │   PDFContent │──▶│   Content    │──▶│   DetectionResult    │   │   │
│  │  │   Analyzer   │  │   Classifier │  │   Repository         │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│  │           │                  │                  │                │   │
│  │           ▼                  ▼                  ▼                │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │              ParserSelector (Integration)                 │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      API Layer (FastAPI)                          │   │
│  │  POST /api/v1/detect                                              │   │
│  │  POST /api/v1/detect/url                                          │   │
│  │  POST /api/v1/detect/batch                                        │   │
│  │  GET  /api/v1/jobs/{id}/parser-selection                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
└────────────────────────────────────┼─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Data Layer                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │   PostgreSQL     │  │      Redis       │  │   File Storage  │ │
│  │   (Results)      │  │     (Cache)      │  │   (Temp Files)  │ │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### PDF Content Analyzer Detail

```
┌─────────────────────────────────────────────────────────┐
│              PDFContentAnalyzer                          │
│                                                          │
│  Input: file_path (Path)                                │
│  Output: ContentAnalysisResult                          │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Step 1: Text Layer Extraction                    │   │
│  │ - Use pdfplumber to extract text               │   │
│  │ - Get character count per page                 │   │
│  │ - Detect encoding and fonts                    │   │
│  │ Library: pdfplumber                            │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Step 2: Image Analysis                          │   │
│  │ - Use PyMuPDF to get image list               │   │
│  │ - Calculate image area ratio                   │   │
│  │ - Get image DPI and dimensions                 │   │
│  │ Library: PyMuPDF (fitz)                        │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Step 3: Calculate Statistics                    │   │
│  │ - text_ratio = text_chars / total_area        │   │
│  │ - image_ratio = image_area / page_area        │   │
│  │ - page-by-page breakdown                       │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Step 4: Classification                          │   │
│  │ - Apply decision matrix                        │   │
│  │ - Calculate confidence score                   │   │
│  │ - Determine content type                       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Approach

### Phase 1: Core Analysis Engine (Week 1)

1. **Create `PDFContentAnalyzer` class**
   - Location: `src/core/content_detection/analyzer.py`
   - Dependencies: pdfplumber, PyMuPDF
   - Methods: `analyze(file_path) -> ContentAnalysisResult`

2. **Define data models**
   - Location: `src/core/content_detection/models.py`
   - Classes: `ContentType` (Enum), `ContentAnalysisResult`, `PageAnalysis`

3. **Implement decision matrix**
   - Text ratio > 0.95 → TEXT_BASED
   - Text ratio < 0.05 AND image_ratio > 0.90 → SCANNED
   - Else → MIXED

4. **Unit tests**
   - Location: `tests/unit/content_detection/test_analyzer.py`
   - Test cases: pure text PDF, pure scanned PDF, mixed content, edge cases

### Phase 2: API Layer (Week 1-2)

1. **Create FastAPI routes**
   - Location: `src/api/routes/detection.py`
   - Endpoints: POST /detect, POST /detect/url, POST /detect/batch

2. **Request/Response models**
   - Location: `src/api/models/detection.py`
   - Pydantic models for API contracts

3. **Integration with existing auth**
   - Use existing API key auth from `src/auth/`
   - Rate limiting: 60 req/min per API key

4. **API tests**
   - Location: `tests/integration/test_detection_api.py`

### Phase 3: Storage & Caching (Week 2)

1. **Database schema**
   - Migration: `migrations/versions/xxx_add_content_detection.py`
   - Table: `content_detection_results`

2. **Repository pattern**
   - Location: `src/db/repositories/detection_result.py`
   - Methods: `get_by_hash()`, `save()`, `increment_access()`

3. **Redis caching layer**
   - Location: `src/core/content_detection/cache.py`
   - TTL: 30 days for detection results
   - Cache key: file SHA-256 hash

4. **Database integration tests**
   - Location: `tests/integration/test_detection_storage.py`

### Phase 4: Parser Selection Integration (Week 2-3)

1. **ParserSelector service**
   - Location: `src/core/parser_selection.py`
   - Integrate with existing pipeline at Stage 3

2. **Modify pipeline flow**
   - Location: `src/core/pipeline.py`
   - Add detection stage (Stage 2) before parser selection
   - Store detection result in job context

3. **Job context enhancement**
   - Location: `src/core/job_context.py`
   - Add `content_detection_result` field

4. **Pipeline integration tests**
   - Location: `tests/integration/test_pipeline_detection.py`

### Phase 5: Observability (Week 3)

1. **Metrics**
   - Counter: `content_detection_total{content_type}`
   - Histogram: `content_detection_duration_seconds`
   - Gauge: `content_detection_cache_hit_ratio`

2. **Tracing**
   - Spans for each analysis step
   - Attributes: file_size, page_count, detected_type

3. **Structured logging**
   - Events: detection_completed, cache_hit, cache_miss

## Decisions

### DEC-1: pdfplumber + PyMuPDF Combination
**Choice:** Use pdfplumber for text extraction and PyMuPDF for image analysis

**Rationale:**
- pdfplumber: Better text layout analysis, table detection (future use)
- PyMuPDF: Superior image extraction and metadata access
- Both are already in the project or commonly used

**Alternative Considered:** 
- Only PyMuPDF - insufficient text layout analysis
- pdfminer.six - lower-level, more complex API

**Trade-off:** Two dependencies vs one, but both are lightweight and well-maintained

### DEC-2: SHA-256 for File Hashing
**Choice:** Use SHA-256 for file hash in cache key

**Rationale:**
- Collision resistant for our use case (not a security application)
- Fast enough for files up to 100MB
- Standard algorithm, widely supported

**Alternative Considered:**
- MD5 - faster but collision-prone
- SHA-3 - overkill for this use case
- Content-based chunking - too complex

### DEC-3: Cache Duration: 30 Days
**Choice:** Cache detection results for 30 days

**Rationale:**
- PDF content doesn't change (immutable files)
- Long enough to benefit from cache for repeated processing
- Short enough to prevent unbounded storage growth

**Alternative Considered:**
- Permanent cache - storage concerns
- 7 days - too short for monthly batch processing patterns

### DEC-4: Low Confidence Threshold: 0.70
**Choice:** Use 0.70 as threshold for conservative strategy

**Rationale:**
- Below 0.70: Too uncertain, use both parsers
- Above 0.70: Confident enough for single parser
- Balances cost vs accuracy

**Alternative Considered:**
- 0.50 - too aggressive, more errors
- 0.90 - too conservative, unnecessary costs

### DEC-5: No Re-analysis on Hash Collision
**Choice:** If hash matches, trust cache (with rare re-analysis on content mismatch)

**Rationale:**
- SHA-256 collisions are astronomically unlikely
- Re-analyzing on every access defeats caching purpose
- Can add periodic re-validation if needed later

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/core/content_detection/__init__.py` | Package init |
| `src/core/content_detection/models.py` | Data models |
| `src/core/content_detection/analyzer.py` | PDF analysis engine |
| `src/core/content_detection/classifier.py` | Content classification |
| `src/core/content_detection/cache.py` | Redis caching |
| `src/api/routes/detection.py` | API endpoints |
| `src/api/models/detection.py` | API request/response models |
| `src/db/repositories/detection_result.py` | Database repository |
| `src/core/parser_selection.py` | Parser selection service |

### Modified Files

| File | Change |
|------|--------|
| `src/core/pipeline.py` | Add detection stage |
| `src/core/job_context.py` | Add detection result field |
| `src/main.py` | Register detection routes |
| `api/openapi.yaml` | Add detection endpoints |
| `pyproject.toml` | Add pdfplumber dependency |

### Migrations

| File | Purpose |
|------|---------|
| `migrations/versions/xxx_add_content_detection.py` | Create detection_results table |

## Dependencies

```toml
[project.dependencies]
# Existing
pymupdf = "^1.23.0"

# New
pdfplumber = "^0.10.0"
```

## Performance Considerations

| Metric | Target | Notes |
|--------|--------|-------|
| Analysis time | < 500ms | For PDFs up to 10MB |
| Memory usage | < 200MB | Peak during analysis |
| Cache hit latency | < 10ms | Redis + DB query |
| Concurrent analysis | 10 req/s | Per worker instance |

## Error Handling

| Error | Handling |
|-------|----------|
| Corrupted PDF | Return 400 with code CORRUPTED_PDF |
| Password protected | Return 400 with code PROTECTED_PDF |
| Analysis timeout | Return 504, log for investigation |
| Cache unavailable | Degrade gracefully, re-analyze |

## Testing Strategy

1. **Unit Tests**: Analyzer logic, classification matrix
2. **Integration Tests**: API endpoints, database operations
3. **E2E Tests**: Full pipeline flow with detection
4. **Performance Tests**: Benchmark with 1000 PDF corpus
5. **Contract Tests**: API schema validation
