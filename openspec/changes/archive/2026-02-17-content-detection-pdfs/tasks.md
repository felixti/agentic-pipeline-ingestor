# Tasks: Content Detection for PDFs

## Implementation Tasks

### Phase 1: Core Analysis Engine

- [x] **TASK-1**: Create content detection package structure
  - Location: `src/core/content_detection/__init__.py`, `models.py`
  - Create enums: `ContentType` (TEXT_BASED, SCANNED, MIXED)
  - Create models: `ContentAnalysisResult`, `PageAnalysis`, `TextStatistics`, `ImageStatistics`
  - Acceptance: Models pass Pydantic validation

- [x] **TASK-2**: Implement PDF text extraction
  - Location: `src/core/content_detection/analyzer.py` (TextExtractor class)
  - Use pdfplumber to extract text per page
  - Calculate character count, word count, encoding
  - Acceptance: Extracts text from test PDFs accurately

- [x] **TASK-3**: Implement PDF image analysis
  - Location: `src/core/content_detection/analyzer.py` (ImageAnalyzer class)
  - Use PyMuPDF to list images and calculate area ratios
  - Extract image metadata (DPI, dimensions)
  - Acceptance: Correctly identifies image-heavy pages

- [x] **TASK-4**: Implement content classifier
  - Location: `src/core/content_detection/classifier.py`
  - Implement decision matrix:
    - text_ratio > 0.95 → TEXT_BASED
    - text_ratio < 0.05 AND image_ratio > 0.90 → SCANNED
    - Else → MIXED
  - Calculate confidence score
  - Acceptance: 98%+ accuracy on test corpus

- [x] **TASK-5**: Integrate analyzer components
  - Location: `src/core/content_detection/analyzer.py` (PDFContentAnalyzer class)
  - Orchestrate text extraction → image analysis → classification
  - Return complete ContentAnalysisResult
  - Acceptance: End-to-end analysis works for all PDF types

- [x] **TASK-6**: Add analyzer unit tests
  - Location: `tests/unit/content_detection/test_analyzer.py`
  - Test cases:
    - Pure text PDF (research paper)
    - Pure scanned PDF (invoices)
    - Mixed content (brochures)
    - Edge cases (image-heavy textbooks)
  - Acceptance: > 90% code coverage

### Phase 2: API Layer

- [x] **TASK-7**: Create API request/response models
  - Location: `src/api/models/detection.py`
  - Pydantic models for DetectionRequest, DetectionResponse, BatchDetectionRequest
  - Include validation rules
  - Acceptance: Models serialize/deserialize correctly

- [x] **TASK-8**: Implement POST /api/v1/detect endpoint
  - Location: `src/api/routes/detection.py`
  - Accept multipart/form-data with PDF file
  - Return detection result with 200 OK
  - Handle errors with appropriate status codes
  - Acceptance: API contract tests pass

- [x] **TASK-9**: Implement POST /api/v1/detect/url endpoint
  - Location: `src/api/routes/detection.py`
  - Accept JSON with URL and optional headers
  - Download file, analyze, return result
  - Handle download errors (404, timeout, etc.)
  - Acceptance: Works with HTTP and HTTPS URLs

- [x] **TASK-10**: Implement POST /api/v1/detect/batch endpoint
  - Location: `src/api/routes/detection.py`
  - Accept up to 10 files in multipart request
  - Process in parallel using asyncio
  - Return aggregated results
  - Acceptance: Batch of 10 processes in < 3s

- [x] **TASK-11**: Add rate limiting to detection endpoints
  - Location: `src/api/routes/detection.py`
  - Limit: 60 requests/minute per API key
  - Return 429 with Retry-After header
  - Acceptance: Rate limit enforced in tests

- [x] **TASK-12**: Write API integration tests
  - Location: `tests/integration/test_detection_api.py`
  - Test all endpoints with various inputs
  - Test error scenarios (invalid files, rate limits)
  - Acceptance: All tests pass

### Phase 3: Storage & Caching

- [x] **TASK-13**: Create database migration for detection results
  - Location: `migrations/versions/xxx_add_content_detection.py`
  - Table: `content_detection_results` with all columns
  - Indexes: file_hash (unique), content_type, expires_at
  - Acceptance: Migration runs successfully

- [x] **TASK-14**: Implement DetectionResultRepository
  - Location: `src/db/repositories/detection_result.py`
  - Methods:
    - `get_by_hash(file_hash) -> Optional[ContentDetectionResult]`
    - `save(result) -> ContentDetectionResult`
    - `increment_access(id) -> None`
  - Acceptance: Repository unit tests pass

- [x] **TASK-15**: Implement Redis caching layer
  - Location: `src/core/content_detection/cache.py`
  - Cache key: `detection:{file_hash}`
  - TTL: 30 days
  - Fallback to DB on cache miss
  - Acceptance: Cache hit returns in < 10ms

- [x] **TASK-16**: Integrate caching into detection flow
  - Location: `src/core/content_detection/service.py`
  - Check cache → Return cached if hit → Analyze if miss → Store result
  - Handle cache failures gracefully (degrade to DB)
  - Acceptance: 90%+ cache hit rate in repeated tests

- [x] **TASK-17**: Add storage integration tests
  - Location: `tests/integration/test_detection_storage.py`
  - Test cache hit, cache miss, cache expiration
  - Test database operations
  - Acceptance: All tests pass

### Phase 4: Parser Selection Integration

- [x] **TASK-18**: Implement ParserSelector service
  - Location: `src/core/parser_selection.py`
  - Method: `select_parser(detection_result, explicit_config) -> ParserSelection`
  - Logic:
    - Explicit config takes priority
    - Low confidence (< 0.70) → conservative (both parsers)
    - Content type → appropriate parser mapping
  - Acceptance: Unit tests for all scenarios

- [x] **TASK-19**: Modify pipeline to add detection stage
  - Location: `src/core/pipeline.py`
  - Insert detection as Stage 2 (after Ingest, before Parse)
  - Store detection result in job context
  - Skip detection if already cached
  - Acceptance: Pipeline flow tests pass

- [x] **TASK-20**: Update parser selection stage
  - Location: `src/core/pipeline.py` (Stage 3)
  - Read detection result from job context
  - Call ParserSelector to determine parsers
  - Pass selection to parser execution
  - Acceptance: Correct parser selected based on content type

- [x] **TASK-21**: Update JobContext model
  - Location: `src/core/job_context.py`
  - Add `content_detection_result` field
  - Include in serialization
  - Acceptance: Job context includes detection result

- [x] **TASK-22**: Add pipeline integration tests
  - Location: `tests/integration/test_pipeline_detection.py`
  - Test full flow: upload → detect → select parser → process
  - Test with text-based, scanned, and mixed PDFs
  - Acceptance: All scenarios pass

### Phase 5: Observability

- [x] **TASK-23**: Add Prometheus metrics
  - Location: `src/observability/metrics.py`
  - Counters:
    - `content_detection_total{content_type}`
    - `content_detection_cache_hit_total`
    - `content_detection_cache_miss_total`
  - Histograms:
    - `content_detection_duration_seconds`
  - Acceptance: Metrics visible at `/metrics`

- [x] **TASK-24**: Add OpenTelemetry spans
  - Location: `src/core/content_detection/analyzer.py`, `service.py`
  - Spans: text_extraction, image_analysis, classification
  - Attributes: file_size, page_count, detected_type, confidence
  - Acceptance: Traces visible in Jaeger

- [x] **TASK-25**: Add structured logging
  - Location: `src/core/content_detection/service.py`
  - Events:
    - `content_detection.completed` (with result summary)
    - `content_detection.cache_hit`
    - `content_detection.cache_miss`
  - Include file_hash, content_type, confidence, duration_ms
  - Acceptance: Logs in JSON format

- [x] **TASK-26**: Create Grafana dashboard
  - Location: `config/grafana/dashboards/content-detection.json`
  - Panels:
    - Detection requests per minute
    - Content type distribution (pie chart)
    - Cache hit ratio over time
    - Average detection duration
    - Error rate
  - Acceptance: Dashboard imports successfully

### Phase 6: Documentation & Polish

- [x] **TASK-27**: Update OpenAPI specification
  - Location: `api/openapi.yaml`
  - Add detection endpoints
  - Add schemas for all request/response models
  - Add error response examples
  - Acceptance: Spec validates with Swagger UI

- [x] **TASK-28**: Add API documentation
  - Location: `docs/content-detection.md`
  - Usage examples with curl
  - Explanation of content types
  - Best practices for batch detection
  - Acceptance: Documentation is clear and complete

- [x] **TASK-29**: Performance benchmarking
  - Location: `tests/performance/test_detection.py`
  - Benchmark with 1000 PDF corpus
  - Measure: latency, throughput, memory usage
  - Acceptance: Meets performance targets

- [x] **TASK-30**: End-to-end testing
  - Location: `tests/e2e/test_detection_flow.py`
  - Full user journey: upload → detect → process → verify
  - Test with production-like data
  - Acceptance: E2E tests pass in CI

## Verification Checklist

Before archiving this change:

- [ ] All unit tests pass (>90% coverage on new code)
- [ ] All integration tests pass
- [ ] API contract tests pass
- [ ] E2E tests pass
- [ ] Performance benchmarks meet targets
- [ ] Documentation updated (API docs, OpenAPI spec)
- [ ] Grafana dashboard created and tested
- [ ] Manual testing completed:
  - [ ] Text-based PDF detected correctly
  - [ ] Scanned PDF detected correctly
  - [ ] Mixed content detected correctly
  - [ ] Cache hit/miss works correctly
  - [ ] Parser selection routes correctly
- [ ] Code review completed
- [ ] Migration tested on staging database

## Task Dependencies

```
TASK-1 (Models)
    │
    ├──▶ TASK-2 (Text Extraction)
    │       │
    │       └──▶ TASK-6 (Analyzer Tests)
    │
    ├──▶ TASK-3 (Image Analysis)
    │       │
    │       └──▶ TASK-6
    │
    └──▶ TASK-4 (Classifier)
            │
            └──▶ TASK-5 (Integrate Analyzer)
                    │
                    └──▶ TASK-6

TASK-7 (API Models)
    │
    ├──▶ TASK-8 (POST /detect)
    ├──▶ TASK-9 (POST /detect/url)
    ├──▶ TASK-10 (POST /detect/batch)
    └──▶ TASK-11 (Rate Limiting)
            │
            └──▶ TASK-12 (API Tests)

TASK-13 (Migration)
    │
    ├──▶ TASK-14 (Repository)
    │       │
    │       └──▶ TASK-17 (Storage Tests)
    │
    └──▶ TASK-15 (Cache)
            │
            └──▶ TASK-16 (Integrate Cache)
                    │
                    └──▶ TASK-17

TASK-18 (ParserSelector)
    │
    ├──▶ TASK-21 (JobContext Update)
    │       │
    │       └──▶ TASK-19 (Pipeline Detection)
    │               │
    │               └──▶ TASK-20 (Parser Selection Stage)
    │                       │
    │                       └──▶ TASK-22 (Pipeline Tests)
    │
    └──▶ TASK-22
```

## Estimated Effort

| Phase | Tasks | Estimated Days |
|-------|-------|----------------|
| Phase 1: Core Engine | 6 | 3-4 days |
| Phase 2: API Layer | 6 | 3-4 days |
| Phase 3: Storage | 5 | 2-3 days |
| Phase 4: Integration | 5 | 3-4 days |
| Phase 5: Observability | 4 | 2 days |
| Phase 6: Docs & Polish | 4 | 2-3 days |
| **Total** | **30** | **15-20 days** |
