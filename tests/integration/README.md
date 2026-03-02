# Integration Tests for Ralph Loop Pipeline

This directory contains comprehensive end-to-end integration tests for the Ralph Loop document processing pipeline.

## Overview

These tests verify all three critical fixes work together:
1. **Parser text extraction** (docling + fallbacks)
2. **Neo4j deadlock handling** (retry + locking)
3. **Database transactions** (proper commit boundaries)

## Test Scenarios

### 1. Single File Upload (Happy Path)
- **File**: `tests/integration/test_end_to_end.py::TestSingleFileUpload`
- **Scenario**: Upload PDF → Parse → Chunk → Embed → Store in PostgreSQL → Store in Cognee → Store in HippoRAG
- **Expected Results**:
  - Job status: completed
  - Chunks in PostgreSQL: > 0
  - Job shows chunks_count > 0
  - Cognee graph updated
  - HippoRAG graph updated

### 2. Multiple Concurrent Uploads (Stress Test)
- **File**: `tests/integration/test_end_to_end.py::TestConcurrentUploads`
- **Scenario**: Upload 5 files simultaneously
- **Expected Results**:
  - All jobs complete successfully
  - No deadlock errors
  - No transaction errors
  - All chunks saved correctly

### 3. Different File Types
- **File**: `tests/integration/test_end_to_end.py::TestFileTypeParsing`
- **Scenario**: Upload PDF, DOCX, PPTX sequentially
- **Expected Results**:
  - All files parsed successfully
  - Text extracted from each (> 500 chars)
  - Chunks created for each

### 4. Search Validation
- **File**: `tests/integration/test_end_to_end.py::TestSearchAfterIngestion`
- **Scenario**: After file processing, search with Cognee hybrid search and HippoRAG retrieval
- **Expected Results**:
  - Search returns results
  - Results contain entities
  - Source documents referenced

## Running the Tests

### Prerequisites

Set environment variables:
```bash
export E2E_BASE_URL="http://localhost:8000"  # or your API URL
export E2E_API_KEY="test-api-key"
```

### Run All Integration Tests

```bash
pytest tests/integration/test_end_to_end.py -v
```

### Run Specific Test Classes

```bash
# Single file upload tests
pytest tests/integration/test_end_to_end.py::TestSingleFileUpload -v

# Concurrent upload stress tests
pytest tests/integration/test_end_to_end.py::TestConcurrentUploads -v

# File type parsing tests
pytest tests/integration/test_end_to_end.py::TestFileTypeParsing -v

# Search validation tests
pytest tests/integration/test_end_to_end.py::TestSearchAfterIngestion -v
```

### Run Individual Tests

```bash
pytest tests/integration/test_end_to_end.py::TestSingleFileUpload::test_single_pdf_upload_happy_path -v
```

## Test Data

Test files are located in `tests/data/`:
- `sample.pdf` - Test PDF parsing
- `sample.docx` - Test DOCX parsing
- `sample.pptx` - Test PPTX parsing
- `sample.txt` - Test text parsing

## Validation Checklist

| Test | Metric | Target |
|------|--------|--------|
| Single PDF | Chunks created | > 5 chunks |
| Single PDF | Processing time | < 60 seconds |
| Concurrent (5x) | Success rate | 100% |
| Concurrent (5x) | Deadlock errors | 0 |
| Concurrent (5x) | FK violations | 0 |
| DOCX parsing | Text extracted | > 500 chars |
| PPTX parsing | Text extracted | > 500 chars |
| Cognee search | Results returned | > 0 |
| HippoRAG search | Entities returned | > 0 |

## Manual Testing

Use the manual test script for interactive testing:

```bash
# Test single file
python scripts/test_pipeline_manual.py --file tests/data/sample.pdf

# Test with wait for completion
python scripts/test_pipeline_manual.py --file tests/data/sample.pdf --wait

# Test directory of files
python scripts/test_pipeline_manual.py --dir tests/data/ --pattern "*.pdf"

# Check job status
python scripts/test_pipeline_manual.py --job-id <uuid> --check
```

## Load Testing

Use the load test script for performance testing:

```bash
# Run 10 concurrent uploads
python scripts/load_test.py --count 10

# Run 20 uploads with max 5 concurrent
python scripts/load_test.py --count 20 --concurrent 5

# Generate JSON report
python scripts/load_test.py --count 10 --report json --output results.json

# Generate CSV report
python scripts/load_test.py --count 10 --report csv --output results.csv
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `E2E_BASE_URL` | `http://localhost:8000` | API base URL |
| `E2E_API_KEY` | `test-api-key` | API key for authentication |
| `E2E_REQUEST_TIMEOUT` | `60` | HTTP request timeout (seconds) |
| `E2E_POLL_INTERVAL` | `2.0` | Seconds between job status polls |
| `E2E_MAX_POLL_ATTEMPTS` | `60` | Maximum polling attempts |
| `API_URL` | `https://pipeline-api.felixtek.cloud/api/v1` | Manual test API URL |
| `API_KEY` | `test-api-key-123` | Manual test API key |

## CI/CD Integration

These tests can be run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Integration Tests
  run: |
    pytest tests/integration/test_end_to_end.py -v --tb=short
  env:
    E2E_BASE_URL: ${{ secrets.API_URL }}
    E2E_API_KEY: ${{ secrets.API_KEY }}
```

## Troubleshooting

### Tests Skipping

If tests are skipped, check:
1. Test files exist in `tests/data/`
2. API is running and accessible
3. API key is valid
4. Cognee/HippoRAG endpoints are available

### Timeout Errors

If jobs timeout:
1. Increase `E2E_MAX_POLL_ATTEMPTS`
2. Check worker is processing jobs
3. Verify database connections

### Connection Errors

If connection fails:
1. Verify `E2E_BASE_URL` is correct
2. Check network connectivity
3. Ensure API is running
