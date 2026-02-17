# E2E Test Suite for Agentic Data Pipeline Ingestor

This directory contains a comprehensive End-to-End (E2E) test suite for the Agentic Data Pipeline Ingestor, following **shift-left engineering** principles. All tests are designed to be runnable locally with Docker.

## 📁 Directory Structure

```
tests/e2e/
├── fixtures/                   # Test data and sample documents
│   ├── documents/              # Sample documents for testing
│   │   ├── pdf/                # PDF test files
│   │   ├── office/             # Office documents (Word, Excel, PowerPoint)
│   │   ├── images/             # Image files for OCR testing
│   │   └── archives/           # Archive files (ZIP, etc.)
│   ├── datasets/
│   │   └── test-dataset.json   # Test case configurations
│   └── documents/
│       └── generate_test_docs.py  # Script to generate test documents
├── http/                       # HTTP request files for manual testing
│   ├── 01_health.http
│   ├── 02_auth.http
│   ├── 03_jobs.http
│   ├── 04_upload.http
│   ├── 05_sources.http
│   ├── 06_destinations.http
│   ├── 07_audit.http
│   └── 08_bulk.http
├── scenarios/                  # E2E test scenarios
│   ├── test_full_pipeline.py
│   ├── test_retry_mechanisms.py
│   ├── test_dlq_workflow.py
│   ├── test_auth_flow.py
│   └── test_performance.py
├── docker/                     # E2E Docker configuration
│   ├── docker-compose.e2e.yml
│   └── Dockerfile.test
├── reports/                    # Test output directory
├── conftest.py                 # Pytest configuration and fixtures
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- curl (for health checks)

### Running Tests

#### Option 1: Using the Helper Script (Recommended)

```bash
# Run all E2E tests (excluding performance tests)
./scripts/run-e2e-tests.sh

# Run full test suite including performance tests
./scripts/run-e2e-tests.sh --full

# Run quick smoke tests only
./scripts/run-e2e-tests.sh --quick

# Run specific test categories
./scripts/run-e2e-tests.sh --auth        # Authentication tests
./scripts/run-e2e-tests.sh --retry       # Retry mechanism tests
./scripts/run-e2e-tests.sh --dlq         # DLQ tests
./scripts/run-e2e-tests.sh --performance # Performance tests

# Keep containers running after tests
./scripts/run-e2e-tests.sh --keep

# Show API logs during test execution
./scripts/run-e2e-tests.sh --logs
```

#### Option 2: Manual Docker Compose

```bash
# Start the E2E stack
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml up -d

# Wait for services to be ready (check health endpoint)
curl http://localhost:8001/health/live

# Run tests
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml \
    exec test-runner pytest /tests/e2e/scenarios -v

# Cleanup
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml down -v
```

#### Option 3: Local pytest (Requires Local Stack)

```bash
# Install dependencies
pip install -e ".[dev]"

# Start local services (PostgreSQL, Redis)
# Then run the API locally
python -m src.main

# Run E2E tests against local API
E2E_BASE_URL=http://localhost:8000 \
E2E_API_KEY=test-api-key \
pytest tests/e2e/scenarios -v -m "e2e and not performance"
```

## 📊 Test Coverage

### Test Categories

| Category | Description | File |
|----------|-------------|------|
| **Full Pipeline** | End-to-end document processing workflows | `test_full_pipeline.py` |
| **Retry Mechanisms** | All 4 retry strategies | `test_retry_mechanisms.py` |
| **DLQ Workflow** | Dead Letter Queue functionality | `test_dlq_workflow.py` |
| **Auth Flow** | Authentication & authorization | `test_auth_flow.py` |
| **Performance** | Latency, throughput, concurrency | `test_performance.py` |

### Test Markers

Tests are marked with pytest markers for selective execution:

- `e2e` - All E2E tests
- `performance` - Performance and load tests (may be slow)
- `slow` - Tests that take longer to run
- `auth` - Authentication and authorization tests
- `retry` - Retry mechanism tests
- `dlq` - Dead Letter Queue tests

Run specific markers:
```bash
pytest tests/e2e/scenarios -m "auth"
pytest tests/e2e/scenarios -m "not performance"
pytest tests/e2e/scenarios -m "e2e and not slow"
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `E2E_BASE_URL` | `http://localhost:8000` | Base URL of the API |
| `E2E_API_KEY` | `test-api-key` | Standard API key |
| `E2E_ADMIN_API_KEY` | `admin-api-key` | Admin API key |
| `E2E_VIEWER_API_KEY` | `viewer-api-key` | Viewer (read-only) API key |
| `E2E_REQUEST_TIMEOUT` | `30` | HTTP request timeout in seconds |
| `E2E_POLL_INTERVAL` | `2.0` | Polling interval for job status |
| `E2E_MAX_POLL_ATTEMPTS` | `60` | Maximum polling attempts |

### Test Data Configuration

Test cases are defined in `fixtures/datasets/test-dataset.json`:

```json
{
  "test_cases": [
    {
      "id": "TC001",
      "name": "Text PDF Processing",
      "file": "pdf/sample-text.pdf",
      "expected_parser": "docling",
      "assertions": {
        "status": "completed",
        "quality_score": ">=0.8"
      }
    }
  ]
}
```

## 🔌 HTTP Request Files

The `http/` directory contains `.http` files that can be used with:

- **IntelliJ IDEA / PyCharm** (HTTP Client plugin)
- **VS Code** (REST Client extension)
- **curl** (with manual conversion)

### Using with IntelliJ/PyCharm

1. Open any `.http` file
2. Click the green play button next to a request
3. View responses in the Run tool window

### Using with VS Code

1. Install "REST Client" extension
2. Open any `.http` file
3. Click "Send Request" above any request
4. View responses inline

### Example: Run All Health Checks

Open `http/01_health.http` and run all requests to verify the system is healthy.

## 📄 Test Documents

### Generating Test Documents

```bash
cd tests/e2e/fixtures/documents
python generate_test_docs.py
```

**Requirements:**
```bash
pip install reportlab pillow openpyxl python-pptx
```

### Document Types

| Type | Files | Purpose |
|------|-------|---------|
| **PDF** | `sample-text.pdf`, `sample-scanned.pdf`, `sample-mixed.pdf`, `large-document.pdf` | Text extraction, OCR, performance |
| **Office** | `sample.docx`, `sample.xlsx`, `sample.pptx` | Office document processing |
| **Images** | `receipt.jpg`, `document.png` | OCR testing |
| **Archives** | `documents.zip` | Bulk processing |

## 🧪 Writing E2E Tests

### Test Structure

```python
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

class TestFeatureName:
    """Test description."""
    
    async def test_specific_scenario(self, auth_client):
        """E2E: Test description."""
        # Arrange
        payload = {"key": "value"}
        
        # Act
        response = await auth_client.post("/api/v1/jobs", json=payload)
        
        # Assert
        assert response.status_code == 202
        assert "id" in response.json()["data"]
```

### Available Fixtures

| Fixture | Description |
|---------|-------------|
| `client` | Unauthenticated HTTP client |
| `auth_client` | Client with standard API key |
| `admin_client` | Client with admin API key |
| `viewer_client` | Client with viewer (read-only) API key |
| `unauth_client` | Client without any authentication |
| `test_documents` | Dictionary of test document paths |
| `test_dataset` | Test dataset configuration |
| `sample_job_payload` | Sample job creation payload |
| `job_helper` | Helper class for job operations |
| `assert_helper` | Custom assertion helpers |

### Helper Functions

```python
# Wait for job completion
from conftest import wait_for_job_completion

job_data = await wait_for_job_completion(auth_client, job_id, timeout=120)

# Create job
from conftest import create_job

job_id = await create_job(auth_client, payload)

# Upload file
from conftest import upload_file

job_id = await upload_file(auth_client, file_path, metadata={"test": True})
```

## 🐳 Docker Services

### Available Services

| Service | Port | Description |
|---------|------|-------------|
| `api-e2e` | 8001 | Main API service |
| `postgres-e2e` | 5433 | Test database |
| `redis-e2e` | 6380 | Cache and job queue |
| `test-runner` | - | Test execution container |

### Profiles

Use `--profile` to start additional services:

```bash
# Include mock LLM service
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml --profile with-mocks up -d

# Include worker service
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml --profile with-worker up -d

# Include monitoring (Prometheus, Grafana)
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml --profile with-monitoring up -d

# Include S3-compatible storage (MinIO)
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml --profile with-storage up -d
```

## 📈 Performance Testing

### Running Performance Tests

```bash
# Run all performance tests
./scripts/run-e2e-tests.sh --performance

# Or with pytest directly
pytest tests/e2e/scenarios/test_performance.py -v
```

### Performance Metrics

Tests verify:

- **Latency**: p99 < 2s for health checks
- **Throughput**: Job creation > 10 jobs/second
- **Concurrency**: Handle 50+ concurrent requests
- **Scalability**: Stateless API design indicators

### Custom Performance Test

```python
@pytest.mark.performance
async def test_custom_latency(self, auth_client):
    latencies = []
    
    for _ in range(100):
        start = time.time()
        response = await auth_client.get("/api/v1/jobs")
        end = time.time()
        latencies.append((end - start) * 1000)
    
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    assert p99 < 1000, f"p99 latency {p99}ms exceeds threshold"
```

## 🔍 Debugging Tests

### View API Logs

```bash
# During test execution
./scripts/run-e2e-tests.sh --logs

# Or manually
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml logs -f api-e2e
```

### Interactive Debugging

```bash
# Start stack
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml up -d

# Run a single test with maximum verbosity
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml \
    run --rm test-runner \
    pytest /tests/e2e/scenarios/test_full_pipeline.py::TestFullPipeline::test_health_check \
    -vvs --tb=long

# Access API directly
curl http://localhost:8001/health
```

### Test Reports

Reports are generated in `tests/e2e/reports/`:

- `report.html` - HTML test report with details
- `junit.xml` - JUnit XML format for CI/CD

Open HTML report:
```bash
open tests/e2e/reports/report.html  # macOS
xdg-open tests/e2e/reports/report.html  # Linux
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run E2E Tests
        run: ./scripts/run-e2e-tests.sh --quick
      
      - name: Upload Test Reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: e2e-reports
          path: tests/e2e/reports/
```

### GitLab CI Example

```yaml
e2e-tests:
  stage: test
  script:
    - ./scripts/run-e2e-tests.sh --quick
  artifacts:
    when: always
    paths:
      - tests/e2e/reports/
```

## 📝 Best Practices

### Do's

✅ **Do** use the fixtures provided in `conftest.py`
✅ **Do** mark tests with appropriate markers (`e2e`, `performance`, `slow`)
✅ **Do** use descriptive test names and docstrings
✅ **Do** clean up resources (jobs, uploads) after tests
✅ **Do** use async/await for all async operations
✅ **Do** assert both status codes and response content

### Don'ts

❌ **Don't** hardcode URLs - use `client.base_url`
❌ **Don't** rely on specific IDs between tests
❌ **Don't** run performance tests in quick mode
❌ **Don't** skip error handling in tests
❌ **Don't** create inter-test dependencies

## 🆘 Troubleshooting

### Common Issues

**Issue**: API fails to start
```bash
# Check logs
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml logs api-e2e

# Check port conflicts
lsof -i :8001
```

**Issue**: Tests fail with 401
```bash
# Verify API keys are configured
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml exec api-e2e env | grep API
```

**Issue**: Database connection errors
```bash
# Check database is ready
docker-compose -f tests/e2e/docker/docker-compose.e2e.yml exec postgres-e2e pg_isready
```

**Issue**: Tests timeout
```bash
# Increase timeout
E2E_REQUEST_TIMEOUT=60 ./scripts/run-e2e-tests.sh
```

### Getting Help

1. Check the [troubleshooting section](#common-issues)
2. Review test logs in `tests/e2e/reports/`
3. Open an issue with:
   - Test command used
   - Error message
   - Environment details (OS, Docker version)

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [HTTP Client Plugin](https://www.jetbrains.com/help/idea/http-client-in-product-code-editor.html)
- [Docker Compose](https://docs.docker.com/compose/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

## 📄 License

This E2E test suite is part of the Agentic Data Pipeline Ingestor project and follows the same license terms.

---

**Shift-left engineering**: These tests are designed to catch issues early in the development cycle, before they reach production. Run them frequently during development!
