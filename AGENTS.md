# Agentic Data Pipeline Ingestor - Agent Guide

> **Last Updated**: 2026-02-17  
> **Language**: English  
> **Project Status**: Phase 4 Complete (Enterprise Features)

---

## 1. Project Overview

**Agentic Data Pipeline Ingestor** is an enterprise-grade document processing pipeline that uses AI-driven decision making for intelligent content routing and processing. The system follows a **Spec-Driven Development (SDD)** workflow and an **API-First architecture**.

### 1.1 Core Purpose

The system ingests documents from various sources (S3, Azure Blob, SharePoint), processes them through a 7-stage pipeline with intelligent parser selection, and outputs enriched data to destination-agnostic stores (Cognee, GraphRAG, webhooks).

### 1.2 Key Features

- **Universal File Support**: PDF, Office documents, images, archives
- **Agentic Processing**: AI-driven decision making for parser selection
- **Dual Parsing Strategy**: Docling primary + Azure OCR fallback
- **LLM-Agnostic**: Azure OpenAI + OpenRouter fallback via litellm
- **Enterprise Security**: RBAC, audit logging, data lineage
- **20GB/day Throughput**: Near-realtime + batch processing split

### 1.3 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    API LAYER (OpenAPI 3.1)                       │
├─────────────────────────────────────────────────────────────────┤
│                    CORE ORCHESTRATION ENGINE                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              AGENTIC DECISION ENGINE                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              7-STAGE PROCESSING PIPELINE                  │   │
│  │  Ingest → Detect → Parse → Enrich → Quality → Transform → │   │
│  │  Output                                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           LLM ADAPTER (litellm)                           │   │
│  │   ┌──────────────┐  ┌──────────────────────────┐         │   │
│  │   │ Azure GPT-4  │→ │ OpenRouter Claude-3      │         │   │
│  │   │  (Primary)   │  │   (Fallback)             │         │   │
│  │   └──────────────┘  └──────────────────────────┘         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PLUGIN ECOSYSTEM                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Sources   │  │   Parsers   │  │      Destinations       │  │
│  │  S3, Blob   │  │  Docling    │  │      Cognee             │  │
│  │  SharePoint │  │  Azure OCR  │  │      GraphRAG           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack

### 2.1 Core Technologies

| Category | Technology | Version |
|----------|------------|---------|
| Language | Python | 3.11+ |
| Web Framework | FastAPI | 0.104+ |
| Data Validation | Pydantic | 2.5+ |
| Database | PostgreSQL | 15+ |
| ORM | SQLAlchemy | 2.0+ |
| Cache | Redis | 7+ |
| LLM Router | litellm | 1.0+ |

### 2.2 Document Processing

| Technology | Purpose |
|------------|---------|
| Docling | Primary document parser |
| Azure AI Vision | OCR fallback for scanned documents |
| PyMuPDF | PDF text extraction |
| Pillow | Image processing |
| Tesseract | OCR fallback |

### 2.3 Observability

| Technology | Purpose |
|------------|---------|
| OpenTelemetry | Distributed tracing |
| Prometheus | Metrics collection |
| Grafana | Metrics visualization |
| Jaeger | Trace visualization |
| structlog | Structured logging |

### 2.4 Security & Enterprise

| Technology | Purpose |
|------------|---------|
| python-jose | JWT handling |
| passlib | Password hashing |
| Azure AD | Enterprise SSO |
| OpenSearch | Audit log storage |

---

## 3. Project Structure

```
agentic-pipeline-ingestion/
├── api/
│   └── openapi.yaml              # OpenAPI 3.1 specification (850+ lines)
├── src/                          # Main source code (83 Python files)
│   ├── api/                      # API layer (routes, models, dependencies)
│   │   ├── routes/               # FastAPI route handlers
│   │   ├── models.py             # Pydantic models
│   │   └── dependencies.py       # FastAPI dependencies
│   ├── auth/                     # Authentication & authorization
│   │   ├── base.py               # Base auth classes
│   │   ├── jwt.py                # JWT handling
│   │   ├── api_key.py            # API key auth
│   │   ├── oauth2.py             # OAuth2/OIDC
│   │   ├── azure_ad.py           # Azure AD integration
│   │   ├── rbac.py               # Role-based access control
│   │   └── dependencies.py       # Auth dependencies
│   ├── audit/                    # Audit logging
│   │   ├── models.py             # Audit event models
│   │   └── logger.py             # Audit logger implementation
│   ├── core/                     # Core orchestration engine
│   │   ├── engine.py             # Orchestration engine
│   │   ├── pipeline.py           # 7-stage pipeline executor
│   │   ├── detection.py          # Content type detection
│   │   ├── decisions.py          # Agentic decision engine
│   │   ├── quality.py            # Quality assessment
│   │   ├── retry.py              # Retry logic
│   │   ├── dlq.py                # Dead letter queue
│   │   ├── routing.py            # Destination routing
│   │   ├── webhooks.py           # Webhook handling
│   │   ├── learning.py           # Learning feedback loop
│   │   ├── healing.py            # Self-healing mechanisms
│   │   ├── optimizations.py      # Performance optimizations
│   │   ├── entity_extraction.py  # Entity extraction
│   │   ├── enrichment/           # Advanced enrichment
│   │   └── graphrag/             # GraphRAG integration
│   ├── db/                       # Database models
│   │   ├── models.py             # SQLAlchemy models
│   │   └── migrations/           # Alembic migrations
│   ├── llm/                      # LLM abstraction
│   │   ├── provider.py           # litellm integration
│   │   └── config.py             # LLM configuration loader
│   ├── lineage/                  # Data lineage tracking
│   │   ├── models.py             # Lineage models
│   │   └── tracker.py            # Lineage tracker
│   ├── observability/            # Observability stack
│   │   ├── tracing.py            # OpenTelemetry tracing
│   │   ├── metrics.py            # Prometheus metrics
│   │   ├── logging.py            # Structured logging
│   │   ├── middleware.py         # Observability middleware
│   │   └── genai_spans.py        # GenAI-specific spans
│   ├── plugins/                  # Plugin system
│   │   ├── base.py               # Plugin ABCs
│   │   ├── registry.py           # Plugin registry
│   │   ├── loaders.py            # Plugin discovery
│   │   ├── sources/              # Source plugins
│   │   │   ├── s3_source.py
│   │   │   ├── azure_blob_source.py
│   │   │   └── sharepoint_source.py
│   │   ├── parsers/              # Parser plugins
│   │   │   ├── docling_parser.py
│   │   │   ├── azure_ocr_parser.py
│   │   │   ├── csv_parser.py
│   │   │   ├── json_parser.py
│   │   │   ├── xml_parser.py
│   │   │   └── email_parser.py
│   │   └── destinations/         # Destination plugins
│   │       ├── cognee.py
│   │       ├── graphrag.py
│   │       ├── neo4j.py
│   │       ├── pinecone.py
│   │       └── weaviate.py
│   ├── retention/                # Data retention
│   │   └── manager.py
│   ├── worker/                   # Background worker
│   │   ├── main.py               # Worker entry point
│   │   └── processor.py          # Job processor
│   ├── config.py                 # Configuration management
│   └── main.py                   # FastAPI application entry
├── tests/                        # Test suite (11 Python files)
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── contract/                 # API contract tests
├── docker/                       # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── config/                       # Configuration files
│   ├── llm.yaml                  # LLM provider configuration
│   ├── prometheus.yml
│   └── otel-collector-config.yaml
├── azure/                        # Azure deployment
│   ├── aks-deployment.yaml
│   └── README.md
├── sdks/                         # Auto-generated SDKs
│   ├── generate.py
│   └── README.md
├── specs/                        # SDD specifications
│   └── spec-2026-02-16-agentic-pipeline-ingestor.md
├── sdd-agents/                   # Spec-Driven Development agents
│   ├── sdd-orchestrator.yaml
│   └── README.md
├── pyproject.toml                # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # Main documentation
```

---

## 4. Build and Development Commands

### 4.1 Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,docling,azure]"

# Copy environment file
cp .env.example .env
# Edit .env with your configuration
```

### 4.2 Running the Application

```bash
# Using Docker Compose (Recommended)
cd docker
docker-compose up -d
docker-compose logs -f api

# Direct (development mode)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Using the CLI
pipeline-ingestor
```

### 4.3 Testing Commands

```bash
# Run all tests
pytest

# Run with coverage (minimum 80%)
pytest --cov=src --cov-report=html

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m contract
pytest -m slow

# Run contract tests only
pytest tests/contract/

# Run with parallel execution
pytest -x
```

### 4.4 Code Quality Commands

```bash
# Linting with ruff
ruff check .
ruff check . --fix

# Formatting
ruff format .

# Type checking
mypy src/

# Security scanning
bandit -r src/
safety check

# Pre-commit hooks
pre-commit run --all-files
```

### 4.5 SDK Generation

```bash
# Install OpenAPI Generator
npm install -g @openapitools/openapi-generator-cli

# Generate all SDKs
python sdks/generate.py

# Generate specific SDK
python sdks/generate.py --language python
python sdks/generate.py --language typescript

# Dry run (preview commands)
python sdks/generate.py --dry-run
```

---

## 4.6 Makefile Tasks (Recommended)

This project includes a comprehensive Makefile for automation. **Use `make` commands instead of direct tool invocations** for consistency.

### Quick Reference

```bash
make help              # Show all available commands
make up                # Start all services
make down              # Stop all services
make test              # Run all tests
make lint              # Run linter
make format            # Format code
```

### Installation & Setup

| Command | Description |
|---------|-------------|
| `make install` | Install production dependencies |
| `make dev-install` | Install development dependencies |
| `make install-all` | Install all dependencies (dev, docling, azure, cognee) |
| `make sync` | Sync dependencies with uv.lock |

### Docker Operations

| Command | Description |
|---------|-------------|
| `make build` | Build Docker images |
| `make up` | Start all services in detached mode |
| `make down` | Stop all services |
| `make stop` | Stop services without removing containers |
| `make restart` | Restart all services |
| `make logs` | Show logs from all services |
| `make logs-api` | Show logs from API service only |
| `make ps` | Show running containers |

### Testing

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests only |
| `make test-contract` | Run contract tests only |
| `make test-coverage` | Run tests with coverage report |
| `make test-coverage-open` | Run tests and open coverage report in browser |

### E2E Testing

| Command | Description |
|---------|-------------|
| `make e2e-up` | Start E2E test environment |
| `make e2e-down` | Stop E2E test environment |
| `make e2e-test` | Run E2E tests |
| `make e2e-test-quick` | Run quick E2E smoke tests |
| `make e2e-test-auth` | Run E2E auth tests only |
| `make e2e-test-performance` | Run E2E performance tests |
| `make e2e-logs` | Show E2E test logs |

### Code Quality

| Command | Description |
|---------|-------------|
| `make lint` | Run linter (ruff) |
| `make lint-fix` | Run linter and fix issues |
| `make format` | Format code with ruff |
| `make type-check` | Run type checker (mypy) |
| `make type-check-strict` | Run type checker in strict mode |
| `make security` | Run security checks (bandit, safety) |
| `make quality` | Run all quality checks (lint + type-check + security) |
| `make format-and-lint` | Format code and run linter |

### Database

| Command | Description |
|---------|-------------|
| `make migrate` | Run database migrations |
| `make migrate-create MESSAGE="desc"` | Create new migration |
| `make migrate-downgrade` | Downgrade database by one revision |
| `make migrate-history` | Show migration history |
| `make db-reset` | Reset database (drop and recreate) |

### SDK Generation

| Command | Description |
|---------|-------------|
| `make sdk` | Generate SDKs from OpenAPI spec |
| `make sdk-python` | Generate Python SDK only |
| `make sdk-typescript` | Generate TypeScript SDK only |
| `make sdk-dry-run` | Preview SDK generation commands |

### Documentation

| Command | Description |
|---------|-------------|
| `make docs` | Build documentation |
| `make docs-serve` | Serve documentation locally |
| `make docs-deploy` | Deploy documentation |

### API Testing

| Command | Description |
|---------|-------------|
| `make api-health` | Test health endpoints |
| `make api-docs` | Open API documentation in browser |
| `make api-spec` | Download OpenAPI specification |

### Cleanup

| Command | Description |
|---------|-------------|
| `make clean` | Clean up build artifacts and cache |
| `make clean-docker` | Clean up Docker resources |
| `make clean-all` | Clean everything |

### Development Utilities

| Command | Description |
|---------|-------------|
| `make run` | Run the application locally (without Docker) |
| `make run-prod` | Run the application in production mode |
| `make shell` | Open a shell in the API container |
| `make db-shell` | Open PostgreSQL shell |
| `make redis-cli` | Open Redis CLI |

### CI/CD

| Command | Description |
|---------|-------------|
| `make ci` | Run CI checks (lint + test) |
| `make ci-full` | Run full CI pipeline |

### Release

| Command | Description |
|---------|-------------|
| `make version` | Show current version |
| `make bump-patch` | Bump patch version |
| `make bump-minor` | Bump minor version |
| `make bump-major` | Bump major version |

### Information

| Command | Description |
|---------|-------------|
| `make status` | Show project status (git, docker) |
| `make info` | Show project information |

---

## 4.7 Makefile Guidelines

### When to Use Make Commands

**Always use `make` commands when:**
- Starting/stopping services (`make up`, `make down`)
- Running tests (`make test`, `make test-unit`)
- Code quality checks (`make lint`, `make format`)
- Database operations (`make migrate`)
- Any operation that involves multiple steps or Docker

**Direct commands are acceptable for:**
- One-off Python script execution
- Debugging specific issues
- Exploring the codebase

### Common Workflows

#### Starting Development

```bash
# 1. Start services
make up

# 2. Check health
make api-health

# 3. Open API docs
make api-docs
```

#### Before Committing

```bash
# Run all quality checks
make format-and-lint
make type-check

# Run tests
make test

# Or run the full CI pipeline
make ci-full
```

#### Working with Database

```bash
# Create a migration after model changes
make migrate-create MESSAGE="add user table"

# Apply migrations
make migrate

# Reset database (use with caution)
make db-reset
```

#### Debugging

```bash
# Check service status
make ps

# View logs
make logs-api

# Enter container shell
make shell

# Check database
make db-shell
```

### Makefile Best Practices

1. **Use `make help`** to discover available commands
2. **Chain commands**: `make down up` for quick restart
3. **Use descriptive messages** when creating migrations
4. **Run `make clean`** periodically to remove cache files
5. **Always run `make ci`** before pushing changes

---

## 5. Configuration

### 5.1 Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required for core functionality
DEBUG=true
ENV=development
HOST=0.0.0.0
PORT=8000

# Database (Required)
DB_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline

# Redis (Required)
REDIS_URL=redis://localhost:6379/0

# LLM Providers (At least one required)
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-azure-key
OPENROUTER_API_KEY=your-openrouter-key

# Security (Required)
SECURITY_SECRET_KEY=your-secret-key

# OpenSearch (Optional - for audit logs)
OPENSEARCH_HOSTS=["http://localhost:9200"]

# Azure Services (Optional)
AZURE_TENANT_ID=
AZURE_CLIENT_ID=
AZURE_CLIENT_SECRET=
```

### 5.2 LLM Configuration

The LLM configuration is in `config/llm.yaml`:

```yaml
llm:
  router:
    - model_name: "agentic-decisions"
      litellm_params:
        model: "azure/gpt-4"
        api_base: "${AZURE_OPENAI_API_BASE}"
        api_key: "${AZURE_OPENAI_API_KEY}"
      fallback_models:
        - model: "openrouter/anthropic/claude-3-opus"
          api_key: "${OPENROUTER_API_KEY}"
```

---

## 6. Code Style Guidelines

### 6.1 Formatting

- **Line length**: 100 characters
- **Quote style**: Double quotes
- **Indent**: 4 spaces
- **Import style**: isort with `src` as first-party

### 6.2 Type Hints

All code must use strict type hints:

```python
from typing import Optional, List, Dict, Any

async def process_job(
    job_id: str,
    config: Dict[str, Any],
    timeout: Optional[int] = None
) -> JobResult:
    ...
```

### 6.3 Docstrings

Use Google-style docstrings:

```python
def analyze_content(
    file_path: Path,
    mime_type: str
) -> ContentAnalysis:
    """Analyze content type and recommend parser.
    
    Args:
        file_path: Path to the file to analyze.
        mime_type: MIME type of the file.
        
    Returns:
        ContentAnalysis with detected type and confidence.
        
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If mime_type is unsupported.
    """
```

### 6.4 Error Handling

Use structured error handling with request IDs:

```python
from structlog import get_logger

logger = get_logger(__name__)

try:
    result = await process()
except Exception as e:
    logger.error("processing_failed", error=str(e), job_id=job_id)
    raise PipelineError(
        message="Processing failed",
        code="PROCESSING_ERROR",
        details={"job_id": job_id}
    ) from e
```

### 6.5 Logging

Use structured logging with `structlog`:

```python
logger.info(
    "job_started",
    job_id=job_id,
    source_type=source_type,
    file_size=file_size,
)
```

---

## 7. Testing Instructions

### 7.1 Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `tests/unit/` | Test individual functions/classes in isolation |
| Integration | `tests/integration/` | Test API endpoints and database interactions |
| Contract | `tests/contract/` | Validate API against OpenAPI spec using Schemathesis |
| E2E | `tests/test_pipeline_integration.py` | Test full pipeline flow |

### 7.2 Test Markers

```python
import pytest

@pytest.mark.unit
def test_simple_function():
    pass

@pytest.mark.integration
def test_api_endpoint():
    pass

@pytest.mark.slow
def test_large_file_processing():
    pass
```

### 7.3 Writing Tests

```python
# Unit test example
import pytest
from src.core.detection import ContentDetector

@pytest.mark.unit
class TestContentDetector:
    def test_detects_text_pdf(self, tmp_path):
        detector = ContentDetector()
        result = detector.analyze(tmp_path / "test.pdf")
        assert result.content_type == "text"

# Integration test example
@pytest.mark.integration
async def test_create_job(client):
    response = await client.post("/api/v1/jobs", json={...})
    assert response.status_code == 202
```

### 7.4 Coverage Requirements

- Minimum coverage: **80%**
- Critical paths (pipeline, auth): **90%**
- Run: `pytest --cov=src --cov-report=html`

---

## 8. API Endpoints

The API follows OpenAPI 3.1 specification with 28 endpoints:

### 8.1 Job Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/jobs` | Submit new ingestion job |
| GET | `/api/v1/jobs` | List jobs with filtering |
| GET | `/api/v1/jobs/{id}` | Get job details |
| POST | `/api/v1/jobs/{id}/cancel` | Cancel running job |
| POST | `/api/v1/jobs/{id}/retry` | Retry failed job |
| GET | `/api/v1/jobs/{id}/result` | Get job result |
| GET | `/api/v1/jobs/{id}/events` | Get job events (SSE) |

### 8.2 File Upload

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/upload` | Multipart file upload |
| POST | `/api/v1/upload/url` | Upload from URL |
| POST | `/api/v1/upload/stream` | Streaming upload |

### 8.3 System & Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health status |
| GET | `/health/ready` | Readiness probe |
| GET | `/health/live` | Liveness probe |
| GET | `/metrics` | Prometheus metrics |
| GET | `/api/v1/openapi.yaml` | OpenAPI specification |

---

## 9. Security Considerations

### 9.1 Authentication Methods

The system supports multiple authentication methods:

1. **API Key**: `X-API-Key` header for service accounts
2. **OAuth2**: `Authorization: Bearer <token>` for user auth
3. **Azure AD**: Enterprise SSO integration
4. **JWT**: Internal token-based auth

### 9.2 RBAC Roles

| Role | Permissions |
|------|-------------|
| admin | All operations |
| operator | READ, CREATE, CANCEL, RETRY, CREATE_JOBS |
| developer | READ, CREATE_JOBS, READ_SOURCES |
| viewer | READ only |

### 9.3 Security Best Practices

- **Never commit secrets**: Use `.env` files (already in `.gitignore`)
- **Rotate API keys**: Regular rotation of Azure keys
- **Use HTTPS**: Always in production
- **Validate inputs**: All inputs validated via Pydantic
- **Rate limiting**: Default 100 req/min per API key
- **Audit logging**: All operations logged

### 9.4 Required Secrets (Production)

```bash
# Database
DB_URL

# LLM
AZURE_OPENAI_API_KEY
OPENROUTER_API_KEY

# Security
SECURITY_SECRET_KEY

# Azure Services
AZURE_AI_VISION_API_KEY
AZURE_STORAGE_KEY
AZURE_CLIENT_SECRET
```

---

## 10. Development Workflow

### 10.1 Spec-Driven Development (SDD)

This project uses SDD. **No code without spec.**

1. **Phase 1: Discovery** - Clarify requirements
2. **Phase 2: Specification** - Create spec in `specs/` (MANDATORY)
3. **Phase 3: Implementation** - Follow spec phases
4. **Phase 4: Validation** - Review against spec

See `sdd-agents/README.md` for details.

### 10.2 Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes, following code style
# Run tests
pytest

# Run linting
ruff check . --fix

# Commit with descriptive message
git commit -m "feat: add new parser support"

# Push and create PR
git push origin feature/my-feature
```

### 10.3 Commit Message Format

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `chore:` Maintenance

---

## 11. Deployment

### 11.1 Docker Deployment

```bash
cd docker
docker-compose up -d
```

Services started:
- API: http://localhost:8000
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- OpenSearch: localhost:9200
- Prometheus: localhost:9090
- Grafana: localhost:3000

### 11.2 Azure Kubernetes Service (AKS)

```bash
# Deploy to AKS
kubectl apply -f azure/aks-deployment.yaml

# Check status
kubectl get pods -n pipeline-ingestor
```

### 11.3 Health Check Endpoints

```bash
# Liveness (k8s probe)
curl http://localhost:8000/health/live

# Readiness (k8s probe)
curl http://localhost:8000/health/ready

# Full health check
curl http://localhost:8000/health
```

---

## 12. Troubleshooting

### 12.1 Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -e ".[dev,docling,azure]"` |
| Database connection failed | Check `DB_URL` in `.env` |
| Redis connection failed | Ensure Redis is running: `docker-compose up redis` |
| LLM timeout | Check Azure/OpenRouter credentials |
| Port 8000 in use | Change `PORT` in `.env` or kill existing process |

### 12.2 Debug Mode

```python
# Enable debug logging
DEBUG=true

# Enable SQL echo
DB_ECHO=true

# Enable OTEL debug
OTEL_LOG_LEVEL=DEBUG
```

### 12.3 Log Locations

- Application logs: stdout (structured JSON)
- Audit logs: OpenSearch (when configured)
- Database logs: PostgreSQL logs

---

## 13. Additional Resources

- **Main README**: `README.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Phase 2 Summary**: `PHASE2_IMPLEMENTATION.md`
- **Phase 4 Summary**: `PHASE4_IMPLEMENTATION_SUMMARY.md`
- **OpenAPI Spec**: `api/openapi.yaml`
- **SDD Guide**: `sdd-agents/README.md`
- **SDK Guide**: `sdks/README.md`
- **Azure Deployment**: `azure/README.md`

---

## 14. Contact

- **Issues**: https://github.com/example/agentic-pipeline-ingestor/issues
- **Email**: api-support@example.com
- **Documentation**: https://docs.pipeline.example.com

---

*This file is intended for AI coding agents. Keep it updated as the project evolves.*
