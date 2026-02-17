# Agentic Data Pipeline Ingestor

Enterprise-grade agentic data pipeline for document ingestion with intelligent content routing, dual parsing strategy, and destination-agnostic output.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🔌 **Universal File Support**: PDF, Office documents, images, archives
- 🤖 **Agentic Processing**: AI-driven decision making for parser selection
- 🔍 **Intelligent Content Detection**: Automatic scanned vs. text-based detection
- 🔄 **Dual Parsing Strategy**: Docling primary + Azure OCR fallback
- 🎯 **Destination-Agnostic**: Pluggable output system (Cognee, GraphRAG, webhooks)
- 🧠 **LLM-Agnostic**: Azure OpenAI + OpenRouter fallback via litellm
- 📊 **20GB/day Throughput**: Near-realtime + batch processing
- 🔒 **Enterprise Security**: RBAC, audit logging, data lineage
- 📡 **API-First**: OpenAPI 3.1 with auto-generated SDKs
- 🔍 **Observability**: OpenTelemetry + Prometheus metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    API LAYER (OpenAPI 3.1)                       │
├─────────────────────────────────────────────────────────────────┤
│                    CORE ORCHESTRATION ENGINE                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              AGENTIC DECISION ENGINE                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              7-STAGE PROCESSING PIPELINE                  │   │
│  │  Ingest → Detect → Parse → Enrich → Quality → Transform → │   │
│  │  Output                                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
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

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for local development)
- PostgreSQL 17+
- Redis 7+

### Installation

```bash
# Clone the repository
git clone https://github.com/example/agentic-pipeline-ingestor.git
cd agentic-pipeline-ingestor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,docling,azure]"

# Copy environment file
cp .env.example .env
# Edit .env with your configuration
```

### Local Development with Docker

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f api

# Stop all services
docker-compose -f docker/docker-compose.yml down
```

### Running the API

```bash
# Using uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Or using the CLI
pipeline-ingestor
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- OpenAPI Spec: http://localhost:8000/api/v1/openapi.yaml

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `false` |
| `DB_URL` | PostgreSQL connection URL | `postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | - |
| `AZURE_OPENAI_API_BASE` | Azure OpenAI endpoint | - |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `SECRET_KEY` | JWT signing key | - |

### LLM Configuration

Edit `config/llm.yaml` to configure LLM providers:

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

## API Usage

### Submit a Job

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "source_type": "upload",
    "source_uri": "/uploads/document.pdf",
    "file_name": "document.pdf",
    "mode": "async"
  }'
```

### Upload a File

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -H "X-API-Key: your-api-key" \
  -F "files=@document.pdf" \
  -F "priority=5"
```

### Check Job Status

```bash
curl http://localhost:8000/api/v1/jobs/{job_id} \
  -H "X-API-Key: your-api-key"
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m contract

# Run contract tests only
pytest tests/contract/
```

## SDK Generation

Generate client SDKs from the OpenAPI specification:

```bash
# Install OpenAPI Generator
npm install -g @openapitools/openapi-generator-cli

# Generate all SDKs
python sdks/generate.py

# Generate specific SDK
python sdks/generate.py --language python
python sdks/generate.py --language typescript
```

## Project Structure

```
agentic-pipeline-ingestor/
├── api/
│   └── openapi.yaml              # OpenAPI 3.1 specification
├── src/
│   ├── api/                      # API layer (routes, models)
│   ├── core/                     # Core orchestration engine
│   ├── db/                       # SQLAlchemy models
│   ├── llm/                      # LLM abstraction (litellm)
│   └── plugins/                  # Plugin system
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── contract/                 # API contract tests
├── docker/                       # Docker configuration
├── config/                       # Configuration files
├── sdks/                         # Auto-generated SDKs
└── pyproject.toml                # Python dependencies
```

## Implementation Phases

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| Phase 1 | Weeks 1-4 | Foundation: OpenAPI, FastAPI, Plugin System, LLM Abstraction |
| Phase 2 | Weeks 5-9 | Core Pipeline: 7 Stages, Content Detection, Parsers |
| Phase 3 | Weeks 9-12 | Agentic Features: Decision Engine, Retry, DLQ |
| Phase 4 | Weeks 13-16 | Enterprise: Auth, Audit, Lineage |
| Phase 5 | Weeks 17-20 | Observability & Scale |
| Phase 6 | Weeks 21-24 | Advanced Features |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: https://docs.pipeline.example.com
- Issues: https://github.com/example/agentic-pipeline-ingestor/issues
- Email: api-support@example.com

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [litellm](https://github.com/BerriAI/litellm) - LLM abstraction
- [Docling](https://github.com/docling/docling) - Document parsing
- [OpenAPI Generator](https://openapi-generator.tech/) - SDK generation
