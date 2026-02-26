# Agentic Data Pipeline Ingestor

Enterprise-grade agentic data pipeline for document ingestion with intelligent content routing, dual parsing strategy, destination-agnostic output, and pgvector-powered semantic search.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17+-336791.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-enabled-00A896.svg)](https://github.com/pgvector/pgvector)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🔌 **Universal File Support**: PDF, Office documents, images, archives
- 🤖 **Agentic Processing**: AI-driven decision making for parser selection
- 🔍 **Intelligent Content Detection**: Automatic scanned vs. text-based detection
- 🧠 **Semantic Search**: pgvector-powered vector similarity search (NEW)
- 🔎 **Hybrid Search**: Combined vector + full-text search with fusion ranking (NEW)
- 📝 **Document Chunking with Embeddings**: Automatic text segmentation and embedding generation (NEW)
- 🔄 **Dual Parsing Strategy**: Docling primary + Azure OCR fallback
- 🎯 **Destination-Agnostic**: Pluggable output system (Cognee, GraphRAG, webhooks)
- 🧠 **LLM-Agnostic**: Azure OpenAI + OpenRouter fallback via litellm
- 📊 **20GB/day Throughput**: Near-realtime + batch processing
- 🔒 **Enterprise Security**: RBAC, audit logging, data lineage
- 📡 **API-First**: OpenAPI 3.1 with 50+ endpoints and auto-generated SDKs
- 🔍 **Observability**: OpenTelemetry + Prometheus metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    API LAYER (OpenAPI 3.1)                       │
│  Jobs │ Upload │ Search │ Chunks │ Auth │ Audit │ Health         │
├─────────────────────────────────────────────────────────────────┤
│                    CORE ORCHESTRATION ENGINE                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              AGENTIC DECISION ENGINE                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              7-STAGE PROCESSING PIPELINE                  │   │
│  │  Ingest → Detect → Parse → Chunk → Embed → Quality →      │   │
│  │  Output                                                       │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────────────────┐  ┌──────────────────────────────┐   │
│  │    SEARCH SERVICES     │  │      LLM ADAPTER (litellm)   │   │
│  │  VectorSearchService   │  │  ┌──────────┐ ┌──────────┐  │   │
│  │  TextSearchService     │  │  │ Azure    │ │ OpenRouter│  │   │
│  │  HybridSearchService   │  │  │ GPT-4    │ │ Claude-3  │  │   │
│  │  EmbeddingService      │  │  └──────────┘ └──────────┘  │   │
│  └────────────────────────┘  └──────────────────────────────┘   │
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
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER (PostgreSQL)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    Jobs     │  │   Chunks    │  │      pgvector           │  │
│  │   Tables    │  │   + VECTOR  │  │   HNSW Indexes          │  │
│  │             │  │  Embeddings │  │   Similarity Search     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for local development)
- PostgreSQL 17+ with pgvector extension
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

### Setup pgvector Extension

```bash
# Connect to PostgreSQL
psql -U postgres -d pipeline

# Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

# Verify installation
SELECT * FROM pg_extension WHERE extname IN ('vector', 'pg_trgm');
```

### Local Development with Docker

```bash
# Start all services (includes pgvector-enabled PostgreSQL)
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

## Vector Search Quick Start

### 1. Semantic Search (Vector Similarity)

Find documents semantically similar to a query embedding:

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query_embedding": [0.023, -0.045, 0.128, 0.091, -0.034, ...],
    "top_k": 10,
    "min_similarity": 0.7,
    "filters": {
      "job_id": "123e4567-e89b-12d3-a456-426614174000"
    }
  }'
```

**Example Response:**
```json
{
  "results": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 0,
      "content": "Machine learning is a subset of artificial intelligence...",
      "metadata": {"page": 1, "source": "ml-overview.pdf"},
      "similarity_score": 0.9234,
      "rank": 1
    }
  ],
  "total": 15,
  "query_time_ms": 25.5
}
```

### 2. Semantic Search (Text Query)

Find documents semantically similar to a natural-language query. The API generates
the query embedding server-side:

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic/text" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "machine learning architecture",
    "top_k": 10,
    "min_similarity": 0.7,
    "filters": {
      "job_id": "123e4567-e89b-12d3-a456-426614174000"
    }
  }'
```

### 3. Text Search (Full-Text + Fuzzy)

Search using PostgreSQL full-text search with BM25 ranking:

```bash
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "machine learning architecture",
    "top_k": 10,
    "use_fuzzy": true,
    "highlight": true
  }'
```

**Example Response:**
```json
{
  "results": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
      "content": "Neural network architecture is fundamental to deep learning...",
      "metadata": {"page": 5},
      "similarity_score": 0.8567,
      "rank": 1,
      "highlighted_content": "Neural <mark>network architecture</mark> is fundamental...",
      "matched_terms": ["network", "architecture"]
    }
  ],
  "total": 12,
  "query_time_ms": 15.3
}
```

### 4. Hybrid Search (Vector + Text Fusion)

Combine semantic and text search for best results:

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "neural network architecture in healthcare",
    "top_k": 10,
    "vector_weight": 0.7,
    "text_weight": 0.3,
    "fusion_method": "weighted_sum",
    "min_similarity": 0.5
  }'
```

**Example Response:**
```json
{
  "results": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440002",
      "content": "Neural networks are transforming healthcare diagnostics...",
      "metadata": {"page": 12, "category": "healthcare"},
      "hybrid_score": 0.8856,
      "vector_score": 0.9234,
      "text_score": 0.7834,
      "vector_rank": 1,
      "text_rank": 3,
      "rank": 1,
      "fusion_method": "weighted_sum"
    }
  ],
  "total": 18,
  "query_time_ms": 45.2
}
```

### 5. List Document Chunks

Retrieve chunks for a processed document:

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}/chunks?limit=10&offset=0" \
  -H "X-API-Key: your-api-key"
```

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
| `VECTOR_STORE_ENABLED` | Enable vector store | `true` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `EMBEDDING_DIMENSIONS` | Embedding dimensions | `1536` |

### Vector Store Configuration

Edit `config/vector_store.yaml` to configure vector search:

```yaml
vector_store:
  enabled: true
  
  embedding:
    model: "text-embedding-3-small"
    dimensions: 1536
    batch_size: 100
    
  search:
    default_top_k: 10
    max_top_k: 100
    default_min_similarity: 0.7
    
  index:
    hnsw_m: 16
    hnsw_ef_construction: 64
    hnsw_ef_search: 32
    
  hybrid:
    default_vector_weight: 0.7
    default_text_weight: 0.3
    rrf_k: 60
    
  pipeline:
    auto_generate_embeddings: true
    chunking_strategy: "semantic"
    chunk_size: 1000
    chunk_overlap: 200
```

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

### Vector Store Health Check

```bash
curl http://localhost:8000/health/vector
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
│   ├── api/                      # API layer (routes, models, validators)
│   ├── core/                     # Core orchestration engine
│   ├── db/                       # SQLAlchemy models and repositories
│   ├── services/                 # Search services (NEW: vector, text, hybrid)
│   ├── vector_store_config/      # Vector store configuration (NEW)
│   ├── auth/                     # Authentication and authorization
│   ├── observability/            # Logging, metrics, tracing
│   ├── plugins/                  # Plugin system
│   ├── audit/                    # Audit logging
│   ├── lineage/                  # Data lineage tracking
│   ├── llm/                      # LLM abstraction layer
│   ├── worker/                   # Background job processor
│   └── retention/                # Data retention management
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── e2e/                      # End-to-end tests
│   ├── contract/                 # API contract tests
│   ├── performance/              # Performance tests
│   └── functional/               # Functional tests
├── docker/                       # Docker configuration
├── config/                       # Configuration files
├── sdks/                         # Auto-generated SDKs
├── docs/                         # Documentation
└── pyproject.toml                # Python dependencies
```

## Technology Stack

| Component | Technology | Count |
|-----------|------------|-------|
| Language | Python 3.11+ | - |
| Web Framework | FastAPI | - |
| Database | PostgreSQL 17+ with pgvector | - |
| Cache | Redis 7+ | - |
| LLM Router | litellm | - |
| Source Files | Python modules | 118 |
| Test Files | Test suites | 70 |
| API Endpoints | REST endpoints | 50 |

## Implementation Phases

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| Phase 1 | Weeks 1-4 | Foundation: OpenAPI, FastAPI, Plugin System, LLM Abstraction |
| Phase 2 | Weeks 5-9 | Core Pipeline: 7 Stages, Content Detection, Parsers |
| Phase 3 | Weeks 9-12 | Agentic Features: Decision Engine, Retry, DLQ |
| Phase 4 | Weeks 13-16 | Enterprise: Auth, Audit, Lineage |
| Phase 5 | Weeks 17-20 | **Vector Search: pgvector, Semantic/Hybrid Search, Embeddings** |
| Phase 6 | Weeks 21-24 | Observability & Scale |

## Documentation

- [Vector Store API](docs/vector_store_api.md) - Complete vector search API reference
- [Vector Store Usage](docs/VECTOR_STORE_API_USAGE.md) - Usage examples and best practices
- [API Documentation](http://localhost:8000/docs) - Interactive OpenAPI docs (when running)

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
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search
- [OpenAPI Generator](https://openapi-generator.tech/) - SDK generation
