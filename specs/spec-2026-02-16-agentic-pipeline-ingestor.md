# Spec: Agentic Data Pipeline Ingestor

**Status:** Draft  
**Created:** 2026-02-16  
**Author:** SDD Agent  
**Spec ID:** spec-2026-02-16-agentic-pipeline-ingestor  
**Version:** 1.0

---

## 1. Overview

### 1.1 Vision Statement

Build a next-generation, agentic data pipeline system that serves as the intelligent backbone for enterprise document ingestion. The system will autonomously handle diverse data sources, intelligently route content through optimal processing paths, and deliver high-quality, enriched data to destination-agnostic stores.

### 1.2 Problem Statement

Enterprises face critical challenges in document processing pipelines:

- **Fragmented Processing**: Different file types require different tools (PDF parsers, OCR engines, Excel handlers) with no unified orchestration
- **Scanned vs. Text Detection**: No reliable mechanism to detect scanned PDFs vs. text-based PDFs, leading to suboptimal processing choices
- **Quality Uncertainty**: No systematic quality assessment or automatic retry strategies when processing fails
- **Integration Complexity**: Each new data source or destination requires custom integration code
- **Lack of Observability**: Limited visibility into processing decisions, especially for AI/LLM-based operations
- **Compliance Gaps**: Insufficient audit trails and data lineage for enterprise compliance requirements

### 1.3 Solution Summary

The Agentic Data Pipeline Ingestor provides:

1. **Universal Connector Framework**: Plugin-based architecture supporting all enterprise file types and sources
2. **Intelligent Content Routing**: Automatic detection of content type (scanned vs. text) with optimal parser selection
3. **Agentic Processing Engine**: AI-driven decision making for processing strategy, quality gates, and self-healing
4. **Destination-Agnostic Output**: Pluggable destination system starting with Cognee, extensible to GraphRAG, Neo4j, vector stores
5. **LLM-Agnostic Architecture**: Chat Completions API standard for seamless provider switching (OpenAI, Azure, Anthropic, Local models)
6. **Enterprise-Grade Features**: Comprehensive auth/authz, audit logging, data lineage, and retention policies
7. **Cloud-Native Observability**: OpenTelemetry with GenAI-specific instrumentation for complete visibility

### 1.4 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Daily Throughput | 20GB/day | Bytes processed per 24h window |
| Near-Realtime Processing | 10% of jobs | Jobs completed < 30s from trigger |
| Batch Processing | 90% of jobs | Jobs completed via scheduled queues |
| OCR Accuracy (Docling) | > 95% | Correctly extracted text / Total text |
| OCR Accuracy (Azure Fallback) | > 90% | Correctly extracted text / Total text |
| Content Type Detection | > 98% | Correct routing decisions / Total |
| API Response Time (p99) | < 2s | Sync ingest API response time |
| Job Success Rate | > 99.5% | Successful jobs / Total jobs |
| Mean Time to Recovery | < 5 min | Auto-retry recovery for transient failures |
| Audit Log Completeness | 100% | All operations logged with required fields |

---

## 2. Goals & Non-Goals

### 2.1 Goals

- [ ] **G1**: Support all enterprise file types (PDF, Office, images, archives)
- [ ] **G2**: Plugin-based architecture for sources, parsers, and destinations
- [ ] **G3**: Intelligent content detection (scanned vs. text-based PDFs)
- [ ] **G4**: Dual parsing strategy (Docling primary, Azure OCR fallback)
- [ ] **G5**: Agentic decision engine for processing strategy selection
- [ ] **G6**: Quality gates with configurable thresholds and auto-retry
- [ ] **G7**: Destination-agnostic output with Cognee as first implementation
- [ ] **G8**: 20GB/day throughput with 10/90 near-realtime/batch split
- [ ] **G9**: Comprehensive authentication and authorization framework
- [ ] **G10**: Full audit logging with data lineage tracking
- [ ] **G11**: OpenTelemetry instrumentation with GenAI spans
- [ ] **G12**: Docker local dev and Azure production deployment
- [ ] **G13**: Dead letter queue for failed items with manual retry capability
- [ ] **G14**: **API-First Architecture** - Complete OpenAPI 3.1 specification driving all implementation
- [ ] **G15**: **Auto-Generated SDKs** - Python and TypeScript SDKs generated from OpenAPI spec
- [ ] **G16**: **Contract Testing** - API contracts validated independently of implementation
- [ ] **G17**: **LLM-Agnostic** - Chat Completions API standard supporting multiple providers (OpenAI, Azure, Anthropic, local models)
- [ ] **G18**: **Provider Hot-Swapping** - Change LLM provider via configuration without code changes

### 2.2 Non-Goals (Explicitly Out of Scope)

- **NG1**: Multi-tenancy (system is single-tenant by design)
- **NG2**: Built-in vector database (uses external destinations like Cognee)
- **NG3**: Custom model training (uses pre-trained Docling/Azure models)
- **NG4**: Real-time collaborative editing (read-only document processing)
- **NG5**: Mobile application (web API only)
- **NG6**: On-premise deployment (cloud-native, Azure-focused)
- **NG7**: Automatic data classification/sensitivity detection (phase 2)
- **NG8**: Content-based access control (handled by source systems)
- **NG9**: Document versioning (processes current version only)
- **NG10**: Built-in workflow builder (programmatic pipeline configuration only)

---

## 3. Architecture Design

### 3.1 System Architecture Overview (API-First)

The system follows a **modular, API-First architecture** with clear separation of concerns. The API layer is the primary contract that drives all implementation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    API LAYER (Contract)                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  OpenAPI Spec → Mock Server → FastAPI → Auto-Generated     │ │
│  │  Documentation → Client SDKs → Contract Tests              │ │
│  └────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    API GATEWAY                                   │
│         Auth, Rate Limiting, Load Balancing, CORS                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CORE ORCHESTRATION ENGINE                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              AGENTIC DECISION ENGINE                      │   │
│  │   Router │ Planner │ Monitor │ Healer                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              7-STAGE PROCESSING PIPELINE                  │   │
│  │  Ingest → Detect → Parse → Enrich → Quality → Transform → │   │
│  │  Output                                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           LLM ADAPTER (Chat Completions API)              │   │
│  │   ┌──────────────────────────────────────────────────┐   │   │
│  │   │              litellm Router                       │   │   │
│  │   │  ┌──────────────┐  ┌──────────────────────────┐  │   │   │
│  │   │  │ Azure GPT-4  │  │ OpenRouter Claude-3      │  │   │   │
│  │   │  │  (Primary)   │  │   (Fallback)             │  │   │   │
│  │   │  └──────────────┘  └──────────────────────────┘  │   │   │
│  │   │  Automatic Fallback Chain + Retry Logic          │   │   │
│  │   └──────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PLUGIN ECOSYSTEM                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Sources   │  │   Parsers   │  │      Destinations       │ │
│  │  S3, Blob   │  │  Docling    │  │      Cognee             │ │
│  │  SharePoint │  │  Azure OCR  │  │      GraphRAG           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**API-First Architecture Principles:**
1. **OpenAPI is Source of Truth**: All APIs defined in `/api/openapi.yaml` before implementation
2. **Contract-Driven Development**: API spec drives FastAPI routes, Pydantic models, and client SDKs
3. **Consumer-First Design**: APIs designed for clients (web, CLI, integrations) not internal convenience
4. **Version Stability**: API versions maintained for 12+ months after deprecation notice
5. **Documentation as Code**: OpenAPI spec generates interactive docs, SDKs, and tests

**LLM-Agnostic Principles:**
1. **Chat Completions API Standard**: Universal interface for all LLM providers
2. **Provider Abstraction**: Single adapter layer supports multiple backends
3. **Configuration-Driven**: Switch providers via config (env vars, not code changes)
4. **Fallback Chain**: Primary → Secondary → Local model fallback

### 3.2 Component Breakdown

| Component | Responsibility | Technology |
|-----------|----------------|------------|
| API Gateway | Request routing, auth, rate limiting | FastAPI + NGINX |
| Orchestration Engine | Job orchestration, state management | Python asyncio |
| Agentic Decision Engine | Intelligent routing, retry logic | Rule engine + ML |
| Pipeline Engine | Stage execution, error handling | Async workers |
| **LLM Adapter** | **Multi-provider LLM abstraction** | **litellm** |
| Plugin Registry | Plugin discovery and lifecycle | Python ABC |
| Source Plugins | Data source connectivity | Pluggable |
| Parser Plugins | Content extraction | Docling, Azure OCR |
| Destination Plugins | Output routing | Cognee, etc. |
| Job Queue | Async job distribution | Azure Queue |
| Metadata Store | Job and configuration data | PostgreSQL |
| Blob Storage | File storage | Azure Blob |
| Audit Store | Audit logs | OpenSearch |
| Cache | Session and rate limiting | Redis |

### 3.3 Data Flow

1. **Ingestion**: File uploaded or pulled from source
2. **Validation**: File format, size, security checks
3. **Staging**: Store in blob storage, create job record
4. **Content Detection**: Analyze file to determine content type
5. **Parser Selection**: Agent selects optimal parser(s)
6. **Parsing**: Extract text and structure
7. **Enrichment**: Add metadata, extract entities
8. **Quality Assessment**: Validate extraction quality
9. **Transformation**: Chunk, generate embeddings
10. **Output**: Route to configured destinations
11. **Audit**: Log all operations

---

## 4. Data Models

### 4.1 Core Entities

#### Job Entity

```python
class Job(BaseModel):
    id: UUID
    external_id: Optional[str]
    source_type: str
    source_uri: str
    file_name: str
    file_size: int
    file_hash: str
    mime_type: str
    mode: ProcessingMode  # sync or async
    priority: int  # 1-10
    pipeline_config: PipelineConfig
    destinations: List[DestinationConfig]
    status: JobStatus
    current_stage: Optional[str]
    stage_progress: Dict[str, StageProgress]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    expires_at: Optional[datetime]
    result: Optional[JobResult]
    error: Optional[JobError]
    retry_count: int
    retry_history: List[RetryRecord]
    created_by: str
    source_ip: Optional[str]
```

#### Pipeline Configuration

```python
class PipelineConfig(BaseModel):
    content_detection: ContentDetectionConfig
    parser: ParserConfig
    enrichment: EnrichmentConfig
    quality: QualityConfig
    transformation: TransformationConfig
    enabled_stages: List[str]

class ParserConfig(BaseModel):
    primary_parser: str = "docling"
    fallback_parser: Optional[str] = "azure_ocr"
    parser_options: Dict[str, Any]
```

### 4.2 Content Detection Result

```python
class ContentDetectionResult(BaseModel):
    detected_type: ContentType
    confidence: float
    detection_method: str
    page_count: Optional[int]
    has_text_layer: bool
    has_images: bool
    image_count: int
    text_statistics: TextStatistics
    recommended_parser: str
    alternative_parsers: List[str]
    preprocessing_required: bool
```

### 4.3 Database Schema

**Key Tables:**
- `jobs`: Job metadata and status
- `audit_logs`: Audit trail (partitioned by month)
- `data_lineage`: Processing lineage
- `source_credentials`: Encrypted source credentials
- `plugin_configs`: Plugin configurations

See full schema in Section 4.6 of detailed spec.

---

## 5. API Design (API-First Architecture)

### 5.1 API-First Design Philosophy

This system follows **API-First** principles where:
1. **API is the Contract**: The API specification is the single source of truth
2. **Design Before Implementation**: API is designed and documented before any backend code
3. **Consumer-Centric**: APIs are designed for consumers (clients, plugins, integrations)
4. **OpenAPI Specification**: Complete OpenAPI 3.1 spec drives implementation
5. **Contract Testing**: API contracts are tested independently of implementation

**API Specification Location**: `/api/openapi.yaml` (OpenAPI 3.1)
**API Documentation**: Auto-generated from OpenAPI spec at `/docs`

### 5.2 API Architecture Principles

| Principle | Implementation |
|-----------|----------------|
| **Resource-Oriented** | Nouns for resources (`/jobs`, `/sources`), verbs via HTTP methods |
| **Versioning** | URL path versioning (`/api/v1/`, `/api/v2/`) |
| **Consistency** | Standard request/response formats across all endpoints |
| **Discoverability** | HATEOAS links in responses, OpenAPI spec available |
| **Stateless** | No server-side session state; JWT or API keys for auth |
| **Idempotency** | Safe retries via `Idempotency-Key` header for POST/PUT |

### 5.3 OpenAPI Specification Structure

```yaml
# /api/openapi.yaml - Core API Contract
openapi: 3.1.0
info:
  title: Agentic Data Pipeline Ingestor API
  version: 1.0.0
  description: |
    Enterprise-grade agentic data pipeline for document ingestion.
    
    ## Features
    - Universal file type support
    - Intelligent parser selection (Docling + Azure OCR)
    - Destination-agnostic output
    - Real-time and batch processing
    - Comprehensive audit and lineage
    
    ## Authentication
    - API Key: `X-API-Key` header
    - OAuth2: `Authorization: Bearer <token>`
    - Azure AD: Enterprise SSO

servers:
  - url: https://api.pipeline.example.com
    description: Production
  - url: http://localhost:8000
    description: Local Development

paths:
  /api/v1/jobs:
    post:
      operationId: createJob
      summary: Submit a new ingestion job
      requestBody:
        $ref: '#/components/requestBodies/JobCreate'
      responses:
        '202':
          $ref: '#/components/responses/JobAccepted'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
    get:
      operationId: listJobs
      summary: List jobs with filtering
      parameters:
        - $ref: '#/components/parameters/PageParam'
        - $ref: '#/components/parameters/LimitParam'
        - $ref: '#/components/parameters/StatusFilter'
        - $ref: '#/components/parameters/DateRangeFilter'
      responses:
        '200':
          $ref: '#/components/responses/JobList'

  # ... additional paths

components:
  schemas:
    Job:
      type: object
      required: [id, source_type, file_name, status]
      properties:
        id:
          type: string
          format: uuid
          example: "550e8400-e29b-41d4-a716-446655440000"
        source_type:
          type: string
          enum: [upload, url, s3, azure_blob, sharepoint]
        file_name:
          type: string
          example: "document.pdf"
        status:
          $ref: '#/components/schemas/JobStatus'
        # ... additional properties
```

### 5.4 REST API Endpoints

#### Job Management

| Endpoint | Method | Description | OpenAPI Operation |
|----------|--------|-------------|-------------------|
| `/api/v1/jobs` | POST | Submit new job | `createJob` |
| `/api/v1/jobs` | GET | List jobs with filters | `listJobs` |
| `/api/v1/jobs/{id}` | GET | Get job details | `getJob` |
| `/api/v1/jobs/{id}` | DELETE | Cancel job | `cancelJob` |
| `/api/v1/jobs/{id}/retry` | POST | Retry failed job | `retryJob` |
| `/api/v1/jobs/{id}/result` | GET | Get result | `getJobResult` |
| `/api/v1/jobs/{id}/events` | GET | SSE stream of events | `streamJobEvents` |

#### File Upload

| Endpoint | Method | Description | OpenAPI Operation |
|----------|--------|-------------|-------------------|
| `/api/v1/upload` | POST | Upload file(s) multipart/form-data | `uploadFiles` |
| `/api/v1/upload/url` | POST | Ingest from URL | `ingestFromUrl` |
| `/api/v1/upload/stream` | POST | Stream upload for large files | `streamUpload` |

#### Pipeline Configuration

| Endpoint | Method | Description | OpenAPI Operation |
|----------|--------|-------------|-------------------|
| `/api/v1/pipelines` | GET | List pipeline configurations | `listPipelines` |
| `/api/v1/pipelines` | POST | Create pipeline config | `createPipeline` |
| `/api/v1/pipelines/{id}` | GET/PUT/DELETE | CRUD operations | `get/update/deletePipeline` |
| `/api/v1/pipelines/{id}/validate` | POST | Validate config | `validatePipeline` |

#### Sources & Destinations

| Endpoint | Method | Description | OpenAPI Operation |
|----------|--------|-------------|-------------------|
| `/api/v1/sources` | GET | List source plugins | `listSources` |
| `/api/v1/sources` | POST | Register/configure source | `createSource` |
| `/api/v1/sources/{id}/test` | POST | Test source connection | `testSource` |
| `/api/v1/destinations` | GET | List destination plugins | `listDestinations` |
| `/api/v1/destinations` | POST | Register/configure destination | `createDestination` |
| `/api/v1/destinations/{id}/test` | POST | Test destination connection | `testDestination` |

#### Audit & Compliance

| Endpoint | Method | Description | OpenAPI Operation |
|----------|--------|-------------|-------------------|
| `/api/v1/audit/logs` | GET | Query audit logs | `queryAuditLogs` |
| `/api/v1/audit/lineage/{job_id}` | GET | Get data lineage | `getLineage` |
| `/api/v1/audit/export` | POST | Export audit data | `exportAudit` |

#### System & Health

| Endpoint | Method | Description | OpenAPI Operation |
|----------|--------|-------------|-------------------|
| `/api/v1/health` | GET | Comprehensive health status | `getHealth` |
| `/api/v1/health/ready` | GET | K8s readiness probe | `getReadiness` |
| `/api/v1/health/live` | GET | K8s liveness probe | `getLiveness` |
| `/api/v1/metrics` | GET | Prometheus metrics | `getMetrics` |
| `/api/v1/openapi.yaml` | GET | OpenAPI specification | `getOpenApiSpec` |

### 5.5 Async APIs

#### Server-Sent Events (SSE)

```
GET /api/v1/jobs/{id}/events
Content-Type: text/event-stream

# Events:
event: created
data: {"job_id": "...", "status": "created", "timestamp": "..."}

event: stage_started
data: {"job_id": "...", "stage": "parse", "timestamp": "..."}

event: stage_progress
data: {"job_id": "...", "stage": "parse", "progress": 45, "timestamp": "..."}

event: stage_completed
data: {"job_id": "...", "stage": "parse", "result": {...}, "timestamp": "..."}

event: completed
data: {"job_id": "...", "status": "completed", "result_url": "...", "timestamp": "..."}

event: error
data: {"job_id": "...", "error": {...}, "timestamp": "..."}
```

#### Webhooks

```yaml
POST {configured_webhook_url}
Content-Type: application/json
X-Pipeline-Signature: sha256=<hmac>
X-Event-Type: job.completed

{
  "event": "job.completed",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result_url": "/api/v1/jobs/550e.../result",
  "processed_at": "2026-02-16T12:00:00Z",
  "timestamp": "2026-02-16T12:00:01Z"
}
```

### 5.6 Request/Response Standards

#### Standard Request Headers

```
X-API-Key: <api_key>           # or Authorization: Bearer <token>
X-Idempotency-Key: <uuid>      # For safe retries
X-Request-ID: <uuid>           # Request tracing
Content-Type: application/json
Accept: application/json
```

#### Standard Response Structure

```json
{
  "data": { ... },           # Resource data or null for errors
  "meta": {
    "request_id": "uuid",
    "timestamp": "2026-02-16T12:00:00Z",
    "api_version": "v1"
  },
  "links": {
    "self": "/api/v1/jobs/550e...",
    "next": "/api/v1/jobs?page=2",
    "prev": null
  }
}
```

#### Error Response Structure

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid file format",
    "details": [
      {
        "field": "file",
        "code": "UNSUPPORTED_FORMAT",
        "message": "File type .xyz is not supported"
      }
    ],
    "documentation_url": "https://docs.pipeline.example.com/errors/VALIDATION_ERROR"
  },
  "meta": {
    "request_id": "uuid",
    "timestamp": "2026-02-16T12:00:00Z"
  }
}
```

### 5.7 API Versioning Strategy

| Version | Status | Base Path | Description |
|---------|--------|-----------|-------------|
| v1 | Current | `/api/v1/` | Initial stable release |
| v2 | Planned | `/api/v2/` | Breaking changes (1 year+) |

**Deprecation Policy**:
- 6-month notice before deprecation
- 12-month support after deprecation
- Sunset header in responses: `Sunset: Sat, 31 Dec 2026 23:59:59 GMT`

### 5.8 SDK & Client Generation

From OpenAPI spec, generate:
- **Python SDK**: `pipeline-ingestor-client`
- **TypeScript SDK**: For frontend integrations
- **CLI Tool**: Command-line interface
- **Postman Collection**: For testing

### 5.9 API Security

| Security Layer | Implementation |
|----------------|----------------|
| Authentication | API Keys, OAuth2, Azure AD |
| Authorization | RBAC via scopes in JWT |
| Rate Limiting | Per-key limits: 100 req/min default |
| Input Validation | JSON Schema validation (Pydantic) |
| Output Sanitization | Automatic PII redaction in logs |
| HTTPS Only | TLS 1.3 required in production |
| CORS | Configurable allowed origins |

### 5.10 API-Driven Development Workflow

```
1. Define API Contract (OpenAPI YAML)
        ↓
2. Generate Mock Server from OpenAPI
        ↓
3. Client Development (can start immediately)
        ↓
4. Contract Testing (validate against OpenAPI)
        ↓
5. Backend Implementation (FastAPI from OpenAPI)
        ↓
6. Integration Testing
        ↓
7. Deploy API + Documentation
```

---

## 6. Processing Pipeline

### 6.1 Stage Overview

| Stage | Purpose | Key Actions |
|-------|---------|-------------|
| 1. Ingest | Receive & validate | Auth, validation, staging |
| 2. Detect | Content analysis | PDF analysis, type detection |
| 3. Parse | Text extraction | Docling/Azure OCR |
| 4. Enrich | Add metadata | Entities, classification |
| 5. Quality | Validate quality | Scoring, threshold checks |
| 6. Transform | Prepare output | Chunking, embeddings |
| 7. Output | Route to destinations | Format, write, confirm |

### 6.2 Content Detection Algorithm

For PDFs, the system analyzes:
- Text-to-image ratio per page
- Text layer presence
- Image characteristics
- Font information

Decision matrix:
- Text ratio > 95% → Text-based PDF → Docling
- Text ratio < 5%, images > 90% → Scanned PDF → Azure OCR
- Mixed → Docling with OCR fallback

### 6.3 Parser Selection Logic

```python
if content_type == TEXT_BASED_PDF:
    primary = "docling"
    fallback = "azure_ocr"
elif content_type == SCANNED_PDF:
    primary = "azure_ocr"
    fallback = "docling"
elif content_type == OFFICE_DOC:
    primary = "docling"
    fallback = None
```

---

## 6.5 LLM Provider Abstraction (Chat Completions API)

### 6.5.1 LLM-Agnostic Design

The system uses the **Chat Completions API** (OpenAI standard) as the universal interface for all LLM operations, enabling seamless provider switching without code changes.

### 6.5.2 Architecture: litellm Proxy Library (Selected)

**Decision**: Use **litellm** as the LLM abstraction library.

**Rationale**:
- Handles 100+ providers with unified interface
- Automatic fallback and routing support
- Built-in retry, rate limiting, and caching
- Active community maintenance
- OpenAI-compatible proxy mode

```python
# Using litellm for automatic multi-provider support
import litellm
from litellm import acompletion

class LLMProvider:
    def __init__(self, config: LLMConfig):
        self.config = config
        # litellm automatically routes to correct provider
        litellm.set_verbose = config.debug
        
    async def chat_completion(
        self, 
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None  # Override model per-request
    ) -> ChatCompletionResponse:
        # litellm handles provider-specific differences internally
        # Supports: azure/gpt-4, openrouter/anthropic/claude-3, etc.
        response = await acompletion(
            model=model or self.config.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.config.api_key,
            api_base=self.config.api_base,
            # Automatic fallback configured at litellm proxy level
        )
        return ChatCompletionResponse.from_litellm(response)
```

**Alternative: Custom HTTP Proxy Adapter** (if litellm doesn't support specific needs)
```python
# Custom implementation with HTTP-level adaptation
class ChatCompletionsAdapter(ABC):
    """Base adapter for Chat Completions API standard"""
    
    @abstractmethod
    async def create_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        pass

class OpenAIAdapter(ChatCompletionsAdapter):
    """Direct OpenAI API"""
    async def create_completion(self, request):
        # Native OpenAI format
        response = await self.http_client.post(
            "https://api.openai.com/v1/chat/completions",
            json=request.to_openai_dict()
        )
        return ChatCompletionResponse.from_openai(response.json())

class AzureOpenAIAdapter(ChatCompletionsAdapter):
    """Azure OpenAI Service"""
    async def create_completion(self, request):
        # Azure-specific endpoint format
        url = f"{self.base_url}/openai/deployments/{self.deployment}/chat/completions"
        response = await self.http_client.post(
            url,
            headers={"api-key": self.api_key},
            json=request.to_azure_dict()
        )
        return ChatCompletionResponse.from_azure(response.json())

class AnthropicAdapter(ChatCompletionsAdapter):
    """Anthropic Claude (adapted to Chat Completions format)"""
    async def create_completion(self, request):
        # Anthropic uses different format - we adapt
        anthropic_request = request.to_anthropic_dict()
        response = await self.http_client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key},
            json=anthropic_request
        )
        # Normalize back to Chat Completions format
        return ChatCompletionResponse.from_anthropic(response.json())

class OllamaAdapter(ChatCompletionsAdapter):
    """Local models via Ollama"""
    async def create_completion(self, request):
        response = await self.http_client.post(
            f"{self.base_url}/api/chat",
            json=request.to_ollama_dict()
        )
        return ChatCompletionResponse.from_ollama(response.json())

# Factory for provider selection
class LLMProviderFactory:
    ADAPTERS = {
        "openai": OpenAIAdapter,
        "azure_openai": AzureOpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "ollama": OllamaAdapter,
        "custom": CustomHTTPAdapter,  # For any OpenAI-compatible endpoint
    }
    
    @classmethod
    def create(cls, provider: str, config: Dict) -> ChatCompletionsAdapter:
        adapter_class = cls.ADAPTERS.get(provider)
        if not adapter_class:
            raise ValueError(f"Unknown provider: {provider}")
        return adapter_class(**config)
```

### 6.5.3 Configuration (Azure + OpenRouter + Fallback Chains)

```yaml
# config/llm.yaml
llm:
  # litellm router configuration for multi-provider with fallback
  router:
    # Primary model group for agentic decisions
    - model_name: "agentic-decisions"
      litellm_params:
        model: "azure/gpt-4"
        api_base: "https://myresource.openai.azure.com"
        api_key: "${AZURE_OPENAI_API_KEY}"
        api_version: "2024-02-01"
        tpm: 10000  # Tokens per minute limit
        rpm: 60     # Requests per minute limit
      
      # Fallback chain: Azure GPT-4 → OpenRouter Claude → Azure GPT-3.5
      fallback_models:
        - model: "openrouter/anthropic/claude-3-opus"
          api_key: "${OPENROUTER_API_KEY}"
          api_base: "https://openrouter.ai/api/v1"
          tpm: 5000
          
        - model: "azure/gpt-35-turbo"
          api_base: "https://myresource.openai.azure.com"
          api_key: "${AZURE_OPENAI_API_KEY}"
          api_version: "2024-02-01"
          tpm: 20000
    
    # Secondary model group for enrichment (cheaper/faster)
    - model_name: "enrichment"
      litellm_params:
        model: "azure/gpt-35-turbo"
        api_base: "https://myresource.openai.azure.com"
        api_key: "${AZURE_OPENAI_API_KEY}"
      fallback_models:
        - model: "openrouter/anthropic/claude-3-haiku"
          api_key: "${OPENROUTER_API_KEY}"
  
  # litellm proxy settings
  proxy:
    host: "0.0.0.0"
    port: 4000
    
  # Retry and fallback behavior
  retry:
    num_retries: 3
    timeout: 30
    backoff_factor: 2
    
  # Default parameters
  defaults:
    temperature: 0.3
    max_tokens: 2000
    
  # Provider-specific settings
  providers:
    azure:
      cache_responses: true
      cache_ttl: 3600  # 1 hour
    openrouter:
      headers:
        HTTP-Referer: "https://pipeline.example.com"
        X-Title: "Agentic Pipeline Ingestor"
```

### 6.5.4 Fallback Chain Strategy

```python
from litellm import Router

class LLMRouterWithFallback:
    """
    Configured fallback chain:
    1. Azure GPT-4 (primary for complex reasoning)
    2. OpenRouter Claude-3 Opus (fallback if Azure fails/rate-limited)
    3. Azure GPT-3.5 (final fallback for basic tasks)
    """
    
    def __init__(self, config: LLMConfig):
        self.router = Router(
            model_list=config.router,
            default_fallbacks=True,
            # Cooldown period for failed models
            cooldown_time=60,
            # Retry failed requests
            num_retries=3,
        )
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model_group: str = "agentic-decisions",
        **kwargs
    ) -> ChatCompletionResponse:
        """
        Automatically routes through fallback chain:
        - Try primary model first
        - If fails (rate limit, timeout, error), try fallback #1
        - If fails, try fallback #2
        - If all fail, raise exception
        """
        try:
            response = await self.router.acompletion(
                model=model_group,
                messages=[m.to_dict() for m in messages],
                **kwargs
            )
            return ChatCompletionResponse.from_litellm(response)
        except Exception as e:
            # All fallbacks exhausted
            logger.error(f"All LLM providers failed: {e}")
            raise LLMProviderUnavailableException() from e
```

### 6.5.5 Supported Providers (via litellm)

| Priority | Provider | litellm Model String | Use Case |
|----------|----------|---------------------|----------|
| **Primary** | Azure OpenAI | `azure/gpt-4` | Complex reasoning, primary processing |
| **Primary** | Azure OpenAI | `azure/gpt-35-turbo` | Fast enrichment, basic tasks |
| **Fallback 1** | OpenRouter | `openrouter/anthropic/claude-3-opus` | Fallback when Azure rate-limited |
| **Fallback 1** | OpenRouter | `openrouter/anthropic/claude-3-haiku` | Fast fallback for enrichment |
| Extended | OpenAI | `gpt-4`, `gpt-3.5-turbo` | Direct OpenAI API option |
| Extended | Anthropic | `claude-3-opus-20240229` | Direct Anthropic API |
| Extended | Ollama | `ollama/llama2` | Local development, air-gapped |
| Extended | vLLM | `openai/<model>` | Self-hosted models |

**MVP Providers**: Azure OpenAI + OpenRouter (as decided)
**Extended**: Available for future expansion without code changes

### 6.5.5 Usage in Agentic Engine

```python
class AgenticDecisionEngine:
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
    
    async def select_parser(self, content_analysis: ContentDetectionResult) -> ParserSelection:
        """Use LLM to decide optimal parser strategy"""
        messages = [
            ChatMessage(role="system", content="You are a document processing expert..."),
            ChatMessage(role="user", content=f"""
                Analyze this content detection result and select the best parser:
                - Detected type: {content_analysis.detected_type}
                - Has text layer: {content_analysis.has_text_layer}
                - Image ratio: {content_analysis.image_ratio}
                - Confidence: {content_analysis.confidence}
                
                Available parsers: docling, azure_ocr
                Return JSON: {{"parser": "...", "reason": "...", "confidence": 0.95}}
            """)
        ]
        
        response = await self.llm.chat_completion(
            messages=messages,
            temperature=0.1,  # Low temp for deterministic decisions
            response_format={"type": "json_object"}
        )
        
        return ParserSelection.parse(response.choices[0].message.content)
```

### 6.5.6 Implementation Decision ✓

**Selected**: **litellm Proxy Library**

**MVP Providers**: 
- ✅ Azure OpenAI (primary)
- ✅ OpenRouter (fallback for Anthropic Claude)

**Fallback Chain**:
```
Azure GPT-4 → OpenRouter Claude-3 Opus → Azure GPT-3.5
```

**Rationale**:
- ✅ Handles 100+ providers automatically
- ✅ Built-in fallback and retry logic
- ✅ Active community (BerriAI)
- ✅ Enterprise features (caching, rate limiting, spend tracking)
- ✅ No code changes to switch providers

**Configuration-Only Changes**:
- Add new provider: Edit `config/llm.yaml` only
- Change fallback order: Edit YAML only
- Adjust rate limits: Edit YAML only

**Alternative: Custom HTTP Adapter**
When to use:
- litellm doesn't support a specific provider (rare)
- Need custom transformations at HTTP level
- Compliance requires direct HTTP control

---

## 7. Plugin System

### 7.1 Plugin Interfaces

#### Source Plugin

```python
class SourcePlugin(ABC):
    plugin_id: str
    supported_formats: List[str]
    
    async def connect(self, config: Dict) -> Connection
    async def list_files(self, conn: Connection, path: str) -> List[SourceFile]
    async def get_file(self, conn: Connection, path: str) -> RetrievedFile
    async def validate_config(self, config: Dict) -> ValidationResult
```

#### Parser Plugin

```python
class ParserPlugin(ABC):
    plugin_id: str
    supported_formats: List[str]
    
    async def parse(self, file_path: str, options: Dict) -> ParsingResult
    async def supports(self, file_path: str, mime_type: str) -> SupportResult
    async def preprocess(self, file_path: str, steps: List[str]) -> str
```

#### Destination Plugin

```python
class DestinationPlugin(ABC):
    plugin_id: str
    
    async def connect(self, config: Dict) -> Connection
    async def write(self, conn: Connection, data: TransformedData) -> WriteResult
    async def health_check(self, config: Dict) -> HealthStatus
```

### 7.2 Built-in Plugins

| Type | Plugin ID | Formats/Protocols |
|------|-----------|-------------------|
| Source | local | Local filesystem |
| Source | s3 | Amazon S3 |
| Source | azure_blob | Azure Blob Storage |
| Source | sharepoint | SharePoint Online |
| Source | url | HTTP/HTTPS |
| Parser | docling | PDF, DOCX, PPTX, XLSX |
| Parser | azure_ocr | PDF, images |
| Parser | csv | CSV, TSV |
| Parser | archive | ZIP, TAR, 7Z |
| Destination | cognee | Cognee API |
| Destination | webhook | HTTP callback |

---

## 8. Agentic Behavior

### 8.1 Decision Points

1. **Parser Selection**: Based on content detection, parser availability
2. **Quality Gates**: Compare scores to thresholds, decide retry/DLQ
3. **Retry Strategy**: Select appropriate retry approach
4. **Destination Routing**: Apply filters, handle failures

### 8.2 Retry Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| same_parser | Retry with same parser | Transient failures |
| fallback_parser | Switch parser | Parser-specific issues |
| preprocess_then_retry | Enhance image quality | Low-quality scans |
| split_processing | Process pages separately | Large document failures |

### 8.3 Self-Healing

- Exponential backoff with jitter
- Max 3 retries by default
- DLQ after max retries exceeded
- Manual retry from DLQ with modified config

---

## 9. Authentication & Authorization

### 9.1 Authentication Methods

1. **API Keys**: For service accounts, stored hashed
2. **OAuth2/OIDC**: For user authentication
3. **Azure AD**: Enterprise SSO integration

### 9.2 RBAC Model

| Role | Permissions |
|------|-------------|
| Admin | All operations |
| Operator | Create, read, cancel, retry jobs |
| Developer | Create jobs, read sources/destinations |
| Viewer | Read-only job access |

### 9.3 Source Authorization

- SharePoint: Delegate to SharePoint permission system
- S3: Use IAM roles/policies
- Azure Blob: Use SAS tokens or managed identity

---

## 10. Audit & Compliance

### 10.1 Audit Events

All operations logged: job lifecycle, source access, config changes, auth events.

### 10.2 Data Retention

| Data Type | Retention | Action |
|-----------|-----------|--------|
| Raw files | 30 days | Delete |
| Processed data | 90 days | Archive |
| Job metadata | 7 years | Keep |
| Audit logs | 7 years | Keep |

### 10.3 Data Lineage

Track transformations through pipeline: input hashes, output hashes, transformations applied.

---

## 11. Observability

### 11.1 OpenTelemetry

- Distributed tracing across all components
- GenAI-specific spans for parser operations
- Custom attributes for job tracking

### 11.2 Metrics (Prometheus)

- `ingestion_jobs_created_total`
- `ingestion_jobs_completed_total`
- `ingestion_job_duration_seconds`
- `ingestion_stage_duration_seconds`
- `ingestion_quality_score`
- `ingestion_queue_depth`

### 11.3 Health Checks

- `/health` - Comprehensive status
- `/health/ready` - K8s readiness probe
- `/health/live` - K8s liveness probe

---

## 12. Implementation Phases

### Phase 1: Foundation & API Design (Weeks 1-4)
- **OpenAPI 3.1 Specification** - Complete API contract definition
- Mock server generation from OpenAPI (Prism)
- Project scaffolding with FastAPI
- Database schema design
- Plugin system interfaces
- Contract testing framework setup
- SDK generation pipeline (Python, TypeScript)
- Local Docker development environment

### Phase 2: Core Pipeline & LLM Abstraction (Weeks 5-9)
- All 7 pipeline stages
- Content detection
- Azure OCR integration
- Quality assessment
- Cognee destination
- **LLM Provider Abstraction Layer** (llmlite or custom adapter)
- **Multi-provider support**: OpenAI, Azure OpenAI, Anthropic
- **Configuration-driven provider switching**

### Phase 3: Agentic Features (Weeks 9-12)
- Decision engine
- Advanced retry strategies
- Dead letter queue
- Multiple destinations
- S3/Azure Blob sources

### Phase 4: Enterprise Features (Weeks 13-16)
- OAuth2/OAuth2
- RBAC system
- Audit logging
- Data lineage
- SharePoint integration

### Phase 5: Observability & Scale (Weeks 17-20)
- OpenTelemetry
- Azure deployment
- Performance optimization
- 20GB/day target

### Phase 6: Advanced Features (Weeks 21-24)
- Additional parsers (CSV, JSON, Email)
- More destinations (GraphRAG, Neo4j)
- Entity extraction
- Bulk operations

---

## 13. Technology Stack

### Core
- Python 3.11+
- FastAPI 0.104+ (generated from OpenAPI spec)
- PostgreSQL 17+
- Redis 7+
- Azure Queue Storage

### API-First Tooling
- **OpenAPI 3.1**: API specification format
- **OpenAPI Generator**: Client SDK generation
- **Prism**: Mock server from OpenAPI spec
- **Schemathesis**: Property-based API contract testing
- **Spectral**: OpenAPI linting and validation
- **Redocly**: Documentation generation

### LLM Abstraction (litellm)
- **litellm**: Unified interface for 100+ LLM providers
- **litellm Router**: Intelligent routing with fallback chains
- **Azure OpenAI SDK**: Native Azure integration (via litellm)
- **Fallback Logic**: Automatic retry with exponential backoff

### Parsing
- docling 1.x
- azure-ai-vision 1.x
- PyMuPDF 1.23+

### Infrastructure
- Docker, Kubernetes (AKS)
- Azure Application Gateway
- Azure Monitor
- OpenSearch

---

## 14. Deployment

### Local Development
```bash
docker-compose up -d
```

Services: API, Worker (x2), PostgreSQL, Redis, OpenSearch

### Azure Production
- AKS cluster with 3 API replicas, 5-10 workers
- Azure DB for PostgreSQL (Primary + Replica)
- Azure Cache for Redis
- Azure Blob Storage
- Azure Queue Storage
- Azure Monitor for observability

---

## 15. Acceptance Criteria

### Functional
- Support all specified file types
- Automatic parser selection > 98% accuracy
- Quality gates with retry/DLQ
- SharePoint integration
- Multiple destination routing
- Complete OpenAPI 3.1 specification published
- Auto-generated SDKs (Python, TypeScript) available
- All API endpoints follow OpenAPI contract
- **LLM-agnostic**: Works with OpenAI, Azure, Anthropic without code changes
- **Provider hot-swapping**: Configuration-only provider switching

### Performance
- 20GB/day throughput
- p99 API latency < 2s
- 99.5% job success rate

### Security
- All endpoints require auth
- Audit logs complete
- Data encrypted in transit and at rest

### Observability
- Traces in Azure Monitor
- Prometheus metrics
- Structured logging

---

## 16. Dependencies & Integrations

| Service | Purpose | Critical |
|---------|---------|----------|
| Docling | Document parsing | Yes |
| Azure AI Vision | OCR fallback | Yes |
| Cognee | Primary destination | Yes |
| Azure Storage | Blob, Queue | Yes |
| PostgreSQL | Relational data | Yes |
| Redis | Cache | Yes |
| OpenSearch | Audit logs | No |
| Azure AD | Authentication | No |
| OpenAPI Generator | SDK generation | No |
| Prism | Mock server | No |
| **LLM Providers (via litellm)** | | |
| Azure OpenAI | **Primary MVP provider** | **Yes** |
| OpenRouter | **Fallback MVP provider** | **Yes** |
| Anthropic Claude | Available via OpenRouter | No |
| OpenAI | Extended option | No |
| Ollama | Local development | No |
| litellm | LLM abstraction library | **Yes** |

---

## 17. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Docling accuracy below target | Medium | High | Azure OCR fallback, quality gates |
| Azure OCR service limits | Medium | High | Rate limiting, caching, batching |
| Cognee API changes | Low | High | Version pinning, abstraction layer |
| 20GB/day throughput not met | Low | Medium | Horizontal scaling, optimization |
| SharePoint auth complexity | Medium | Medium | Clear documentation, test harness |
| Memory issues with large files | Medium | High | Streaming, chunking, limits |

---

## 18. Open Questions

1. What is the maximum file size limit? (Current assumption: 100MB)
2. Should we support password-protected documents?
3. What is the SLA requirement for near-realtime processing?
4. Should we implement automatic PII redaction?
5. What is the disaster recovery RTO/RPO?
6. Should we support custom parser training?
7. **API-First**: Should we publish the OpenAPI spec to a public API registry (e.g., SwaggerHub)?
8. **API-First**: What versioning strategy for SDKs - match API versions or semantic versioning?
9. **API-First**: GraphQL support as alternative to REST - should we add it in Phase 2?
10. ~~**LLM**: Preferred approach~~ ✅ **DECIDED: litellm library**
11. ~~**LLM**: Which providers to support in MVP?~~ ✅ **DECIDED: Azure OpenAI + OpenRouter**
12. ~~**LLM**: Should we support model fallback chains?~~ ✅ **DECIDED: Yes, automatic fallback chains**

---

## Appendix: Full Technical Details

For detailed code examples, database schemas, and API specifications, see the implementation guide.
