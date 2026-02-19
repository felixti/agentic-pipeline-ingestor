# System Architecture

**Agentic Data Pipeline Ingestor**  
*Enterprise-grade agentic data pipeline for document ingestion with intelligent content routing, vector search capabilities, and destination-agnostic output.*

> **Version**: 1.0.0  
> **Last Updated**: 2026-02-18  
> **OpenSpec Change**: review-update-architecture-docs

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Component Architecture](#2-component-architecture)
3. [Data Flow Architecture](#3-data-flow-architecture)
4. [API Architecture](#4-api-architecture)
5. [Database Architecture](#5-database-architecture)
6. [Security Architecture](#6-security-architecture)
7. [Integration Architecture](#7-integration-architecture)
8. [Technology Stack](#8-technology-stack)
9. [Deployment Architecture](#9-deployment-architecture)

---

## 1. System Overview

### 1.1 System Context and Boundaries

The Agentic Data Pipeline Ingestor is an enterprise-grade document processing system that bridges the gap between raw document sources and structured, searchable data destinations. It operates as a middleware service that:

- **Ingests** documents from multiple sources (S3, Azure Blob, SharePoint, direct uploads)
- **Processes** documents through an AI-driven 7-stage pipeline
- **Transforms** unstructured content into structured, searchable chunks with embeddings
- **Delivers** processed data to configurable destinations (Cognee, GraphRAG, webhooks, Neo4j)

### 1.2 Primary Purpose and Goals

| Goal | Description |
|------|-------------|
| **Universal Ingestion** | Support any document format (PDF, Office, images, archives) |
| **Intelligent Processing** | Use AI to select optimal parsing strategies |
| **Semantic Search** | Enable vector-based similarity search across documents |
| **Destination Agnostic** | Route output to any configured destination |
| **Enterprise Scale** | Process 20GB/day with near-realtime capabilities |
| **Observability** | Full audit trails, lineage tracking, and metrics |

### 1.3 Stakeholders and Users

| Stakeholder | Role | Primary Use |
|-------------|------|-------------|
| **Data Engineers** | Configure pipelines, manage sources/destinations | API integration, monitoring |
| **System Operators** | Monitor processing, handle failures | Dashboard, DLQ management |
| **Developers** | Build applications on top of processed data | SDK, search APIs |
| **Security Auditors** | Review access patterns, compliance | Audit logs, lineage |
| **End Users** | Search and retrieve processed documents | Search APIs |

### 1.4 System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL SYSTEMS                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │     S3       │  │ Azure Blob   │  │ SharePoint   │  │   Upload     │    │
│  │   Buckets    │  │  Storage     │  │   Online     │  │   Client     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼─────────────────┼─────────────────┼─────────────────┼────────────┘
          │                 │                 │                 │
          └─────────────────┴──────────┬──────┴─────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │   AGENTIC DATA PIPELINE INGESTOR    │
                    │                                     │
                    │  ┌─────────────┐  ┌─────────────┐  │
                    │  │   API Layer │  │  Pipeline   │  │
                    │  │  (FastAPI)  │  │   Engine    │  │
                    │  └─────────────┘  └─────────────┘  │
                    │                                     │
                    │  ┌─────────────┐  ┌─────────────┐  │
                    │  │    Search   │  │   Plugin    │  │
                    │  │  Services   │  │   System    │  │
                    │  └─────────────┘  └─────────────┘  │
                    └──────────────────┬──────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
┌─────────▼──────────┐      ┌──────────▼──────────┐      ┌──────────▼──────────┐
│     COGNEE         │      │     GraphRAG        │      │      Neo4j          │
│  (Memory System)   │      │  (Knowledge Graph)  │      │   (Graph DB)        │
└────────────────────┘      └─────────────────────┘      └─────────────────────┘
```

---

## 2. Component Architecture

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │ /jobs       │ │ /upload     │ │ /search     │ │ /chunks     │ │ /health   │  │
│  │ (7 endpoints)│ │ (3 endpoints)│ │ (4 endpoints)│ │ (2 endpoints)│ │ (5 endpoints)│
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └─────┬─────┘  │
└─────────┼───────────────┼───────────────┼───────────────┼──────────────┼────────┘
          │               │               │               │              │
          └───────────────┴───────────────┴───────────────┴──────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CORE ORCHESTRATION ENGINE                                │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    AGENTIC DECISION ENGINE                               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │    │
│  │  │  Detection  │  │   Parser    │  │   Quality   │  │   Routing   │    │    │
│  │  │   Service   │  │  Selection  │  │  Assessment │  │   Engine    │    │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    7-STAGE PROCESSING PIPELINE                           │    │
│  │                                                                          │    │
│  │   Ingest → Detect → Select Parser → Parse → Chunk → Embed → Output      │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    SEARCH SERVICES LAYER (NEW)                           │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │    │
│  │  │    Vector    │  │     Text     │  │    Hybrid    │  │   Embedding  │ │    │
│  │  │    Search    │  │    Search    │  │    Search    │  │   Service    │ │    │
│  │  │   Service    │  │   Service    │  │   Service    │  │              │ │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    LLM ADAPTER (litellm)                                 │    │
│  │       ┌──────────────┐         ┌──────────────────────────────┐         │    │
│  │       │ Azure GPT-4  │ ←────→ │ OpenRouter Claude-3 (fallback)│         │    │
│  │       │  (Primary)   │         │                               │         │    │
│  │       └──────────────┘         └──────────────────────────────┘         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            PLUGIN ECOSYSTEM                                      │
│                                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────────┐ │
│  │     SOURCES      │  │     PARSERS      │  │         DESTINATIONS           │ │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐ ┌────────────┐ │ │
│  │  │ S3 Source  │  │  │  │  Docling   │  │  │  │   Cognee   │ │  GraphRAG  │ │ │
│  │  │ Azure Blob │  │  │  │ Azure OCR  │  │  │  │   Neo4j    │ │  Pinecone  │ │ │
│  │  │ SharePoint │  │  │  │  PyMuPDF   │  │  │  │  Weaviate  │ │  Webhook   │ │ │
│  │  └────────────┘  │  │  └────────────┘  │  │  └────────────┘ └────────────┘ │ │
│  └──────────────────┘  └──────────────────┘  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │   PostgreSQL     │  │    pgvector      │  │         Redis                │  │
│  │   ├─ jobs        │  │   ├─ VECTOR      │  │    ├─ Job Queue              │  │
│  │   ├─ pipelines   │  │   ├─ HNSW Index  │  │    ├─ Cache                  │  │
│  │   ├─ chunks      │  │   ├─ Cosine Sim  │  │    └─ Rate Limiting          │  │
│  │   ├─ audit_logs  │  │   └─ pg_trgm     │  │                              │  │
│  │   └─ api_keys    │  │                  │  │                              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 API Layer (FastAPI Routes)

| Module | File | Endpoints | Description |
|--------|------|-----------|-------------|
| Jobs | `src/main.py` | 7 | Job submission, listing, status, retry, cancel |
| Upload | `src/main.py` | 3 | File upload, URL ingestion, streaming |
| Search | `src/api/routes/search.py` | 4 | Semantic, text, hybrid search, similar chunks |
| Chunks | `src/api/routes/chunks.py` | 2 | List chunks, get chunk details |
| Auth | `src/api/routes/auth.py` | 10 | Login, tokens, API keys, OAuth2 |
| Audit | `src/api/routes/audit.py` | 5 | Query logs, export, summaries |
| DLQ | `src/api/routes/dlq.py` | 7 | Dead letter queue management |
| Lineage | `src/api/routes/lineage.py` | 7 | Data lineage tracking |
| Health | `src/api/routes/health.py` | 5 | Health checks, vector store status |
| Detection | `src/api/routes/detection.py` | 3 | Content type detection |
| Bulk | `src/api/routes/bulk.py` | 7 | Bulk operations |

**Total: 50 API Endpoints**

### 2.3 Core Engine (Pipeline & Decisions)

#### Pipeline Stages (7-Stage Flow)

```python
class Pipeline:
    STAGES = [
        IngestStage,         # Stage 1: File validation
        DetectStage,         # Stage 2: Content detection (AI-driven)
        SelectParserStage,   # Stage 3: Parser selection
        ParseStage,          # Stage 4: Document parsing
        ChunkStage,          # Stage 5: Text chunking
        EmbedStage,          # Stage 6: Embedding generation
        # EnrichStage,       # Stage 7: Content enrichment (future)
        # QualityStage,      # Stage 8: Quality assessment (future)
        # OutputStage,       # Stage 9: Destination output (future)
    ]
```

#### Agentic Decision Engine Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Content Detection | `src/core/content_detection/service.py` | Detect scanned vs. text PDFs |
| Parser Selection | `src/core/parser_selection.py` | Choose optimal parser chain |
| Decision Engine | `src/core/decisions.py` | AI-driven routing decisions |
| Quality Assessment | `src/core/quality.py` | Score extraction quality |
| Retry Logic | `src/core/retry.py` | Exponential backoff, circuit breaker |
| DLQ Management | `src/core/dlq.py` | Failed job handling |

### 2.4 Service Layer (Search Services)

The search services layer provides vector and text search capabilities:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEARCH SERVICES LAYER                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              VectorSearchService                          │  │
│  │  - Cosine similarity search using pgvector                │  │
│  │  - HNSW index for approximate nearest neighbor            │  │
│  │  - Metadata filtering (job_id, chunk_index)               │  │
│  │  - File: src/services/vector_search_service.py            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              TextSearchService                            │  │
│  │  - Full-text search with PostgreSQL tsvector              │  │
│  │  - BM25 ranking algorithm                                 │  │
│  │  - Fuzzy matching with pg_trgm                            │  │
│  │  - Highlighting support                                   │  │
│  │  - File: src/services/text_search_service.py              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              HybridSearchService                          │  │
│  │  - Combines vector + text search                          │  │
│  │  - Weighted sum fusion                                    │  │
│  │  - Reciprocal Rank Fusion (RRF)                           │  │
│  │  - Configurable weights                                   │  │
│  │  - File: src/services/hybrid_search_service.py            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              EmbeddingService                             │  │
│  │  - Text embedding generation via litellm                  │  │
│  │  - Batch processing support                               │  │
│  │  - Multiple provider support (OpenAI, Azure)              │  │
│  │  - File: src/services/embedding_service.py                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 Repository Layer (Data Access)

| Repository | File | Purpose |
|------------|------|---------|
| JobRepository | `src/db/repositories/job.py` | Job CRUD, status management |
| PipelineRepository | `src/db/repositories/pipeline.py` | Pipeline config management |
| DocumentChunkRepository | `src/db/repositories/document_chunk_repository.py` | Chunk CRUD with embeddings |
| AuditRepository | `src/db/repositories/audit.py` | Audit log operations |
| ApiKeyRepository | `src/db/repositories/api_key.py` | API key management |

### 2.6 Plugin System

The plugin system uses an Abstract Base Class (ABC) pattern with auto-discovery:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PLUGIN SYSTEM                                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Base Classes (ABC)                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │ SourcePlugin │  │ ParserPlugin │  │DestinationPlugin│  │
│  │  │ (abstract)   │  │ (abstract)   │  │ (abstract)    │   │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │  │
│  └─────────┼─────────────────┼─────────────────┼───────────┘  │
│            │                 │                 │              │
│            ▼                 ▼                 ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Plugin Registry                         │  │
│  │              (src/plugins/registry.py)                   │  │
│  │  - Plugin registration                                   │  │
│  │  - Lifecycle management (init/shutdown)                  │  │
│  │  - Health checking                                       │  │
│  │  - Auto-discovery via src/plugins/loaders.py             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   SOURCES   │  │   PARSERS   │  │      DESTINATIONS       │ │
│  │  s3_source  │  │  docling    │  │       cognee            │ │
│  │azure_blob   │  │ azure_ocr  │  │      graphrag           │ │
│  │sharepoint   │  │  csv       │  │       neo4j             │ │
│  │             │  │  json      │  │      pinecone           │ │
│  │             │  │  xml       │  │      weaviate           │ │
│  │             │  │ email      │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Architecture

### 3.1 Document Ingestion Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────→│   API    │────→│  Queue   │────→│  Worker  │────→│ Pipeline │
│          │     │ (FastAPI)│     │ (Redis)  │     │Processor │     │ Engine   │
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └────┬─────┘
     │                                                                   │
     │  POST /api/v1/jobs                                                │
     │  {                                                                │
     │    "source_type": "upload",                                       │
     │    "source_uri": "/uploads/doc.pdf",                              │
     │    "file_name": "doc.pdf"                                         │
     │  }                                                                │
     │────────────────────────────────────────────────────────────────────→│
     │                                                                   │
     │  Response 202 Accepted                                            │
     │  { "job_id": "uuid", "status": "pending" }                        │
     │←────────────────────────────────────────────────────────────────────│
     │                                                                   │
     │                                                                   ▼
     │                                                          ┌──────────┐
     │                                                          │   Job    │
     │                                                          │  Record  │
     │                                                          │ (DB)     │
     │                                                          └────┬─────┘
     │                                                               │
     │  GET /api/v1/jobs/{id}                                       │
     │───────────────────────────────────────────────────────────────→│
     │                                                               │
     │  { "status": "processing", "progress": 45% }                  │
     │←───────────────────────────────────────────────────────────────│
     │                                                               │
     │  GET /api/v1/jobs/{id}/result                                │
     │───────────────────────────────────────────────────────────────→│
     │                                                               │
     │  { "status": "completed", "chunks": 15 }                      │
     │←───────────────────────────────────────────────────────────────│
```

### 3.2 Pipeline Processing Flow (7 Stages)

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Ingest  │───→│ Detect  │───→│ Select  │───→│  Parse  │───→│  Chunk  │───→│  Embed  │───→│ Output  │
│ Stage   │    │ Stage   │    │ Parser  │    │ Stage   │    │ Stage   │    │ Stage   │    │ Stage   │
└────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
     │              │              │              │              │              │              │
     │ Validate     │ Analyze      │ Choose       │ Extract      │ Split text   │ Generate     │ Route to
     │ file exists  │ content      │ parser       │ text         │ into chunks  │ embeddings   │ destination
     │              │ type         │ chain        │              │              │              │
     │              │              │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼              ▼              ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ File    │    │Content  │    │ Primary │    │Extracted│    │Chunk[]  │    │Embedding│    │Output   │
│Metadata │    │Analysis │    │Fallback │    │Text     │    │         │    │Vectors  │    │Result   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘

Stage Details:
─────────────
1. INGEST:     File validation, size checks, metadata extraction
2. DETECT:     AI-driven content detection (scanned vs. text)
3. SELECT:     Parser selection based on detection + config
4. PARSE:      Document parsing with fallback chain
5. CHUNK:      Text segmentation (fixed/semantic/hierarchical)
6. EMBED:      Vector embedding generation via litellm
7. OUTPUT:     Delivery to configured destinations
```

### 3.3 Search Query Flows

#### Semantic Search Flow

```
┌──────────┐     ┌──────────┐     ┌───────────────┐     ┌─────────────────┐     ┌──────────┐
│  Client  │────→│   API    │────→│VectorSearch   │────→│DocumentChunk    │────→│PostgreSQL│
│          │     │/search/  │     │   Service     │     │  Repository     │     │+pgvector │
│          │     │semantic  │     │               │     │                 │     │          │
└──────────┘     └──────────┘     └───────────────┘     └─────────────────┘     └──────────┘
      │                                                              │                │
      │ POST {                                                      │                │
      │   "query_embedding": [0.023, ...],                          │                │
      │   "top_k": 10,                                              │                │
      │   "min_similarity": 0.7                                     │                │
      │ }                                                           │                │
      │─────────────────────────────────────────────────────────────→│                │
      │                                                             │                │
      │                        SELECT *, embedding <=> query_vector │                │
      │                        FROM document_chunks                 │                │
      │                        ORDER BY embedding <=> query_vector  │                │
      │                        LIMIT 10                             │                │
      │                                                             │───────────────→│
      │                                                             │                │
      │                                                             │←───────────────│
      │                                                             │                │
      │  { "results": [...], "total": 15, "query_time_ms": 25.5 }   │                │
      │←─────────────────────────────────────────────────────────────│                │
```

#### Hybrid Search Flow

```
┌──────────┐     ┌──────────┐     ┌─────────────────────────────────┐     ┌──────────┐
│  Client  │────→│   API    │────→│       HybridSearchService       │────→│ Results  │
│          │     │/search/  │     │                                 │     │          │
│          │     │ hybrid   │     │  ┌─────────────┐ ┌────────────┐ │     │          │
└──────────┘     └──────────┘     │  │ VectorSearch │ │TextSearch  │ │     │          │
                                  │  │   Service    │ │  Service   │ │     │          │
                                  │  └──────┬───────┘ └─────┬──────┘ │     │          │
                                  │         │               │        │     │          │
                                  │         ▼               ▼        │     │          │
                                  │  ┌────────────────────────────┐  │     │          │
                                  │  │      Fusion Engine         │  │     │          │
                                  │  │  - Weighted Sum            │  │     │          │
                                  │  │  - Reciprocal Rank Fusion  │  │     │          │
                                  │  └─────────────┬──────────────┘  │     │          │
                                  └────────────────┼─────────────────┘     │          │
                                                   │                       │          │
                                                   ▼                       │          │
                                            ┌──────────┐                   │          │
                                            │ Ranked   │───────────────────→│          │
                                            │ Results  │                   │          │
                                            └──────────┘                   │          │
```

### 3.4 Data Persistence Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Pipeline   │────→│  JobContext  │────→│ Repositories │────→│  PostgreSQL  │
│   Stages     │     │              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                   │                    │                    │
       │                   │                    │                    │
       ▼                   ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Job Status   │────→│ JobContext   │────→│JobRepository │────→│ jobs table   │
│ Updates      │     │  job_id      │     │  update()    │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Embed      │────→│ ChunkModels  │────→│DocumentChunk │────→│document_chunks│
│   Stage      │     │  List[Chunk] │     │Repository    │     │  table +     │
│              │     │              │     │  bulk_save() │     │  HNSW index  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 4. API Architecture

### 4.1 REST API Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Resource-Oriented** | Nouns for resources (`/jobs`, `/chunks`, `/search`) |
| **HTTP Verbs** | GET (read), POST (create), PUT (update), DELETE (remove) |
| **Versioning** | URL-based versioning (`/api/v1/...`) |
| **Content Negotiation** | JSON default, YAML for OpenAPI spec |
| **Status Codes** | Proper HTTP semantics (200, 201, 202, 400, 401, 404, 429, 500) |
| **Pagination** | Cursor-based and offset-based options |
| **Filtering** | Query parameters for list endpoints |

### 4.2 Endpoint Organization

```
/api/v1/
├── jobs                    # Job management
│   ├── GET    /           # List jobs
│   ├── POST   /           # Create job
│   ├── GET    /{id}       # Get job details
│   ├── DELETE /{id}       # Cancel job
│   ├── POST   /{id}/retry # Retry failed job
│   ├── GET    /{id}/result # Get job result
│   └── GET    /{id}/chunks # List chunks (nested resource)
│
├── upload                  # File upload
│   ├── POST   /           # Multipart upload
│   └── POST   /url        # URL-based ingestion
│
├── search                  # Search operations
│   ├── POST   /semantic   # Vector similarity search
│   ├── POST   /text       # Full-text search
│   ├── POST   /hybrid     # Combined search
│   └── GET    /similar/{id} # Find similar chunks
│
├── auth                    # Authentication
│   ├── POST   /login
│   ├── POST   /token/refresh
│   ├── GET    /me
│   └── ...
│
├── audit                   # Audit logging
│   ├── GET    /logs
│   └── ...
│
└── system                  # System operations
    ├── GET    /health
    └── GET    /metrics

/health                     # Health checks
├── GET /                  # Overall health
├── GET /ready             # Kubernetes readiness
├── GET /live              # Kubernetes liveness
└── GET /vector            # Vector store health

/metrics                    # Prometheus metrics
```

### 4.3 Authentication & Authorization Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUTHENTICATION METHODS                               │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │   API Key      │  │    OAuth2      │  │   Azure AD     │  │    JWT     │ │
│  │  X-API-Key     │  │  Bearer Token  │  │  Enterprise    │  │  Internal  │ │
│  │                │  │                │  │     SSO        │  │            │ │
│  │  Service-to-   │  │  User auth     │  │  Azure Active  │  │  Internal  │ │
│  │  service       │  │  Flow          │  │  Directory     │  │  services  │ │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘  └─────┬──────┘ │
│          │                   │                   │                 │        │
│          └───────────────────┴─────────┬─────────┴─────────────────┘        │
│                                        │                                     │
│                                        ▼                                     │
│                          ┌─────────────────────────┐                         │
│                          │   Auth Middleware       │                         │
│                          │   (FastAPI Depends)     │                         │
│                          └────────────┬────────────┘                         │
│                                       │                                      │
│                                       ▼                                      │
│                          ┌─────────────────────────┐                         │
│                          │   RBACManager           │                         │
│                          │   - Role checking       │                         │
│                          │   - Permission eval     │                         │
│                          └────────────┬────────────┘                         │
│                                       │                                      │
│                                       ▼                                      │
│                          ┌─────────────────────────┐                         │
│                          │   Permission Decision   │                         │
│                          │   Allow / Deny (403)    │                         │
│                          └─────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Request/Response Patterns

#### Standard Response Wrapper

```python
class ApiResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool
    data: Any | None
    error: ErrorDetail | None
    meta: ResponseMeta | None
    request_id: str
    timestamp: datetime

class ErrorDetail(BaseModel):
    """Error details."""
    
    code: str
    message: str
    details: dict[str, Any] | None

class ResponseMeta(BaseModel):
    """Response metadata for list endpoints."""
    
    page: int
    per_page: int
    total: int
    total_pages: int
    links: PaginationLinks
```

#### Example Request/Response

```http
# Request
POST /api/v1/jobs
Content-Type: application/json
X-API-Key: your-api-key

{
  "source_type": "upload",
  "source_uri": "/uploads/document.pdf",
  "file_name": "document.pdf",
  "mode": "async",
  "priority": "high"
}

# Response 202 Accepted
{
  "success": true,
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "pending",
    "source_type": "upload",
    "source_uri": "/uploads/document.pdf",
    "file_name": "document.pdf",
    "priority": 10,
    "mode": "async",
    "created_at": "2026-02-18T10:30:00.000000",
    "updated_at": "2026-02-18T10:30:00.000000"
  },
  "error": null,
  "request_id": "req_abc123",
  "timestamp": "2026-02-18T10:30:00.000000"
}
```

### 4.5 Error Handling Strategy

| Layer | Error Type | HTTP Status | Response Format |
|-------|------------|-------------|-----------------|
| Validation | Pydantic ValidationError | 400 | `{"detail": [...]}` |
| Authentication | Invalid API key | 401 | `{"detail": "Unauthorized"}` |
| Authorization | Insufficient permissions | 403 | `{"detail": "Forbidden"}` |
| Not Found | Resource doesn't exist | 404 | `{"detail": "Not found"}` |
| Rate Limit | Too many requests | 429 | `{"detail": "Rate limit exceeded"}` |
| Server Error | Unexpected exception | 500 | `{"detail": "Internal server error"}` |

**Error Logging Strategy:**
- All errors logged with `structlog` including request_id
- Sensitive data never logged
- Stack traces only in debug mode

---

## 5. Database Architecture

### 5.1 PostgreSQL Schema Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATABASE SCHEMA                                        │
│                                                                                  │
│  ┌──────────────────┐                                                            │
│  │      jobs        │◄──────────────────┐                                        │
│  │  - id (PK)       │                   │                                        │
│  │  - status        │                   │                                        │
│  │  - source_type   │                   │                                        │
│  │  - source_uri    │                   │                                        │
│  │  - pipeline_id   │─────────────┐     │                                        │
│  │  - metadata_json │             │     │                                        │
│  └────────┬─────────┘             │     │                                        │
│           │                       │     │                                        │
│           │ 1:N                   │     │                                        │
│           ▼                       │     │                                        │
│  ┌──────────────────┐             │     │                                        │
│  │  document_chunks │             │     │                                        │
│  │  - id (PK)       │             │     │                                        │
│  │  - job_id (FK)   │─────────────┘     │                                        │
│  │  - chunk_index   │                   │                                        │
│  │  - content       │                   │                                        │
│  │  - embedding     │◄─── VECTOR(1536)  │                                        │
│  │  - metadata      │                   │                                        │
│  └──────────────────┘                   │                                        │
│                                         │                                        │
│  ┌──────────────────┐                   │     ┌──────────────────┐              │
│  │   job_results    │◄──────────────────┘     │    pipelines     │              │
│  │  - job_id (FK)   │                         │  - id (PK)       │              │
│  │  - extracted_text│                         │  - config (JSONB)│              │
│  │  - output_data   │                         │  - is_active     │              │
│  │  - quality_score │                         └──────────────────┘              │
│  └──────────────────┘                                                            │
│                                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐               │
│  │   audit_logs     │  │    api_keys      │  │webhook_subscriptions             │
│  │  - timestamp     │  │  - key_hash      │  │  - url           │               │
│  │  - user_id       │  │  - permissions   │  │  - events[]      │               │
│  │  - action        │  │  - is_active     │  │  - secret        │               │
│  │  - resource_type │  │  - expires_at    │  │  - is_active     │               │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 pgvector Integration

The `document_chunks` table stores vector embeddings using pgvector:

```sql
-- Table definition (simplified)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    embedding VECTOR(1536),  -- pgvector type
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint: one chunk_index per job
    UNIQUE(job_id, chunk_index)
);

-- HNSW index for approximate nearest neighbor search
CREATE INDEX idx_document_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Composite index for job + chunk queries
CREATE INDEX idx_document_chunks_job_chunk 
ON document_chunks (job_id, chunk_index);

-- Index for deduplication lookups
CREATE INDEX idx_document_chunks_content_hash 
ON document_chunks (content_hash);
```

### 5.3 HNSW Indexes for Vector Search

| Parameter | Value | Description |
|-----------|-------|-------------|
| `m` | 16 | Number of bi-directional links for each node |
| `ef_construction` | 64 | Size of dynamic candidate list during index construction |
| `ef_search` | 32 | Size of dynamic candidate list during search |
| Distance Metric | Cosine Similarity | `1 - (embedding <=> query_vector)` |

**Performance Characteristics:**
- Query time: O(log N) for approximate search
- Recall: ~95% with default parameters
- Build time: O(N log N)

### 5.4 pg_trgm for Text Search

```sql
-- Enable pg_trgm extension
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- GIN index for trigram similarity search
CREATE INDEX idx_document_chunks_content_trgm 
ON document_chunks 
USING gin (content gin_trgm_ops);

-- Example usage: fuzzy text search
SELECT *, similarity(content, 'machine learning') as sim
FROM document_chunks
WHERE content % 'machine learning'  -- trigram similarity operator
ORDER BY sim DESC
LIMIT 10;
```

### 5.5 Entity Relationship Diagram

```
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│      jobs           │       │   document_chunks   │       │    job_results      │
├─────────────────────┤       ├─────────────────────┤       ├─────────────────────┤
│ id (PK)             │──1:N──│ job_id (FK)         │       │ job_id (FK, PK)     │
│ status              │       │ id (PK)             │       │ extracted_text      │
│ source_type         │       │ chunk_index         │       │ output_data         │
│ source_uri          │       │ content             │       │ quality_score       │
│ file_name           │       │ embedding (VECTOR)  │       │ processing_time_ms  │
│ file_size           │       │ content_hash        │       │ created_at          │
│ mime_type           │       │ chunk_metadata      │       └─────────────────────┘
│ priority            │       │ created_at          │
│ mode                │       └─────────────────────┘
│ metadata_json       │
│ pipeline_id (FK)    │─────┐
│ created_at          │     │
│ updated_at          │     │
└─────────────────────┘     │       ┌─────────────────────┐
                            │       │     pipelines       │
                            │       ├─────────────────────┤
                            └────N:1│ id (PK)             │
                                    │ name                │
                                    │ config (JSONB)      │
                                    │ version             │
                                    │ is_active           │
                                    └─────────────────────┘
```

### 5.6 Migration Strategy

**Tools:** Alembic for SQLAlchemy migrations

```bash
# Create migration
make migrate-create MESSAGE="add vector store support"

# Apply migrations
make migrate

# Rollback
make migrate-downgrade
```

**Migration Principles:**
1. **Backward Compatible**: New columns nullable, old code still works
2. **Incremental**: Small, focused migrations
3. **Reversible**: All migrations have downgrade paths
4. **Tested**: Migrations tested in staging before production

---

## 6. Security Architecture

### 6.1 Authentication Methods

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AUTHENTICATION ARCHITECTURE                           │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        API Key Authentication                        │  │
│  │  Header: X-API-Key: <key>                                            │  │
│  │  - SHA-256 hash stored in database                                   │  │
│  │  - Per-key permission configuration                                  │  │
│  │  - Expiration support                                                │  │
│  │  - Usage tracking                                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        OAuth2 / JWT Authentication                   │  │
│  │  Header: Authorization: Bearer <token>                               │  │
│  │  - Access tokens (short-lived, 30 min)                               │  │
│  │  - Refresh tokens (long-lived, 7 days)                               │  │
│  │  - RS256 signing algorithm                                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Azure AD Integration                          │  │
│  │  - Enterprise SSO                                                    │  │
│  │  - Group-based role mapping                                          │  │
│  │  - Token validation via Microsoft identity platform                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 RBAC Roles and Permissions

| Role | Permissions | Typical User |
|------|-------------|--------------|
| **admin** | All permissions | System administrators |
| **operator** | READ, CREATE, CANCEL, RETRY, CREATE_JOBS | Operations team |
| **developer** | READ, CREATE_JOBS, READ_SOURCES | Integration developers |
| **viewer** | READ only | Auditors, read-only access |

**Permission Hierarchy:**
```
ADMIN
├── READ
├── CREATE
├── UPDATE
├── DELETE
├── CANCEL
├── RETRY
├── CREATE_JOBS
├── READ_SOURCES
├── MANAGE_CONFIG
├── MANAGE_USERS
├── VIEW_AUDIT
└── EXPORT_DATA
```

### 6.3 Audit Logging Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUDIT LOGGING FLOW                                   │
│                                                                              │
│   Request ──► Middleware ──► Audit Logger ──► PostgreSQL + OpenSearch       │
│                               (src/audit/)                                   │
│                                                                              │
│  Logged Events:                                                              │
│  - job.created, job.completed, job.failed                                    │
│  - auth.login, auth.logout, auth.token_refresh                               │
│  - search.semantic, search.text, search.hybrid                               │
│  - chunk.accessed, chunk.downloaded                                          │
│  - api_key.created, api_key.revoked                                          │
│                                                                              │
│  Logged Fields:                                                              │
│  - timestamp, user_id, api_key_id, action, resource_type                     │
│  - resource_id, request_method, request_path, request_details                │
│  - success/failure, error_message, ip_address, user_agent                    │
│  - duration_ms                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Data Protection Strategies

| Layer | Strategy | Implementation |
|-------|----------|----------------|
| **Transport** | TLS 1.3 | All API endpoints HTTPS |
| **Authentication** | API Keys + JWT | SHA-256 hashed keys, RS256 tokens |
| **Authorization** | RBAC | Role-based access control |
| **Data at Rest** | Encryption | PostgreSQL encryption TDE |
| **Secrets** | Environment Variables | Never commit secrets, use `.env` |
| **Input Validation** | Pydantic | All inputs validated |
| **Rate Limiting** | Per-key limits | Default 100 req/min |

---

## 7. Integration Architecture

### 7.1 LLM Provider Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LLM INTEGRATION ARCHITECTURE                           │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      LLM Router (litellm)                            │  │
│  │  - Unified interface for multiple providers                          │  │
│  │  - Automatic failover                                                │  │
│  │  - Rate limiting and retry logic                                     │  │
│  │  - Cost tracking                                                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│           ┌────────────────────────┼────────────────────────┐                │
│           │                        │                        │                │
│           ▼                        ▼                        ▼                │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │   Azure OpenAI   │   │    OpenRouter    │   │  Other Providers │        │
│  │  ├─ GPT-4        │   │  ├─ Claude-3     │   │  ├─ Local models │        │
│  │  ├─ GPT-4 Turbo  │   │  ├─ Llama 3      │   │  ├─ Ollama       │        │
│  │  └─ Embeddings   │   │  └─ Mistral      │   │  └─ etc.         │        │
│  │     (Primary)    │   │     (Fallback)   │   │                  │        │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘        │
│                                                                              │
│  Configuration: config/llm.yaml                                              │
│  - Primary: azure/gpt-4                                                     │
│  - Fallback: openrouter/anthropic/claude-3-opus                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 External Storage Integration

| Storage Type | Source Plugin | Destination Support |
|--------------|---------------|---------------------|
| **Amazon S3** | `S3Source` | Yes (via presigned URLs) |
| **Azure Blob** | `AzureBlobSource` | Native |
| **SharePoint** | `SharePointSource` | Via Microsoft Graph |
| **Local Files** | Upload endpoint | Staging directory |

### 7.3 Destination Plugins

| Destination | Plugin | Use Case |
|-------------|--------|----------|
| **Cognee** | `CogneeDestination` | AI memory system integration |
| **GraphRAG** | `GraphRAGDestination` | Knowledge graph construction |
| **Neo4j** | `Neo4jDestination` | Graph database storage |
| **Pinecone** | `PineconeDestination` | External vector database |
| **Weaviate** | `WeaviateDestination` | Vector search engine |
| **Webhook** | `WebhookDestination` | Custom HTTP callbacks |

### 7.4 Webhook Delivery System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       WEBHOOK DELIVERY ARCHITECTURE                          │
│                                                                              │
│  Event ──► Webhook Manager ──► Queue ──► Delivery Worker ──► HTTP POST     │
│                                (Redis)       (src/core/webhook_delivery.py) │
│                                                                              │
│  Delivery Flow:                                                              │
│  1. Event generated (job.completed, job.failed, etc.)                       │
│  2. Match against active webhook subscriptions                               │
│  3. Sign payload with HMAC-SHA256                                            │
│  4. Queue delivery attempt                                                   │
│  5. Retry with exponential backoff (max 5 attempts)                          │
│  6. Record delivery status in database                                       │
│                                                                              │
│  Signature Header:                                                           │
│  X-Webhook-Signature: sha256=<hmac_digest>                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.5 Integration Patterns

| Pattern | Usage | Implementation |
|---------|-------|----------------|
| **Circuit Breaker** | LLM calls, external APIs | `src/core/retry.py` |
| **Retry with Backoff** | All external calls | Exponential backoff, max 3 retries |
| **Fallback Chain** | Parser selection | Docling → Azure OCR → Tesseract |
| **Async Processing** | Job processing | Redis queue + worker processors |
| **Webhook** | Event notifications | HMAC-signed HTTP POSTs |

---

## 8. Technology Stack

### 8.1 Core Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.11+ | Application code |
| **Web Framework** | FastAPI | 0.104+ | REST API |
| **Data Validation** | Pydantic | 2.5+ | Request/response models |
| **Database** | PostgreSQL | 17+ | Primary datastore |
| **Vector Extension** | pgvector | 0.8+ | Vector storage & search |
| **Text Search** | pg_trgm | 1.6+ | Fuzzy text search |
| **ORM** | SQLAlchemy | 2.0+ | Database abstraction |
| **Cache/Queue** | Redis | 7+ | Job queue, caching |
| **LLM Router** | litellm | 1.0+ | Multi-provider LLM access |
| **Migrations** | Alembic | 1.12+ | Database migrations |

### 8.2 Document Processing Stack

| Technology | Purpose |
|------------|---------|
| **Docling** | Primary document parser (PDF, Office) |
| **Azure AI Vision** | OCR fallback for scanned documents |
| **PyMuPDF** | PDF text extraction fallback |
| **Pillow** | Image processing |
| **Tesseract** | OCR fallback |
| **pandas** | CSV/Excel parsing |
| **lxml** | XML parsing |

### 8.3 Observability Stack

| Technology | Purpose |
|------------|---------|
| **OpenTelemetry** | Distributed tracing |
| **Prometheus** | Metrics collection |
| **Grafana** | Metrics visualization |
| **Jaeger** | Trace visualization |
| **structlog** | Structured logging |
| **OpenSearch** | Audit log storage |

### 8.4 Security Stack

| Technology | Purpose |
|------------|---------|
| **python-jose** | JWT handling |
| **passlib** | Password hashing |
| **bcrypt** | Secure hashing |
| **python-multipart** | Form data parsing |
| **slowapi** (optional) | Rate limiting |

---

## 9. Deployment Architecture

### 9.1 Docker Compose Setup

The project includes a comprehensive Docker Compose configuration in `docker/docker-compose.yml`:

```yaml
# Core Services (always running)
services:
  api:           # FastAPI application
  worker:        # Background job processor (2 replicas)
  postgres:      # PostgreSQL 17 with pgvector
  redis:         # Redis cache and queue
  opensearch:    # Audit log storage

# Optional Services (profiles)
  litellm:       # LLM proxy (--profile litellm)
  prometheus:    # Metrics (--profile monitoring)
  grafana:       # Dashboards (--profile monitoring)
  jaeger:        # Tracing (--profile monitoring)
  otel-collector:# OpenTelemetry (--profile monitoring)
```

**Quick Start:**
```bash
# Start core services
make up

# Start with monitoring
make up-monitoring

# View logs
make logs-api
make logs-worker
```

### 9.2 Kubernetes Deployment

```yaml
# azure/aks-deployment.yaml (simplified)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pipeline-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pipeline-api
  template:
    spec:
      containers:
      - name: api
        image: pipeline-ingestor:latest
        ports:
        - containerPort: 8000
        env:
        - name: DB_URL
          valueFrom:
            secretKeyRef:
              name: pipeline-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
---
# Worker deployment with HPA
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pipeline-worker
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: worker
        image: pipeline-ingestor:latest
        command: ["python", "-m", "src.worker.main"]
```

### 9.3 Service Dependencies

```
Startup Order:
────────────────
1. postgres      (PostgreSQL with pgvector)
2. redis         (Cache and job queue)
3. opensearch    (Audit logs)
4. api           (FastAPI application)
5. worker        (Background processors)

Health Check Dependencies:
──────────────────────────
api depends on:
  - postgres (healthy)
  - redis (healthy)
  - opensearch (healthy)

worker depends on:
  - postgres (healthy)
  - redis (healthy)
  - api (healthy)
```

### 9.4 Health Checks and Monitoring

| Endpoint | Purpose | Kubernetes |
|----------|---------|------------|
| `/health` | Overall system health | - |
| `/health/ready` | Ready to accept traffic | Readiness probe |
| `/health/live` | Process is running | Liveness probe |
| `/health/vector` | Vector store health | Custom probe |
| `/metrics` | Prometheus metrics | Scraping |

**Health Check Implementation:**

```python
# From src/main.py

@app.get("/health/ready")
async def get_readiness() -> dict:
    """Kubernetes readiness probe."""
    return {"status": "ready"}

@app.get("/health/live")
async def get_liveness() -> dict:
    """Kubernetes liveness probe."""
    return {"status": "alive"}

@app.get("/health")
async def get_health(request: Request) -> dict:
    """Comprehensive health check."""
    components = {
        "api": ComponentHealth(status=HealthStatus.HEALTHY),
        "database": await check_database(),
        "plugins": await check_plugins(),
    }
    # Return overall status based on components
```

**Prometheus Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `pipeline_jobs_total` | Counter | Total jobs processed |
| `pipeline_job_duration_seconds` | Histogram | Job processing time |
| `vector_search_requests_total` | Counter | Search requests by type |
| `vector_search_duration_seconds` | Histogram | Search latency |
| `embedding_generation_total` | Counter | Embeddings generated |
| `api_requests_total` | Counter | API requests by endpoint |

---

## Appendix A: File Structure Reference

```
agentic-pipeline-ingestor/
├── src/                          # 118 Python files
│   ├── api/                      # 19 files - API layer
│   │   ├── routes/               # FastAPI route handlers
│   │   ├── models.py             # Pydantic models
│   │   ├── dependencies.py       # FastAPI dependencies
│   │   └── middleware/           # Rate limiting, etc.
│   ├── auth/                     # 8 files - Authentication
│   ├── core/                     # 29 files - Pipeline engine
│   │   ├── pipeline.py           # 7-stage pipeline
│   │   ├── content_detection/    # AI detection
│   │   └── decisions.py          # Decision engine
│   ├── db/                       # 11 files - Database
│   │   ├── models.py             # SQLAlchemy models
│   │   └── repositories/         # Repository pattern
│   ├── services/                 # 5 files - Search services (NEW)
│   │   ├── vector_search_service.py
│   │   ├── text_search_service.py
│   │   ├── hybrid_search_service.py
│   │   └── embedding_service.py
│   ├── vector_store_config/      # 2 files - Vector config (NEW)
│   ├── plugins/                  # 21 files - Plugin system
│   │   ├── sources/              # S3, Azure, SharePoint
│   │   ├── parsers/              # Docling, OCR, etc.
│   │   └── destinations/         # Cognee, GraphRAG, Neo4j
│   ├── observability/            # 6 files - Logging, metrics, tracing
│   ├── audit/                    # 3 files - Audit logging
│   ├── lineage/                  # 3 files - Data lineage
│   ├── llm/                      # 3 files - LLM abstraction
│   ├── worker/                   # 3 files - Background processing
│   ├── config.py                 # Configuration management
│   └── main.py                   # FastAPI application entry
├── tests/                        # 70 test files
├── docker/                       # Docker configuration
├── config/                       # YAML configurations
├── docs/                         # Documentation
└── api/                          # OpenAPI specification
```

---

## Appendix B: API Endpoint Summary

| Category | Count | Endpoints |
|----------|-------|-----------|
| Jobs | 7 | POST, GET (list), GET (detail), DELETE, POST (retry), GET (result), GET (events) |
| Upload | 3 | POST (multipart), POST (url), POST (stream) |
| Search | 4 | POST (semantic), POST (text), POST (hybrid), GET (similar) |
| Chunks | 2 | GET (list), GET (detail) |
| Auth | 10 | login, logout, refresh, me, oauth2, api-keys CRUD |
| Audit | 5 | logs, export, summary, events |
| DLQ | 7 | list, get, retry, review, delete, stats, archive |
| Lineage | 7 | graph, summary, verify, stages |
| Health | 5 | health, ready, live, vector, detailed |
| **Total** | **50** | |

---

*End of Architecture Document*
