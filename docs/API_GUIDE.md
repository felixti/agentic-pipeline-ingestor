# API Guide

**Agentic Data Pipeline Ingestor**  
*Complete reference for the Ralph Loop REST API*

> **Version**: 1.0.0  
> **Base URL**: `http://localhost:8000/api/v1`  
> **OpenAPI Spec**: `/openapi.json` or `/api/v1/openapi.yaml`  
> **OpenSpec Change**: review-update-architecture-docs

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Job Management Endpoints](#job-management-endpoints)
4. [File Upload Endpoints](#file-upload-endpoints)
5. [Document Chunk Endpoints (NEW)](#document-chunk-endpoints-new)
6. [Search Endpoints (NEW)](#search-endpoints-new)
7. [Pipeline Configuration Endpoints](#pipeline-configuration-endpoints)
8. [Authentication Endpoints](#authentication-endpoints)
9. [System & Health Endpoints](#system--health-endpoints)
10. [Common Workflows](#common-workflows)
11. [Error Reference](#error-reference)
12. [Rate Limiting](#rate-limiting)

---

## Overview

### Base URL

```
http://localhost:8000/api/v1
```

### Content Types

All API requests and responses use JSON unless otherwise specified:

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |
| `Accept` | `application/json` |

### Standard Response Format

All API responses follow a consistent envelope format:

```json
{
  "data": { ... },
  "meta": {
    "request_id": "uuid",
    "timestamp": "2026-02-18T10:30:00.000000",
    "api_version": "v1",
    "total_count": 100
  },
  "links": {
    "self": "...",
    "next": "...",
    "prev": "..."
  }
}
```

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": [...],
    "documentation_url": "..."
  },
  "meta": {
    "request_id": "uuid",
    "timestamp": "2026-02-18T10:30:00.000000"
  }
}
```

### Response Headers

Every response includes:

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier for tracing |
| `X-API-Version` | API version (v1) |
| `X-RateLimit-Limit` | Maximum requests allowed |
| `X-RateLimit-Remaining` | Remaining requests in window |
| `X-RateLimit-Reset` | Unix timestamp when limit resets |

---

## Authentication

The API supports multiple authentication methods:

### 1. API Key Authentication (Recommended for Services)

Include your API key in the request header:

```bash
curl -H "X-API-Key: your-api-key-here" \
  http://localhost:8000/api/v1/jobs
```

### 2. Bearer Token Authentication (OAuth2/JWT)

```bash
curl -H "Authorization: Bearer your-access-token" \
  http://localhost:8000/api/v1/jobs
```

### 3. Azure AD Authentication (Enterprise SSO)

For Azure AD, use the OAuth2 flow:

```bash
# 1. Get authorization URL
GET /api/v1/auth/oauth2/authorize?redirect_uri=...&provider=azure_ad

# 2. Exchange code for token
POST /api/v1/auth/oauth2/callback
```

### Authentication Examples

#### Python

```python
import requests

headers = {
    "X-API-Key": "your-api-key",
    "Content-Type": "application/json"
}

response = requests.get(
    "http://localhost:8000/api/v1/jobs",
    headers=headers
)
```

#### JavaScript/TypeScript

```typescript
const headers = {
  "X-API-Key": "your-api-key",
  "Content-Type": "application/json"
};

const response = await fetch("http://localhost:8000/api/v1/jobs", {
  headers
});
```

### Permission Levels

| Role | Description | Typical Use |
|------|-------------|-------------|
| `admin` | Full access | System administrators |
| `operator` | Read, Create, Cancel, Retry | Operations team |
| `developer` | Read, Create Jobs | Integration developers |
| `viewer` | Read only | Auditors, read-only access |

---

## Job Management Endpoints

### Create Job

Submit a new document ingestion job.

**Endpoint:** `POST /jobs`

**Authentication:** Required (API Key or Bearer Token)

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_type` | string | Yes | `upload`, `url`, `s3`, `azure_blob`, `sharepoint` |
| `source_uri` | string | Yes | URI to the document |
| `file_name` | string | No | Original filename |
| `file_size` | integer | No | File size in bytes |
| `mime_type` | string | No | MIME type |
| `priority` | string | No | `low`, `normal`, `high` (default: `normal`) |
| `mode` | string | No | `sync` or `async` (default: `async`) |
| `external_id` | string | No | Your reference ID |
| `pipeline_id` | string | No | Pipeline configuration UUID |
| `destination_ids` | array | No | Destination UUIDs |
| `metadata` | object | No | Custom metadata |

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "upload",
    "source_uri": "/tmp/uploads/document.pdf",
    "file_name": "document.pdf",
    "file_size": 1024567,
    "priority": "high",
    "mode": "async",
    "external_id": "doc-12345",
    "metadata": {"department": "finance", "year": 2024}
  }'
```

**Success Response (202 Accepted):**

```json
{
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "created",
    "source_type": "upload",
    "source_uri": "/tmp/uploads/document.pdf",
    "file_name": "document.pdf",
    "file_size": 1024567,
    "mime_type": null,
    "priority": 10,
    "mode": "async",
    "external_id": "doc-12345",
    "retry_count": 0,
    "max_retries": 3,
    "created_at": "2026-02-18T10:30:00.000000",
    "updated_at": "2026-02-18T10:30:00.000000",
    "started_at": null,
    "completed_at": null,
    "error": null
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000",
    "api_version": "v1"
  }
}
```

**Error Response (400 Bad Request):**

```json
{
  "error": {
    "code": "MISSING_SOURCE_TYPE",
    "message": "source_type is required",
    "details": null
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000"
  }
}
```

---

### List Jobs

Retrieve a paginated list of jobs with optional filtering.

**Endpoint:** `GET /jobs`

**Authentication:** Required

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `limit` | integer | 20 | Items per page (max 100) |
| `status` | string | - | Filter by status: `created`, `processing`, `completed`, `failed` |
| `source_type` | string | - | Filter by source type |
| `sort_by` | string | `created_at` | Sort field |
| `sort_order` | string | `desc` | `asc` or `desc` |

**cURL Example:**

```bash
curl "http://localhost:8000/api/v1/jobs?page=1&limit=20&status=completed" \
  -H "X-API-Key: your-api-key"
```

**Success Response (200 OK):**

```json
{
  "data": {
    "items": [
      {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "completed",
        "source_type": "upload",
        "source_uri": "/tmp/uploads/document.pdf",
        "file_name": "document.pdf",
        "file_size": 1024567,
        "priority": 10,
        "mode": "async",
        "retry_count": 0,
        "created_at": "2026-02-18T10:30:00.000000",
        "updated_at": "2026-02-18T10:35:00.000000",
        "started_at": "2026-02-18T10:30:05.000000",
        "completed_at": "2026-02-18T10:35:00.000000"
      }
    ],
    "total": 150,
    "page": 1,
    "page_size": 20
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000",
    "total_count": 150
  },
  "links": {
    "self": "http://localhost:8000/api/v1/jobs?page=1&limit=20",
    "next": "http://localhost:8000/api/v1/jobs?page=2&limit=20"
  }
}
```

---

### Get Job

Retrieve detailed information about a specific job.

**Endpoint:** `GET /jobs/{job_id}`

**Authentication:** Required

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string (UUID) | Job identifier |

**cURL Example:**

```bash
curl "http://localhost:8000/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000" \
  -H "X-API-Key: your-api-key"
```

**Success Response (200 OK):**

```json
{
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "completed",
    "source_type": "upload",
    "source_uri": "/tmp/uploads/document.pdf",
    "file_name": "document.pdf",
    "file_size": 1024567,
    "mime_type": "application/pdf",
    "priority": 10,
    "mode": "async",
    "external_id": "doc-12345",
    "retry_count": 0,
    "max_retries": 3,
    "created_at": "2026-02-18T10:30:00.000000",
    "updated_at": "2026-02-18T10:35:00.000000",
    "started_at": "2026-02-18T10:30:05.000000",
    "completed_at": "2026-02-18T10:35:00.000000",
    "error": null
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000"
  }
}
```

**Error Response (404 Not Found):**

```json
{
  "detail": "Job with ID '123e4567-e89b-12d3-a456-426614174000' not found"
}
```

---

### Cancel Job

Cancel a job that is not yet completed.

**Endpoint:** `DELETE /jobs/{job_id}`

**Authentication:** Required (operator or admin)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string (UUID) | Job identifier |

**cURL Example:**

```bash
curl -X DELETE "http://localhost:8000/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000" \
  -H "X-API-Key: your-api-key"
```

**Success Response (204 No Content)**

**Error Response (400 Bad Request):**

```json
{
  "detail": "Cannot cancel job with status 'completed'"
}
```

---

### Retry Job

Retry a failed job with a new job instance.

**Endpoint:** `POST /jobs/{job_id}/retry`

**Authentication:** Required (operator or admin)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string (UUID) | Job identifier |

**Request Body (Optional):**

| Field | Type | Description |
|-------|------|-------------|
| `force_parser` | string | Override parser for retry |
| `updated_config` | object | Updated pipeline configuration |

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000/retry" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "force_parser": "azure_ocr"
  }'
```

**Success Response (202 Accepted):**

```json
{
  "data": {
    "original_job_id": "123e4567-e89b-12d3-a456-426614174000",
    "new_job_id": "987fcdeb-51a2-43d4-b567-890123456789",
    "status": "created",
    "message": "Job retry initiated",
    "job": { ... }
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000"
  }
}
```

---

### Get Job Result

Retrieve the processing result for a completed job.

**Endpoint:** `GET /jobs/{job_id}/result`

**Authentication:** Required

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string (UUID) | Job identifier |

**cURL Example:**

```bash
curl "http://localhost:8000/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000/result" \
  -H "X-API-Key: your-api-key"
```

**Success Response (200 OK):**

```json
{
  "data": {
    "job_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "completed",
    "success": true,
    "extracted_text": "Full extracted text content...",
    "output_data": {
      "pages": 10,
      "chunks": 25,
      "entities": [...]
    },
    "metadata": {
      "parser": "docling",
      "processing_time_seconds": 45
    },
    "quality_score": 0.92,
    "processing_time_ms": 45000,
    "output_uri": "s3://bucket/output/123e4567/result.json",
    "created_at": "2026-02-18T10:35:00.000000"
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000"
  }
}
```

---

## File Upload Endpoints

### Upload File(s)

Upload one or more files for processing.

**Endpoint:** `POST /upload`

**Authentication:** Required

**Content-Type:** `multipart/form-data`

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | file(s) | Yes | File(s) to upload |
| `priority` | string | No | `low`, `normal`, `high` |
| `pipeline_id` | string | No | Pipeline configuration UUID |
| `external_id` | string | No | Your reference ID |

**cURL Example - Single File:**

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "X-API-Key: your-api-key" \
  -F "files=@document.pdf" \
  -F "priority=high"
```

**cURL Example - Multiple Files:**

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "X-API-Key: your-api-key" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "priority=normal"
```

**Success Response (202 Accepted) - Single File:**

```json
{
  "data": {
    "message": "File uploaded successfully",
    "job_id": "123e4567-e89b-12d3-a456-426614174000",
    "file_name": "document.pdf",
    "file_size": 1024567
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000"
  }
}
```

**Success Response (202 Accepted) - Multiple Files:**

```json
{
  "data": {
    "message": "2 files uploaded successfully",
    "job_ids": [
      "123e4567-e89b-12d3-a456-426614174000",
      "987fcdeb-51a2-43d4-b567-890123456789"
    ],
    "files": ["document1.pdf", "document2.pdf"]
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000"
  }
}
```

---

### Ingest from URL

Ingest a document from a remote URL.

**Endpoint:** `POST /upload/url`

**Authentication:** Required

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | URL to download from |
| `filename` | string | No | Override filename |
| `priority` | string | No | `low`, `normal`, `high` |
| `mode` | string | No | `sync` or `async` |
| `external_id` | string | No | Your reference ID |
| `headers` | object | No | Custom HTTP headers |

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/upload/url" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "priority": "normal",
    "mode": "async",
    "headers": {
      "Authorization": "Bearer token"
    }
  }'
```

**Success Response (202 Accepted):**

```json
{
  "data": {
    "message": "URL ingestion started",
    "job_id": "123e4567-e89b-12d3-a456-426614174000",
    "file_name": "document.pdf",
    "file_size": 1024567
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000"
  }
}
```

---

## Document Chunk Endpoints (NEW)

These endpoints provide access to document chunks generated during the Chunk stage of the 7-stage pipeline. Chunks are text segments extracted from documents with optional vector embeddings for semantic search.

### List Chunks

Retrieve all chunks for a specific job with pagination.

**Endpoint:** `GET /jobs/{job_id}/chunks`

**Authentication:** Required

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string (UUID) | Job identifier |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 100 | Maximum results (1-1000) |
| `offset` | integer | 0 | Number of chunks to skip |
| `include_embedding` | boolean | false | Include vector embedding (performance impact) |

**cURL Example:**

```bash
curl "http://localhost:8000/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000/chunks?limit=10&offset=0" \
  -H "X-API-Key: your-api-key"
```

**Success Response (200 OK):**

```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 0,
      "content": "This is the first chunk of text content extracted from the document...",
      "content_hash": "a1b2c3d4e5f6...",
      "metadata": {
        "start_char": 0,
        "end_char": 1000,
        "char_count": 1000,
        "page": 1
      },
      "created_at": "2026-02-18T10:30:00.000000"
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440001",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 1,
      "content": "Second chunk of text content...",
      "content_hash": "b2c3d4e5f6a7...",
      "metadata": {
        "start_char": 1000,
        "end_char": 2000,
        "char_count": 1000,
        "page": 1
      },
      "created_at": "2026-02-18T10:30:00.000000"
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

**Error Response (404 Not Found):**

```json
{
  "detail": "Job with ID '123e4567-e89b-12d3-a456-426614174000' not found"
}
```

**Error Response (400 Bad Request):**

```json
{
  "detail": "Invalid UUID format"
}
```

**Rate Limit:** 120 requests per 60 seconds

---

### Get Chunk

Retrieve a specific chunk by ID with optional embedding data.

**Endpoint:** `GET /jobs/{job_id}/chunks/{chunk_id}`

**Authentication:** Required

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string (UUID) | Job identifier |
| `chunk_id` | string (UUID) | Chunk identifier |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_embedding` | boolean | false | Include vector embedding |

**cURL Example:**

```bash
curl "http://localhost:8000/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000/chunks/550e8400-e29b-41d4-a716-446655440000?include_embedding=true" \
  -H "X-API-Key: your-api-key"
```

**Success Response (200 OK):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "chunk_index": 0,
  "content": "This is the first chunk of text content extracted from the document...",
  "content_hash": "a1b2c3d4e5f6...",
  "embedding": [0.023, -0.045, 0.128, 0.003, -0.087, ...],
  "metadata": {
    "start_char": 0,
    "end_char": 1000,
    "char_count": 1000,
    "page": 1,
    "embedding_model": "text-embedding-3-small"
  },
  "created_at": "2026-02-18T10:30:00.000000"
}
```

**Error Response (404 Not Found):**

```json
{
  "detail": "Chunk with ID '550e8400-e29b-41d4-a716-446655440000' not found"
}
```

**Error Response (400 Bad Request):**

```json
{
  "detail": "Chunk does not belong to the specified job"
}
```

**Rate Limit:** 120 requests per 60 seconds

---

## Search Endpoints (NEW)

These endpoints enable semantic and text search across all document chunks using pgvector and PostgreSQL full-text search capabilities.

### Semantic Search

Search for similar chunks using a pre-computed embedding vector with cosine similarity.

**Endpoint:** `POST /search/semantic`

**Authentication:** Required

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query_embedding` | array[float] | Yes | Query embedding vector (1536 dimensions for text-embedding-3-small) |
| `top_k` | integer | No | Maximum results (1-100, default: 10) |
| `min_similarity` | float | No | Minimum similarity 0-1 (default: 0.7) |
| `filters` | object | No | Optional filters (e.g., `job_id`) |

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query_embedding": [0.023, -0.045, 0.128, 0.003, -0.087, ...],
    "top_k": 10,
    "min_similarity": 0.75,
    "filters": {
      "job_id": "123e4567-e89b-12d3-a456-426614174000"
    }
  }'
```

**Success Response (200 OK):**

```json
{
  "results": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 0,
      "content": "Neural network architecture involves layers of interconnected nodes...",
      "metadata": {"page": 5, "section": "architecture"},
      "similarity_score": 0.9234,
      "rank": 1
    },
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 3,
      "content": "Deep learning models use multiple layers to extract features...",
      "metadata": {"page": 7, "section": "deep_learning"},
      "similarity_score": 0.8912,
      "rank": 2
    }
  ],
  "total": 8,
  "query_time_ms": 25.5
}
```

**Error Response (400 Bad Request):**

```json
{
  "detail": "Invalid embedding vector: dimension mismatch"
}
```

**Error Response (422 Validation Error):**

```json
{
  "detail": "query_embedding cannot be empty"
}
```

**Rate Limit:** 60 requests per 60 seconds

---

### Text Search

Full-text search using PostgreSQL's tsvector/tsquery with BM25 ranking and optional fuzzy trigram matching.

**Endpoint:** `POST /search/text`

**Authentication:** Required

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query text (1-1024 characters) |
| `top_k` | integer | No | Maximum results (1-100, default: 10) |
| `language` | string | No | Text search language (default: "english") |
| `use_fuzzy` | boolean | No | Include fuzzy trigram matching (default: true) |
| `highlight` | boolean | No | Include highlighted snippets (default: false) |
| `filters` | object | No | Optional filters |

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning architecture",
    "top_k": 10,
    "language": "english",
    "use_fuzzy": true,
    "highlight": true,
    "filters": {}
  }'
```

**Success Response (200 OK):**

```json
{
  "results": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 0,
      "content": "Machine learning architecture is fundamental to building effective AI systems...",
      "metadata": {"page": 1},
      "similarity_score": 0.8567,
      "rank": 1,
      "highlighted_content": "<mark>Machine learning</mark> <mark>architecture</mark> is fundamental to building effective AI systems...",
      "matched_terms": ["machine", "learning", "architecture"]
    },
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440003",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 2,
      "content": "Neural networks are a subset of machine learning...",
      "metadata": {"page": 3},
      "similarity_score": 0.7934,
      "rank": 2,
      "highlighted_content": "Neural networks are a subset of <mark>machine learning</mark>...",
      "matched_terms": ["machine", "learning"]
    }
  ],
  "total": 12,
  "query_time_ms": 15.3
}
```

**Error Response (400 Bad Request):**

```json
{
  "detail": "Query must be at least 2 characters"
}
```

**Rate Limit:** 100 requests per 60 seconds

---

### Hybrid Search

Combine vector similarity and text search using weighted sum or Reciprocal Rank Fusion (RRF).

**Endpoint:** `POST /search/hybrid`

**Authentication:** Required

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query text |
| `top_k` | integer | No | Maximum results (1-100, default: 10) |
| `vector_weight` | float | No | Vector score weight 0-1 (default: 0.7) |
| `text_weight` | float | No | Text score weight 0-1 (default: 0.3) |
| `fusion_method` | string | No | `weighted_sum` or `rrf` (default: `weighted_sum`) |
| `min_similarity` | float | No | Minimum similarity 0-1 (default: 0.5) |
| `filters` | object | No | Optional filters |

**Validation Rules:**
- `vector_weight + text_weight` must equal approximately 1.0
- Weights must be between 0.0 and 1.0

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural network architecture",
    "top_k": 10,
    "vector_weight": 0.7,
    "text_weight": 0.3,
    "fusion_method": "weighted_sum",
    "min_similarity": 0.5,
    "filters": {}
  }'
```

**Success Response (200 OK):**

```json
{
  "results": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 0,
      "content": "Neural network architecture involves multiple layers...",
      "metadata": {"page": 5},
      "hybrid_score": 0.8856,
      "vector_score": 0.9234,
      "text_score": 0.7834,
      "vector_rank": 1,
      "text_rank": 3,
      "rank": 1,
      "fusion_method": "weighted_sum"
    },
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440005",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 5,
      "content": "Deep learning architectures build upon neural networks...",
      "metadata": {"page": 8},
      "hybrid_score": 0.8543,
      "vector_score": 0.8912,
      "text_score": 0.7654,
      "vector_rank": 2,
      "text_rank": 2,
      "rank": 2,
      "fusion_method": "weighted_sum"
    }
  ],
  "total": 15,
  "query_time_ms": 45.2
}
```

**Error Response (400 Bad Request):**

```json
{
  "detail": "vector_weight + text_weight must equal 1.0, got 1.2"
}
```

**Error Response (422 Validation Error):**

```json
{
  "detail": [
    {
      "loc": ["body", "fusion_method"],
      "msg": "String should match pattern '^(weighted_sum|rrf)$'",
      "type": "string_pattern_mismatch"
    }
  ]
}
```

**Rate Limit:** 30 requests per 60 seconds

---

### Find Similar Chunks

Find chunks semantically similar to a reference chunk using its stored embedding.

**Endpoint:** `GET /search/similar/{chunk_id}`

**Authentication:** Required

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `chunk_id` | string (UUID) | Reference chunk UUID |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | integer | 10 | Maximum results (1-100) |
| `exclude_self` | boolean | true | Exclude the reference chunk from results |

**cURL Example:**

```bash
curl "http://localhost:8000/api/v1/search/similar/550e8400-e29b-41d4-a716-446655440000?top_k=5&exclude_self=true" \
  -H "X-API-Key: your-api-key"
```

**Success Response (200 OK):**

```json
{
  "results": [
    {
      "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 1,
      "content": "Related content about neural networks and their applications...",
      "metadata": {"page": 2},
      "similarity_score": 0.8934,
      "rank": 1
    },
    {
      "chunk_id": "660e8400-e29b-41d4-a716-446655440002",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 4,
      "content": "Similar concepts regarding deep learning architectures...",
      "metadata": {"page": 6},
      "similarity_score": 0.8567,
      "rank": 2
    }
  ],
  "total": 5,
  "query_time_ms": 18.7
}
```

**Error Response (404 Not Found):**

```json
{
  "detail": "Reference chunk with ID '550e8400-e29b-41d4-a716-446655440000' not found"
}
```

**Error Response (400 Bad Request):**

```json
{
  "detail": "Reference chunk does not have an embedding"
}
```

**Rate Limit:** 60 requests per 60 seconds

---

## Pipeline Configuration Endpoints

### List Pipelines

Retrieve all pipeline configurations.

**Endpoint:** `GET /pipelines`

**Authentication:** Required

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `limit` | integer | 20 | Items per page |

**cURL Example:**

```bash
curl "http://localhost:8000/api/v1/pipelines?page=1&limit=20" \
  -H "X-API-Key: your-api-key"
```

**Success Response (200 OK):**

```json
{
  "data": {
    "items": [
      {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "name": "default-pipeline",
        "description": "Default document processing pipeline",
        "config": {
          "content_detection": {...},
          "parser": {...},
          "transformation": {...}
        },
        "version": 1,
        "is_active": true,
        "created_by": "admin",
        "created_at": "2026-02-18T10:30:00.000000",
        "updated_at": "2026-02-18T10:30:00.000000"
      }
    ],
    "total": 5,
    "page": 1,
    "page_size": 20
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000"
  }
}
```

---

### Create Pipeline

Create a new pipeline configuration.

**Endpoint:** `POST /pipelines`

**Authentication:** Required (admin)

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique pipeline name |
| `description` | string | No | Pipeline description |
| `config` | object | Yes | Pipeline configuration |

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/pipelines" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ocr-pipeline",
    "description": "Optimized for scanned documents",
    "config": {
      "content_detection": {
        "auto_detect": true,
        "detection_method": "hybrid"
      },
      "parser": {
        "primary_parser": "azure_ocr",
        "fallback_parser": "tesseract"
      },
      "transformation": {
        "chunking": {
          "enabled": true,
          "strategy": "semantic",
          "chunk_size": 1000,
          "chunk_overlap": 200
        },
        "generate_embeddings": true,
        "embedding_model": "text-embedding-3-small"
      }
    }
  }'
```

**Success Response (201 Created):**

```json
{
  "data": {
    "id": "987fcdeb-51a2-43d4-b567-890123456789",
    "name": "ocr-pipeline",
    "description": "Optimized for scanned documents",
    "config": {...},
    "version": 1,
    "created_at": "2026-02-18T10:30:00.000000"
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2026-02-18T10:30:00.000000"
  }
}
```

---

## Authentication Endpoints

### Get Current User

Retrieve information about the currently authenticated user.

**Endpoint:** `GET /auth/me`

**Authentication:** Required

**cURL Example:**

```bash
curl "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer your-access-token"
```

**Success Response (200 OK):**

```json
{
  "id": "user-123",
  "email": "user@example.com",
  "username": "john.doe",
  "role": "operator",
  "roles": ["operator", "viewer"],
  "permissions": ["READ", "CREATE", "CANCEL"],
  "auth_provider": "oauth2",
  "is_service_account": false
}
```

---

### Refresh Token

Refresh an expired access token using a refresh token.

**Endpoint:** `POST /auth/token/refresh`

**Authentication:** None (requires valid refresh token)

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `refresh_token` | string | Yes | Valid refresh token |

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/auth/token/refresh" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "your-refresh-token"
  }'
```

**Success Response (200 OK):**

```json
{
  "access_token": "new-access-token",
  "token_type": "bearer",
  "expires_in": 1800
}
```

---

### OAuth2 Authorize

Initiate OAuth2 authorization flow.

**Endpoint:** `GET /auth/oauth2/authorize`

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `redirect_uri` | string | Yes | Callback URL |
| `provider` | string | No | `oauth2` or `azure_ad` (default: `oauth2`) |

**cURL Example:**

```bash
curl "http://localhost:8000/api/v1/auth/oauth2/authorize?redirect_uri=http://localhost:3000/callback&provider=azure_ad"
```

**Success Response (200 OK):**

```json
{
  "authorization_url": "https://login.microsoftonline.com/.../oauth2/authorize?...",
  "state": "random-state-string"
}
```

---

### Create API Key

Create a new API key (admin only).

**Endpoint:** `POST /auth/api-keys`

**Authentication:** Required (admin)

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Key name |
| `description` | string | No | Key description |
| `role` | string | No | `admin`, `operator`, `developer`, `viewer` |
| `expires_in_days` | integer | No | Expiration in days (1-365) |
| `permissions` | array | No | Custom permissions |

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/auth/api-keys" \
  -H "X-API-Key: your-admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "production-service",
    "description": "API key for production service",
    "role": "operator",
    "expires_in_days": 90
  }'
```

**Success Response (200 OK):**

```json
{
  "id": "key-123",
  "key": "sk_live_abc123xyz789...",
  "name": "production-service",
  "role": "operator",
  "expires_at": "2026-05-18T10:30:00.000000"
}
```

**Important:** The actual API key is only returned once. Store it securely.

---

## System & Health Endpoints

### Health Check

Comprehensive health check of all system components.

**Endpoint:** `GET /health`

**Authentication:** None (public)

**cURL Example:**

```bash
curl "http://localhost:8000/health"
```

**Success Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-02-18T10:30:00.000000",
  "components": {
    "api": {
      "status": "healthy"
    },
    "database": {
      "status": "healthy",
      "latency_ms": 2.5
    },
    "redis": {
      "status": "healthy",
      "latency_ms": 1.2
    },
    "plugins": {
      "status": "healthy",
      "message": "5/5 plugins healthy"
    },
    "vector_store": {
      "status": "healthy",
      "message": "pgvector and pg_trgm extensions available"
    }
  }
}
```

---

### Readiness Probe

Kubernetes readiness probe.

**Endpoint:** `GET /health/ready`

**Authentication:** None

**cURL Example:**

```bash
curl "http://localhost:8000/health/ready"
```

**Success Response (200 OK):**

```json
{
  "ready": true,
  "timestamp": "2026-02-18T10:30:00.000000",
  "checks": {
    "database": true,
    "redis": true
  }
}
```

**Error Response (503 Service Unavailable):**

```json
{
  "ready": false,
  "timestamp": "2026-02-18T10:30:00.000000",
  "checks": {
    "database": false,
    "redis": true
  }
}
```

---

### Liveness Probe

Kubernetes liveness probe.

**Endpoint:** `GET /health/live`

**Authentication:** None

**cURL Example:**

```bash
curl "http://localhost:8000/health/live"
```

**Success Response (200 OK):**

```json
{
  "alive": true,
  "timestamp": "2026-02-18T10:30:00.000000"
}
```

---

### Vector Store Health

Specific health check for vector store (pgvector).

**Endpoint:** `GET /health/vector`

**Authentication:** None

**cURL Example:**

```bash
curl "http://localhost:8000/health/vector"
```

**Success Response (200 OK):**

```json
{
  "healthy": true,
  "status": "healthy",
  "message": "pgvector and pg_trgm extensions available",
  "latency_ms": 2.5,
  "extensions": {
    "vector": "0.8.0",
    "pg_trgm": "1.6"
  },
  "pgvector_version": "0.8.0",
  "pg_trgm_version": "1.6",
  "timestamp": "2026-02-18T10:30:00.000000"
}
```

---

### Prometheus Metrics

Retrieve Prometheus metrics for monitoring.

**Endpoint:** `GET /metrics`

**Authentication:** None (internal network recommended)

**cURL Example:**

```bash
curl "http://localhost:8000/metrics"
```

**Success Response (200 OK):**

```
# HELP pipeline_jobs_total Total jobs processed
# TYPE pipeline_jobs_total counter
pipeline_jobs_total{status="completed"} 150
pipeline_jobs_total{status="failed"} 10

# HELP vector_search_duration_seconds Search latency
# TYPE vector_search_duration_seconds histogram
...
```

---

## Common Workflows

### 1. Basic Document Ingestion Flow

Upload a document, monitor processing, and retrieve results:

```bash
# Step 1: Upload file
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "X-API-Key: your-api-key" \
  -F "files=@document.pdf" \
  -F "priority=high"

# Response: {"job_id": "123e4567-e89b-12d3-a456-426614174000", ...}

# Step 2: Poll for completion
while true; do
  STATUS=$(curl -s "http://localhost:8000/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000" \
    -H "X-API-Key: your-api-key" | jq -r '.data.status')
  echo "Status: $STATUS"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  sleep 5
done

# Step 3: Get results
curl "http://localhost:8000/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000/result" \
  -H "X-API-Key: your-api-key"
```

### 2. Search Workflow (Ingest → Search)

Process a document and then search within it:

```python
import requests
import time

API_BASE = "http://localhost:8000/api/v1"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY}

# Step 1: Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{API_BASE}/upload",
        headers=headers,
        files={"files": f}
    )
job_id = response.json()["data"]["job_id"]

# Step 2: Wait for completion
while True:
    response = requests.get(f"{API_BASE}/jobs/{job_id}", headers=headers)
    status = response.json()["data"]["status"]
    if status in ["completed", "failed"]:
        break
    time.sleep(5)

# Step 3: Search the document
search_response = requests.post(
    f"{API_BASE}/search/text",
    headers={**headers, "Content-Type": "application/json"},
    json={
        "query": "machine learning",
        "top_k": 10,
        "filters": {"job_id": job_id},
        "highlight": True
    }
)

results = search_response.json()["results"]
for result in results:
    print(f"Score: {result['similarity_score']:.2f}")
    print(f"Content: {result['highlighted_content']}")
    print()
```

### 3. Semantic Search Workflow

Search using pre-computed embeddings:

```python
# Generate embedding using your embedding service
query_embedding = generate_embedding("neural network architecture")

# Search for similar chunks
response = requests.post(
    f"{API_BASE}/search/semantic",
    headers={**headers, "Content-Type": "application/json"},
    json={
        "query_embedding": query_embedding,
        "top_k": 10,
        "min_similarity": 0.7,
        "filters": {"job_id": "specific-job-uuid"}
    }
)

results = response.json()["results"]
```

### 4. Hybrid Search for Best Results

Combine semantic and text search:

```python
response = requests.post(
    f"{API_BASE}/search/hybrid",
    headers={**headers, "Content-Type": "application/json"},
    json={
        "query": "neural network architecture",
        "top_k": 10,
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "fusion_method": "weighted_sum",
        "min_similarity": 0.5
    }
)

for result in response.json()["results"]:
    print(f"Rank: {result['rank']}")
    print(f"Hybrid Score: {result['hybrid_score']:.3f}")
    print(f"Vector Score: {result['vector_score']:.3f}")
    print(f"Text Score: {result['text_score']:.3f}")
    print(f"Content: {result['content'][:200]}...")
    print()
```

### 5. Pagination Pattern

Handle paginated responses:

```python
def get_all_jobs():
    all_jobs = []
    page = 1
    
    while True:
        response = requests.get(
            f"{API_BASE}/jobs",
            headers=headers,
            params={"page": page, "limit": 100}
        )
        data = response.json()["data"]
        all_jobs.extend(data["items"])
        
        # Check if we've reached the end
        if page * data["page_size"] >= data["total"]:
            break
        page += 1
    
    return all_jobs
```

---

## Error Reference

### HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| `200` | OK | Successful GET/PUT/DELETE |
| `201` | Created | Successful POST (resource created) |
| `202` | Accepted | Job submitted for async processing |
| `204` | No Content | Successful DELETE with no body |
| `400` | Bad Request | Invalid request body, validation errors |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Resource doesn't exist |
| `409` | Conflict | Resource conflict (e.g., duplicate) |
| `422` | Unprocessable | Validation error, semantic errors |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Error | Server-side error |
| `503` | Service Unavailable | Dependencies unavailable |

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `MISSING_SOURCE_TYPE` | Required field missing | Include `source_type` in request |
| `INVALID_PIPELINE` | Pipeline not found | Check `pipeline_id` is valid |
| `INVALID_EMBEDDING` | Embedding dimension mismatch | Ensure embedding has correct dimensions |
| `INVALID_QUERY` | Empty or invalid query | Provide non-empty query string |
| `INVALID_WEIGHTS` | Vector + text weights != 1.0 | Adjust weights to sum to 1.0 |
| `JOB_NOT_COMPLETE` | Job still processing | Wait for job to complete |
| `CHUNK_NOT_FOUND` | Chunk doesn't exist | Verify `chunk_id` |
| `NO_EMBEDDING` | Chunk has no embedding | Ensure chunk was processed with embeddings |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |
| `UPLOAD_FAILED` | File upload error | Check file format and size |

### Error Handling Pattern

```python
import requests
from requests.exceptions import HTTPError

def api_request(method, endpoint, **kwargs):
    try:
        response = requests.request(
            method,
            f"{API_BASE}{endpoint}",
            headers=headers,
            **kwargs
        )
        response.raise_for_status()
        return response.json()
    except HTTPError as e:
        if e.response.status_code == 429:
            # Rate limited - implement backoff
            retry_after = int(e.response.headers.get('X-RateLimit-Reset', 60))
            print(f"Rate limited. Retry after {retry_after} seconds")
        elif e.response.status_code == 404:
            print("Resource not found")
        else:
            error_data = e.response.json()
            print(f"Error: {error_data.get('error', {}).get('message', str(e))}")
        raise
```

---

## Rate Limiting

The API implements tiered rate limiting based on endpoint type:

| Endpoint Category | Limit | Window |
|-------------------|-------|--------|
| Job Management (list/get) | 100 | 60 seconds |
| Job Creation | 30 | 60 seconds |
| File Upload | 10 | 60 seconds |
| List/Get Chunks | 120 | 60 seconds |
| Semantic Search | 60 | 60 seconds |
| Text Search | 100 | 60 seconds |
| Hybrid Search | 30 | 60 seconds |
| Similar Chunks | 60 | 60 seconds |
| Health Checks | 60 | 60 seconds |

### Rate Limit Headers

Every response includes rate limit information:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1708255800
```

### Rate Limit Exceeded Response

```json
{
  "detail": "Rate limit exceeded",
  "retry_after": 45
}
```

---

## SDK Examples

### Python Client

```python
import requests
from typing import Optional, List

class RalphLoopClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def create_job(self, source_type: str, source_uri: str, **kwargs):
        """Create a new ingestion job."""
        response = requests.post(
            f"{self.base_url}/jobs",
            headers=self.headers,
            json={
                "source_type": source_type,
                "source_uri": source_uri,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()["data"]
    
    def get_job(self, job_id: str):
        """Get job details."""
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["data"]
    
    def list_chunks(self, job_id: str, limit: int = 100):
        """List chunks for a job."""
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}/chunks",
            headers=self.headers,
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()["items"]
    
    def semantic_search(self, query_embedding: List[float], **kwargs):
        """Perform semantic search."""
        response = requests.post(
            f"{self.base_url}/search/semantic",
            headers=self.headers,
            json={
                "query_embedding": query_embedding,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()["results"]
    
    def text_search(self, query: str, **kwargs):
        """Perform text search."""
        response = requests.post(
            f"{self.base_url}/search/text",
            headers=self.headers,
            json={
                "query": query,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()["results"]
    
    def hybrid_search(self, query: str, **kwargs):
        """Perform hybrid search."""
        response = requests.post(
            f"{self.base_url}/search/hybrid",
            headers=self.headers,
            json={
                "query": query,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()["results"]

# Usage
client = RalphLoopClient(
    base_url="http://localhost:8000/api/v1",
    api_key="your-api-key"
)

# Upload and search
job = client.create_job("upload", "/path/to/file.pdf")
# ... wait for completion ...
chunks = client.list_chunks(job["id"])
```

### JavaScript/TypeScript Client

```typescript
interface SearchResult {
  chunk_id: string;
  job_id: string;
  content: string;
  similarity_score: number;
  rank: number;
}

class RalphLoopClient {
  constructor(
    private baseUrl: string,
    private apiKey: string
  ) {}

  private async request<T>(
    method: string,
    endpoint: string,
    body?: object
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method,
      headers: {
        "X-API-Key": this.apiKey,
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  }

  async createJob(params: {
    source_type: string;
    source_uri: string;
    [key: string]: any;
  }) {
    const result = await this.request<any>("POST", "/jobs", params);
    return result.data;
  }

  async getJob(jobId: string) {
    const result = await this.request<any>("GET", `/jobs/${jobId}`);
    return result.data;
  }

  async listChunks(jobId: string, limit = 100) {
    const result = await this.request<any>(
      "GET",
      `/jobs/${jobId}/chunks?limit=${limit}`
    );
    return result.items;
  }

  async textSearch(query: string, options: { top_k?: number; filters?: object } = {}) {
    const result = await this.request<{ results: SearchResult[] }>(
      "POST",
      "/search/text",
      { query, ...options }
    );
    return result.results;
  }

  async hybridSearch(query: string, options: { 
    top_k?: number;
    vector_weight?: number;
    text_weight?: number;
    filters?: object;
  } = {}) {
    const result = await this.request<{ results: SearchResult[] }>(
      "POST",
      "/search/hybrid",
      { query, ...options }
    );
    return result.results;
  }
}

// Usage
const client = new RalphLoopClient(
  "http://localhost:8000/api/v1",
  "your-api-key"
);

const results = await client.hybridSearch("neural networks", {
  top_k: 10,
  vector_weight: 0.7,
  text_weight: 0.3
});
```

---

## Additional Resources

- **OpenAPI Spec**: `/openapi.json` or `/api/v1/openapi.yaml`
- **Interactive Docs**: `/docs` (Swagger UI)
- **ReDoc**: `/redoc`
- **Vector Store Guide**: [vector_store_api.md](./vector_store_api.md)
- **Usage Examples**: [VECTOR_STORE_API_USAGE.md](./VECTOR_STORE_API_USAGE.md)
- **Architecture**: [ARCHITECTURE.md](../ARCHITECTURE.md)

---

## API Endpoint Summary

| Category | Count | Endpoints |
|----------|-------|-----------|
| Jobs | 6 | POST, GET (list), GET (detail), DELETE, POST (retry), GET (result) |
| Upload | 2 | POST (file), POST (url) |
| Chunks | 2 | GET (list), GET (detail) |
| Search | 4 | POST (semantic), POST (text), POST (hybrid), GET (similar) |
| Pipelines | 5 | GET (list), GET (detail), POST, PUT, DELETE |
| Sources | 1 | GET (list) |
| Destinations | 1 | GET (list) |
| Auth | 7 | GET (me), POST (refresh), GET/POST (oauth2), GET/POST/DELETE (api-keys) |
| Health | 5 | GET /health, /health/ready, /health/live, /health/vector, /health/detailed/{component} |
| System | 2 | GET /metrics, GET /api/v1/openapi.yaml |
| **Total** | **33** | *Core endpoints documented. System has 50+ total endpoints.* |

---

*For support, contact your system administrator or refer to the project documentation.*
