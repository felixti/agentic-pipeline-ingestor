# Vector Store API Documentation

## Overview

The Vector Store API provides semantic search capabilities using PostgreSQL with pgvector extension. It enables:

- **Semantic Search**: Find similar content using vector embeddings
- **Text Search**: Full-text search with BM25 ranking
- **Hybrid Search**: Combine vector and text search for better results
- **Document Chunking**: Automatic text segmentation and embedding generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│  GET /jobs/{id}/chunks  │  POST /search/{semantic|text|hybrid} │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     Service Layer                           │
│  VectorSearchService │ TextSearchService │ HybridSearchService │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Repository Layer                          │
│              DocumentChunkRepository                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Database Layer                            │
│  PostgreSQL + pgvector + pg_trgm + HNSW/GIN indexes        │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Vector Store Configuration File

The vector store is configured via `config/vector_store.yaml`:

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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VECTOR_STORE_ENABLED` | Enable/disable vector store | `true` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `EMBEDDING_API_KEY` | API key for embedding provider | - |
| `EMBEDDING_API_BASE` | Base URL for embedding provider | - |
| `EMBEDDING_DIMENSIONS` | Embedding dimensions | `1536` |

## API Endpoints

### Document Chunks

#### List Chunks for a Job

```http
GET /api/v1/jobs/{job_id}/chunks
```

Retrieve all document chunks for a specific job with pagination.

**Parameters:**

| Name | Type | In | Required | Description |
|------|------|-----|----------|-------------|
| `job_id` | string (UUID) | path | Yes | Job UUID |
| `limit` | integer | query | No | Maximum results (1-1000, default: 100) |
| `offset` | integer | query | No | Skip offset (default: 0) |
| `include_embedding` | boolean | query | No | Include vector embedding (default: false) |

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000/chunks?limit=10&offset=0" \
  -H "X-API-Key: your-api-key"
```

**Example Response:**

```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 0,
      "content": "This is the first chunk of text content...",
      "content_hash": "a1b2c3d4e5f6...",
      "metadata": {
        "start_char": 0,
        "end_char": 1000,
        "char_count": 1000
      },
      "created_at": "2026-02-18T10:30:00.000000"
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

**Error Responses:**

| Status | Description | Example Response |
|--------|-------------|------------------|
| `400` | Invalid UUID | `{"detail": "Invalid UUID format"}` |
| `404` | Job not found | `{"detail": "Job with ID '...' not found"}` |
| `429` | Rate limit exceeded | `{"detail": "Rate limit exceeded", "retry_after": 60}` |

---

#### Get Single Chunk

```http
GET /api/v1/jobs/{job_id}/chunks/{chunk_id}
```

Retrieve a specific chunk by ID with optional embedding data.

**Parameters:**

| Name | Type | In | Required | Description |
|------|------|-----|----------|-------------|
| `job_id` | string (UUID) | path | Yes | Job UUID |
| `chunk_id` | string (UUID) | path | Yes | Chunk UUID |
| `include_embedding` | boolean | query | No | Include vector embedding (default: false) |

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000/chunks/550e8400-e29b-41d4-a716-446655440000?include_embedding=true" \
  -H "X-API-Key: your-api-key"
```

**Example Response (with embedding):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "chunk_index": 0,
  "content": "This is the first chunk of text content...",
  "content_hash": "a1b2c3d4e5f6...",
  "embedding": [0.023, -0.045, 0.128, ...],  // 1536 dimensions
  "metadata": {
    "start_char": 0,
    "end_char": 1000,
    "char_count": 1000,
    "embedding_model": "text-embedding-3-small"
  },
  "created_at": "2026-02-18T10:30:00.000000"
}
```

**Error Responses:**

| Status | Description | Example Response |
|--------|-------------|------------------|
| `400` | Chunk doesn't belong to job | `{"detail": "Chunk does not belong to the specified job"}` |
| `404` | Chunk not found | `{"detail": "Chunk with ID '...' not found"}` |
| `429` | Rate limit exceeded | `{"detail": "Rate limit exceeded"}` |

---

### Search Endpoints

#### Semantic Search

```http
POST /api/v1/search/semantic
```

Search for similar chunks using a pre-computed embedding vector with cosine similarity.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query_embedding` | array[float] | Yes | Query embedding vector |
| `top_k` | integer | No | Maximum results (1-100, default: 10) |
| `min_similarity` | float | No | Minimum similarity 0-1 (default: 0.7) |
| `filters` | object | No | Optional filters (job_id, metadata) |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query_embedding": [0.023, -0.045, 0.128, ...],
    "top_k": 10,
    "min_similarity": 0.75,
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
      "content": "This is the matching chunk content...",
      "metadata": {"page": 1},
      "similarity_score": 0.9234,
      "rank": 1
    }
  ],
  "total": 5,
  "query_time_ms": 25.5
}
```

**Error Responses:**

| Status | Description | Example Response |
|--------|-------------|------------------|
| `400` | Invalid embedding | `{"detail": "Invalid embedding vector: dimension mismatch"}` |
| `422` | Validation error | `{"detail": "query_embedding cannot be empty"}` |
| `429` | Rate limit exceeded | `{"detail": "Rate limit exceeded"}` |

---

#### Text Search

```http
POST /api/v1/search/text
```

Search using full-text search with BM25 ranking and optional fuzzy matching.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query text (1-1024 chars) |
| `top_k` | integer | No | Maximum results (1-100, default: 10) |
| `language` | string | No | Language config (default: "english") |
| `use_fuzzy` | boolean | No | Include fuzzy matching (default: true) |
| `highlight` | boolean | No | Highlight matched terms (default: false) |
| `filters` | object | No | Optional filters |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "machine learning architecture",
    "top_k": 10,
    "use_fuzzy": true,
    "highlight": true,
    "filters": {}
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
      "content": "Neural network architecture is...",
      "metadata": {"page": 5},
      "similarity_score": 0.8567,
      "rank": 1,
      "highlighted_content": "Neural <mark>network architecture</mark> is...",
      "matched_terms": ["network", "architecture"]
    }
  ],
  "total": 12,
  "query_time_ms": 15.3
}
```

---

#### Hybrid Search

```http
POST /api/v1/search/hybrid
```

Combine vector similarity and text search using weighted sum or Reciprocal Rank Fusion (RRF).

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query text |
| `top_k` | integer | No | Maximum results (1-100, default: 10) |
| `vector_weight` | float | No | Vector score weight 0-1 (default: 0.7) |
| `text_weight` | float | No | Text score weight 0-1 (default: 0.3) |
| `fusion_method` | string | No | "weighted_sum" or "rrf" (default: "weighted_sum") |
| `min_similarity` | float | No | Minimum similarity 0-1 (default: 0.5) |
| `filters` | object | No | Optional filters |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "neural network architecture",
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
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 0,
      "content": "Neural network architecture is...",
      "metadata": {"page": 5},
      "hybrid_score": 0.8856,
      "vector_score": 0.9234,
      "text_score": 0.7834,
      "vector_rank": 1,
      "text_rank": 3,
      "rank": 1,
      "fusion_method": "weighted_sum"
    }
  ],
  "total": 15,
  "query_time_ms": 45.2
}
```

**Validation:**
- `vector_weight + text_weight` must equal approximately 1.0
- Weights must be between 0.0 and 1.0

---

#### Find Similar Chunks

```http
GET /api/v1/search/similar/{chunk_id}
```

Find chunks semantically similar to a reference chunk using its stored embedding.

**Parameters:**

| Name | Type | In | Required | Description |
|------|------|-----|----------|-------------|
| `chunk_id` | string (UUID) | path | Yes | Reference chunk UUID |
| `top_k` | integer | query | No | Maximum results (1-100, default: 10) |
| `exclude_self` | boolean | query | No | Exclude reference chunk (default: true) |

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/search/similar/550e8400-e29b-41d4-a716-446655440000?top_k=5" \
  -H "X-API-Key: your-api-key"
```

**Example Response:**

```json
{
  "results": [
    {
      "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 1,
      "content": "Related content about...",
      "metadata": {"page": 2},
      "similarity_score": 0.8934,
      "rank": 1
    }
  ],
  "total": 5,
  "query_time_ms": 18.7
}
```

**Error Responses:**

| Status | Description | Example Response |
|--------|-------------|------------------|
| `400` | No embedding | `{"detail": "Reference chunk does not have an embedding"}` |
| `404` | Chunk not found | `{"detail": "Reference chunk with ID '...' not found"}` |

---

### Health Check

#### Vector Store Health

```http
GET /health/vector
```

Check vector store health including pgvector and pg_trgm extensions.

**Example Request:**

```bash
curl -X GET "http://localhost:8000/health/vector"
```

**Example Response:**

```json
{
  "healthy": true,
  "status": "healthy",
  "message": "pgvector and pg_trgm extensions available",
  "latency_ms": 2.5,
  "extensions": {
    "vector": "0.7.0",
    "pg_trgm": "1.6"
  },
  "pgvector_version": "0.7.0",
  "pg_trgm_version": "1.6",
  "timestamp": "2026-02-18T10:30:00.000000"
}
```

---

## Rate Limits

All search endpoints have tiered rate limiting:

| Endpoint | Limit | Window |
|----------|-------|--------|
| List/Get Chunks | 120 | 60 seconds |
| Semantic Search | 60 | 60 seconds |
| Text Search | 100 | 60 seconds |
| Hybrid Search | 30 | 60 seconds |
| Similar Chunks | 60 | 60 seconds |

Rate limit headers are returned with each response:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1708255800
```

When rate limit is exceeded (429):

```json
{
  "detail": "Rate limit exceeded",
  "retry_after": 45
}
```

---

## Performance

### Expected Latencies

| Operation | p50 | p99 |
|-----------|-----|-----|
| List Chunks | 10ms | 50ms |
| Get Chunk | 5ms | 20ms |
| Semantic Search | 30ms | 100ms |
| Text Search | 15ms | 50ms |
| Hybrid Search | 50ms | 150ms |
| Similar Chunks | 20ms | 80ms |

### Throughput

- Semantic search: ~1000 QPS (1M chunks, HNSW index)
- Text search: ~2000 QPS (GIN index)
- Hybrid search: ~500 QPS

---

## Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_EMBEDDING` | Embedding dimension mismatch | 400 |
| `INVALID_QUERY` | Empty or invalid query | 400 |
| `INVALID_WEIGHTS` | Vector + text weights != 1.0 | 400 |
| `CHUNK_NOT_FOUND` | Chunk doesn't exist | 404 |
| `NO_EMBEDDING` | Chunk has no embedding vector | 400 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `SEARCH_ERROR` | Internal search failure | 500 |

---

## SDK Examples

### Python

```python
import requests

API_BASE = "http://localhost:8000/api/v1"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY}

# List chunks for a job
def list_chunks(job_id: str, limit: int = 10):
    response = requests.get(
        f"{API_BASE}/jobs/{job_id}/chunks",
        headers=headers,
        params={"limit": limit}
    )
    response.raise_for_status()
    return response.json()

# Semantic search
def semantic_search(embedding: list[float], top_k: int = 10):
    response = requests.post(
        f"{API_BASE}/search/semantic",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "query_embedding": embedding,
            "top_k": top_k,
            "min_similarity": 0.7
        }
    )
    response.raise_for_status()
    return response.json()

# Text search
def text_search(query: str, highlight: bool = True):
    response = requests.post(
        f"{API_BASE}/search/text",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "query": query,
            "top_k": 10,
            "highlight": highlight
        }
    )
    response.raise_for_status()
    return response.json()

# Hybrid search
def hybrid_search(query: str, vector_weight: float = 0.7):
    response = requests.post(
        f"{API_BASE}/search/hybrid",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "query": query,
            "top_k": 10,
            "vector_weight": vector_weight,
            "text_weight": 1.0 - vector_weight,
            "fusion_method": "weighted_sum"
        }
    )
    response.raise_for_status()
    return response.json()

# Find similar chunks
def find_similar(chunk_id: str, top_k: int = 5):
    response = requests.get(
        f"{API_BASE}/search/similar/{chunk_id}",
        headers=headers,
        params={"top_k": top_k}
    )
    response.raise_for_status()
    return response.json()
```

### JavaScript/TypeScript

```typescript
const API_BASE = "http://localhost:8000/api/v1";
const API_KEY = "your-api-key";

// List chunks
async function listChunks(jobId: string, limit: number = 10) {
  const response = await fetch(
    `${API_BASE}/jobs/${jobId}/chunks?limit=${limit}`,
    { headers: { "X-API-Key": API_KEY } }
  );
  return response.json();
}

// Semantic search
async function semanticSearch(embedding: number[], topK: number = 10) {
  const response = await fetch(`${API_BASE}/search/semantic`, {
    method: "POST",
    headers: {
      "X-API-Key": API_KEY,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query_embedding: embedding,
      top_k: topK,
      min_similarity: 0.7,
    }),
  });
  return response.json();
}

// Text search
async function textSearch(query: string, highlight: boolean = true) {
  const response = await fetch(`${API_BASE}/search/text`, {
    method: "POST",
    headers: {
      "X-API-Key": API_KEY,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      top_k: 10,
      highlight,
    }),
  });
  return response.json();
}
```

---

## Pipeline Integration

The vector store is automatically integrated into the document processing pipeline:

```
Stage 1: Ingest → File validation
Stage 2: Detect → Content detection
Stage 3: SelectParser → Parser selection
Stage 4: Parse → Document parsing
Stage 5: Chunk → Text chunking (NEW)
Stage 6: Embed → Embedding generation (NEW)
Stage 7: Output → Results
```

Chunks and embeddings are generated automatically when:
- `vector_store.enabled: true`
- `vector_store.pipeline.auto_generate_embeddings: true`
- Document parsing produces extracted text

---

## Monitoring

### Metrics

Available at `/metrics` endpoint:

- `vector_search_requests_total` - Search request count by type
- `vector_search_duration_seconds` - Search latency
- `vector_search_results_count` - Results returned per query
- `embedding_generation_total` - Embeddings generated
- `embedding_generation_duration_seconds` - Embedding latency
- `chunk_storage_total` - Chunks stored

### Grafana Dashboard

Dashboard available at: `config/grafana/dashboards/vector-store.json`

Shows:
- Search volume and latency
- Cache hit ratios
- Embedding generation stats
- Error rates

---

## Support

For issues or questions:
- API documentation: `/docs` endpoint
- OpenAPI spec: `/api/v1/openapi.json`
- Health status: `/health/vector`
- Metrics: `/metrics` endpoint
