# RAG Search Strategies - cURL Examples

Complete cURL examples for all available RAG (Retrieval-Augmented Generation) search strategies in the Agentic Data Pipeline Ingestor.

**Base URL**: `http://localhost:8000/api/v1`  
**Authentication**: Include `X-API-Key` header with your API key

---

## Table of Contents

1. [Semantic Search (Vector Search)](#1-semantic-search-vector-search)
2. [Text Search (Lexical/BM25)](#2-text-search-lexicalbm25)
3. [Hybrid Search](#3-hybrid-search)
4. [Find Similar Chunks](#4-find-similar-chunks)
5. [Cognee GraphRAG Search](#5-cognee-graphrag-search)
6. [Advanced Usage Examples](#6-advanced-usage-examples)

---

## 1. Semantic Search (Vector Search)

Pure semantic similarity search using embedding vectors and cosine similarity.

**Best for**: Conceptual similarity, natural language queries, finding documents with similar meaning but different wording.

### Basic Semantic Search

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query_embedding": [0.023, -0.045, 0.128, 0.003, -0.087, ...],
    "top_k": 10
  }'
```

### Semantic Search with Filters

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

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query_embedding` | array[float] | Yes | - | Query embedding vector (1536 dimensions) |
| `top_k` | integer | No | 10 | Maximum results (1-100) |
| `min_similarity` | float | No | 0.7 | Minimum similarity threshold (0-1) |
| `filters` | object | No | {} | Optional filters (e.g., `job_id`) |

---

## 2. Text Search (Lexical/BM25)

Traditional full-text search using PostgreSQL's tsvector/tsquery with BM25 ranking and optional fuzzy trigram matching.

**Best for**: Specific keywords, exact term matching, product codes, names, identifiers, highlighting matched terms.

### Basic Text Search

```bash
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning architecture",
    "top_k": 10
  }'
```

### Text Search with Highlighting and Fuzzy Matching

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
    "filters": {
      "job_id": "123e4567-e89b-12d3-a456-426614174000"
    }
  }'
```

### Text Search with Job Filter Only

```bash
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "kubernetes deployment strategies",
    "top_k": 10,
    "highlight": true,
    "use_fuzzy": true,
    "language": "english",
    "filters": {}
  }'
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text (1-1024 characters) |
| `top_k` | integer | No | 10 | Maximum results (1-100) |
| `language` | string | No | "english" | Text search language |
| `use_fuzzy` | boolean | No | true | Include fuzzy trigram matching |
| `highlight` | boolean | No | false | Include highlighted snippets |
| `filters` | object | No | {} | Optional filters |

---

## 3. Hybrid Search

Combines vector similarity and text search using weighted sum or Reciprocal Rank Fusion (RRF).

**Best for**: Best of both semantic and lexical search, complex queries with both concepts and specific terms, production systems.

### Basic Hybrid Search (Weighted Sum)

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural network architecture",
    "top_k": 10,
    "vector_weight": 0.7,
    "text_weight": 0.3,
    "fusion_method": "weighted_sum"
  }'
```

### Hybrid Search with RRF Fusion

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning applications in healthcare",
    "top_k": 10,
    "vector_weight": 0.7,
    "text_weight": 0.3,
    "fusion_method": "rrf",
    "min_similarity": 0.5,
    "filters": {
      "source_type": "pdf"
    }
  }'
```

### Semantic-Focused Hybrid Search

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deep learning neural networks",
    "top_k": 10,
    "vector_weight": 0.9,
    "text_weight": 0.1,
    "fusion_method": "weighted_sum",
    "min_similarity": 0.7
  }'
```

### Lexical-Focused Hybrid Search

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "API-KEY-123 kubernetes deployment",
    "top_k": 10,
    "vector_weight": 0.3,
    "text_weight": 0.7,
    "fusion_method": "weighted_sum",
    "min_similarity": 0.5
  }'
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text |
| `top_k` | integer | No | 10 | Maximum results (1-100) |
| `vector_weight` | float | No | 0.7 | Vector score weight (0-1) |
| `text_weight` | float | No | 0.3 | Text score weight (0-1) |
| `fusion_method` | string | No | "weighted_sum" | Fusion method: "weighted_sum" or "rrf" |
| `min_similarity` | float | No | 0.5 | Minimum similarity threshold (0-1) |
| `filters` | object | No | {} | Optional filters |

**Weight Presets:**
- **Semantic Focus**: vector=0.9, text=0.1
- **Balanced**: vector=0.7, text=0.3 (default)
- **Lexical Focus**: vector=0.3, text=0.7

---

## 4. Find Similar Chunks

Find chunks semantically similar to a reference chunk using its stored embedding.

**Best for**: Finding related content, exploring similar documents, content recommendation.

### Basic Similar Chunk Search

```bash
curl "http://localhost:8000/api/v1/search/similar/550e8400-e29b-41d4-a716-446655440000?top_k=5" \
  -H "X-API-Key: your-api-key"
```

### Similar Chunks Excluding Reference

```bash
curl "http://localhost:8000/api/v1/search/similar/550e8400-e29b-41d4-a716-446655440000?top_k=10&exclude_self=true" \
  -H "X-API-Key: your-api-key"
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `chunk_id` | string (UUID) | Yes | - | Reference chunk UUID (path parameter) |
| `top_k` | integer | No | 10 | Maximum results (1-100) |
| `exclude_self` | boolean | No | true | Exclude the reference chunk from results |

---

## 5. Cognee GraphRAG Search

Knowledge graph search using Cognee with Neo4j backend. Provides entity-aware search with relationship traversal.

**Best for**: Finding relationships between entities, exploring knowledge graphs, complex multi-hop queries, entity-based exploration.

### Prerequisites

Documents must be processed with Cognee destination:
```json
{"destinations": [{"type": "cognee_local", "auto_cognify": true}]}
```

### Basic Cognee Hybrid Search

```bash
curl -X POST "http://localhost:8000/api/v1/cognee/search" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What'\''s the relationship between Caio and Felipe?",
    "dataset_id": "tee34",
    "search_type": "hybrid",
    "top_k": 5
  }'
```

### Cognee Graph Search (Relationship Focus)

```bash
curl -X POST "http://localhost:8000/api/v1/cognee/search" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about software architecture patterns",
    "dataset_id": "my-dataset",
    "search_type": "graph",
    "top_k": 3
  }'
```

### Cognee Vector Search

```bash
curl -X POST "http://localhost:8000/api/v1/cognee/search" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks deep learning",
    "dataset_id": "research-papers",
    "search_type": "vector",
    "top_k": 10
  }'
```

### Extract Entities from Text

```bash
curl -X POST "http://localhost:8000/api/v1/cognee/extract-entities" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Microsoft was founded by Bill Gates and Paul Allen in 1975. It is headquartered in Redmond, Washington.",
    "dataset_id": "test-dataset"
  }'
```

### Get Cognee Graph Statistics

```bash
curl "http://localhost:8000/api/v1/cognee/stats?dataset_id=tee34" \
  -H "X-API-Key: your-api-key"
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language query |
| `dataset_id` | string | Yes | - | Dataset identifier (namespace) |
| `search_type` | string | No | "hybrid" | Search type: "hybrid", "graph", "vector" |
| `top_k` | integer | No | 5 | Maximum results (1-20) |

### Search Type Guide

| Type | Best For | Latency |
|------|----------|---------|
| `hybrid` | General queries, best overall results | ~10s |
| `graph` | Relationship questions, entity exploration | ~15s |
| `vector` | Semantic similarity, fast retrieval | ~5s |

### Cognee Search Response

```json
{
  "results": [
    {
      "chunk_id": "result_0",
      "content": "Felipe is the person who wrote a reference or recommendation letter for Caio...",
      "score": 1.0,
      "source_document": "text_b5c97ef719658e4460c8e6dd9f097b83",
      "entities": [
        "caio",
        "caio amaral correa",
        "felipe augusto felix"
      ],
      "metadata": {}
    }
  ],
  "search_type": "hybrid",
  "dataset_id": "tee34",
  "query_time_ms": 10279.51,
  "message": null
}
```

### Entity Extraction Response

```json
{
  "entities": [
    {
      "name": "Microsoft",
      "type": "Organization",
      "description": "Technology company founded in 1975"
    },
    {
      "name": "Bill Gates",
      "type": "Person",
      "description": "Co-founder of Microsoft"
    },
    {
      "name": "Paul Allen",
      "type": "Person",
      "description": "Co-founder of Microsoft"
    },
    {
      "name": "Redmond",
      "type": "Location",
      "description": "City in Washington state"
    }
  ],
  "relationships": [
    {
      "source": "Bill Gates",
      "target": "Microsoft",
      "type": "FOUNDED"
    }
  ],
  "query_time_ms": 1250
}
```

### Stats Response

```json
{
  "dataset_id": "tee34",
  "total_nodes": 111,
  "total_relationships": 263,
  "entity_count": 45,
  "document_count": 3,
  "last_updated": "2026-03-01T17:00:00Z"
}
```

---

## 6. Advanced Usage Examples

### Complete Search Workflow

```bash
#!/bin/bash

API_KEY="your-api-key"
BASE_URL="http://localhost:8000/api/v1"

# 1. First, list chunks from a job to get a reference chunk ID
echo "=== Listing chunks ==="
CHUNKS=$(curl -s "${BASE_URL}/jobs/123e4567-e89b-12d3-a456-426614174000/chunks?limit=1" \
  -H "X-API-Key: ${API_KEY}")

echo "$CHUNKS" | jq '.items[0].id'

# 2. Perform text search with highlighting
echo "=== Text Search ==="
curl -s -X POST "${BASE_URL}/search/text" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning architecture",
    "top_k": 5,
    "highlight": true,
    "use_fuzzy": true
  }' | jq '.results[] | {rank: .rank, score: .similarity_score, content: .content[:100]}'

# 3. Perform hybrid search with RRF
echo "=== Hybrid Search (RRF) ==="
curl -s -X POST "${BASE_URL}/search/hybrid" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "top_k": 5,
    "fusion_method": "rrf",
    "vector_weight": 0.6,
    "text_weight": 0.4
  }' | jq '.results[] | {rank: .rank, hybrid_score: .hybrid_score, content: .content[:100]}'

# 4. Find similar chunks to a reference
echo "=== Similar Chunks ==="
curl -s "${BASE_URL}/search/similar/550e8400-e29b-41d4-a716-446655440000?top_k=3&exclude_self=true" \
  -H "X-API-Key: ${API_KEY}" | jq '.results[] | {rank: .rank, score: .similarity_score, content: .content[:100]}'
```

### Search with Metadata Filtering

```bash
# Filter by job_id and custom metadata
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "top_k": 10,
    "vector_weight": 0.7,
    "text_weight": 0.3,
    "filters": {
      "job_id": "550e8400-e29b-41d4-a716-446655440001",
      "metadata": {"category": "research", "year": 2024}
    }
  }'
```

### Multi-Language Text Search

```bash
# Search in Spanish
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "aprendizaje automático",
    "top_k": 10,
    "language": "spanish",
    "use_fuzzy": true,
    "highlight": true
  }'

# Search in German
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "maschinelles lernen",
    "top_k": 10,
    "language": "german",
    "use_fuzzy": true
  }'
```

### Testing Search Performance

```bash
# Time semantic search
time curl -s -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query_embedding": [0.023, -0.045, 0.128, ...],
    "top_k": 10
  }' | jq '.query_time_ms'

# Time text search
time curl -s -X POST "http://localhost:8000/api/v1/search/text" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "top_k": 10
  }' | jq '.query_time_ms'

# Time hybrid search
time curl -s -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "top_k": 10
  }' | jq '.query_time_ms'
```

---

## Rate Limits

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `POST /search/semantic` | 60/min | 60 seconds |
| `POST /search/text` | 100/min | 60 seconds |
| `POST /search/hybrid` | 30/min | 60 seconds |
| `GET /search/similar/{chunk_id}` | 60/min | 60 seconds |

**Rate Limit Headers** (included in all responses):
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

---

## Strategy Selection Guide

### When to Use Each Strategy

| Strategy | Use When | Don't Use When |
|----------|----------|----------------|
| **Semantic Search** | Conceptual similarity needed, natural language queries | Exact keyword matching needed |
| **Text Search** | Specific keywords, identifiers, highlighting needed | Semantic understanding needed |
| **Hybrid Search** | Best overall retrieval, complex queries | Ultra-low latency required |
| **Similar Chunks** | Finding related content, exploration | Initial search from query |

### Decision Flow

```
Query Type?
├── Simple factual lookup → Hybrid Search (balanced preset)
├── Complex analytical → Hybrid Search (semantic focus) + Re-ranking
├── Vague/ambiguous → Hybrid Search with higher vector weight
├── Specific identifiers → Text Search with highlighting
├── Find related content → Similar Chunks API
└── Production system → Hybrid Search (default)
```

---

## Response Formats

### Semantic Search Response

```json
{
  "results": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 0,
      "content": "Neural network architecture involves layers...",
      "metadata": {"page": 5, "section": "architecture"},
      "similarity_score": 0.9234,
      "rank": 1
    }
  ],
  "total": 8,
  "query_time_ms": 25.5
}
```

### Text Search Response

```json
{
  "results": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 0,
      "content": "Machine learning architecture is fundamental...",
      "metadata": {"page": 1},
      "similarity_score": 0.8567,
      "rank": 1,
      "highlighted_content": "<mark>Machine learning</mark> <mark>architecture</mark> is fundamental...",
      "matched_terms": ["machine", "learning", "architecture"]
    }
  ],
  "total": 12,
  "query_time_ms": 15.3
}
```

### Hybrid Search Response

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
    }
  ],
  "total": 15,
  "query_time_ms": 45.2
}
```

---

## Additional Resources

- [RAG Strategy Guide](./RAG_STRATEGY_GUIDE.md) - Detailed strategy explanations
- [API Guide](./API_GUIDE.md) - Complete API reference
- [Vector Store API Usage](./VECTOR_STORE_API_USAGE.md) - Python SDK examples
