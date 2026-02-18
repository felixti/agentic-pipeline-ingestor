# Vector Store API Usage Guide

Complete examples for using the vector store search endpoints.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Include API key in headers:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/...
```

## Chunk Endpoints

### 1. List Chunks for a Job

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}/chunks?limit=10&offset=0" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "job_id": "550e8400-e29b-41d4-a716-446655440001",
      "chunk_index": 0,
      "content": "This is the first chunk of text...",
      "content_hash": "a1b2c3d4e5f6...",
      "metadata": {"page": 1, "source": "document.pdf"},
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 50,
  "limit": 10,
  "offset": 0
}
```

### 2. Get Specific Chunk

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}/chunks/{chunk_id}" \
  -H "X-API-Key: your-api-key"
```

With embedding:

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}/chunks/{chunk_id}?include_embedding=true" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "chunk_index": 0,
  "content": "This is the first chunk of text...",
  "content_hash": "a1b2c3d4e5f6...",
  "embedding": [0.023, -0.045, 0.012, ...],  // Only if requested
  "metadata": {"page": 1},
  "created_at": "2024-01-15T10:30:00Z"
}
```

## Search Endpoints

### 3. Semantic Search (Vector Similarity)

Find chunks semantically similar to a query embedding:

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query_embedding": [0.023, -0.045, 0.012, ...],
    "top_k": 10,
    "min_similarity": 0.7,
    "filters": {
      "job_id": "550e8400-e29b-41d4-a716-446655440001"
    }
  }'
```

**Response:**
```json
{
  "results": [
    {
      "chunk": {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "content": "Relevant content here...",
        "metadata": {"page": 1}
      },
      "similarity_score": 0.89,
      "rank": 1
    }
  ],
  "total": 10,
  "query_time_ms": 45
}
```

### 4. Text Search (BM25 + Fuzzy)

Full-text search with BM25 ranking:

```bash
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "machine learning applications",
    "top_k": 10,
    "language": "english",
    "use_fuzzy": true,
    "highlight": true,
    "filters": {}
  }'
```

**Response:**
```json
{
  "results": [
    {
      "chunk": {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "content": "Machine learning has many applications..."
      },
      "rank_score": 0.75,
      "rank": 1,
      "highlighted_content": "<mark>Machine learning</mark> has many applications...",
      "matched_terms": ["machine", "learning"]
    }
  ]
}
```

### 5. Hybrid Search (Vector + Text)

Combine semantic and text search:

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "neural networks in healthcare",
    "top_k": 10,
    "vector_weight": 0.7,
    "text_weight": 0.3,
    "fusion_method": "weighted_sum",
    "min_similarity": 0.5
  }'
```

**Fusion Methods:**
- `weighted_sum` - Linear combination (default)
- `rrf` - Reciprocal Rank Fusion

**Response:**
```json
{
  "results": [
    {
      "chunk": {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "content": "Neural networks are transforming healthcare..."
      },
      "hybrid_score": 0.82,
      "vector_score": 0.85,
      "text_score": 0.73,
      "vector_rank": 2,
      "text_rank": 1,
      "fusion_method": "weighted_sum",
      "rank": 1
    }
  ]
}
```

### 6. Find Similar Chunks

Find chunks similar to a reference chunk:

```bash
curl "http://localhost:8000/api/v1/search/similar/{chunk_id}?top_k=5&exclude_self=true" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "results": [
    {
      "chunk": {
        "id": "550e8400-e29b-41d4-a716-446655440002",
        "content": "Similar content..."
      },
      "similarity_score": 0.92,
      "rank": 1
    }
  ]
}
```

## Python SDK Example

```python
import requests

class VectorStoreClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}
    
    def list_chunks(self, job_id: str, limit: int = 100):
        """List chunks for a job."""
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}/chunks",
            params={"limit": limit},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def semantic_search(self, query_embedding: list[float], top_k: int = 10):
        """Search by vector similarity."""
        response = requests.post(
            f"{self.base_url}/search/semantic",
            json={
                "query_embedding": query_embedding,
                "top_k": top_k,
                "min_similarity": 0.7
            },
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def text_search(self, query: str, highlight: bool = True):
        """Full-text search."""
        response = requests.post(
            f"{self.base_url}/search/text",
            json={
                "query": query,
                "top_k": 10,
                "highlight": highlight,
                "use_fuzzy": True
            },
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def hybrid_search(self, query: str, vector_weight: float = 0.7):
        """Hybrid vector + text search."""
        response = requests.post(
            f"{self.base_url}/search/hybrid",
            json={
                "query": query,
                "top_k": 10,
                "vector_weight": vector_weight,
                "text_weight": 1.0 - vector_weight,
                "fusion_method": "weighted_sum"
            },
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
client = VectorStoreClient(
    base_url="http://localhost:8000/api/v1",
    api_key="your-api-key"
)

# Search for documents about AI
results = client.text_search("artificial intelligence applications", highlight=True)
for result in results["results"]:
    print(f"Score: {result['rank_score']:.2f}")
    print(f"Content: {result['highlighted_content']}")
    print()
```

## Advanced Usage

### Filtering by Metadata

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query_embedding": [...],
    "filters": {
      "job_id": "specific-job-uuid",
      "metadata": {"category": "research", "year": 2024}
    }
  }'
```

### Pagination

```bash
# Get page 2 with 20 items per page
curl "http://localhost:8000/api/v1/jobs/{job_id}/chunks?limit=20&offset=20"
```

### Combining Multiple Filters

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "top_k": 10,
    "filters": {
      "job_id": "550e8400...",
      "chunk_index": {"gte": 0, "lte": 100}
    },
    "vector_weight": 0.6,
    "text_weight": 0.4
  }'
```

## Rate Limits

| Endpoint | Rate Limit |
|----------|------------|
| List chunks | 100/min |
| Get chunk | 200/min |
| Semantic search | 30/min |
| Text search | 60/min |
| Hybrid search | 20/min |
| Similar chunks | 40/min |

Headers in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Error Handling

**400 Bad Request:**
```json
{
  "error": "INVALID_QUERY",
  "message": "Query must be at least 2 characters",
  "details": {}
}
```

**404 Not Found:**
```json
{
  "error": "CHUNK_NOT_FOUND",
  "message": "Chunk not found",
  "chunk_id": "550e8400..."
}
```

**429 Rate Limited:**
```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Try again in 30 seconds."
}
```

## Best Practices

1. **Use filters** to narrow search scope and improve performance
2. **Don't request embeddings** in list calls unless needed
3. **Use highlight** for better UX in text search
4. **Start with hybrid search** for best results
5. **Cache embeddings** if doing repeated similar searches
