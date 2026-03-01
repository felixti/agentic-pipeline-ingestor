# Specification: HippoRAG API Endpoints

## Overview

Implement 3 REST API endpoints for HippoRAG multi-hop retrieval functionality.

## Endpoints

### 1. POST /api/v1/hipporag/retrieve

**Purpose**: Perform multi-hop retrieval using HippoRAG Personalized PageRank algorithm.

**Request Model**:
```python
class HippoRAGRetrieveRequest(BaseModel):
    queries: list[str]  # Required, list of query strings
    num_to_retrieve: int = 10  # min: 1, max: 50
```

**Response Model**:
```python
class HippoRAGRetrievalResult(BaseModel):
    query: str
    passages: list[str]
    scores: list[float]
    source_documents: list[str]
    entities: list[str]

class HippoRAGRetrieveResponse(BaseModel):
    results: list[HippoRAGRetrievalResult]
    query_time_ms: float
```

### 2. POST /api/v1/hipporag/qa

**Purpose**: Complete RAG pipeline with HippoRAG multi-hop retrieval and answer generation.

**Request Model**:
```python
class HippoRAGQARequest(BaseModel):
    queries: list[str]  # Required, list of questions
    num_to_retrieve: int = 10
```

**Response Model**:
```python
class HippoRAGQAResult(BaseModel):
    query: str
    answer: str
    sources: list[str]
    confidence: float
    retrieval_results: HippoRAGRetrievalResult

class HippoRAGQAResponse(BaseModel):
    results: list[HippoRAGQAResult]
    total_tokens: int
    query_time_ms: float
```

### 3. POST /api/v1/hipporag/extract-triples

**Purpose**: Extract subject-predicate-object triples from text using OpenIE.

**Request Model**:
```python
class HippoRAGExtractTriplesRequest(BaseModel):
    text: str  # Required
```

**Response Model**:
```python
class HippoRAGTriple(BaseModel):
    subject: str
    predicate: str
    object: str

class HippoRAGExtractTriplesResponse(BaseModel):
    triples: list[HippoRAGTriple]
```

## Implementation Notes

1. Use existing `HippoRAGDestination` plugin from `src/plugins/destinations/hipporag.py`
2. Create new route file: `src/api/routes/hipporag.py`
3. Follow patterns from `src/api/routes/rag.py` and `src/api/routes/search.py`
4. HippoRAG uses neurobiological memory model - single-step multi-hop reasoning
5. Add proper error handling for storage directory issues
6. Include authentication/authorization if required by existing patterns
