# Specification: Cognee API Endpoints

## Overview

Implement 3 REST API endpoints for Cognee GraphRAG functionality.

## Endpoints

### 1. POST /api/v1/cognee/search

**Purpose**: Search the Cognee knowledge graph using vector, graph, or hybrid search.

**Request Model**:
```python
class CogneeSearchRequest(BaseModel):
    query: str  # Search query text
    search_type: str = "hybrid"  # enum: [vector, graph, hybrid]
    top_k: int = 10  # min: 1, max: 100
    dataset_id: str = "default"
```

**Response Model**:
```python
class CogneeSearchResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    source_document: str
    entities: list[str]

class CogneeSearchResponse(BaseModel):
    results: list[CogneeSearchResult]
    search_type: str
    dataset_id: str
    query_time_ms: float
```

### 2. POST /api/v1/cognee/extract-entities

**Purpose**: Extract entities and relationships from text using Cognee LLM provider.

**Request Model**:
```python
class CogneeExtractEntitiesRequest(BaseModel):
    text: str  # Required
    dataset_id: str = "default"
```

**Response Model**:
```python
class CogneeEntity(BaseModel):
    name: str
    type: str  # e.g., "PERSON"
    description: str

class CogneeRelationship(BaseModel):
    source: str
    target: str
    type: str  # e.g., "FOUNDED"

class CogneeExtractEntitiesResponse(BaseModel):
    entities: list[CogneeEntity]
    relationships: list[CogneeRelationship]
```

### 3. GET /api/v1/cognee/stats

**Purpose**: Retrieve statistics about the Cognee knowledge graph.

**Query Parameters**:
- `dataset_id`: str = "default" (optional)

**Response Model**:
```python
class CogneeStatsResponse(BaseModel):
    dataset_id: str
    document_count: int
    chunk_count: int
    entity_count: int
    relationship_count: int
    graph_density: float
    last_updated: datetime
```

## Implementation Notes

1. Use existing `CogneeLocalGraph` plugin from `src/plugins/destinations/cognee_local.py`
2. Create new route file: `src/api/routes/cognee.py`
3. Follow patterns from `src/api/routes/rag.py` and `src/api/routes/search.py`
4. Add proper error handling for Neo4j connection issues
5. Include authentication/authorization if required by existing patterns
