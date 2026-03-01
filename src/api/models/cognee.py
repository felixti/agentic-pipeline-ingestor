"""Pydantic models for Cognee API requests and responses.

These models define the request/response schemas for the Cognee knowledge graph
endpoints, including semantic search, entity extraction, and graph statistics.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Search Models
# ============================================================================


class CogneeSearchRequest(BaseModel):
    """Request model for Cognee search endpoint.

    Attributes:
        query: The search query string
        search_type: Type of search to perform (vector, graph, or hybrid)
        top_k: Maximum number of results to return
        dataset_id: Dataset identifier to search within
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "query": "What is machine learning?",
                "search_type": "hybrid",
                "top_k": 10,
                "dataset_id": "default",
            }
        },
    )

    query: str = Field(..., min_length=1, description="The search query string")
    search_type: str = Field(
        default="hybrid",
        pattern="^(vector|graph|hybrid)$",
        description="Type of search: vector, graph, or hybrid",
    )
    top_k: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results to return (1-100)"
    )
    dataset_id: str = Field(default="default", description="Dataset identifier to search within")


class CogneeSearchResult(BaseModel):
    """Individual search result from Cognee.

    Attributes:
        chunk_id: Unique identifier of the retrieved chunk
        content: Text content of the chunk
        score: Relevance score (0-1)
        source_document: Name or identifier of the source document
        entities: List of entities found in this chunk
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "chunk_id": "chunk_001",
                "content": "Machine learning is a subset of artificial intelligence...",
                "score": 0.92,
                "source_document": "ai_overview.pdf",
                "entities": ["machine learning", "artificial intelligence"],
            }
        },
    )

    chunk_id: str = Field(..., description="Unique identifier of the retrieved chunk")
    content: str = Field(..., description="Text content of the chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    source_document: str = Field(..., description="Name or identifier of the source document")
    entities: list[str] = Field(default_factory=list, description="List of entities found in this chunk")


class CogneeSearchResponse(BaseModel):
    """Response model for Cognee search endpoint.

    Attributes:
        results: List of search results
        search_type: Type of search that was performed
        dataset_id: Dataset identifier that was searched
        query_time_ms: Time taken to execute the query in milliseconds
        message: Optional message (e.g., when knowledge graph is empty)
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "chunk_id": "chunk_001",
                        "content": "Machine learning is a subset of artificial intelligence...",
                        "score": 0.92,
                        "source_document": "ai_overview.pdf",
                        "entities": ["machine learning", "artificial intelligence"],
                    }
                ],
                "search_type": "hybrid",
                "dataset_id": "default",
                "query_time_ms": 145.5,
                "message": None,
            }
        },
    )

    results: list[CogneeSearchResult] = Field(default_factory=list, description="List of search results")
    search_type: str = Field(..., description="Type of search that was performed")
    dataset_id: str = Field(..., description="Dataset identifier that was searched")
    query_time_ms: float = Field(..., description="Time taken to execute the query in milliseconds")
    message: str | None = Field(default=None, description="Optional message (e.g., when knowledge graph is empty)")


# ============================================================================
# Entity Extraction Models
# ============================================================================


class CogneeExtractEntitiesRequest(BaseModel):
    """Request model for Cognee entity extraction endpoint.

    Attributes:
        text: Text content to extract entities from
        dataset_id: Dataset identifier for context
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "text": "Apple Inc. was founded by Steve Jobs and Steve Wozniak.",
                "dataset_id": "default",
            }
        },
    )

    text: str = Field(..., min_length=1, description="Text content to extract entities from")
    dataset_id: str = Field(default="default", description="Dataset identifier for context")


class CogneeEntity(BaseModel):
    """Entity extracted from text.

    Attributes:
        name: Name of the entity
        type: Type of entity (e.g., PERSON, ORGANIZATION, LOCATION)
        description: Description or additional information about the entity
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "name": "Steve Jobs",
                "type": "PERSON",
                "description": "Co-founder of Apple Inc.",
            }
        },
    )

    name: str = Field(..., description="Name of the entity")
    type: str = Field(..., description="Type of entity (e.g., PERSON, ORGANIZATION, LOCATION)")
    description: str = Field(..., description="Description or additional information about the entity")


class CogneeRelationship(BaseModel):
    """Relationship between two entities.

    Attributes:
        source: Source entity name
        target: Target entity name
        type: Type of relationship (e.g., FOUNDED, WORKS_AT, LOCATED_IN)
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "source": "Steve Jobs",
                "target": "Apple Inc.",
                "type": "FOUNDED",
            }
        },
    )

    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    type: str = Field(..., description="Type of relationship (e.g., FOUNDED, WORKS_AT, LOCATED_IN)")


class CogneeExtractEntitiesResponse(BaseModel):
    """Response model for Cognee entity extraction endpoint.

    Attributes:
        entities: List of extracted entities
        relationships: List of relationships between entities
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "entities": [
                    {"name": "Steve Jobs", "type": "PERSON", "description": "Co-founder of Apple Inc."},
                    {"name": "Apple Inc.", "type": "ORGANIZATION", "description": "Technology company"},
                ],
                "relationships": [
                    {"source": "Steve Jobs", "target": "Apple Inc.", "type": "FOUNDED"}
                ],
            }
        },
    )

    entities: list[CogneeEntity] = Field(default_factory=list, description="List of extracted entities")
    relationships: list[CogneeRelationship] = Field(
        default_factory=list, description="List of relationships between entities"
    )


# ============================================================================
# Statistics Models
# ============================================================================


class CogneeStatsResponse(BaseModel):
    """Response model for Cognee statistics endpoint.

    Attributes:
        dataset_id: Dataset identifier
        document_count: Total number of documents in the dataset
        chunk_count: Total number of chunks in the dataset
        entity_count: Total number of entities in the knowledge graph
        relationship_count: Total number of relationships in the knowledge graph
        graph_density: Graph density metric (0-1)
        last_updated: Timestamp of last update
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "dataset_id": "default",
                "document_count": 150,
                "chunk_count": 1250,
                "entity_count": 450,
                "relationship_count": 890,
                "graph_density": 0.34,
                "last_updated": "2024-01-15T10:30:00Z",
            }
        },
    )

    dataset_id: str = Field(..., description="Dataset identifier")
    document_count: int = Field(..., ge=0, description="Total number of documents in the dataset")
    chunk_count: int = Field(..., ge=0, description="Total number of chunks in the dataset")
    entity_count: int = Field(..., ge=0, description="Total number of entities in the knowledge graph")
    relationship_count: int = Field(
        ..., ge=0, description="Total number of relationships in the knowledge graph"
    )
    graph_density: float = Field(..., ge=0.0, le=1.0, description="Graph density metric (0-1)")
    last_updated: datetime = Field(..., description="Timestamp of last update")
