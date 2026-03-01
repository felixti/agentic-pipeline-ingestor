"""Pydantic models for HippoRAG API requests and responses.

These models define the request/response schemas for the HippoRAG knowledge graph
RAG endpoints, including retrieval-based search, question-answering, and triple extraction.
"""

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Retrieval Models
# ============================================================================


class HippoRAGRetrieveRequest(BaseModel):
    """Request model for HippoRAG retrieval endpoint.

    Attributes:
        queries: List of query strings to retrieve passages for
        num_to_retrieve: Maximum number of passages to retrieve per query
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "queries": ["What is machine learning?", "How does neural network work?"],
                "num_to_retrieve": 10,
            }
        },
    )

    queries: list[str] = Field(
        ..., min_length=1, description="List of query strings"
    )
    num_to_retrieve: int = Field(
        default=10, ge=1, le=50, description="Maximum number of passages to retrieve per query"
    )


class HippoRAGRetrievalResult(BaseModel):
    """Individual retrieval result from HippoRAG.

    Attributes:
        query: The query string used for retrieval
        passages: List of retrieved passage texts
        scores: List of relevance scores corresponding to passages
        source_documents: List of source document identifiers
        entities: List of relevant entities extracted for the query
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "query": "What is machine learning?",
                "passages": [
                    "Machine learning is a subset of artificial intelligence...",
                    "ML algorithms learn patterns from training data...",
                ],
                "scores": [0.92, 0.85],
                "source_documents": ["ai_overview.pdf", "ml_basics.pdf"],
                "entities": ["machine learning", "artificial intelligence", "algorithm"],
            }
        },
    )

    query: str = Field(..., description="The query string used for retrieval")
    passages: list[str] = Field(default_factory=list, description="List of retrieved passage texts")
    scores: list[float] = Field(default_factory=list, description="List of relevance scores")
    source_documents: list[str] = Field(
        default_factory=list, description="List of source document identifiers"
    )
    entities: list[str] = Field(
        default_factory=list, description="List of relevant entities extracted for the query"
    )


class HippoRAGRetrieveResponse(BaseModel):
    """Response model for HippoRAG retrieval endpoint.

    Attributes:
        results: List of retrieval results for each query
        query_time_ms: Time taken to execute the query in milliseconds
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "query": "What is machine learning?",
                        "passages": [
                            "Machine learning is a subset of artificial intelligence..."
                        ],
                        "scores": [0.92],
                        "source_documents": ["ai_overview.pdf"],
                        "entities": ["machine learning", "artificial intelligence"],
                    }
                ],
                "query_time_ms": 245.5,
            }
        },
    )

    results: list[HippoRAGRetrievalResult] = Field(
        default_factory=list, description="List of retrieval results"
    )
    query_time_ms: float = Field(..., description="Time taken to execute the query in milliseconds")


# ============================================================================
# QA Models
# ============================================================================


class HippoRAGQARequest(BaseModel):
    """Request model for HippoRAG question-answering endpoint.

    Attributes:
        queries: List of questions to answer
        num_to_retrieve: Maximum number of passages to retrieve per question
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "queries": [
                    "What is the capital of France?",
                    "Who invented the telephone?",
                ],
                "num_to_retrieve": 10,
            }
        },
    )

    queries: list[str] = Field(..., min_length=1, description="List of questions")
    num_to_retrieve: int = Field(
        default=10, ge=1, le=50, description="Maximum number of passages to retrieve per question"
    )


class HippoRAGQAResult(BaseModel):
    """Individual QA result from HippoRAG.

    Attributes:
        query: The question that was asked
        answer: Generated answer text
        sources: List of source document identifiers used for the answer
        confidence: Confidence score for the answer (0-1)
        retrieval_results: Underlying retrieval results used for answering
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "query": "What is the capital of France?",
                "answer": "The capital of France is Paris.",
                "sources": ["geography_basics.pdf", "europe_facts.pdf"],
                "confidence": 0.95,
                "retrieval_results": {
                    "query": "What is the capital of France?",
                    "passages": ["Paris is the capital and most populous city of France."],
                    "scores": [0.98],
                    "source_documents": ["geography_basics.pdf"],
                    "entities": ["Paris", "France"],
                },
            }
        },
    )

    query: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="Generated answer text")
    sources: list[str] = Field(
        default_factory=list, description="List of source document identifiers"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    retrieval_results: HippoRAGRetrievalResult = Field(
        ..., description="Underlying retrieval results"
    )


class HippoRAGQAResponse(BaseModel):
    """Response model for HippoRAG question-answering endpoint.

    Attributes:
        results: List of QA results for each question
        total_tokens: Total number of tokens used for generation
        query_time_ms: Time taken to execute the query in milliseconds
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "query": "What is the capital of France?",
                        "answer": "The capital of France is Paris.",
                        "sources": ["geography_basics.pdf"],
                        "confidence": 0.95,
                        "retrieval_results": {
                            "query": "What is the capital of France?",
                            "passages": ["Paris is the capital and most populous city of France."],
                            "scores": [0.98],
                            "source_documents": ["geography_basics.pdf"],
                            "entities": ["Paris", "France"],
                        },
                    }
                ],
                "total_tokens": 450,
                "query_time_ms": 520.0,
            }
        },
    )

    results: list[HippoRAGQAResult] = Field(default_factory=list, description="List of QA results")
    total_tokens: int = Field(..., ge=0, description="Total number of tokens used for generation")
    query_time_ms: float = Field(..., description="Time taken to execute the query in milliseconds")


# ============================================================================
# Triple Extraction Models
# ============================================================================


class HippoRAGTriple(BaseModel):
    """Knowledge graph triple extracted from text.

    Attributes:
        subject: Subject entity of the triple
        predicate: Relationship or predicate
        object: Object entity of the triple
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "subject": "Steve Jobs",
                "predicate": "founded",
                "object": "Apple Inc.",
            }
        },
    )

    subject: str = Field(..., description="Subject entity of the triple")
    predicate: str = Field(..., description="Relationship or predicate")
    object: str = Field(..., description="Object entity of the triple")


class HippoRAGExtractTriplesRequest(BaseModel):
    """Request model for HippoRAG triple extraction endpoint.

    Attributes:
        text: Text content to extract knowledge graph triples from
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "text": "Steve Jobs founded Apple Inc. in 1976. He served as CEO until 2011.",
            }
        },
    )

    text: str = Field(..., min_length=1, description="Text content to extract triples from")


class HippoRAGExtractTriplesResponse(BaseModel):
    """Response model for HippoRAG triple extraction endpoint.

    Attributes:
        triples: List of extracted knowledge graph triples
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "triples": [
                    {
                        "subject": "Steve Jobs",
                        "predicate": "founded",
                        "object": "Apple Inc.",
                    },
                    {
                        "subject": "Steve Jobs",
                        "predicate": "served as",
                        "object": "CEO",
                    },
                ]
            }
        },
    )

    triples: list[HippoRAGTriple] = Field(
        default_factory=list, description="List of extracted knowledge graph triples"
    )
