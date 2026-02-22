"""Pydantic models for RAG functionality."""

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContextType(str, Enum):
    """Context enhancement strategies for contextual retrieval.
    
    These strategies determine how chunk content is enhanced with
    surrounding context before embedding for improved retrieval.
    
    Attributes:
        PARENT_DOCUMENT: Adds document title and metadata as context
        WINDOW: Includes neighboring chunks (previous/next) as context
        HIERARCHICAL: Uses document hierarchy (sections/subsections) as context
    """
    
    PARENT_DOCUMENT = "parent_document"
    WINDOW = "window"
    HIERARCHICAL = "hierarchical"


class QueryType(str, Enum):
    """Query classification types for RAG strategy selection.
    
    This enum defines the different types of queries that can be classified
    to determine the optimal RAG retrieval strategy.
    
    Attributes:
        FACTUAL: Simple fact lookup (e.g., "What is X?", "Who is Y?")
        ANALYTICAL: Requires explanation or synthesis (e.g., "Explain X", "How does Y work?")
        COMPARATIVE: Compares multiple items (e.g., "Compare X and Y", "Pros/cons")
        VAGUE: Unclear or broad queries (e.g., "Tell me about...", "Information on...")
        MULTI_HOP: Requires multiple retrieval steps (e.g., "Author of X on topic Y")
    """
    
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    VAGUE = "vague"
    MULTI_HOP = "multi_hop"


class QueryClassification(BaseModel):
    """Result of query classification.
    
    This model represents the structured output from the QueryClassifier,
    containing the query type, confidence score, reasoning, and suggested strategies.
    
    Attributes:
        query_type: The classified type of the query
        confidence: Confidence score (0-1) of the classification
        reasoning: Explanation of why this classification was chosen
        suggested_strategies: List of RAG strategies recommended for this query type
    """
    
    query_type: QueryType = Field(
        description="The classified type of the query",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the classification",
    )
    reasoning: str = Field(
        description="Explanation of why this classification was chosen",
    )
    suggested_strategies: list[str] = Field(
        default_factory=list,
        description="List of RAG strategies recommended for this query type",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_type": "factual",
                    "confidence": 0.95,
                    "reasoning": "Query asks for a specific fact with clear entity",
                    "suggested_strategies": ["query_rewrite", "reranking", "hybrid_search"],
                }
            ]
        }
    }


class RankedChunk(BaseModel):
    """Result of re-ranking a chunk using cross-encoder.
    
    This model represents a document chunk that has been scored and ranked
    by a cross-encoder model for relevance to a specific query.
    
    Attributes:
        chunk_id: Unique identifier of the chunk
        content: Text content of the chunk
        score: Cross-encoder relevance score (0-1, higher is better)
        rank: Position after re-ranking (1-indexed)
        metadata: Optional metadata about the chunk (source, page numbers, etc.)
    """
    
    chunk_id: str = Field(
        description="Unique identifier of the chunk",
    )
    content: str = Field(
        description="Text content of the chunk",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cross-encoder relevance score (0-1, higher is better)",
    )
    rank: int = Field(
        ge=1,
        description="Position after re-ranking (1-indexed)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the chunk (source, page numbers, etc.)",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                    "content": "Vibe coding is a programming approach that emphasizes intuition...",
                    "score": 0.92,
                    "rank": 1,
                    "metadata": {"source": "doc1.pdf", "page": 3},
                }
            ]
        }
    }


class QueryRewriteResult(BaseModel):
    """Result of query rewriting operation.
    
    This model represents the structured output from the QueryRewriter,
    separating user intent into search-optimized and generation-optimized components.
    
    Attributes:
        search_rag: Whether to search the knowledge base (true if @knowledgebase in query)
        embedding_source_text: Clean keywords for vector/embedding search
        llm_query: Clear instruction for the LLM response generation
    """
    
    search_rag: bool = Field(
        default=False,
        description="True if query contains '@knowledgebase' trigger",
    )
    embedding_source_text: str = Field(
        default="",
        description="Core topic keywords for embedding/vector search, without instruction words",
    )
    llm_query: str = Field(
        default="",
        description="Clear instruction for LLM with context reference",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "search_rag": True,
                    "embedding_source_text": "vibe coding programming approach",
                    "llm_query": (
                        "Based on the provided context, explain what vibe coding is, "
                        "including its pros and cons, and cite sources."
                    ),
                }
            ]
        }
    }


class ConversationContext(BaseModel):
    """Context for conversational query rewriting.
    
    Represents the conversation history and context for follow-up queries.
    
    Attributes:
        previous_queries: List of previous user queries in the conversation
        previous_responses: List of previous assistant responses
        session_id: Optional session identifier for the conversation
    """
    
    previous_queries: list[str] = Field(
        default_factory=list,
        description="Previous user queries in the conversation",
    )
    previous_responses: list[str] = Field(
        default_factory=list,
        description="Previous assistant responses",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional conversation session identifier",
    )


class RAGConfig(BaseModel):
    """Configuration for RAG pipeline execution.
    
    This model defines which strategies to enable for a RAG query,
    allowing fine-grained control over the retrieval and generation process.
    
    Attributes:
        query_rewrite: Whether to apply query rewriting
        hyde: Whether to use HyDE (Hypothetical Document Embeddings)
        reranking: Whether to apply cross-encoder re-ranking
        hybrid_search: Whether to use hybrid search (vector + text)
        strategy_preset: Preset configuration name (fast, balanced, thorough)
    """
    
    query_rewrite: bool = Field(
        default=True,
        description="Enable query rewriting for better retrieval",
    )
    hyde: bool = Field(
        default=False,
        description="Enable HyDE for vague/complex queries",
    )
    reranking: bool = Field(
        default=True,
        description="Enable cross-encoder re-ranking",
    )
    hybrid_search: bool = Field(
        default=True,
        description="Enable hybrid search (vector + text)",
    )
    strategy_preset: str = Field(
        default="balanced",
        description="Strategy preset: fast, balanced, thorough, or auto",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query_rewrite": True,
                    "hyde": False,
                    "reranking": True,
                    "hybrid_search": True,
                    "strategy_preset": "balanced",
                }
            ]
        }
    )


class Source(BaseModel):
    """Source document reference in RAG results.
    
    Represents a source document that was used to generate the answer,
    including relevance score and metadata.
    
    Attributes:
        chunk_id: Unique identifier of the source chunk
        content: Text content from the source
        score: Relevance score (0-1)
        metadata: Optional metadata about the source (document name, page, etc.)
    """
    
    chunk_id: str = Field(
        description="Unique identifier of the source chunk",
    )
    content: str = Field(
        description="Text content from the source",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score (0-1)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the source",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                    "content": "Vibe coding is a programming approach...",
                    "score": 0.92,
                    "metadata": {"source": "doc1.pdf", "page": 3},
                }
            ]
        }
    )


class RAGMetrics(BaseModel):
    """Metrics for RAG pipeline execution.
    
    Tracks performance and quality metrics for a RAG query execution,
    useful for monitoring, optimization, and debugging.
    
    Attributes:
        latency_ms: Total query processing time in milliseconds
        tokens_used: Total tokens consumed by LLM calls
        retrieval_score: Average relevance score of retrieved chunks
        classification_confidence: Confidence score of query classification
        rewrite_time_ms: Time spent on query rewriting
        retrieval_time_ms: Time spent on document retrieval
        reranking_time_ms: Time spent on re-ranking (if enabled)
        generation_time_ms: Time spent on answer generation
        chunks_retrieved: Number of chunks retrieved
        chunks_used: Number of chunks used in final answer
        self_correction_iterations: Number of self-correction iterations
    """
    
    latency_ms: float = Field(
        description="Total query processing time in milliseconds",
    )
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed by LLM calls",
    )
    retrieval_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Average relevance score of retrieved chunks",
    )
    classification_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of query classification",
    )
    rewrite_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on query rewriting",
    )
    retrieval_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on document retrieval",
    )
    reranking_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on re-ranking (if enabled)",
    )
    generation_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on answer generation",
    )
    chunks_retrieved: int = Field(
        default=0,
        ge=0,
        description="Number of chunks retrieved",
    )
    chunks_used: int = Field(
        default=0,
        ge=0,
        description="Number of chunks used in final answer",
    )
    self_correction_iterations: int = Field(
        default=0,
        ge=0,
        description="Number of self-correction iterations",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "latency_ms": 245.5,
                    "tokens_used": 850,
                    "retrieval_score": 0.82,
                    "classification_confidence": 0.95,
                    "rewrite_time_ms": 45.2,
                    "retrieval_time_ms": 120.0,
                    "reranking_time_ms": 35.5,
                    "generation_time_ms": 44.8,
                    "chunks_retrieved": 20,
                    "chunks_used": 5,
                    "self_correction_iterations": 0,
                }
            ]
        }
    )


class RAGResult(BaseModel):
    """Result of RAG pipeline execution.
    
    Contains the generated answer, source documents, and execution metrics.
    
    Attributes:
        answer: Generated answer text
        sources: List of source documents used
        metrics: Execution metrics
        strategy_used: Strategy preset used for this query
        query_type: Type of query classified
    """
    
    answer: str = Field(
        description="Generated answer text",
    )
    sources: list[Source] = Field(
        default_factory=list,
        description="List of source documents used",
    )
    metrics: RAGMetrics = Field(
        description="Execution metrics",
    )
    strategy_used: str = Field(
        default="balanced",
        description="Strategy preset used for this query",
    )
    query_type: QueryType = Field(
        default=QueryType.FACTUAL,
        description="Type of query classified",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "Vibe coding is a programming approach that emphasizes...",
                    "sources": [
                        {
                            "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                            "content": "Vibe coding is a programming approach...",
                            "score": 0.92,
                            "metadata": {"source": "doc1.pdf", "page": 3},
                        }
                    ],
                    "metrics": {
                        "latency_ms": 245.5,
                        "tokens_used": 850,
                        "retrieval_score": 0.82,
                        "classification_confidence": 0.95,
                    },
                    "strategy_used": "balanced",
                    "query_type": "factual",
                }
            ]
        }
    )


class WeightPreset(str, Enum):
    """Predefined weight configurations for hybrid search.
    
    These presets control the balance between semantic (vector) and
    lexical (text) search components in hybrid search.
    
    Attributes:
        SEMANTIC_FOCUS: Prioritizes semantic similarity (vector: 0.9, text: 0.1)
        BALANCED: Balanced approach (vector: 0.7, text: 0.3)
        LEXICAL_FOCUS: Prioritizes lexical matching (vector: 0.3, text: 0.7)
    """
    
    SEMANTIC_FOCUS = "semantic_focus"
    BALANCED = "balanced"
    LEXICAL_FOCUS = "lexical_focus"


class FusionMethod(str, Enum):
    """Fusion methods for combining vector and text search results.
    
    Attributes:
        WEIGHTED_SUM: Linear combination of normalized scores
        RECIPROCAL_RANK_FUSION: Rank-based fusion using RRF formula with weights
    """
    
    WEIGHTED_SUM = "weighted_sum"
    RECIPROCAL_RANK_FUSION = "rrf"


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search API.
    
    Attributes:
        query: Text query for full-text search
        embedding: Vector embedding for similarity search (optional if using text-only)
        filters: Metadata filters for result filtering
        limit: Number of results to return (default: 10)
        vector_weight: Weight for vector scores (0-1, overrides preset)
        text_weight: Weight for text scores (0-1, overrides preset)
        weight_preset: Predefined weight configuration
        fusion_method: Method for combining results (rrf or weighted_sum)
        use_query_expansion: Whether to expand query for better lexical matching
    """
    
    query: str = Field(
        ...,
        description="Text query for full-text search",
        min_length=1,
        max_length=2000,
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding for similarity search",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters (e.g., {'source_type': 'pdf', 'author': 'John'})",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    vector_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weight for vector scores (overrides preset)",
    )
    text_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weight for text scores (overrides preset)",
    )
    weight_preset: WeightPreset = Field(
        default=WeightPreset.BALANCED,
        description="Predefined weight configuration",
    )
    fusion_method: FusionMethod = Field(
        default=FusionMethod.RECIPROCAL_RANK_FUSION,
        description="Method for combining vector and text results",
    )
    use_query_expansion: bool = Field(
        default=True,
        description="Whether to expand query for better lexical matching",
    )
    
    @field_validator("text_weight", "vector_weight")
    @classmethod
    def validate_weight_range(cls, v: float | None) -> float | None:
        """Validate weight is in valid range."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "machine learning applications in healthcare",
                    "embedding": [0.1, 0.2, 0.3] * 512,  # Example 1536-dim embedding
                    "filters": {"source_type": "pdf", "document_type": "research"},
                    "limit": 10,
                    "weight_preset": "balanced",
                    "fusion_method": "rrf",
                    "use_query_expansion": True,
                }
            ]
        }
    )


class HybridSearchResultItem(BaseModel):
    """Individual result item in hybrid search response.
    
    Attributes:
        chunk_id: Unique identifier of the document chunk
        content: Text content of the chunk
        hybrid_score: Final combined relevance score (0-1)
        vector_score: Cosine similarity from vector search (0-1), or None
        text_score: BM25 score from text search (0-1), or None
        vector_rank: Rank in vector search results (1-based), or None
        text_rank: Rank in text search results (1-based), or None
        metadata: Additional metadata about the chunk
    """
    
    chunk_id: str = Field(
        ...,
        description="Unique identifier of the document chunk",
    )
    content: str = Field(
        ...,
        description="Text content of the chunk",
    )
    hybrid_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final combined relevance score (0-1)",
    )
    vector_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity from vector search (0-1)",
    )
    text_score: float | None = Field(
        default=None,
        ge=0.0,
        description="BM25 score from text search (normalized 0-1)",
    )
    vector_rank: int | None = Field(
        default=None,
        ge=1,
        description="Rank in vector search results (1-based)",
    )
    text_rank: int | None = Field(
        default=None,
        ge=1,
        description="Rank in text search results (1-based)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the chunk",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                    "content": "Machine learning has revolutionized healthcare by enabling...",
                    "hybrid_score": 0.85,
                    "vector_score": 0.82,
                    "text_score": 0.88,
                    "vector_rank": 2,
                    "text_rank": 1,
                    "metadata": {"source": "paper.pdf", "page": 5, "author": "Dr. Smith"},
                }
            ]
        }
    )


class HybridSearchResponse(BaseModel):
    """Response model for hybrid search API.
    
    Attributes:
        results: List of search results
        total_results: Total number of results found
        query: Original search query
        expanded_query: Query after expansion (if enabled)
        fusion_method: Method used for fusion
        weight_preset: Preset used for weights
        vector_weight: Actual vector weight used
        text_weight: Actual text weight used
        filters_applied: Filters that were applied
        latency_ms: Query processing time in milliseconds
    """
    
    results: list[HybridSearchResultItem] = Field(
        ...,
        description="List of search results",
    )
    total_results: int = Field(
        ...,
        ge=0,
        description="Total number of results found",
    )
    query: str = Field(
        ...,
        description="Original search query",
    )
    expanded_query: str | None = Field(
        default=None,
        description="Query after expansion (if enabled)",
    )
    fusion_method: FusionMethod = Field(
        ...,
        description="Method used for fusion",
    )
    weight_preset: WeightPreset = Field(
        ...,
        description="Preset used for weights",
    )
    vector_weight: float = Field(
        ...,
        description="Actual vector weight used",
    )
    text_weight: float = Field(
        ...,
        description="Actual text weight used",
    )
    filters_applied: dict[str, Any] = Field(
        default_factory=dict,
        description="Filters that were applied",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Query processing time in milliseconds",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "results": [
                        {
                            "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                            "content": "Machine learning has revolutionized healthcare...",
                            "hybrid_score": 0.85,
                            "vector_score": 0.82,
                            "text_score": 0.88,
                            "vector_rank": 2,
                            "text_rank": 1,
                            "metadata": {"source": "paper.pdf", "page": 5},
                        }
                    ],
                    "total_results": 10,
                    "query": "machine learning healthcare",
                    "expanded_query": "machine learning healthcare ai medical",
                    "fusion_method": "rrf",
                    "weight_preset": "balanced",
                    "vector_weight": 0.7,
                    "text_weight": 0.3,
                    "filters_applied": {"source_type": "pdf"},
                    "latency_ms": 45.2,
                }
            ]
        }
    )



class ContextualContext(BaseModel):
    """Context information for enhanced chunk retrieval.
    
    Contains contextual information about a chunk's position within
    a document, including parent document info, section headers,
    neighboring chunks, and hierarchical relationships.
    
    Attributes:
        document_id: ID of the parent document
        document_title: Title of the parent document
        section_headers: List of section headers containing this chunk
        document_metadata: Additional document-level metadata
        previous_chunk_content: Content of the previous chunk (for window strategy)
        next_chunk_content: Content of the next chunk (for window strategy)
        hierarchy_path: Hierarchical path as a list (e.g., ["Section 1", "Subsection 1.1"])
        hierarchy_level: Depth level in the document hierarchy
    """
    
    document_id: str | None = Field(
        default=None,
        description="ID of the parent document",
    )
    document_title: str | None = Field(
        default=None,
        description="Title of the parent document",
    )
    section_headers: list[str] = Field(
        default_factory=list,
        description="List of section headers containing this chunk",
    )
    document_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document-level metadata (author, category, etc.)",
    )
    previous_chunk_content: str | None = Field(
        default=None,
        description="Content of the previous chunk (for window strategy)",
    )
    next_chunk_content: str | None = Field(
        default=None,
        description="Content of the next chunk (for window strategy)",
    )
    hierarchy_path: list[str] = Field(
        default_factory=list,
        description="Hierarchical path as a list of section/subsection names",
    )
    hierarchy_level: int | None = Field(
        default=None,
        ge=0,
        description="Depth level in the document hierarchy (0=document, 1=section, etc.)",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "550e8400-e29b-41d4-a716-446655440000",
                    "document_title": "Database Architecture Guide",
                    "section_headers": ["Vector Storage", "Implementation Details"],
                    "document_metadata": {
                        "author": "John Smith",
                        "category": "technical",
                        "version": "2.0",
                    },
                    "previous_chunk_content": "...semantic search requires vector storage.",
                    "next_chunk_content": "This enables efficient similarity queries.",
                    "hierarchy_path": ["Vector Storage", "Implementation Details"],
                    "hierarchy_level": 2,
                }
            ]
        }
    )


class EnhancedChunk(BaseModel):
    """A chunk enhanced with contextual information for improved embedding.
    
    This model represents a document chunk that has been enriched with
    contextual information (parent document, neighboring chunks, hierarchy)
    to improve semantic understanding and retrieval quality.
    
    Attributes:
        chunk_id: Unique identifier of the original chunk
        original_content: Original chunk content without context
        enhanced_text: Content enhanced with context for embedding
        context: Contextual information about the chunk
        embedding: Optional vector embedding of the enhanced text
        context_type: Strategy used for context enhancement
        enhanced_at: Timestamp of enhancement
    """
    
    chunk_id: str = Field(
        ...,
        description="Unique identifier of the original chunk",
    )
    original_content: str = Field(
        ...,
        description="Original chunk content without context",
    )
    enhanced_text: str = Field(
        ...,
        description="Content enhanced with context for embedding",
    )
    context: ContextualContext = Field(
        default_factory=ContextualContext,
        description="Contextual information about the chunk",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Optional vector embedding of the enhanced text",
    )
    context_type: ContextType = Field(
        default=ContextType.PARENT_DOCUMENT,
        description="Strategy used for context enhancement",
    )
    enhanced_at: str | None = Field(
        default=None,
        description="ISO timestamp of when the enhancement was created",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                    "original_content": "The system uses pgvector for vector storage.",
                    "enhanced_text": (
                        "Document: Database Architecture Guide\n"
                        "Section: Vector Storage\n"
                        "Content: The system uses pgvector for vector storage."
                    ),
                    "context": {
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "document_title": "Database Architecture Guide",
                        "section_headers": ["Vector Storage"],
                        "document_metadata": {"author": "John Smith"},
                        "hierarchy_level": 2,
                    },
                    "context_type": "parent_document",
                    "enhanced_at": "2026-02-20T12:00:00Z",
                }
            ]
        }
    )


class ContextualRetrievalRequest(BaseModel):
    """Request model for contextual retrieval enhancement.
    
    Attributes:
        chunk_id: ID of the chunk to enhance
        context_type: Strategy to use for enhancement
        include_metadata: Whether to include document metadata
        metadata_fields: Specific metadata fields to include
    """
    
    chunk_id: str = Field(
        ...,
        description="ID of the chunk to enhance",
    )
    context_type: ContextType = Field(
        default=ContextType.PARENT_DOCUMENT,
        description="Strategy to use for enhancement",
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include document metadata",
    )
    metadata_fields: list[str] | None = Field(
        default=None,
        description="Specific metadata fields to include (None = all)",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                    "context_type": "parent_document",
                    "include_metadata": True,
                    "metadata_fields": ["title", "author", "category"],
                }
            ]
        }
    )


class ContextualRetrievalResult(BaseModel):
    """Result of contextual retrieval enhancement.
    
    Attributes:
        success: Whether enhancement was successful
        enhanced_chunk: The enhanced chunk (if successful)
        error: Error message (if failed)
        latency_ms: Processing time in milliseconds
    """
    
    success: bool = Field(
        ...,
        description="Whether enhancement was successful",
    )
    enhanced_chunk: EnhancedChunk | None = Field(
        default=None,
        description="The enhanced chunk (if successful)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (if failed)",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "success": True,
                    "enhanced_chunk": {
                        "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                        "original_content": "The system uses pgvector for vector storage.",
                        "enhanced_text": "Document: Database Architecture Guide\n...",
                        "context_type": "parent_document",
                    },
                    "latency_ms": 5.2,
                }
            ]
        }
    )


class DocumentSection(BaseModel):
    """Represents a section within a document for hierarchical chunking.
    
    Attributes:
        header: Section header/title
        level: Hierarchy level (1 for top-level, 2 for subsection, etc.)
        content: Text content of the section
        parent_id: Optional ID of parent section
        subsections: List of child subsections
        metadata: Additional section metadata
    """
    
    header: str = Field(
        ...,
        description="Section header/title",
    )
    level: int = Field(
        ...,
        ge=1,
        description="Hierarchy level (1 for top-level, 2 for subsection, etc.)",
    )
    content: str = Field(
        ...,
        description="Text content of the section",
    )
    parent_id: str | None = Field(
        default=None,
        description="Optional ID of parent section",
    )
    subsections: list["DocumentSection"] = Field(
        default_factory=list,
        description="List of child subsections",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional section metadata",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "header": "Introduction",
                    "level": 1,
                    "content": "This is the introduction content.",
                    "parent_id": None,
                    "subsections": [],
                    "metadata": {"page": 1},
                }
            ]
        }
    )


class Document(BaseModel):
    """Represents a document for chunking operations.
    
    Attributes:
        id: Unique document identifier
        title: Document title
        content: Full document text content
        sections: Document sections for hierarchical chunking
        metadata: Document metadata
        doc_type: Document type (e.g., 'technical', 'narrative', 'legal')
        has_structure: Whether document has clear structural elements
    """
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique document identifier",
    )
    title: str | None = Field(
        default=None,
        description="Document title",
    )
    content: str = Field(
        ...,
        description="Full document text content",
    )
    sections: list[DocumentSection] = Field(
        default_factory=list,
        description="Document sections for hierarchical chunking",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata",
    )
    doc_type: str | None = Field(
        default=None,
        description="Document type (e.g., 'technical', 'narrative', 'legal')",
    )
    
    def has_clear_structure(self) -> bool:
        """Check if document has clear structural elements."""
        return len(self.sections) > 0 or self._has_headers()
    
    def is_technical(self) -> bool:
        """Check if document appears to be technical content."""
        technical_indicators = [
            "code", "api", "function", "class", "module",
            "implementation", "configuration", "parameter",
        ]
        content_lower = self.content.lower()
        indicator_count = sum(
            1 for indicator in technical_indicators
            if indicator in content_lower
        )
        # If more than 3 technical indicators, consider it technical
        return indicator_count > 3
    
    def _has_headers(self) -> bool:
        """Check if content has markdown-style headers."""
        import re
        header_pattern = r"^#{1,6}\s+.+$"
        return bool(re.search(header_pattern, self.content, re.MULTILINE))
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "title": "API Documentation",
                    "content": "# Introduction\nThis is the API documentation.",
                    "sections": [],
                    "metadata": {"author": "John Doe"},
                    "doc_type": "technical",
                }
            ]
        }
    )


class Chunk(BaseModel):
    """Represents a document chunk produced by chunking strategies.
    
    Attributes:
        id: Unique chunk identifier
        content: Text content of the chunk
        index: Position of chunk within document
        metadata: Chunk metadata including source info
        parent_section_id: ID of parent section (for hierarchical chunks)
        hierarchy_level: Depth level in document hierarchy
        embedding: Optional vector embedding
        token_count: Number of tokens in chunk
    """
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique chunk identifier",
    )
    content: str = Field(
        ...,
        description="Text content of the chunk",
    )
    index: int = Field(
        default=0,
        ge=0,
        description="Position of chunk within document",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata including source info",
    )
    parent_section_id: str | None = Field(
        default=None,
        description="ID of parent section (for hierarchical chunks)",
    )
    hierarchy_level: int | None = Field(
        default=None,
        ge=0,
        description="Depth level in document hierarchy",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Optional vector embedding",
    )
    token_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of tokens in chunk",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440001",
                    "content": "This is a chunk of text.",
                    "index": 0,
                    "metadata": {"source": "doc1.pdf", "page": 1},
                    "parent_section_id": None,
                    "hierarchy_level": None,
                    "token_count": 5,
                }
            ]
        }
    )


class ChunkingStrategy(str, Enum):
    """Available chunking strategies.
    
    Attributes:
        SEMANTIC: Chunk based on semantic boundaries
        HIERARCHICAL: Chunk based on document structure
        FIXED: Fixed-size chunking with overlap
        AGENTIC: Agentically select best strategy
    """
    
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    FIXED = "fixed"
    AGENTIC = "agentic"


class ChunkingResult(BaseModel):
    """Result of a chunking operation.
    
    Attributes:
        success: Whether chunking was successful
        chunks: List of produced chunks
        strategy_used: Strategy that was used
        error: Error message if failed
        metrics: Performance metrics
    """
    
    success: bool = Field(
        ...,
        description="Whether chunking was successful",
    )
    chunks: list[Chunk] = Field(
        default_factory=list,
        description="List of produced chunks",
    )
    strategy_used: str = Field(
        default="",
        description="Strategy that was used",
    )
    error: str | None = Field(
        default=None,
        description="Error message if failed",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance metrics",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "success": True,
                    "chunks": [
                        {
                            "id": "chunk-1",
                            "content": "First chunk",
                            "index": 0,
                        }
                    ],
                    "strategy_used": "semantic",
                    "metrics": {
                        "total_chunks": 1,
                        "processing_time_ms": 50.0,
                    },
                }
            ]
        }
    )



class EmbeddingModelInfo(BaseModel):
    """Information about an embedding model.
    
    Attributes:
        name: Model identifier
        dimensions: Number of output dimensions
        speed: Speed category (very_fast, fast, medium, slow)
        quality: Quality category (good, excellent)
        provider: Provider name (openai, sentence_transformers, etc.)
        batch_size: Recommended batch size
    """
    
    name: str = Field(..., description="Model identifier")
    dimensions: int = Field(..., description="Number of output dimensions")
    speed: str = Field(..., description="Speed category")
    quality: str = Field(..., description="Quality category")
    provider: str = Field(..., description="Provider name")
    batch_size: int = Field(default=100, description="Recommended batch size")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "text-embedding-3-small",
                    "dimensions": 1536,
                    "speed": "fast",
                    "quality": "good",
                    "provider": "openai",
                    "batch_size": 100,
                }
            ]
        }
    )


class EmbeddingResult(BaseModel):
    """Result of embedding generation.
    
    Attributes:
        vector: The embedding vector
        model: Model used for generation
        dimensions: Vector dimensions
        text_hash: Hash of original text
        metadata: Additional metadata (latency, tokens)
    """
    
    vector: list[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used for generation")
    dimensions: int = Field(..., description="Vector dimensions")
    text_hash: str | None = Field(default=None, description="Hash of original text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "vector": [0.1, 0.2, 0.3],
                    "model": "text-embedding-3-small",
                    "dimensions": 1536,
                    "text_hash": "abc123",
                    "metadata": {"latency_ms": 45.2, "tokens": 10},
                }
            ]
        }
    )


class EmbeddingBatchResult(BaseModel):
    """Result of batch embedding generation.
    
    Attributes:
        embeddings: List of embedding results
        model: Model used
        total_tokens: Total tokens consumed
        cache_hits: Number of cache hits
        latency_ms: Total latency in milliseconds
    """
    
    embeddings: list[EmbeddingResult] = Field(default_factory=list)
    model: str = Field(default="")
    total_tokens: int = Field(default=0)
    cache_hits: int = Field(default=0)
    latency_ms: float = Field(default=0.0)
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "embeddings": [],
                    "model": "text-embedding-3-small",
                    "total_tokens": 150,
                    "cache_hits": 2,
                    "latency_ms": 45.2,
                }
            ]
        }
    )


class QuantizedEmbedding(BaseModel):
    """Quantized embedding for storage efficiency.
    
    Attributes:
        data: Quantized byte data
        scale: Min/max values for dequantization
        original_dims: Original dimensions before quantization
        compression_ratio: Achieved compression ratio
    """
    
    data: str = Field(..., description="Base64-encoded quantized data")
    scale: tuple[float, float] = Field(..., description="(min, max) values")
    original_dims: int = Field(..., description="Original dimensions")
    compression_ratio: float = Field(default=4.0, description="Compression ratio achieved")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "data": "base64encodedstring...",
                    "scale": [-0.5, 0.5],
                    "original_dims": 1536,
                    "compression_ratio": 4.0,
                }
            ]
        }
    )


class EmbeddingCacheStats(BaseModel):
    """Statistics for embedding cache.
    
    Attributes:
        size: Current cache size
        max_size: Maximum cache size
        hits: Number of cache hits
        misses: Number of cache misses
        hit_rate: Cache hit rate (0-1)
        ttl_seconds: Time-to-live setting
    """
    
    size: int = Field(..., description="Current cache size")
    max_size: int = Field(..., description="Maximum cache size")
    hits: int = Field(..., description="Number of cache hits")
    misses: int = Field(..., description="Number of cache misses")
    hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate")
    ttl_seconds: int = Field(..., description="Time-to-live setting")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "size": 50000,
                    "max_size": 100000,
                    "hits": 750000,
                    "misses": 250000,
                    "hit_rate": 0.75,
                    "ttl_seconds": 86400,
                }
            ]
        }
    )


class DimensionalityReductionConfig(BaseModel):
    """Configuration for dimensionality reduction.
    
    Attributes:
        enabled: Whether reduction is enabled
        target_dimensions: Target number of dimensions
        preserve_threshold: Minimum quality preservation threshold
        method: Reduction method (pca, autoencoder)
    """
    
    enabled: bool = Field(default=True, description="Whether reduction is enabled")
    target_dimensions: int = Field(default=256, ge=64, le=1024, description="Target dimensions")
    preserve_threshold: float = Field(
        default=0.95, ge=0.8, le=1.0, description="Quality preservation threshold"
    )
    method: str = Field(default="pca", description="Reduction method")
    
    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate reduction method."""
        valid_methods = {"pca", "autoencoder"}
        if v.lower() not in valid_methods:
            raise ValueError(f"Invalid method: {v}. Must be one of: {valid_methods}")
        return v.lower()
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "enabled": True,
                    "target_dimensions": 256,
                    "preserve_threshold": 0.95,
                    "method": "pca",
                }
            ]
        }
    )


class EmbeddingOptimizationConfig(BaseModel):
    """Configuration for embedding optimization.
    
    Attributes:
        dimensionality_reduction: Dimensionality reduction settings
        quantization: Quantization settings
        caching: Caching settings
        auto_selection: Auto model selection settings
    """
    
    dimensionality_reduction: DimensionalityReductionConfig = Field(
        default_factory=DimensionalityReductionConfig,
        description="Dimensionality reduction settings",
    )
    quantization: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "bits": 8,
            "compression_ratio": 4.0,
        },
        description="Quantization settings",
    )
    caching: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "ttl": 86400,
            "max_size": 100000,
        },
        description="Caching settings",
    )
    auto_selection: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "rules": [
                {"condition": "technical_content AND length > 500", "model": "bge-large"},
                {"condition": "length > 2000", "model": "text-embedding-3-large"},
                {"condition": "default", "model": "text-embedding-3-small"},
            ],
        },
        description="Auto model selection settings",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "dimensionality_reduction": {
                        "enabled": True,
                        "target_dimensions": 256,
                        "preserve_threshold": 0.95,
                    },
                    "quantization": {"enabled": True, "bits": 8},
                    "caching": {"enabled": True, "ttl": 86400},
                }
            ]
        }
    )


class ModelSelectionRule(BaseModel):
    """Rule for automatic model selection.
    
    Attributes:
        condition: Condition expression
        model: Model to select when condition matches
        priority: Rule priority (higher = evaluated first)
    """
    
    condition: str = Field(..., description="Condition expression")
    model: str = Field(..., description="Model to select")
    priority: int = Field(default=0, description="Rule priority")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "condition": "technical_content AND length > 500",
                    "model": "bge-large",
                    "priority": 10,
                }
            ]
        }
    )


class EmbeddingRequest(BaseModel):
    """Request for embedding generation.
    
    Attributes:
        texts: Texts to embed
        model: Model or alias to use
        use_cache: Whether to use caching
        reduce: Whether to apply dimensionality reduction
        quantize: Whether to apply quantization
    """
    
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Texts to embed",
    )
    model: str = Field(default="auto", description="Model or alias")
    use_cache: bool = Field(default=True, description="Whether to use caching")
    reduce: bool | None = Field(default=None, description="Apply dimensionality reduction")
    quantize: bool | None = Field(default=None, description="Apply quantization")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "texts": ["Hello world", "Test text"],
                    "model": "auto",
                    "use_cache": True,
                }
            ]
        }
    )
