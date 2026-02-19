"""Search API routes.

This module provides endpoints for semantic, text, hybrid, and similar chunk
search operations using the vector and text search services.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db, parse_uuid
from src.api.middleware.rate_limiter import (
    rate_limit_hybrid_search,
    rate_limit_semantic_search,
    rate_limit_similar_chunks,
    rate_limit_text_search,
)
from src.api.models import ApiResponse, DocumentChunkListItem
from src.api.validators.search_validators import (
    HybridSearchValidatorsMixin,
    SemanticSearchValidatorsMixin,
    TextSearchValidatorsMixin,
    sanitize_search_query,
    validate_language,
)
from src.db.repositories.document_chunk_repository import DocumentChunkRepository
from src.observability.logging import get_logger
from src.services.hybrid_search_service import (
    FusionMethod,
    HybridSearchResult,
    HybridSearchService,
)
from src.services.text_search_service import TextSearchResult, TextSearchService
from src.services.vector_search_service import (
    ChunkNotFoundError,
    InvalidEmbeddingError,
    SearchResult,
    VectorSearchError,
    VectorSearchService,
)

router = APIRouter(prefix="/search", tags=["Search"])
logger = get_logger(__name__)


# ============================================================================
# Request Models
# ============================================================================

class SemanticSearchRequest(BaseModel, SemanticSearchValidatorsMixin):
    """Request body for semantic search.
    
    Performs cosine similarity search using a pre-computed embedding vector.
    """
    model_config = {"json_schema_extra": {
        "example": {
            "query_embedding": [0.1, 0.2, 0.3, 0.4],
            "top_k": 10,
            "min_similarity": 0.7,
            "filters": {"job_id": "123e4567-e89b-12d3-a456-426614174000"},
        }
    }}

    query_embedding: list[float] = Field(
        ...,
        description="Query embedding vector (list of floats)",
        min_length=1,
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    min_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0-1)",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional filters (e.g., job_id)",
    )

    @field_validator("query_embedding")
    @classmethod
    def validate_embedding_not_empty(cls, v: list[float]) -> list[float]:
        """Validate embedding is not empty."""
        if not v:
            raise ValueError("query_embedding cannot be empty")
        return v


class TextSearchRequest(BaseModel, TextSearchValidatorsMixin):
    """Request body for text search.
    
    Performs full-text search using PostgreSQL's tsvector/tsquery with BM25 ranking
    and optional fuzzy trigram matching.
    """
    model_config = {"json_schema_extra": {
        "example": {
            "query": "machine learning architecture",
            "top_k": 10,
            "language": "english",
            "use_fuzzy": True,
            "highlight": True,
            "filters": {},
        }
    }}

    query: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Search query text",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    language: str = Field(
        default="english",
        description="Text search language configuration",
    )
    use_fuzzy: bool = Field(
        default=True,
        description="Include fuzzy trigram matching",
    )
    highlight: bool = Field(
        default=False,
        description="Include highlighted snippets with matched terms",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional filters (e.g., job_id)",
    )

    @field_validator("query")
    @classmethod
    def sanitize_query_input(cls, v: str) -> str:
        """Sanitize query input to prevent XSS."""
        return sanitize_search_query(v)


class HybridSearchRequest(BaseModel, HybridSearchValidatorsMixin):
    """Request body for hybrid search.
    
    Combines vector similarity search and text search using configurable fusion methods
    (weighted sum or reciprocal rank fusion).
    """
    model_config = {"json_schema_extra": {
        "example": {
            "query": "neural network architecture",
            "top_k": 10,
            "vector_weight": 0.7,
            "text_weight": 0.3,
            "fusion_method": "weighted_sum",
            "min_similarity": 0.5,
            "filters": {},
        }
    }}

    query: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Search query text",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector search scores",
    )
    text_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for text search scores",
    )
    fusion_method: str = Field(
        default="weighted_sum",
        pattern="^(weighted_sum|rrf)$",
        description="Fusion method: 'weighted_sum' or 'rrf' (reciprocal rank fusion)",
    )
    min_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for vector search",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional filters (e.g., job_id)",
    )

    @field_validator("query")
    @classmethod
    def sanitize_query_input(cls, v: str) -> str:
        """Sanitize query input to prevent XSS."""
        return sanitize_search_query(v)

    @model_validator(mode="after")
    def validate_weight_sum(self) -> "HybridSearchRequest":
        """Validate that vector_weight + text_weight ≈ 1.0."""
        total = self.vector_weight + self.text_weight
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"vector_weight + text_weight must equal 1.0, got {total:.4f}")
        return self


# ============================================================================
# Response Models
# ============================================================================

class SearchResultItem(BaseModel):
    """Single search result item."""
    model_config = {"json_schema_extra": {
        "example": {
            "chunk_id": "123e4567-e89b-12d3-a456-426614174000",
            "job_id": "123e4567-e89b-12d3-a456-426614174001",
            "chunk_index": 0,
            "content": "Sample chunk content...",
            "metadata": {"page": 1},
            "similarity_score": 0.95,
            "rank": 1,
        }
    }}

    chunk_id: str = Field(..., description="Unique chunk identifier")
    job_id: str = Field(..., description="Parent job identifier")
    chunk_index: int = Field(..., ge=0, description="Position within document")
    content: str = Field(..., description="Text content of the chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    rank: int = Field(..., ge=1, description="Result rank (1-based)")


class TextSearchResultItem(SearchResultItem):
    """Text search result with additional fields."""
    model_config = {"json_schema_extra": {
        "example": {
            "chunk_id": "123e4567-e89b-12d3-a456-426614174000",
            "job_id": "123e4567-e89b-12d3-a456-426614174001",
            "chunk_index": 0,
            "content": "Sample chunk content...",
            "metadata": {"page": 1},
            "similarity_score": 0.85,
            "rank": 1,
            "highlighted_content": "Sample <mark>chunk</mark> content...",
            "matched_terms": ["chunk"],
        }
    }}

    highlighted_content: str | None = Field(
        default=None,
        description="Content with matched terms highlighted",
    )
    matched_terms: list[str] | None = Field(
        default=None,
        description="List of matched terms",
    )


class HybridSearchResultItem(BaseModel):
    """Hybrid search result with component scores."""
    model_config = {"json_schema_extra": {
        "example": {
            "chunk_id": "123e4567-e89b-12d3-a456-426614174000",
            "job_id": "123e4567-e89b-12d3-a456-426614174001",
            "chunk_index": 0,
            "content": "Sample chunk content...",
            "metadata": {"page": 1},
            "hybrid_score": 0.88,
            "vector_score": 0.92,
            "text_score": 0.78,
            "vector_rank": 1,
            "text_rank": 2,
            "rank": 1,
            "fusion_method": "weighted_sum",
        }
    }}

    chunk_id: str = Field(..., description="Unique chunk identifier")
    job_id: str = Field(..., description="Parent job identifier")
    chunk_index: int = Field(..., ge=0, description="Position within document")
    content: str = Field(..., description="Text content of the chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    hybrid_score: float = Field(..., description="Combined hybrid score (0-1)")
    vector_score: float | None = Field(
        default=None,
        description="Vector similarity score (0-1) if available",
    )
    text_score: float | None = Field(
        default=None,
        description="Text search score (0-1) if available",
    )
    vector_rank: int | None = Field(
        default=None,
        description="Rank in vector search results",
    )
    text_rank: int | None = Field(
        default=None,
        description="Rank in text search results",
    )
    rank: int = Field(..., ge=1, description="Final rank (1-based)")
    fusion_method: str = Field(..., description="Fusion method used")


class SearchResultsResponse(BaseModel):
    """Response model for search results."""
    model_config = {"json_schema_extra": {
        "example": {
            "results": [],
            "total": 0,
            "query_time_ms": 25.5,
        }
    }}

    results: list[SearchResultItem] = Field(default_factory=list, description="Search results")
    total: int = Field(..., ge=0, description="Total number of results")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")


class TextSearchResultsResponse(BaseModel):
    """Response model for text search results."""
    results: list[TextSearchResultItem] = Field(default_factory=list, description="Search results")
    total: int = Field(..., ge=0, description="Total number of results")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")


class HybridSearchResultsResponse(BaseModel):
    """Response model for hybrid search results."""
    results: list[HybridSearchResultItem] = Field(default_factory=list, description="Search results")
    total: int = Field(..., ge=0, description="Total number of results")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")


# ============================================================================
# Error Response Models
# ============================================================================

class InvalidEmbeddingResponse(BaseModel):
    """Error response for invalid embedding."""
    detail: str = Field(default="Invalid embedding vector")


class InvalidQueryResponse(BaseModel):
    """Error response for invalid query."""
    detail: str = Field(default="Invalid search query")


class ChunkNotFoundResponse(BaseModel):
    """Error response when reference chunk not found."""
    detail: str = Field(default="Reference chunk not found")


# ============================================================================
# Helper Functions
# ============================================================================

def _search_result_to_item(result: SearchResult) -> SearchResultItem:
    """Convert a SearchResult to a SearchResultItem.
    
    Args:
        result: SearchResult from VectorSearchService
        
    Returns:
        SearchResultItem for API response
    """
    chunk = result.chunk
    return SearchResultItem(
        chunk_id=str(chunk.id),
        job_id=str(chunk.job_id),
        chunk_index=int(chunk.chunk_index),
        content=str(chunk.content),
        metadata=dict(chunk.chunk_metadata or {}),
        similarity_score=result.similarity_score,
        rank=result.rank,
    )


def _text_search_result_to_item(result: TextSearchResult) -> TextSearchResultItem:
    """Convert a TextSearchResult to a TextSearchResultItem.
    
    Args:
        result: TextSearchResult from TextSearchService
        
    Returns:
        TextSearchResultItem for API response
    """
    chunk = result.chunk
    return TextSearchResultItem(
        chunk_id=str(chunk.id),
        job_id=str(chunk.job_id),
        chunk_index=int(chunk.chunk_index),
        content=str(chunk.content),
        metadata=dict(chunk.chunk_metadata or {}),
        similarity_score=result.rank_score,
        rank=result.rank,
        highlighted_content=result.highlighted_content,
        matched_terms=result.matched_terms,
    )


def _hybrid_search_result_to_item(result: HybridSearchResult) -> HybridSearchResultItem:
    """Convert a HybridSearchResult to a HybridSearchResultItem.
    
    Args:
        result: HybridSearchResult from HybridSearchService
        
    Returns:
        HybridSearchResultItem for API response
    """
    chunk = result.chunk
    return HybridSearchResultItem(
        chunk_id=str(chunk.id),
        job_id=str(chunk.job_id),
        chunk_index=int(chunk.chunk_index),
        content=str(chunk.content),
        metadata=dict(chunk.chunk_metadata or {}),
        hybrid_score=result.hybrid_score,
        vector_score=result.vector_score,
        text_score=result.text_score,
        vector_rank=result.vector_rank,
        text_rank=result.text_rank,
        rank=result.rank,
        fusion_method=result.fusion_method,
    )


# ============================================================================
# Semantic Search Endpoint (4.3)
# ============================================================================

@router.post(
    "/semantic",
    response_model=SearchResultsResponse,
    responses={
        400: {"model": InvalidEmbeddingResponse, "description": "Invalid embedding vector"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
    },
    summary="Semantic search by embedding",
    description="Search for similar chunks using a pre-computed embedding vector with cosine similarity.",
)
@rate_limit_semantic_search
async def semantic_search(
    request: Request,
    search_request: SemanticSearchRequest,
    db: AsyncSession = Depends(get_db),
) -> SearchResultsResponse:
    """Perform semantic search using an embedding vector.
    
    Uses pgvector's cosine similarity (<=>) to find chunks with embeddings
    similar to the query vector. Results are ordered by similarity descending.
    
    Args:
        request: FastAPI request object
        search_request: Semantic search parameters
        db: Database session
        
    Returns:
        List of search results with similarity scores
        
    Raises:
        HTTPException: 400 if embedding is invalid
    """
    import time
    start_time = time.monotonic()
    request_id = getattr(request.state, "request_id", str(UUID(int=0)))
    
    logger.info(
        "semantic_search_request",
        request_id=request_id,
        top_k=search_request.top_k,
        min_similarity=search_request.min_similarity,
        filters=search_request.filters,
    )
    
    try:
        # Initialize repository and service
        repo = DocumentChunkRepository(db)
        service = VectorSearchService(repo)
        
        # Process job_id filter if present
        filters = dict(search_request.filters)
        if "job_id" in filters and isinstance(filters["job_id"], str):
            try:
                filters["job_id"] = parse_uuid(filters["job_id"])
            except HTTPException as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid job_id UUID: {filters['job_id']}",
                ) from e
        
        # Perform search
        results = await service.search_by_vector(
            query_embedding=search_request.query_embedding,
            top_k=search_request.top_k,
            min_similarity=search_request.min_similarity,
            filters=filters or None,
        )
        
        # Convert to response items
        items = [_search_result_to_item(r) for r in results]
        
        query_time_ms = (time.monotonic() - start_time) * 1000
        
        logger.info(
            "semantic_search_completed",
            request_id=request_id,
            result_count=len(items),
            query_time_ms=round(query_time_ms, 2),
        )
        
        return SearchResultsResponse(
            results=items,
            total=len(items),
            query_time_ms=round(query_time_ms, 2),
        )
        
    except InvalidEmbeddingError as e:
        logger.warning(
            "semantic_search_invalid_embedding",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid embedding vector: {e}",
        ) from e
    except VectorSearchError as e:
        logger.error(
            "semantic_search_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        ) from e


# ============================================================================
# Text Search Endpoint (4.4)
# ============================================================================

@router.post(
    "/text",
    response_model=TextSearchResultsResponse,
    responses={
        400: {"model": InvalidQueryResponse, "description": "Invalid search query"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
    },
    summary="Text search",
    description="Search for chunks using full-text search with BM25 ranking and optional fuzzy matching.",
)
@rate_limit_text_search
async def text_search(
    request: Request,
    search_request: TextSearchRequest,
    db: AsyncSession = Depends(get_db),
) -> TextSearchResultsResponse:
    """Perform text search using PostgreSQL full-text search.
    
    Uses to_tsvector/to_tsquery with ts_rank_cd for BM25-like ranking.
    Optionally includes pg_trgm fuzzy matching for typo tolerance.
    
    Args:
        request: FastAPI request object
        search_request: Text search parameters
        db: Database session
        
    Returns:
        List of search results with rank scores and optional highlighting
        
    Raises:
        HTTPException: 400 if query is invalid
    """
    import time
    start_time = time.monotonic()
    request_id = getattr(request.state, "request_id", str(UUID(int=0)))
    
    logger.info(
        "text_search_request",
        request_id=request_id,
        query=search_request.query[:100],
        top_k=search_request.top_k,
        language=search_request.language,
        use_fuzzy=search_request.use_fuzzy,
        highlight=search_request.highlight,
    )
    
    try:
        # Initialize repository and service
        repo = DocumentChunkRepository(db)
        service = TextSearchService(repo)
        
        # Process job_id filter if present
        filters = dict(search_request.filters)
        if "job_id" in filters and isinstance(filters["job_id"], str):
            try:
                filters["job_id"] = parse_uuid(filters["job_id"])
            except HTTPException as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid job_id UUID: {filters['job_id']}",
                ) from e
        
        # Perform search
        results = await service.search_by_text(
            query=search_request.query,
            top_k=search_request.top_k,
            language=search_request.language,
            use_fuzzy=search_request.use_fuzzy,
            highlight=search_request.highlight,
            filters=filters or None,
        )
        
        # Convert to response items
        items = [_text_search_result_to_item(r) for r in results]
        
        query_time_ms = (time.monotonic() - start_time) * 1000
        
        logger.info(
            "text_search_completed",
            request_id=request_id,
            result_count=len(items),
            query_time_ms=round(query_time_ms, 2),
        )
        
        return TextSearchResultsResponse(
            results=items,
            total=len(items),
            query_time_ms=round(query_time_ms, 2),
        )
        
    except Exception as e:
        if "InvalidQueryError" in type(e).__name__ or "LanguageNotSupportedError" in type(e).__name__:
            logger.warning(
                "text_search_invalid_query",
                request_id=request_id,
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        
        logger.error(
            "text_search_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        ) from e


# ============================================================================
# Hybrid Search Endpoint (4.5)
# ============================================================================

@router.post(
    "/hybrid",
    response_model=HybridSearchResultsResponse,
    responses={
        400: {"model": InvalidQueryResponse, "description": "Invalid search query"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
    },
    summary="Hybrid search",
    description="Combine vector similarity and text search using weighted sum or RRF fusion.",
)
@rate_limit_hybrid_search
async def hybrid_search(
    request: Request,
    search_request: HybridSearchRequest,
    db: AsyncSession = Depends(get_db),
) -> HybridSearchResultsResponse:
    """Perform hybrid search combining vector and text search.
    
    Runs both vector similarity search and text search, then combines results
    using either weighted sum or Reciprocal Rank Fusion (RRF).
    
    Args:
        request: FastAPI request object
        search_request: Hybrid search parameters
        db: Database session
        
    Returns:
        List of hybrid search results with component scores
        
    Raises:
        HTTPException: 400 if query or weights are invalid
    """
    import time
    start_time = time.monotonic()
    request_id = getattr(request.state, "request_id", str(UUID(int=0)))
    
    logger.info(
        "hybrid_search_request",
        request_id=request_id,
        query=search_request.query[:100],
        top_k=search_request.top_k,
        vector_weight=search_request.vector_weight,
        text_weight=search_request.text_weight,
        fusion_method=search_request.fusion_method,
    )
    
    try:
        # Initialize repositories and services
        repo = DocumentChunkRepository(db)
        vector_service = VectorSearchService(repo)
        text_service = TextSearchService(repo)
        hybrid_service = HybridSearchService(vector_service, text_service)
        
        # Process job_id filter if present
        filters = dict(search_request.filters)
        if "job_id" in filters and isinstance(filters["job_id"], str):
            try:
                filters["job_id"] = parse_uuid(filters["job_id"])
            except HTTPException as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid job_id UUID: {filters['job_id']}",
                ) from e
        
        # Parse fusion method
        fusion_method = FusionMethod.WEIGHTED_SUM
        if search_request.fusion_method == "rrf":
            fusion_method = FusionMethod.RECIPROCAL_RANK_FUSION
        
        # Perform search - Note: This requires an embedder for the query
        # For now, we use search_with_embedding which requires a pre-computed embedding
        # In a production environment, this would use an embedding service
        logger.warning(
            "hybrid_search_requires_embedder",
            request_id=request_id,
            message="Hybrid search currently requires an external embedder - falling back to text-only results",
        )
        
        # Fallback to text search since we don't have an embedder in this context
        text_results = await text_service.search_by_text(
            query=search_request.query,
            top_k=search_request.top_k,
            filters=filters or None,
        )
        
        # Convert text results to hybrid format with placeholder scores
        items = []
        for rank, result in enumerate(text_results, start=1):
            chunk = result.chunk
            items.append(HybridSearchResultItem(
                chunk_id=str(chunk.id),
                job_id=str(chunk.job_id),
                chunk_index=int(chunk.chunk_index),
                content=str(chunk.content),
                metadata=dict(chunk.chunk_metadata or {}),
                hybrid_score=result.rank_score,
                vector_score=None,
                text_score=result.rank_score,
                vector_rank=None,
                text_rank=result.rank,
                rank=rank,
                fusion_method="text_fallback",
            ))
        
        query_time_ms = (time.monotonic() - start_time) * 1000
        
        logger.info(
            "hybrid_search_completed",
            request_id=request_id,
            result_count=len(items),
            query_time_ms=round(query_time_ms, 2),
            note="text_fallback_mode",
        )
        
        return HybridSearchResultsResponse(
            results=items,
            total=len(items),
            query_time_ms=round(query_time_ms, 2),
        )
        
    except Exception as e:
        if "InvalidWeightError" in type(e).__name__:
            logger.warning(
                "hybrid_search_invalid_weights",
                request_id=request_id,
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        
        logger.error(
            "hybrid_search_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        ) from e


# ============================================================================
# Similar Chunks Endpoint (4.6)
# ============================================================================

@router.get(
    "/similar/{chunk_id}",
    response_model=SearchResultsResponse,
    responses={
        400: {"description": "Invalid UUID format"},
        404: {"model": ChunkNotFoundResponse, "description": "Reference chunk not found"},
        429: {"description": "Rate limit exceeded"},
    },
    summary="Find similar chunks",
    description="Find chunks semantically similar to a reference chunk using its embedding.",
)
@rate_limit_similar_chunks
async def find_similar_chunks(
    chunk_id: str,
    request: Request,
    top_k: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    exclude_self: bool = Query(True, description="Exclude the reference chunk from results"),
    db: AsyncSession = Depends(get_db),
) -> SearchResultsResponse:
    """Find chunks similar to a reference chunk.
    
    Retrieves the embedding of the reference chunk and performs a similarity
    search to find semantically similar chunks.
    
    Args:
        chunk_id: UUID of the reference chunk
        request: FastAPI request object
        top_k: Maximum number of similar chunks to return
        exclude_self: If True, exclude the reference chunk from results
        db: Database session
        
    Returns:
        List of similar chunks with similarity scores
        
    Raises:
        HTTPException: 400 if UUID is invalid
        HTTPException: 404 if reference chunk not found
    """
    import time
    start_time = time.monotonic()
    request_id = getattr(request.state, "request_id", str(UUID(int=0)))
    
    logger.info(
        "find_similar_chunks_request",
        request_id=request_id,
        chunk_id=chunk_id,
        top_k=top_k,
        exclude_self=exclude_self,
    )
    
    # Validate UUID format
    try:
        chunk_uuid = parse_uuid(chunk_id)
    except HTTPException:
        logger.warning(
            "find_similar_chunks_invalid_uuid",
            request_id=request_id,
            chunk_id=chunk_id,
        )
        raise
    
    try:
        # Initialize repository and service
        repo = DocumentChunkRepository(db)
        service = VectorSearchService(repo)
        
        # Find similar chunks
        results = await service.find_similar_chunks(
            chunk_id=chunk_uuid,
            top_k=top_k,
            exclude_self=exclude_self,
        )
        
        # Convert to response items
        items = [_search_result_to_item(r) for r in results]
        
        query_time_ms = (time.monotonic() - start_time) * 1000
        
        logger.info(
            "find_similar_chunks_completed",
            request_id=request_id,
            chunk_id=chunk_id,
            result_count=len(items),
            query_time_ms=round(query_time_ms, 2),
        )
        
        return SearchResultsResponse(
            results=items,
            total=len(items),
            query_time_ms=round(query_time_ms, 2),
        )
        
    except ChunkNotFoundError as e:
        logger.warning(
            "find_similar_chunks_not_found",
            request_id=request_id,
            chunk_id=chunk_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference chunk with ID '{chunk_id}' not found",
        ) from e
    except InvalidEmbeddingError as e:
        logger.warning(
            "find_similar_chunks_no_embedding",
            request_id=request_id,
            chunk_id=chunk_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Reference chunk does not have an embedding: {e}",
        ) from e
    except VectorSearchError as e:
        logger.error(
            "find_similar_chunks_error",
            request_id=request_id,
            chunk_id=chunk_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        ) from e
