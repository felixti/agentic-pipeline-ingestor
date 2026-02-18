"""Document chunk API routes.

This module provides endpoints for retrieving document chunks
with optional embedding data for semantic search use cases.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db, parse_uuid
from src.api.middleware.rate_limiter import (
    rate_limit_get_chunk,
    rate_limit_list_chunks,
)
from src.api.models import (
    ApiResponse,
    DocumentChunkListItem,
    DocumentChunkListResponse,
    DocumentChunkResponse,
)
from src.db.repositories.document_chunk_repository import DocumentChunkRepository
from src.db.repositories.job import JobRepository
from src.observability.logging import get_logger

router = APIRouter(prefix="/jobs", tags=["Document Chunks"])
logger = get_logger(__name__)


# ============================================================================
# Response Models
# ============================================================================

class ChunkNotFoundResponse(BaseModel):
    """Error response when chunk is not found."""
    detail: str = Field(default="Chunk not found")


class JobNotFoundResponse(BaseModel):
    """Error response when job is not found."""
    detail: str = Field(default="Job not found")


class ChunkJobMismatchResponse(BaseModel):
    """Error response when chunk doesn't belong to job."""
    detail: str = Field(default="Chunk does not belong to the specified job")


# ============================================================================
# Helper Functions
# ============================================================================

def _chunk_to_list_item(chunk: Any) -> DocumentChunkListItem:
    """Convert a DocumentChunkModel to a DocumentChunkListItem.
    
    Args:
        chunk: DocumentChunkModel instance
        
    Returns:
        DocumentChunkListItem for list responses
    """
    return DocumentChunkListItem(
        id=chunk.id,
        job_id=chunk.job_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        content_hash=chunk.content_hash,
        metadata=chunk.metadata or {},
        created_at=chunk.created_at,
    )


def _chunk_to_response(
    chunk: Any,
    include_embedding: bool = False,
) -> DocumentChunkResponse:
    """Convert a DocumentChunkModel to a DocumentChunkResponse.
    
    Args:
        chunk: DocumentChunkModel instance
        include_embedding: Whether to include the embedding vector
        
    Returns:
        DocumentChunkResponse with optional embedding
    """
    embedding = None
    if include_embedding and chunk.embedding is not None:
        # Convert vector to list of floats
        embedding = list(chunk.embedding) if hasattr(chunk.embedding, "__iter__") else chunk.embedding
    
    return DocumentChunkResponse(
        id=chunk.id,
        job_id=chunk.job_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        content_hash=chunk.content_hash,
        embedding=embedding,
        metadata=chunk.metadata or {},
        created_at=chunk.created_at,
    )


# ============================================================================
# List Chunks Endpoint
# ============================================================================

@router.get(
    "/{job_id}/chunks",
    response_model=DocumentChunkListResponse,
    responses={
        404: {"model": JobNotFoundResponse, "description": "Job not found"},
        400: {"description": "Invalid UUID format"},
        429: {"description": "Rate limit exceeded"},
    },
    summary="List chunks for a job",
    description="Retrieve paginated list of document chunks for a specific job. Embeddings are excluded by default for performance.",
)
@rate_limit_list_chunks
async def list_chunks(
    job_id: str,
    request: Request,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of chunks to return"),
    offset: int = Query(0, ge=0, description="Number of chunks to skip"),
    include_embedding: bool = Query(False, description="Include vector embedding in response (performance impact)"),
    db: AsyncSession = Depends(get_db),
) -> DocumentChunkListResponse:
    """List chunks for a job with pagination.
    
    Retrieves all chunks belonging to the specified job, ordered by chunk_index.
    By default, embeddings are not included to reduce response size and improve performance.
    
    Args:
        job_id: Job UUID
        limit: Maximum number of chunks to return (1-1000)
        offset: Number of chunks to skip for pagination
        include_embedding: Whether to include embedding vectors
        
    Returns:
        Paginated list of chunks
        
    Raises:
        HTTPException: 400 if job_id is invalid UUID
        HTTPException: 404 if job not found
    """
    request_id = getattr(request.state, "request_id", str(UUID(int=0)))
    
    # Validate job_id UUID format
    try:
        job_uuid = parse_uuid(job_id)
    except HTTPException:
        logger.warning("invalid_job_id_format", job_id=job_id)
        raise
    
    # Verify job exists
    job_repo = JobRepository(db)
    job = await job_repo.get_by_id(job_uuid)
    if job is None:
        logger.info("job_not_found", job_id=str(job_uuid))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found",
        )
    
    # Get chunks from repository
    chunk_repo = DocumentChunkRepository(db)
    chunks = await chunk_repo.get_by_job_id(
        job_id=job_uuid,
        limit=limit,
        offset=offset,
    )
    
    # Get total count
    total = await chunk_repo.count_by_job_id(job_uuid)
    
    # Convert to response models
    # Note: List response never includes embeddings even if requested
    # The include_embedding flag is reserved for future use or detail endpoint
    items = [_chunk_to_list_item(chunk) for chunk in chunks]
    
    logger.info(
        "chunks_listed",
        job_id=str(job_uuid),
        count=len(items),
        total=total,
        offset=offset,
        limit=limit,
    )
    
    return DocumentChunkListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


# ============================================================================
# Get Chunk Endpoint
# ============================================================================

@router.get(
    "/{job_id}/chunks/{chunk_id}",
    response_model=DocumentChunkResponse,
    responses={
        404: {"model": ChunkNotFoundResponse, "description": "Chunk not found"},
        400: {"model": ChunkJobMismatchResponse, "description": "Chunk doesn't belong to job"},
        429: {"description": "Rate limit exceeded"},
    },
    summary="Get a specific chunk",
    description="Retrieve a single document chunk by ID with optional embedding data.",
)
@rate_limit_get_chunk
async def get_chunk(
    job_id: str,
    chunk_id: str,
    request: Request,
    include_embedding: bool = Query(False, description="Include vector embedding in response"),
    db: AsyncSession = Depends(get_db),
) -> DocumentChunkResponse:
    """Get a specific chunk by ID.
    
    Retrieves a single document chunk. Verifies that the chunk belongs to
the specified job for security.
    
    Args:
        job_id: Job UUID
        chunk_id: Chunk UUID
        include_embedding: Whether to include the embedding vector
        
    Returns:
        Chunk details
        
    Raises:
        HTTPException: 400 if UUIDs are invalid
        HTTPException: 404 if chunk not found
        HTTPException: 400 if chunk doesn't belong to specified job
    """
    request_id = getattr(request.state, "request_id", str(UUID(int=0)))
    
    # Validate UUID formats
    try:
        job_uuid = parse_uuid(job_id)
        chunk_uuid = parse_uuid(chunk_id)
    except HTTPException:
        logger.warning("invalid_uuid_format", job_id=job_id, chunk_id=chunk_id)
        raise
    
    # Get chunk from repository
    chunk_repo = DocumentChunkRepository(db)
    chunk = await chunk_repo.get_by_id(chunk_uuid)
    
    if chunk is None:
        logger.info("chunk_not_found", chunk_id=str(chunk_uuid))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with ID '{chunk_id}' not found",
        )
    
    # Security check: verify chunk belongs to the specified job
    if chunk.job_id != job_uuid:
        logger.warning(
            "chunk_job_mismatch",
            chunk_id=str(chunk_uuid),
            chunk_job_id=str(chunk.job_id),
            requested_job_id=str(job_uuid),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chunk does not belong to the specified job",
        )
    
    logger.info(
        "chunk_retrieved",
        chunk_id=str(chunk_uuid),
        job_id=str(job_uuid),
        include_embedding=include_embedding,
    )
    
    return _chunk_to_response(chunk, include_embedding=include_embedding)
