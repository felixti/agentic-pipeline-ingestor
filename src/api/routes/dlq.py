"""DLQ (Dead Letter Queue) API routes.

This module provides REST API endpoints for managing
the Dead Letter Queue, including listing, retrying, and
analyzing failed jobs.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from src.api.models import ApiResponse, PipelineConfig
from src.core.dlq import DLQEntry, DLQEntryStatus, DLQFailureCategory, DLQFilters, get_dlq

router = APIRouter(prefix="/dlq", tags=["dlq"])


# ============================================================================
# Request/Response Models
# ============================================================================

class DLQEntryResponse(BaseModel):
    """Response model for DLQ entry."""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    job_id: str
    failure_category: str
    status: str
    error_code: str
    error_message: str
    retry_count: int
    created_at: str
    file_name: str
    file_type: str


class DLQListResponse(BaseModel):
    """Response model for DLQ list."""
    model_config = ConfigDict(populate_by_name=True)

    entries: list[DLQEntryResponse]
    total_count: int
    page: int
    page_size: int


class DLQRetryRequest(BaseModel):
    """Request to retry a DLQ entry."""
    model_config = ConfigDict(populate_by_name=True)

    updated_config: PipelineConfig | None = Field(
        default=None,
        description="Updated pipeline configuration for retry",
    )
    reviewed_by: str | None = Field(
        default=None,
        description="User initiating the retry",
    )
    resolution_notes: str | None = Field(
        default=None,
        description="Notes about the retry decision",
    )


class DLQRetryResponse(BaseModel):
    """Response for DLQ retry."""
    model_config = ConfigDict(populate_by_name=True)

    success: bool
    entry_id: str
    new_job_id: str | None = None
    message: str
    error: str | None = None


class DLQStatsResponse(BaseModel):
    """Response for DLQ statistics."""
    model_config = ConfigDict(populate_by_name=True)

    total_entries: int
    by_status: dict[str, int]
    by_failure_category: dict[str, int]
    age_distribution: dict[str, int]


class DLQReviewRequest(BaseModel):
    """Request to review/mark a DLQ entry."""
    model_config = ConfigDict(populate_by_name=True)

    reviewed_by: str = Field(..., description="User reviewing the entry")
    notes: str | None = Field(default=None, description="Review notes")
    action: str = Field(
        default="review",
        pattern="^(review|resolve|discard)$",
        description="Action to take: review, resolve, or discard",
    )
    discard_reason: str | None = Field(
        default=None,
        description="Reason for discarding (required if action is discard)",
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _entry_to_response(entry: DLQEntry) -> DLQEntryResponse:
    """Convert DLQEntry to response model.
    
    Args:
        entry: DLQ entry
        
    Returns:
        DLQEntryResponse
    """
    return DLQEntryResponse(
        id=str(entry.id),
        job_id=str(entry.job_id),
        failure_category=entry.failure_category.value,
        status=entry.status.value,
        error_code=entry.error.code if entry.error else "unknown",
        error_message=entry.error.message if entry.error else "",
        retry_count=entry.retry_count,
        created_at=entry.created_at.isoformat(),
        file_name=entry.job.file_name,
        file_type=entry.job.mime_type or "unknown",
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("", response_model=ApiResponse)
async def list_dlq_entries(
    request: Request,
    status: str | None = Query(None, description="Filter by status"),
    failure_category: str | None = Query(None, description="Filter by failure category"),
    source_type: str | None = Query(None, description="Filter by source type"),
    search: str | None = Query(None, description="Free text search"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
) -> ApiResponse:
    """List DLQ entries with filtering and pagination.
    
    Returns a paginated list of dead letter queue entries
    with optional filtering by status, category, and search.
    """
    dlq = get_dlq()

    # Build filters
    filters = DLQFilters()
    if status:
        try:
            filters.status = DLQEntryStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    if failure_category:
        try:
            filters.failure_category = DLQFailureCategory(failure_category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category: {failure_category}")

    if source_type:
        filters.source_type = source_type

    if search:
        filters.search = search

    # Calculate offset
    offset = (page - 1) * page_size

    # Get entries
    entries = await dlq.list_entries(
        filters=filters,
        limit=page_size,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    total_count = await dlq.count_entries(filters)

    # Convert to response models
    response_entries = [_entry_to_response(e) for e in entries]

    data = DLQListResponse(
        entries=response_entries,
        total_count=total_count,
        page=page,
        page_size=page_size,
    )

    return ApiResponse.create(
        data=data.model_dump(),
        request_id=request.state.request_id if hasattr(request.state, "request_id") else UUID(int=0),
        total_count=total_count,
    )


@router.get("/{entry_id}", response_model=ApiResponse)
async def get_dlq_entry(
    request: Request,
    entry_id: UUID,
) -> ApiResponse:
    """Get a single DLQ entry by ID.
    
    Returns detailed information about a specific DLQ entry.
    """
    dlq = get_dlq()
    entry = await dlq.get_entry(entry_id)

    if not entry:
        raise HTTPException(status_code=404, detail=f"DLQ entry not found: {entry_id}")

    return ApiResponse.create(
        data=entry.to_dict(),
        request_id=request.state.request_id if hasattr(request.state, "request_id") else UUID(int=0),
    )


@router.post("/{entry_id}/retry", response_model=ApiResponse)
async def retry_dlq_entry(
    request: Request,
    entry_id: UUID,
    retry_request: DLQRetryRequest,
) -> ApiResponse:
    """Retry a job from the DLQ.
    
    Creates a new job with optional configuration changes
    to retry processing of a failed document.
    """
    dlq = get_dlq()

    # Check entry exists
    entry = await dlq.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"DLQ entry not found: {entry_id}")

    # Check entry can be retried
    if entry.status in (DLQEntryStatus.RESOLVED, DLQEntryStatus.DISCARDED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry entry with status: {entry.status.value}",
        )

    # Execute retry
    result = await dlq.retry_from_dlq(
        entry_id=entry_id,
        new_config=retry_request.updated_config,
        reviewed_by=retry_request.reviewed_by,
        resolution_notes=retry_request.resolution_notes,
    )

    response = DLQRetryResponse(
        success=result.success,
        entry_id=str(result.entry_id),
        new_job_id=str(result.new_job_id) if result.new_job_id else None,
        message=result.message,
        error=result.error,
    )

    return ApiResponse.create(
        data=response.model_dump(),
        request_id=request.state.request_id if hasattr(request.state, "request_id") else UUID(int=0),
    )


@router.post("/{entry_id}/review", response_model=ApiResponse)
async def review_dlq_entry(
    request: Request,
    entry_id: UUID,
    review_request: DLQReviewRequest,
) -> ApiResponse:
    """Review or mark a DLQ entry.
    
    Actions:
    - review: Mark as under review
    - resolve: Mark as resolved (manual success)
    - discard: Mark as permanently discarded
    """
    dlq = get_dlq()

    # Check entry exists
    entry = await dlq.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"DLQ entry not found: {entry_id}")

    # Execute action
    if review_request.action == "review":
        updated = await dlq.mark_reviewed(
            entry_id=entry_id,
            reviewed_by=review_request.reviewed_by,
            notes=review_request.notes,
        )
        message = "Entry marked as under review"

    elif review_request.action == "resolve":
        updated = await dlq.mark_resolved(
            entry_id=entry_id,
            resolution_notes=review_request.notes,
        )
        message = "Entry marked as resolved"

    elif review_request.action == "discard":
        if not review_request.discard_reason:
            raise HTTPException(
                status_code=400,
                detail="discard_reason is required when action is 'discard'",
            )
        updated = await dlq.discard_entry(
            entry_id=entry_id,
            reason=review_request.discard_reason,
            discarded_by=review_request.reviewed_by,
        )
        message = "Entry marked as discarded"

    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {review_request.action}")

    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update entry")

    return ApiResponse.create(
        data={
            "entry_id": str(entry_id),
            "action": review_request.action,
            "message": message,
            "status": updated.status.value,
        },
        request_id=request.state.request_id if hasattr(request.state, "request_id") else UUID(int=0),
    )


@router.delete("/{entry_id}", response_model=ApiResponse)
async def delete_dlq_entry(
    request: Request,
    entry_id: UUID,
) -> ApiResponse:
    """Remove an entry from the DLQ.
    
    Permanently deletes a DLQ entry. Use with caution.
    """
    dlq = get_dlq()

    # Check entry exists
    entry = await dlq.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"DLQ entry not found: {entry_id}")

    # Remove entry
    removed = await dlq.dequeue(entry_id)

    if not removed:
        raise HTTPException(status_code=500, detail="Failed to remove entry")

    return ApiResponse.create(
        data={
            "entry_id": str(entry_id),
            "message": "Entry removed from DLQ",
            "deleted": True,
        },
        request_id=request.state.request_id if hasattr(request.state, "request_id") else UUID(int=0),
    )


@router.get("/stats/summary", response_model=ApiResponse)
async def get_dlq_stats(
    request: Request,
) -> ApiResponse:
    """Get DLQ statistics summary.
    
    Returns statistics about DLQ entries including counts by
    status, failure category, and age distribution.
    """
    dlq = get_dlq()
    stats = await dlq.get_statistics()

    # Convert enum keys to strings for JSON serialization
    by_status = {k.value if hasattr(k, "value") else k: v for k, v in stats["by_status"].items()}
    by_category = {k.value if hasattr(k, "value") else k: v for k, v in stats["by_failure_category"].items()}

    response = DLQStatsResponse(
        total_entries=stats["total_entries"],
        by_status=by_status,
        by_failure_category=by_category,
        age_distribution=stats["age_distribution"],
    )

    return ApiResponse.create(
        data=response.model_dump(),
        request_id=request.state.request_id if hasattr(request.state, "request_id") else UUID(int=0),
    )


@router.post("/archive-old", response_model=ApiResponse)
async def archive_old_entries(
    request: Request,
) -> ApiResponse:
    """Archive old DLQ entries.
    
    Moves entries older than the configured max age to archived status.
    """
    dlq = get_dlq()
    archived_count = await dlq.archive_old_entries()

    return ApiResponse.create(
        data={
            "archived_count": archived_count,
            "message": f"{archived_count} entries archived",
        },
        request_id=request.state.request_id if hasattr(request.state, "request_id") else UUID(int=0),
    )
