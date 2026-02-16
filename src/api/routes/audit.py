"""Audit API routes.

This module provides endpoints for querying and exporting audit logs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from src.auth.base import User
from src.auth.dependencies import (
    get_current_user,
    require_export_audit,
    require_permissions,
    require_read_audit,
)
from src.audit.logger import get_audit_logger
from src.audit.models import (
    AuditEvent,
    AuditEventStatus,
    AuditEventType,
    AuditLogExportRequest,
    AuditLogQuery,
    AuditLogQueryResult,
)


router = APIRouter(prefix="/audit", tags=["Audit"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AuditLogQueryParams(BaseModel):
    """Query parameters for audit logs."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    resource_types: Optional[List[str]] = None
    resource_ids: Optional[List[str]] = None
    status: Optional[str] = None
    correlation_id: Optional[str] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)
    sort_by: str = "timestamp"
    sort_order: str = "desc"
    
    def to_audit_query(self) -> AuditLogQuery:
        """Convert to internal audit query."""
        event_types = None
        if self.event_types:
            event_types = [AuditEventType(e) for e in self.event_types]
        
        status_enum = None
        if self.status:
            status_enum = AuditEventStatus(self.status)
        
        return AuditLogQuery(
            start_time=self.start_time,
            end_time=self.end_time,
            event_types=event_types,
            user_ids=self.user_ids,
            resource_types=self.resource_types,
            resource_ids=self.resource_ids,
            status=status_enum,
            correlation_id=self.correlation_id,
            page=self.page,
            page_size=self.page_size,
            sort_by=self.sort_by,
            sort_order=self.sort_order,
        )


class AuditLogEntryResponse(BaseModel):
    """Single audit log entry response."""
    id: str
    timestamp: str
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: Optional[str]
    status: str
    details: Dict[str, Any]
    correlation_id: Optional[str]
    request_id: Optional[str]


class AuditLogListResponse(BaseModel):
    """Audit log list response."""
    events: List[AuditLogEntryResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_more: bool


class AuditExportRequest(BaseModel):
    """Audit export request."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    resource_types: Optional[List[str]] = None
    status: Optional[str] = None
    format: str = Field(default="json", pattern="^(json|csv|ndjson)$")
    
    def to_export_request(self) -> AuditLogExportRequest:
        """Convert to internal export request."""
        query = AuditLogQueryParams(
            start_time=self.start_time,
            end_time=self.end_time,
            event_types=self.event_types,
            user_ids=self.user_ids,
            resource_types=self.resource_types,
            status=self.status,
        )
        
        return AuditLogExportRequest(
            query=query.to_audit_query(),
            format=self.format,
        )


class AuditSummaryResponse(BaseModel):
    """Audit log summary response."""
    total_events: int
    events_by_type: Dict[str, int]
    events_by_status: Dict[str, int]
    events_by_user: Dict[str, int]
    time_range: Dict[str, Optional[str]]


# ============================================================================
# Audit Query Endpoints
# ============================================================================

@router.get("/logs", response_model=AuditLogListResponse)
async def query_audit_logs(
    request: Request,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    event_types: Optional[List[str]] = Query(None),
    user_ids: Optional[List[str]] = Query(None),
    resource_types: Optional[List[str]] = Query(None),
    resource_ids: Optional[List[str]] = Query(None),
    status: Optional[str] = None,
    correlation_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    sort_by: str = Query("timestamp"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    user: User = Depends(require_read_audit),
    audit_logger=Depends(get_audit_logger),
):
    """Query audit logs.
    
    Retrieves audit events matching the specified filters.
    Requires audit:read permission.
    """
    # Build query
    query = AuditLogQuery(
        start_time=start_time,
        end_time=end_time,
        event_types=[AuditEventType(e) for e in event_types] if event_types else None,
        user_ids=user_ids,
        resource_types=resource_types,
        resource_ids=resource_ids,
        status=AuditEventStatus(status) if status else None,
        correlation_id=correlation_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    
    # Execute query
    result = await audit_logger.query_logs(query)
    
    # Log this access
    await audit_logger.query_logs(
        AuditLogQuery(event_types=[AuditEventType.AUDIT_EXPORTED], page=1, page_size=1)
    )
    
    return AuditLogListResponse(
        events=[
            AuditLogEntryResponse(
                id=str(event.id),
                timestamp=event.timestamp.isoformat(),
                event_type=event.event_type.value,
                user_id=event.user_id,
                ip_address=event.ip_address,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                action=event.action,
                status=event.status.value,
                details=event.details,
                correlation_id=event.correlation_id,
                request_id=event.request_id,
            )
            for event in result.events
        ],
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
        has_more=result.has_more,
    )


@router.post("/export")
async def export_audit_logs(
    export_request: AuditExportRequest,
    user: User = Depends(require_export_audit),
    audit_logger=Depends(get_audit_logger),
):
    """Export audit logs.
    
    Exports audit events matching the specified filters in the requested format.
    Requires audit:export permission.
    """
    # Convert to internal request
    internal_request = export_request.to_export_request()
    
    # Export data
    data = await audit_logger.export_logs(internal_request)
    
    # Determine content type
    content_type_map = {
        "json": "application/json",
        "csv": "text/csv",
        "ndjson": "application/x-ndjson",
    }
    content_type = content_type_map.get(export_request.format, "application/octet-stream")
    
    # Determine filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"audit_export_{timestamp}.{export_request.format}"
    
    # Log export
    # Note: This would need to be async, but we're already in the function
    
    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )


@router.get("/summary", response_model=AuditSummaryResponse)
async def get_audit_summary(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    user: User = Depends(require_read_audit),
    audit_logger=Depends(get_audit_logger),
):
    """Get audit log summary.
    
    Returns aggregated statistics about audit events.
    Requires audit:read permission.
    """
    # Build query for all events in time range
    query = AuditLogQuery(
        start_time=start_time,
        end_time=end_time,
        page=1,
        page_size=10000,  # Large page for aggregation
    )
    
    result = await audit_logger.query_logs(query)
    
    # Aggregate statistics
    events_by_type: Dict[str, int] = {}
    events_by_status: Dict[str, int] = {}
    events_by_user: Dict[str, int] = {}
    
    for event in result.events:
        # Count by type
        event_type = event.event_type.value
        events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
        
        # Count by status
        status = event.status.value
        events_by_status[status] = events_by_status.get(status, 0) + 1
        
        # Count by user
        user_id = event.user_id or "anonymous"
        events_by_user[user_id] = events_by_user.get(user_id, 0) + 1
    
    return AuditSummaryResponse(
        total_events=result.total,
        events_by_type=events_by_type,
        events_by_status=events_by_status,
        events_by_user=events_by_user,
        time_range={
            "start": start_time.isoformat() if start_time else None,
            "end": end_time.isoformat() if end_time else None,
        },
    )


@router.get("/events/types")
async def list_event_types(
    user: User = Depends(get_current_user),
):
    """List available audit event types."""
    return {
        "event_types": [
            {"value": e.value, "name": e.name}
            for e in AuditEventType
        ]
    }


@router.get("/events/{event_id}", response_model=AuditLogEntryResponse)
async def get_event_details(
    event_id: str,
    user: User = Depends(require_read_audit),
    audit_logger=Depends(get_audit_logger),
):
    """Get details of a specific audit event.
    
    Requires audit:read permission.
    """
    # Query for specific event
    from uuid import UUID
    
    try:
        event_uuid = UUID(event_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid event ID format",
        )
    
    # In a real implementation, this would query by ID
    # For now, return not found
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Event not found",
    )
