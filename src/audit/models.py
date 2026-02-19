"""Audit logging models.

This module defines the data models for audit events and queries.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    """Audit event types for comprehensive logging.
    
    These event types cover all operations in the system as specified
    in Section 10.1 of the spec.
    """
    # Job lifecycle events
    JOB_CREATED = "job.created"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"
    JOB_RETRY = "job.retry"
    JOB_DELETED = "job.deleted"

    # Stage events
    STAGE_STARTED = "stage.started"
    STAGE_COMPLETED = "stage.completed"
    STAGE_FAILED = "stage.failed"

    # Source access events
    SOURCE_ACCESSED = "source.accessed"
    SOURCE_CREATED = "source.created"
    SOURCE_UPDATED = "source.updated"
    SOURCE_DELETED = "source.deleted"
    SOURCE_TESTED = "source.tested"

    # Destination events
    DESTINATION_CREATED = "destination.created"
    DESTINATION_UPDATED = "destination.updated"
    DESTINATION_DELETED = "destination.deleted"
    DESTINATION_TESTED = "destination.tested"
    DESTINATION_WRITE = "destination.write"

    # Pipeline config events
    CONFIG_CREATED = "config.created"
    CONFIG_UPDATED = "config.updated"
    CONFIG_DELETED = "config.deleted"

    # Authentication events
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    AUTH_TOKEN_REFRESH = "auth.token_refresh"

    # API key events
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"
    API_KEY_USED = "api_key.used"

    # User management events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"

    # Audit and compliance events
    AUDIT_EXPORTED = "audit.exported"
    LINEAGE_ACCESSED = "lineage.accessed"

    # DLQ events
    DLQ_ENTRY_CREATED = "dlq.entry_created"
    DLQ_ENTRY_RETRIED = "dlq.entry_retried"
    DLQ_ENTRY_ARCHIVED = "dlq.entry_archived"

    # System events
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_METRICS_ACCESSED = "system.metrics_accessed"


class AuditEventStatus(str, Enum):
    """Status of an audit event."""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    WARNING = "warning"


class AuditEvent(BaseModel):
    """Audit event model.
    
    Represents a single auditable action in the system.
    All fields are optional except event_type and timestamp
    to allow for flexible logging.
    
    Attributes:
        id: Unique event identifier
        timestamp: When the event occurred
        event_type: Type of event
        user_id: User who performed the action
        ip_address: Client IP address
        resource_type: Type of resource affected
        resource_id: Identifier of affected resource
        action: Action performed
        status: Success/failure status
        details: Additional event details
        correlation_id: Request correlation ID
        request_id: Unique request identifier
        user_agent: Client user agent
        metadata: Additional metadata
    """

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: AuditEventType
    user_id: str | None = None
    ip_address: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    action: str | None = None
    status: AuditEventStatus = AuditEventStatus.SUCCESS
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    request_id: str | None = None
    user_agent: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        # Handle both enum and string values (use_enum_values config may convert to string)
        event_type_value = self.event_type.value if hasattr(self.event_type, "value") else self.event_type
        status_value = self.status.value if hasattr(self.status, "value") else self.status
        
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "event_type": event_type_value,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "status": status_value,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "user_agent": self.user_agent,
            "metadata": self.metadata,
        }

    @classmethod
    def from_job_event(
        cls,
        event_type: AuditEventType,
        job_id: UUID,
        user_id: str | None = None,
        status: AuditEventStatus = AuditEventStatus.SUCCESS,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "AuditEvent":
        """Create an audit event for a job-related action.
        
        Args:
            event_type: Type of job event
            job_id: Job identifier
            user_id: User who performed the action
            status: Event status
            details: Additional details
            **kwargs: Additional event fields
            
        Returns:
            Audit event
        """
        return cls(
            event_type=event_type,
            user_id=user_id,
            resource_type="job",
            resource_id=str(job_id),
            action=event_type.value.split(".")[-1],
            status=status,
            details=details or {},
            **kwargs,
        )

    @classmethod
    def from_auth_event(
        cls,
        event_type: AuditEventType,
        user_id: str | None = None,
        ip_address: str | None = None,
        status: AuditEventStatus = AuditEventStatus.SUCCESS,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "AuditEvent":
        """Create an audit event for an authentication action.
        
        Args:
            event_type: Type of auth event
            user_id: User identifier
            ip_address: Client IP address
            status: Event status
            details: Additional details
            **kwargs: Additional event fields
            
        Returns:
            Audit event
        """
        return cls(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            resource_type="auth",
            action=event_type.value.split(".")[-1],
            status=status,
            details=details or {},
            **kwargs,
        )

    @classmethod
    def from_source_event(
        cls,
        event_type: AuditEventType,
        source_id: str,
        user_id: str | None = None,
        status: AuditEventStatus = AuditEventStatus.SUCCESS,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "AuditEvent":
        """Create an audit event for a source-related action.
        
        Args:
            event_type: Type of source event
            source_id: Source identifier
            user_id: User identifier
            status: Event status
            details: Additional details
            **kwargs: Additional event fields
            
        Returns:
            Audit event
        """
        return cls(
            event_type=event_type,
            user_id=user_id,
            resource_type="source",
            resource_id=source_id,
            action=event_type.value.split(".")[-1],
            status=status,
            details=details or {},
            **kwargs,
        )


class AuditLogQuery(BaseModel):
    """Query parameters for audit log retrieval.
    
    Attributes:
        start_time: Filter events after this time
        end_time: Filter events before this time
        event_types: Filter by event types
        user_ids: Filter by user IDs
        resource_types: Filter by resource types
        resource_ids: Filter by resource IDs
        status: Filter by event status
        correlation_id: Filter by correlation ID
        page: Page number (1-indexed)
        page_size: Number of results per page
        sort_by: Field to sort by
        sort_order: Sort direction (asc/desc)
    """

    start_time: datetime | None = None
    end_time: datetime | None = None
    event_types: list[AuditEventType] | None = None
    user_ids: list[str] | None = None
    resource_types: list[str] | None = None
    resource_ids: list[str] | None = None
    status: AuditEventStatus | None = None
    correlation_id: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)
    sort_by: str = "timestamp"
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")

    def to_db_filters(self) -> dict[str, Any]:
        """Convert query to database filter dictionary."""
        filters: dict[str, Any] = {}

        if self.start_time:
            filters["timestamp__gte"] = self.start_time

        if self.end_time:
            filters["timestamp__lte"] = self.end_time

        if self.event_types:
            filters["event_type__in"] = [e.value for e in self.event_types]

        if self.user_ids:
            filters["user_id__in"] = self.user_ids

        if self.resource_types:
            filters["resource_type__in"] = self.resource_types

        if self.resource_ids:
            filters["resource_id__in"] = self.resource_ids

        if self.status:
            filters["status"] = self.status.value

        if self.correlation_id:
            filters["correlation_id"] = self.correlation_id

        return filters


class AuditLogQueryResult(BaseModel):
    """Result of an audit log query.
    
    Attributes:
        events: List of audit events
        total: Total number of matching events
        page: Current page number
        page_size: Page size
        has_more: Whether there are more results
    """

    events: list[AuditEvent]
    total: int
    page: int
    page_size: int
    has_more: bool

    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        return (self.total + self.page_size - 1) // self.page_size


class AuditLogExportRequest(BaseModel):
    """Request to export audit logs.
    
    Attributes:
        query: Query parameters
        format: Export format (json, csv, ndjson)
        include_metadata: Whether to include metadata
    """

    query: AuditLogQuery
    format: str = Field(default="json", pattern="^(json|csv|ndjson)$")
    include_metadata: bool = True
