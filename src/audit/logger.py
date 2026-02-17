"""Audit logging implementation.

This module provides the AuditLogger class for logging and querying
audit events in the system.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from src.audit.models import (
    AuditEvent,
    AuditEventStatus,
    AuditEventType,
    AuditLogExportRequest,
    AuditLogQuery,
    AuditLogQueryResult,
)

logger = logging.getLogger(__name__)


class AuditLogStore(Protocol):
    """Protocol for audit log storage backends.
    
    Implementations can use PostgreSQL, OpenSearch, or other storage.
    """

    async def save_event(self, event: AuditEvent) -> None:
        """Save an audit event.
        
        Args:
            event: Event to save
        """
        ...

    async def query_events(
        self,
        query: AuditLogQuery,
    ) -> AuditLogQueryResult:
        """Query audit events.
        
        Args:
            query: Query parameters
            
        Returns:
            Query result
        """
        ...

    async def export_events(
        self,
        request: AuditLogExportRequest,
    ) -> bytes:
        """Export events matching query.
        
        Args:
            request: Export request
            
        Returns:
            Exported data as bytes
        """
        ...


class InMemoryAuditStore:
    """In-memory audit log store for development/testing.
    
    This store keeps events in memory and does not persist them.
    """

    def __init__(self, max_events: int = 10000):
        """Initialize in-memory store.
        
        Args:
            max_events: Maximum number of events to keep
        """
        self._events: list[AuditEvent] = []
        self._max_events = max_events
        self._lock = asyncio.Lock()

    async def save_event(self, event: AuditEvent) -> None:
        """Save an audit event.
        
        Args:
            event: Event to save
        """
        async with self._lock:
            self._events.append(event)

            # Trim if exceeding max
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

    async def query_events(
        self,
        query: AuditLogQuery,
    ) -> AuditLogQueryResult:
        """Query audit events.
        
        Args:
            query: Query parameters
            
        Returns:
            Query result
        """
        async with self._lock:
            # Filter events
            filtered = self._filter_events(query)

            # Sort events
            reverse = query.sort_order == "desc"
            if query.sort_by == "timestamp":
                filtered.sort(key=lambda e: e.timestamp, reverse=reverse)
            elif query.sort_by == "event_type":
                filtered.sort(key=lambda e: e.event_type.value, reverse=reverse)
            elif query.sort_by == "user_id":
                filtered.sort(key=lambda e: e.user_id or "", reverse=reverse)

            # Paginate
            total = len(filtered)
            start = (query.page - 1) * query.page_size
            end = start + query.page_size
            page_events = filtered[start:end]

            return AuditLogQueryResult(
                events=page_events,
                total=total,
                page=query.page,
                page_size=query.page_size,
                has_more=end < total,
            )

    def _filter_events(self, query: AuditLogQuery) -> list[AuditEvent]:
        """Filter events based on query parameters.
        
        Args:
            query: Query parameters
            
        Returns:
            Filtered events
        """
        events = self._events

        if query.start_time:
            events = [e for e in events if e.timestamp >= query.start_time]

        if query.end_time:
            events = [e for e in events if e.timestamp <= query.end_time]

        if query.event_types:
            types = set(query.event_types)
            events = [e for e in events if e.event_type in types]

        if query.user_ids:
            user_ids = set(query.user_ids)
            events = [e for e in events if e.user_id in user_ids]

        if query.resource_types:
            resource_types = set(query.resource_types)
            events = [e for e in events if e.resource_type in resource_types]

        if query.resource_ids:
            resource_ids = set(query.resource_ids)
            events = [e for e in events if e.resource_id in resource_ids]

        if query.status:
            events = [e for e in events if e.status == query.status]

        if query.correlation_id:
            events = [e for e in events if e.correlation_id == query.correlation_id]

        return events

    async def export_events(
        self,
        request: AuditLogExportRequest,
    ) -> bytes:
        """Export events matching query.
        
        Args:
            request: Export request
            
        Returns:
            Exported data as bytes
        """
        # Get all matching events (no pagination for export)
        query = AuditLogQuery(
            start_time=request.query.start_time,
            end_time=request.query.end_time,
            event_types=request.query.event_types,
            user_ids=request.query.user_ids,
            resource_types=request.query.resource_types,
            resource_ids=request.query.resource_ids,
            status=request.query.status,
            correlation_id=request.query.correlation_id,
            page=1,
            page_size=100000,  # Large page size for export
        )

        result = await self.query_events(query)

        if request.format == "json":
            data = {
                "exported_at": datetime.utcnow().isoformat(),
                "total": result.total,
                "events": [e.to_dict() for e in result.events],
            }
            if request.include_metadata:
                data["metadata"] = {
                    "format": "json",
                    "version": "1.0",
                }
            return json.dumps(data, indent=2, default=str).encode("utf-8")

        elif request.format == "ndjson":
            lines = [json.dumps(e.to_dict(), default=str) for e in result.events]
            return "\n".join(lines).encode("utf-8")

        elif request.format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow([
                "id", "timestamp", "event_type", "user_id", "ip_address",
                "resource_type", "resource_id", "action", "status",
            ])

            # Write events
            for e in result.events:
                writer.writerow([
                    str(e.id),
                    e.timestamp.isoformat(),
                    e.event_type.value,
                    e.user_id,
                    e.ip_address,
                    e.resource_type,
                    e.resource_id,
                    e.action,
                    e.status.value,
                ])

            return output.getvalue().encode("utf-8")

        else:
            raise ValueError(f"Unsupported export format: {request.format}")


class AuditLogger:
    """Audit logger for the system.
    
    This class provides methods for logging audit events and querying
    the audit log. It supports multiple storage backends.
    
    Example:
        audit_logger = AuditLogger()
        
        # Log a job creation event
        await audit_logger.log_job_created(
            job_id=job_id,
            user_id=user_id,
            details={"file_name": "document.pdf"},
        )
        
        # Query audit logs
        result = await audit_logger.query_logs(
            AuditLogQuery(start_time=yesterday, event_types=[AuditEventType.JOB_CREATED])
        )
    """

    def __init__(self, store: AuditLogStore | None = None):
        """Initialize audit logger.
        
        Args:
            store: Storage backend (defaults to in-memory)
        """
        self.store = store or InMemoryAuditStore()

    async def log_event(self, event: AuditEvent) -> None:
        """Log an audit event.
        
        Args:
            event: Event to log
        """
        try:
            await self.store.save_event(event)
        except Exception as e:
            # Audit logging should never fail the main operation
            logger.error(f"Failed to log audit event: {e}", exc_info=True)

    # Convenience methods for common events

    async def log_job_created(
        self,
        job_id: UUID,
        user_id: str | None = None,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log job creation."""
        event = AuditEvent.from_job_event(
            event_type=AuditEventType.JOB_CREATED,
            job_id=job_id,
            user_id=user_id,
            details=details,
            **kwargs,
        )
        await self.log_event(event)

    async def log_job_started(
        self,
        job_id: UUID,
        user_id: str | None = None,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log job start."""
        event = AuditEvent.from_job_event(
            event_type=AuditEventType.JOB_STARTED,
            job_id=job_id,
            user_id=user_id,
            details=details,
            **kwargs,
        )
        await self.log_event(event)

    async def log_job_completed(
        self,
        job_id: UUID,
        user_id: str | None = None,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log job completion."""
        event = AuditEvent.from_job_event(
            event_type=AuditEventType.JOB_COMPLETED,
            job_id=job_id,
            user_id=user_id,
            details=details,
            **kwargs,
        )
        await self.log_event(event)

    async def log_job_failed(
        self,
        job_id: UUID,
        user_id: str | None = None,
        error: str | None = None,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log job failure."""
        merged_details = details or {}
        if error:
            merged_details["error"] = error

        event = AuditEvent.from_job_event(
            event_type=AuditEventType.JOB_FAILED,
            job_id=job_id,
            user_id=user_id,
            status=AuditEventStatus.FAILURE,
            details=merged_details,
            **kwargs,
        )
        await self.log_event(event)

    async def log_job_cancelled(
        self,
        job_id: UUID,
        user_id: str | None = None,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log job cancellation."""
        event = AuditEvent.from_job_event(
            event_type=AuditEventType.JOB_CANCELLED,
            job_id=job_id,
            user_id=user_id,
            details=details,
            **kwargs,
        )
        await self.log_event(event)

    async def log_job_retry(
        self,
        job_id: UUID,
        user_id: str | None = None,
        retry_count: int = 0,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log job retry."""
        merged_details = details or {}
        merged_details["retry_count"] = retry_count

        event = AuditEvent.from_job_event(
            event_type=AuditEventType.JOB_RETRY,
            job_id=job_id,
            user_id=user_id,
            details=merged_details,
            **kwargs,
        )
        await self.log_event(event)

    async def log_auth_login(
        self,
        user_id: str,
        ip_address: str | None = None,
        success: bool = True,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log authentication login."""
        event = AuditEvent.from_auth_event(
            event_type=AuditEventType.AUTH_LOGIN,
            user_id=user_id,
            ip_address=ip_address,
            status=AuditEventStatus.SUCCESS if success else AuditEventStatus.FAILURE,
            details=details,
            **kwargs,
        )
        await self.log_event(event)

    async def log_auth_failed(
        self,
        user_id: str | None = None,
        ip_address: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log authentication failure."""
        merged_details = details or {}
        if reason:
            merged_details["failure_reason"] = reason

        event = AuditEvent.from_auth_event(
            event_type=AuditEventType.AUTH_FAILED,
            user_id=user_id,
            ip_address=ip_address,
            status=AuditEventStatus.FAILURE,
            details=merged_details,
            **kwargs,
        )
        await self.log_event(event)

    async def log_source_accessed(
        self,
        source_id: str,
        user_id: str | None = None,
        action: str = "read",
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log source access."""
        event = AuditEvent.from_source_event(
            event_type=AuditEventType.SOURCE_ACCESSED,
            source_id=source_id,
            user_id=user_id,
            details={"action": action, **(details or {})},
            **kwargs,
        )
        await self.log_event(event)

    async def log_api_key_created(
        self,
        key_id: str,
        user_id: str,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log API key creation."""
        event = AuditEvent(
            event_type=AuditEventType.API_KEY_CREATED,
            user_id=user_id,
            resource_type="api_key",
            resource_id=key_id,
            details=details,
            **kwargs,
        )
        await self.log_event(event)

    async def log_api_key_revoked(
        self,
        key_id: str,
        user_id: str,
        details: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log API key revocation."""
        event = AuditEvent(
            event_type=AuditEventType.API_KEY_REVOKED,
            user_id=user_id,
            resource_type="api_key",
            resource_id=key_id,
            details=details,
            **kwargs,
        )
        await self.log_event(event)

    async def query_logs(self, query: AuditLogQuery) -> AuditLogQueryResult:
        """Query audit logs.
        
        Args:
            query: Query parameters
            
        Returns:
            Query result
        """
        return await self.store.query_events(query)

    async def export_logs(self, request: AuditLogExportRequest) -> bytes:
        """Export audit logs.
        
        Args:
            request: Export request
            
        Returns:
            Exported data as bytes
        """
        return await self.store.export_events(request)


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance.
    
    Returns:
        Audit logger singleton
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def set_audit_logger(logger: AuditLogger) -> None:
    """Set the global audit logger.
    
    Args:
        logger: Audit logger instance
    """
    global _audit_logger
    _audit_logger = logger
