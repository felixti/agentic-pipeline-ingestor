"""Unit tests for audit log API routes.

Tests the audit log endpoints including querying logs, exporting logs,
getting summaries, and listing event types.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException, Request, status

from src.api.routes.audit import (
    AuditExportRequest,
    AuditLogQueryParams,
    get_audit_summary,
    get_event_details,
    list_event_types,
    query_audit_logs,
)
from src.audit.models import (
    AuditEvent,
    AuditEventStatus,
    AuditEventType,
    AuditLogQueryResult,
)
from src.auth.base import User


@pytest.mark.unit
class TestAuditLogQueryParams:
    """Tests for AuditLogQueryParams model."""

    def test_to_audit_query_basic(self):
        """Test converting basic params to audit query."""
        params = AuditLogQueryParams(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            page=1,
            page_size=50,
        )
        
        query = params.to_audit_query()
        
        assert query.start_time == datetime(2024, 1, 1)
        assert query.end_time == datetime(2024, 1, 31)
        assert query.page == 1
        assert query.page_size == 50

    def test_to_audit_query_with_event_types(self):
        """Test converting params with event types."""
        params = AuditLogQueryParams(
            event_types=["JOB_CREATED", "JOB_COMPLETED"],
        )
        
        query = params.to_audit_query()
        
        assert len(query.event_types) == 2
        assert AuditEventType.JOB_CREATED in query.event_types
        assert AuditEventType.JOB_COMPLETED in query.event_types

    def test_to_audit_query_with_status(self):
        """Test converting params with status filter."""
        params = AuditLogQueryParams(
            status="SUCCESS",
        )
        
        query = params.to_audit_query()
        
        assert query.status == AuditEventStatus.SUCCESS


@pytest.mark.unit
class TestQueryAuditLogs:
    """Tests for GET /audit/logs - Query audit logs endpoint."""

    @pytest.mark.asyncio
    async def test_query_audit_logs_success(self):
        """Test successful audit log query."""
        mock_request = MagicMock()
        
        # Create mock audit events
        mock_event = AuditEvent(
            id=uuid4(),
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.JOB_CREATED,
            user_id="user-123",
            status=AuditEventStatus.SUCCESS,
            details={"job_id": str(uuid4())},
        )
        
        mock_result = AuditLogQueryResult(
            events=[mock_event],
            total=1,
            page=1,
            page_size=50,
            has_more=False,
        )
        
        mock_audit_logger = AsyncMock()
        mock_audit_logger.query_logs.return_value = mock_result
        
        user = User(
            id=uuid4(),
            email="admin@example.com",
            permissions=["audit:read"],
        )
        
        response = await query_audit_logs(
            request=mock_request,
            user=user,
            audit_logger=mock_audit_logger,
        )
        
        assert response.total == 1
        assert len(response.events) == 1
        assert response.events[0].event_type == "JOB_CREATED"
        assert response.page == 1
        assert response.page_size == 50

    @pytest.mark.asyncio
    async def test_query_audit_logs_with_filters(self):
        """Test audit log query with filters."""
        mock_request = MagicMock()
        
        mock_result = AuditLogQueryResult(
            events=[],
            total=0,
            page=1,
            page_size=50,
            has_more=False,
        )
        
        mock_audit_logger = AsyncMock()
        mock_audit_logger.query_logs.return_value = mock_result
        
        user = User(
            id=uuid4(),
            permissions=["audit:read"],
        )
        
        response = await query_audit_logs(
            request=mock_request,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            event_types=["JOB_CREATED", "JOB_FAILED"],
            user_ids=["user-123"],
            resource_types=["job"],
            status="SUCCESS",
            page=1,
            page_size=25,
            user=user,
            audit_logger=mock_audit_logger,
        )
        
        assert response.total == 0
        # Verify the query was called with correct parameters
        call_args = mock_audit_logger.query_logs.call_args[0][0]
        assert call_args.start_time == datetime(2024, 1, 1)
        assert call_args.end_time == datetime(2024, 1, 31)
        assert call_args.page_size == 25

    @pytest.mark.asyncio
    async def test_query_audit_logs_pagination(self):
        """Test audit log query pagination."""
        mock_request = MagicMock()
        
        mock_result = AuditLogQueryResult(
            events=[],
            total=100,
            page=2,
            page_size=20,
            has_more=True,
        )
        
        mock_audit_logger = AsyncMock()
        mock_audit_logger.query_logs.return_value = mock_result
        
        user = User(
            id=uuid4(),
            permissions=["audit:read"],
        )
        
        response = await query_audit_logs(
            request=mock_request,
            page=2,
            page_size=20,
            user=user,
            audit_logger=mock_audit_logger,
        )
        
        assert response.total == 100
        assert response.page == 2
        assert response.page_size == 20
        assert response.has_more is True
        assert response.total_pages == 5


@pytest.mark.unit
class TestExportAuditLogs:
    """Tests for POST /audit/export - Export audit logs endpoint."""

    @pytest.mark.asyncio
    async def test_export_audit_logs_json(self):
        """Test exporting audit logs as JSON."""
        export_request = AuditExportRequest(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            format="json",
        )
        
        mock_export_data = b'{"exported_at": "2024-01-31T00:00:00", "total": 10}'
        mock_audit_logger = AsyncMock()
        mock_audit_logger.export_logs.return_value = mock_export_data
        
        user = User(
            id=uuid4(),
            permissions=["audit:export"],
        )
        
        from src.api.routes.audit import export_audit_logs
        response = await export_audit_logs(export_request, user, mock_audit_logger)
        
        assert response.body == mock_export_data
        assert response.media_type == "application/json"
        assert "audit_export_" in response.headers["content-disposition"]
        assert ".json" in response.headers["content-disposition"]

    @pytest.mark.asyncio
    async def test_export_audit_logs_csv(self):
        """Test exporting audit logs as CSV."""
        export_request = AuditExportRequest(
            start_time=datetime(2024, 1, 1),
            format="csv",
        )
        
        mock_export_data = b"id,timestamp,event_type\n1,2024-01-01,JOB_CREATED"
        mock_audit_logger = AsyncMock()
        mock_audit_logger.export_logs.return_value = mock_export_data
        
        user = User(
            id=uuid4(),
            permissions=["audit:export"],
        )
        
        from src.api.routes.audit import export_audit_logs
        response = await export_audit_logs(export_request, user, mock_audit_logger)
        
        assert response.body == mock_export_data
        assert response.media_type == "text/csv"
        assert ".csv" in response.headers["content-disposition"]

    @pytest.mark.asyncio
    async def test_export_audit_logs_ndjson(self):
        """Test exporting audit logs as NDJSON."""
        export_request = AuditExportRequest(
            format="ndjson",
        )
        
        mock_export_data = b'{"event_type": "JOB_CREATED"}\n{"event_type": "JOB_COMPLETED"}'
        mock_audit_logger = AsyncMock()
        mock_audit_logger.export_logs.return_value = mock_export_data
        
        user = User(
            id=uuid4(),
            permissions=["audit:export"],
        )
        
        from src.api.routes.audit import export_audit_logs
        response = await export_audit_logs(export_request, user, mock_audit_logger)
        
        assert response.body == mock_export_data
        assert response.media_type == "application/x-ndjson"


@pytest.mark.unit
class TestGetAuditSummary:
    """Tests for GET /audit/summary - Get audit summary endpoint."""

    @pytest.mark.asyncio
    async def test_get_audit_summary_success(self):
        """Test successful audit summary retrieval."""
        # Create mock events
        events = [
            AuditEvent(
                id=uuid4(),
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.JOB_CREATED,
                user_id="user-1",
                status=AuditEventStatus.SUCCESS,
            ),
            AuditEvent(
                id=uuid4(),
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.JOB_CREATED,
                user_id="user-1",
                status=AuditEventStatus.SUCCESS,
            ),
            AuditEvent(
                id=uuid4(),
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.JOB_FAILED,
                user_id="user-2",
                status=AuditEventStatus.FAILURE,
            ),
        ]
        
        mock_result = AuditLogQueryResult(
            events=events,
            total=3,
            page=1,
            page_size=10000,
            has_more=False,
        )
        
        mock_audit_logger = AsyncMock()
        mock_audit_logger.query_logs.return_value = mock_result
        
        user = User(
            id=uuid4(),
            permissions=["audit:read"],
        )
        
        response = await get_audit_summary(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            user=user,
            audit_logger=mock_audit_logger,
        )
        
        assert response.total_events == 3
        assert response.events_by_type["JOB_CREATED"] == 2
        assert response.events_by_type["JOB_FAILED"] == 1
        assert response.events_by_status["SUCCESS"] == 2
        assert response.events_by_status["FAILURE"] == 1
        assert response.events_by_user["user-1"] == 2
        assert response.events_by_user["user-2"] == 1

    @pytest.mark.asyncio
    async def test_get_audit_summary_empty(self):
        """Test audit summary with no events."""
        mock_result = AuditLogQueryResult(
            events=[],
            total=0,
            page=1,
            page_size=10000,
            has_more=False,
        )
        
        mock_audit_logger = AsyncMock()
        mock_audit_logger.query_logs.return_value = mock_result
        
        user = User(
            id=uuid4(),
            permissions=["audit:read"],
        )
        
        response = await get_audit_summary(
            user=user,
            audit_logger=mock_audit_logger,
        )
        
        assert response.total_events == 0
        assert response.events_by_type == {}
        assert response.events_by_status == {}
        assert response.events_by_user == {}

    @pytest.mark.asyncio
    async def test_get_audit_summary_time_range(self):
        """Test audit summary with specific time range."""
        mock_result = AuditLogQueryResult(
            events=[],
            total=0,
            page=1,
            page_size=10000,
            has_more=False,
        )
        
        mock_audit_logger = AsyncMock()
        mock_audit_logger.query_logs.return_value = mock_result
        
        user = User(
            id=uuid4(),
            permissions=["audit:read"],
        )
        
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 31, 23, 59, 59)
        
        response = await get_audit_summary(
            start_time=start_time,
            end_time=end_time,
            user=user,
            audit_logger=mock_audit_logger,
        )
        
        assert response.time_range["start"] == start_time.isoformat()
        assert response.time_range["end"] == end_time.isoformat()


@pytest.mark.unit
class TestListEventTypes:
    """Tests for GET /audit/events/types - List event types endpoint."""

    @pytest.mark.asyncio
    async def test_list_event_types(self):
        """Test listing all event types."""
        user = User(
            id=uuid4(),
            email="test@example.com",
        )
        
        response = await list_event_types(user)
        
        assert "event_types" in response
        assert len(response["event_types"]) > 0
        
        # Check structure of event type
        event_type = response["event_types"][0]
        assert "value" in event_type
        assert "name" in event_type
        
        # Verify some expected event types exist
        event_values = [e["value"] for e in response["event_types"]]
        assert "JOB_CREATED" in event_values
        assert "JOB_COMPLETED" in event_values
        assert "AUTH_LOGIN" in event_values


@pytest.mark.unit
class TestGetEventDetails:
    """Tests for GET /audit/events/{event_id} - Get event details endpoint."""

    @pytest.mark.asyncio
    async def test_get_event_details_invalid_uuid(self):
        """Test getting event details with invalid UUID."""
        user = User(
            id=uuid4(),
            permissions=["audit:read"],
        )
        mock_audit_logger = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await get_event_details("invalid-uuid", user, mock_audit_logger)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid event ID format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_event_details_not_found(self):
        """Test getting details for non-existent event."""
        user = User(
            id=uuid4(),
            permissions=["audit:read"],
        )
        mock_audit_logger = AsyncMock()
        
        event_id = str(uuid4())
        
        with pytest.raises(HTTPException) as exc_info:
            await get_event_details(event_id, user, mock_audit_logger)
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Event not found" in exc_info.value.detail


@pytest.mark.unit
class TestAuditExportRequest:
    """Tests for AuditExportRequest model."""

    def test_to_export_request(self):
        """Test converting to internal export request."""
        request = AuditExportRequest(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            event_types=["JOB_CREATED"],
            user_ids=["user-123"],
            resource_types=["job"],
            status="SUCCESS",
            format="json",
        )
        
        internal = request.to_export_request()
        
        assert internal.query.start_time == datetime(2024, 1, 1)
        assert internal.query.end_time == datetime(2024, 1, 31)
        assert internal.format == "json"

    def test_export_request_default_format(self):
        """Test export request with default format."""
        request = AuditExportRequest()
        
        internal = request.to_export_request()
        
        assert internal.format == "json"
