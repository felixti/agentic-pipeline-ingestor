"""Unit tests for audit models."""

from datetime import datetime
from uuid import uuid4

import pytest

from src.audit.models import (
    AuditEvent,
    AuditEventStatus,
    AuditEventType,
    AuditLogExportRequest,
    AuditLogQuery,
    AuditLogQueryResult,
)


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_job_event_types(self):
        """Test job event types exist."""
        assert AuditEventType.JOB_CREATED == "job.created"
        assert AuditEventType.JOB_STARTED == "job.started"
        assert AuditEventType.JOB_COMPLETED == "job.completed"
        assert AuditEventType.JOB_FAILED == "job.failed"

    def test_auth_event_types(self):
        """Test auth event types exist."""
        assert AuditEventType.AUTH_LOGIN == "auth.login"
        assert AuditEventType.AUTH_LOGOUT == "auth.logout"
        assert AuditEventType.AUTH_FAILED == "auth.failed"

    def test_source_event_types(self):
        """Test source event types exist."""
        assert AuditEventType.SOURCE_CREATED == "source.created"
        assert AuditEventType.SOURCE_ACCESSED == "source.accessed"


class TestAuditEventStatus:
    """Tests for AuditEventStatus enum."""

    def test_status_values(self):
        """Test status values."""
        assert AuditEventStatus.SUCCESS == "success"
        assert AuditEventStatus.FAILURE == "failure"
        assert AuditEventStatus.PENDING == "pending"
        assert AuditEventStatus.WARNING == "warning"


class TestAuditEvent:
    """Tests for AuditEvent model."""

    def test_create_basic_event(self):
        """Test creating a basic audit event."""
        event = AuditEvent(
            event_type=AuditEventType.JOB_CREATED,
        )

        assert event.event_type == AuditEventType.JOB_CREATED
        assert event.status == AuditEventStatus.SUCCESS
        assert event.timestamp is not None

    def test_create_event_with_details(self):
        """Test creating event with all fields."""
        event = AuditEvent(
            event_type=AuditEventType.JOB_COMPLETED,
            user_id="user-123",
            ip_address="192.168.1.1",
            resource_type="job",
            resource_id="job-456",
            action="completed",
            status=AuditEventStatus.SUCCESS,
            details={"duration_ms": 1500},
            correlation_id="corr-789",
            request_id="req-abc",
            user_agent="test-agent",
            metadata={"source": "api"},
        )

        assert event.user_id == "user-123"
        assert event.ip_address == "192.168.1.1"
        assert event.details["duration_ms"] == 1500

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = AuditEvent(
            event_type=AuditEventType.JOB_CREATED,
            user_id="user-123",
            status=AuditEventStatus.SUCCESS,
        )

        d = event.to_dict()

        assert d["event_type"] == "job.created"
        assert d["user_id"] == "user-123"
        assert d["status"] == "success"
        assert "timestamp" in d
        assert "id" in d

    def test_from_job_event(self):
        """Test creating event from job event."""
        job_id = uuid4()
        event = AuditEvent.from_job_event(
            event_type=AuditEventType.JOB_STARTED,
            job_id=job_id,
            user_id="user-123",
            details={"pipeline_id": "pipe-1"},
        )

        assert event.event_type == AuditEventType.JOB_STARTED
        assert event.resource_type == "job"
        assert event.resource_id == str(job_id)
        assert event.action == "started"
        assert event.user_id == "user-123"

    def test_from_auth_event(self):
        """Test creating event from auth event."""
        event = AuditEvent.from_auth_event(
            event_type=AuditEventType.AUTH_LOGIN,
            user_id="user-123",
            ip_address="192.168.1.1",
            status=AuditEventStatus.SUCCESS,
            details={"method": "password"},
        )

        assert event.event_type == AuditEventType.AUTH_LOGIN
        assert event.resource_type == "auth"
        assert event.action == "login"
        assert event.ip_address == "192.168.1.1"

    def test_from_source_event(self):
        """Test creating event from source event."""
        event = AuditEvent.from_source_event(
            event_type=AuditEventType.SOURCE_CREATED,
            source_id="s3-source-1",
            user_id="admin",
            details={"bucket": "my-bucket"},
        )

        assert event.event_type == AuditEventType.SOURCE_CREATED
        assert event.resource_type == "source"
        assert event.resource_id == "s3-source-1"
        assert event.action == "created"


class TestAuditLogQuery:
    """Tests for AuditLogQuery model."""

    def test_default_query(self):
        """Test default query values."""
        query = AuditLogQuery()

        assert query.page == 1
        assert query.page_size == 50
        assert query.sort_by == "timestamp"
        assert query.sort_order == "desc"

    def test_query_with_filters(self):
        """Test query with filters."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        
        query = AuditLogQuery(
            start_time=start,
            end_time=end,
            event_types=[AuditEventType.JOB_CREATED, AuditEventType.JOB_COMPLETED],
            user_ids=["user-1", "user-2"],
            resource_types=["job"],
            status=AuditEventStatus.SUCCESS,
            page=2,
            page_size=100,
        )

        assert query.start_time == start
        assert query.end_time == end
        assert len(query.event_types) == 2
        assert query.page == 2

    def test_to_db_filters(self):
        """Test converting to database filters."""
        query = AuditLogQuery(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 12, 31),
            event_types=[AuditEventType.JOB_CREATED],
            user_ids=["user-1"],
            resource_types=["job"],
            resource_ids=["job-123"],
            status=AuditEventStatus.SUCCESS,
            correlation_id="corr-abc",
        )

        filters = query.to_db_filters()

        assert "timestamp__gte" in filters
        assert "timestamp__lte" in filters
        assert filters["event_type__in"] == ["job.created"]
        assert filters["user_id__in"] == ["user-1"]
        assert filters["status"] == "success"
        assert filters["correlation_id"] == "corr-abc"

    def test_empty_db_filters(self):
        """Test empty query returns empty filters."""
        query = AuditLogQuery()
        filters = query.to_db_filters()

        assert filters == {}

    def test_sort_order_validation(self):
        """Test sort order validation."""
        # Valid sort orders
        query1 = AuditLogQuery(sort_order="asc")
        assert query1.sort_order == "asc"
        
        query2 = AuditLogQuery(sort_order="desc")
        assert query2.sort_order == "desc"


class TestAuditLogQueryResult:
    """Tests for AuditLogQueryResult model."""

    def test_create_result(self):
        """Test creating query result."""
        events = [
            AuditEvent(event_type=AuditEventType.JOB_CREATED),
            AuditEvent(event_type=AuditEventType.JOB_COMPLETED),
        ]
        
        result = AuditLogQueryResult(
            events=events,
            total=100,
            page=1,
            page_size=50,
            has_more=True,
        )

        assert len(result.events) == 2
        assert result.total == 100
        assert result.has_more is True

    def test_total_pages_calculation(self):
        """Test total pages calculation."""
        result = AuditLogQueryResult(
            events=[],
            total=100,
            page=1,
            page_size=50,
            has_more=True,
        )

        assert result.total_pages == 2  # 100 items / 50 per page

    def test_total_pages_with_remainder(self):
        """Test total pages with remainder."""
        result = AuditLogQueryResult(
            events=[],
            total=105,
            page=1,
            page_size=50,
            has_more=True,
        )

        assert result.total_pages == 3  # 105 items needs 3 pages


class TestAuditLogExportRequest:
    """Tests for AuditLogExportRequest model."""

    def test_default_export_request(self):
        """Test default export request."""
        query = AuditLogQuery()
        request = AuditLogExportRequest(query=query)

        assert request.format == "json"
        assert request.include_metadata is True

    def test_csv_export_request(self):
        """Test CSV export request."""
        query = AuditLogQuery()
        request = AuditLogExportRequest(
            query=query,
            format="csv",
            include_metadata=False,
        )

        assert request.format == "csv"
        assert request.include_metadata is False

    def test_ndjson_export_request(self):
        """Test NDJSON export request."""
        query = AuditLogQuery()
        request = AuditLogExportRequest(query=query, format="ndjson")

        assert request.format == "ndjson"
