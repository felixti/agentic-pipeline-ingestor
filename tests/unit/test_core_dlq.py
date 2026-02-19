"""Unit tests for the Dead Letter Queue (DLQ) module.

Tests cover:
- DLQ entry creation
- DLQ manager
- Retry from DLQ
- DLQ entry expiration
- Error classification
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.api.models import (
    Job,
    JobError,
    JobStatus,
    ParserConfig,
    PipelineConfig,
    ProcessingMode,
    QualityConfig,
    RetryRecord,
    SourceType,
)
from src.core.dlq import (
    DeadLetterQueue,
    DLQEntry,
    DLQEntryStatus,
    DLQFailureCategory,
    DLQFilters,
    DLQRetryResult,
    get_dlq,
    set_dlq,
)


def utcnow():
    """Return UTC datetime (naive for compatibility with source code)."""
    return datetime.utcnow()


@pytest.mark.unit
class TestDLQEntryStatus:
    """Tests for DLQEntryStatus enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert DLQEntryStatus.PENDING.value == "pending"
        assert DLQEntryStatus.UNDER_REVIEW.value == "under_review"
        assert DLQEntryStatus.RETRY_SCHEDULED.value == "retry_scheduled"
        assert DLQEntryStatus.RETRYING.value == "retrying"
        assert DLQEntryStatus.RESOLVED.value == "resolved"
        assert DLQEntryStatus.DISCARDED.value == "discarded"
        assert DLQEntryStatus.ARCHIVED.value == "archived"


@pytest.mark.unit
class TestDLQFailureCategory:
    """Tests for DLQFailureCategory enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert DLQFailureCategory.PARSING_ERROR.value == "parsing_error"
        assert DLQFailureCategory.QUALITY_THRESHOLD.value == "quality_threshold"
        assert DLQFailureCategory.TIMEOUT.value == "timeout"
        assert DLQFailureCategory.RESOURCE_EXHAUSTED.value == "resource_exhausted"
        assert DLQFailureCategory.NETWORK_ERROR.value == "network_error"
        assert DLQFailureCategory.AUTHENTICATION_ERROR.value == "authentication_error"
        assert DLQFailureCategory.VALIDATION_ERROR.value == "validation_error"
        assert DLQFailureCategory.UNKNOWN_ERROR.value == "unknown_error"


@pytest.mark.unit
class TestDLQFilters:
    """Tests for DLQFilters dataclass."""

    def test_filters_creation(self):
        """Test creating DLQFilters with all values."""
        from_date = utcnow() - timedelta(days=7)
        to_date = utcnow()

        filters = DLQFilters(
            status=DLQEntryStatus.PENDING,
            failure_category=DLQFailureCategory.PARSING_ERROR,
            source_type="upload",
            date_from=from_date,
            date_to=to_date,
            file_name_pattern="*.pdf",
            search="error",
        )

        assert filters.status == DLQEntryStatus.PENDING
        assert filters.failure_category == DLQFailureCategory.PARSING_ERROR
        assert filters.source_type == "upload"
        assert filters.date_from == from_date
        assert filters.date_to == to_date
        assert filters.file_name_pattern == "*.pdf"
        assert filters.search == "error"

    def test_filters_defaults(self):
        """Test DLQFilters default values."""
        filters = DLQFilters()

        assert filters.status is None
        assert filters.failure_category is None
        assert filters.source_type is None
        assert filters.date_from is None
        assert filters.date_to is None
        assert filters.file_name_pattern is None
        assert filters.search is None


@pytest.mark.unit
class TestDLQEntry:
    """Tests for DLQEntry dataclass."""

    @pytest.fixture
    def sample_job(self):
        """Create a sample job."""
        return Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/file.pdf",
            file_name="test.pdf",
            status=JobStatus.FAILED,
            created_at=utcnow(),
        )

    @pytest.fixture
    def sample_error(self):
        """Create a sample error."""
        return JobError(
            code="PARSING_ERROR",
            message="Failed to parse document",
            failed_stage="parse",
        )

    @pytest.fixture
    def sample_retry_history(self):
        """Create sample retry history."""
        return [
            RetryRecord(
                attempt=1,
                timestamp=utcnow(),
                strategy="same_parser",
                error_code="PARSING_ERROR",
            ),
        ]

    def test_entry_creation(self, sample_job, sample_error, sample_retry_history):
        """Test creating a DLQEntry instance."""
        entry_id = uuid4()
        entry = DLQEntry(
            id=entry_id,
            job_id=sample_job.id,
            job=sample_job,
            error=sample_error,
            failure_category=DLQFailureCategory.PARSING_ERROR,
            retry_history=sample_retry_history,
            status=DLQEntryStatus.PENDING,
            retry_count=1,
            max_retries_reached=True,
        )

        assert entry.id == entry_id
        assert entry.job_id == sample_job.id
        assert entry.job == sample_job
        assert entry.error == sample_error
        assert entry.failure_category == DLQFailureCategory.PARSING_ERROR
        assert entry.retry_history == sample_retry_history
        assert entry.status == DLQEntryStatus.PENDING
        assert entry.retry_count == 1
        assert entry.max_retries_reached is True

    def test_entry_defaults(self, sample_job, sample_error, sample_retry_history):
        """Test DLQEntry default values."""
        entry = DLQEntry(
            id=uuid4(),
            job_id=sample_job.id,
            job=sample_job,
            error=sample_error,
            failure_category=DLQFailureCategory.PARSING_ERROR,
            retry_history=sample_retry_history,
        )

        assert entry.status == DLQEntryStatus.PENDING
        assert entry.created_at is not None
        assert entry.updated_at is not None
        assert entry.reviewed_at is None
        assert entry.reviewed_by is None
        assert entry.retry_count == 0
        assert entry.max_retries_reached is False
        assert entry.resolution_notes is None
        assert entry.archived_at is None
        assert entry.metadata == {}

    def test_entry_to_dict(self, sample_job, sample_error, sample_retry_history):
        """Test converting DLQEntry to dictionary."""
        entry = DLQEntry(
            id=uuid4(),
            job_id=sample_job.id,
            job=sample_job,
            error=sample_error,
            failure_category=DLQFailureCategory.PARSING_ERROR,
            retry_history=sample_retry_history,
        )

        result = entry.to_dict()

        assert isinstance(result, dict)
        assert "id" in result
        assert "job_id" in result
        assert "error" in result
        assert "failure_category" in result
        assert "status" in result
        assert "created_at" in result
        assert result["failure_category"] == "parsing_error"
        assert result["status"] == "pending"


@pytest.mark.unit
class TestDLQRetryResult:
    """Tests for DLQRetryResult dataclass."""

    def test_retry_result_success(self):
        """Test creating a successful DLQRetryResult."""
        entry_id = uuid4()
        new_job_id = uuid4()

        result = DLQRetryResult(
            success=True,
            entry_id=entry_id,
            new_job_id=new_job_id,
            message="Retry scheduled successfully",
        )

        assert result.success is True
        assert result.entry_id == entry_id
        assert result.new_job_id == new_job_id
        assert result.message == "Retry scheduled successfully"
        assert result.error is None

    def test_retry_result_failure(self):
        """Test creating a failed DLQRetryResult."""
        entry_id = uuid4()

        result = DLQRetryResult(
            success=False,
            entry_id=entry_id,
            error="Entry not found",
        )

        assert result.success is False
        assert result.entry_id == entry_id
        assert result.new_job_id is None
        assert result.error == "Entry not found"


@pytest.mark.unit
class TestDeadLetterQueue:
    """Tests for DeadLetterQueue class."""

    @pytest.fixture
    def dlq(self):
        """Create a DeadLetterQueue instance with patched logger."""
        dlq = DeadLetterQueue(max_age_days=30)
        # Patch the instance logger to avoid structlog issues
        dlq.logger = MagicMock()
        return dlq

    @pytest.fixture
    def sample_job(self):
        """Create a sample job."""
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            quality=QualityConfig(max_retries=3),
        )
        return Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/file.pdf",
            file_name="test.pdf",
            status=JobStatus.FAILED,
            created_at=utcnow(),
            pipeline_config=pipeline_config,
            retry_count=3,
            current_stage="parse",
        )

    @pytest.fixture
    def sample_error(self):
        """Create a sample exception."""
        return Exception("Parsing failed")

    def test_dlq_initialization(self, dlq):
        """Test DLQ initialization."""
        assert dlq.max_age_days == 30
        assert dlq._entries == {}
        assert dlq._job_id_to_entry == {}

    def test_categorize_failure_parsing_error(self, dlq):
        """Test categorizing parsing errors."""
        error = JobError(code="PARSING_FAILED", message="Failed to parse")
        history = []

        category = dlq._categorize_failure(error, history)

        assert category == DLQFailureCategory.PARSING_ERROR

    def test_categorize_failure_quality_error(self, dlq):
        """Test categorizing quality threshold errors."""
        error = JobError(code="QUALITY_THRESHOLD", message="Quality too low")
        history = []

        category = dlq._categorize_failure(error, history)

        assert category == DLQFailureCategory.QUALITY_THRESHOLD

    def test_categorize_failure_timeout(self, dlq):
        """Test categorizing timeout errors."""
        error = JobError(code="TIMEOUT_ERROR", message="Processing timed out")
        history = []

        category = dlq._categorize_failure(error, history)

        assert category == DLQFailureCategory.TIMEOUT

    def test_categorize_failure_resource_exhausted(self, dlq):
        """Test categorizing resource exhausted errors."""
        error = JobError(code="MEMORY_ERROR", message="Out of memory")
        history = []

        category = dlq._categorize_failure(error, history)

        assert category == DLQFailureCategory.RESOURCE_EXHAUSTED

    def test_categorize_failure_network_error(self, dlq):
        """Test categorizing network errors."""
        error = JobError(code="NETWORK_ERROR", message="Connection failed")
        history = []

        category = dlq._categorize_failure(error, history)

        assert category == DLQFailureCategory.NETWORK_ERROR

    def test_categorize_failure_auth_error(self, dlq):
        """Test categorizing authentication errors."""
        error = JobError(code="AUTHENTICATION_FAILED", message="Invalid credentials")
        history = []

        category = dlq._categorize_failure(error, history)

        assert category == DLQFailureCategory.AUTHENTICATION_ERROR

    def test_categorize_failure_validation_error(self, dlq):
        """Test categorizing validation errors."""
        error = JobError(code="VALIDATION_ERROR", message="Invalid input")
        history = []

        category = dlq._categorize_failure(error, history)

        assert category == DLQFailureCategory.VALIDATION_ERROR

    def test_categorize_failure_from_message(self, dlq):
        """Test categorizing errors from message when code doesn't match."""
        error = JobError(code="UNKNOWN", message="Quality score below threshold")
        history = []

        category = dlq._categorize_failure(error, history)

        assert category == DLQFailureCategory.QUALITY_THRESHOLD

    def test_categorize_failure_unknown(self, dlq):
        """Test categorizing unknown errors."""
        error = JobError(code="SOMETHING_UNEXPECTED", message="Unknown issue")
        history = []

        category = dlq._categorize_failure(error, history)

        assert category == DLQFailureCategory.UNKNOWN_ERROR

    @pytest.mark.asyncio
    async def test_enqueue_with_exception(self, dlq, sample_job, sample_error):
        """Test enqueueing a job with a standard exception."""
        history = []

        entry = await dlq.enqueue(sample_job, sample_error, history)

        assert isinstance(entry, DLQEntry)
        assert entry.job_id == sample_job.id
        assert entry.failure_category == DLQFailureCategory.UNKNOWN_ERROR
        assert entry.retry_count == 3
        assert entry.max_retries_reached is True
        assert entry.id in dlq._entries
        assert dlq._job_id_to_entry[sample_job.id] == entry.id

    @pytest.mark.asyncio
    async def test_enqueue_with_job_error(self, dlq, sample_job):
        """Test enqueueing a job with a JobError."""
        job_error = JobError(
            code="PARSING_ERROR",
            message="Failed to parse",
            failed_stage="parse",
        )
        history = []

        entry = await dlq.enqueue(sample_job, job_error, history)

        assert entry.error.code == "PARSING_ERROR"
        assert entry.failure_category == DLQFailureCategory.PARSING_ERROR

    @pytest.mark.asyncio
    async def test_enqueue_with_metadata(self, dlq, sample_job, sample_error):
        """Test enqueueing with metadata."""
        history = []
        metadata = {"source": "test", "retry_attempts": 3}

        entry = await dlq.enqueue(sample_job, sample_error, history, metadata)

        assert entry.metadata == metadata

    @pytest.mark.asyncio
    async def test_dequeue_existing_entry(self, dlq, sample_job, sample_error):
        """Test dequeueing an existing entry."""
        entry = await dlq.enqueue(sample_job, sample_error, [])

        result = await dlq.dequeue(entry.id)

        assert result == entry
        assert entry.id not in dlq._entries
        assert sample_job.id not in dlq._job_id_to_entry

    @pytest.mark.asyncio
    async def test_dequeue_nonexistent_entry(self, dlq):
        """Test dequeueing a non-existent entry."""
        result = await dlq.dequeue(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_entry_by_id(self, dlq, sample_job, sample_error):
        """Test getting an entry by ID."""
        entry = await dlq.enqueue(sample_job, sample_error, [])

        result = await dlq.get_entry(entry.id)

        assert result == entry

    @pytest.mark.asyncio
    async def test_get_entry_nonexistent(self, dlq):
        """Test getting a non-existent entry."""
        result = await dlq.get_entry(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_entry_by_job_id(self, dlq, sample_job, sample_error):
        """Test getting an entry by job ID."""
        entry = await dlq.enqueue(sample_job, sample_error, [])

        result = await dlq.get_entry_by_job(sample_job.id)

        assert result == entry

    @pytest.mark.asyncio
    async def test_get_entry_by_job_nonexistent(self, dlq):
        """Test getting an entry by non-existent job ID."""
        result = await dlq.get_entry_by_job(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_list_entries_no_filters(self, dlq, sample_job, sample_error):
        """Test listing entries without filters."""
        entry1 = await dlq.enqueue(sample_job, sample_error, [])

        job2 = sample_job.model_copy()
        job2.id = uuid4()
        entry2 = await dlq.enqueue(job2, sample_error, [])

        entries = await dlq.list_entries()

        assert len(entries) == 2
        assert entry1 in entries
        assert entry2 in entries

    @pytest.mark.asyncio
    async def test_list_entries_with_status_filter(self, dlq, sample_job, sample_error):
        """Test listing entries with status filter."""
        entry = await dlq.enqueue(sample_job, sample_error, [])
        entry.status = DLQEntryStatus.UNDER_REVIEW

        entries = await dlq.list_entries(
            filters=DLQFilters(status=DLQEntryStatus.PENDING)
        )

        assert len(entries) == 0

        entries = await dlq.list_entries(
            filters=DLQFilters(status=DLQEntryStatus.UNDER_REVIEW)
        )

        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_list_entries_with_category_filter(self, dlq, sample_job, sample_error):
        """Test listing entries with failure category filter."""
        await dlq.enqueue(sample_job, sample_error, [])

        entries = await dlq.list_entries(
            filters=DLQFilters(failure_category=DLQFailureCategory.PARSING_ERROR)
        )

        assert len(entries) == 0  # sample_error results in UNKNOWN_ERROR

        entries = await dlq.list_entries(
            filters=DLQFilters(failure_category=DLQFailureCategory.UNKNOWN_ERROR)
        )

        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_list_entries_with_date_filter(self, dlq, sample_job, sample_error):
        """Test listing entries with date filter."""
        await dlq.enqueue(sample_job, sample_error, [])

        now = utcnow()
        entries = await dlq.list_entries(
            filters=DLQFilters(
                date_from=now - timedelta(hours=1),
                date_to=now + timedelta(hours=1),
            )
        )

        assert len(entries) == 1

        entries = await dlq.list_entries(
            filters=DLQFilters(
                date_from=now + timedelta(hours=1),
            )
        )

        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_list_entries_with_search(self, dlq, sample_job, sample_error):
        """Test listing entries with search."""
        await dlq.enqueue(sample_job, sample_error, [])

        entries = await dlq.list_entries(filters=DLQFilters(search="test.pdf"))

        assert len(entries) == 1

        entries = await dlq.list_entries(filters=DLQFilters(search="nonexistent"))

        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_list_entries_pagination(self, dlq, sample_job, sample_error):
        """Test listing entries with pagination."""
        for i in range(5):
            job = sample_job.model_copy()
            job.id = uuid4()
            await dlq.enqueue(job, sample_error, [])

        entries = await dlq.list_entries(limit=2, offset=0)
        assert len(entries) == 2

        entries = await dlq.list_entries(limit=2, offset=2)
        assert len(entries) == 2

        entries = await dlq.list_entries(limit=2, offset=4)
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_count_entries(self, dlq, sample_job, sample_error):
        """Test counting entries."""
        for i in range(3):
            job = sample_job.model_copy()
            job.id = uuid4()
            await dlq.enqueue(job, sample_error, [])

        count = await dlq.count_entries()

        assert count == 3

    @pytest.mark.asyncio
    async def test_retry_from_dlq_success(self, dlq, sample_job, sample_error):
        """Test retrying from DLQ successfully."""
        entry = await dlq.enqueue(sample_job, sample_error, [])

        result = await dlq.retry_from_dlq(
            entry.id,
            reviewed_by="admin",
            resolution_notes="Retrying with new config",
        )

        assert isinstance(result, DLQRetryResult)
        assert result.success is True
        assert result.entry_id == entry.id
        assert result.new_job_id is not None
        assert "scheduled" in result.message.lower()
        assert entry.status == DLQEntryStatus.RETRY_SCHEDULED
        assert entry.reviewed_by == "admin"
        assert entry.resolution_notes == "Retrying with new config"

    @pytest.mark.asyncio
    async def test_retry_from_dlq_not_found(self, dlq):
        """Test retrying from DLQ when entry not found."""
        entry_id = uuid4()

        result = await dlq.retry_from_dlq(entry_id)

        assert result.success is False
        assert result.entry_id == entry_id
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_retry_with_new_config(self, dlq, sample_job, sample_error):
        """Test retrying with new pipeline config."""
        entry = await dlq.enqueue(sample_job, sample_error, [])
        new_config = PipelineConfig(
            name="retry_config",
            parser=ParserConfig(primary_parser="azure_ocr"),
        )

        result = await dlq.retry_from_dlq(entry.id, new_config=new_config)

        assert result.success is True
        assert result.new_job_id is not None

    @pytest.mark.asyncio
    async def test_mark_reviewed(self, dlq, sample_job, sample_error):
        """Test marking an entry as reviewed."""
        entry = await dlq.enqueue(sample_job, sample_error, [])

        result = await dlq.mark_reviewed(entry.id, "admin", "Needs investigation")

        assert result is not None
        assert result.status == DLQEntryStatus.UNDER_REVIEW
        assert result.reviewed_by == "admin"
        assert result.reviewed_at is not None
        assert result.resolution_notes == "Needs investigation"

    @pytest.mark.asyncio
    async def test_mark_reviewed_not_found(self, dlq):
        """Test marking a non-existent entry as reviewed."""
        result = await dlq.mark_reviewed(uuid4(), "admin")

        assert result is None

    @pytest.mark.asyncio
    async def test_mark_resolved(self, dlq, sample_job, sample_error):
        """Test marking an entry as resolved."""
        entry = await dlq.enqueue(sample_job, sample_error, [])

        result = await dlq.mark_resolved(entry.id, "Fixed and processed successfully")

        assert result is not None
        assert result.status == DLQEntryStatus.RESOLVED
        assert result.resolution_notes == "Fixed and processed successfully"

    @pytest.mark.asyncio
    async def test_mark_resolved_not_found(self, dlq):
        """Test marking a non-existent entry as resolved."""
        result = await dlq.mark_resolved(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_discard_entry(self, dlq, sample_job, sample_error):
        """Test discarding an entry."""
        entry = await dlq.enqueue(sample_job, sample_error, [])

        result = await dlq.discard_entry(entry.id, "Corrupted file", "admin")

        assert result is not None
        assert result.status == DLQEntryStatus.DISCARDED
        assert result.reviewed_by == "admin"
        assert "Corrupted file" in result.resolution_notes

    @pytest.mark.asyncio
    async def test_discard_entry_not_found(self, dlq):
        """Test discarding a non-existent entry."""
        result = await dlq.discard_entry(uuid4(), "Reason")

        assert result is None

    @pytest.mark.asyncio
    async def test_archive_old_entries(self, dlq, sample_job, sample_error):
        """Test archiving old entries."""
        # Create an old entry
        old_entry = await dlq.enqueue(sample_job, sample_error, [])
        old_entry.created_at = utcnow() - timedelta(days=31)

        # Create a recent entry
        job2 = sample_job.model_copy()
        job2.id = uuid4()
        recent_entry = await dlq.enqueue(job2, sample_error, [])

        count = await dlq.archive_old_entries()

        assert count == 1
        assert old_entry.status == DLQEntryStatus.ARCHIVED
        assert old_entry.archived_at is not None
        assert recent_entry.status == DLQEntryStatus.PENDING

    @pytest.mark.asyncio
    async def test_archive_old_entries_none_expired(self, dlq, sample_job, sample_error):
        """Test archiving when no entries are expired."""
        await dlq.enqueue(sample_job, sample_error, [])

        count = await dlq.archive_old_entries()

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, dlq):
        """Test getting statistics with empty DLQ."""
        stats = await dlq.get_statistics()

        assert stats["total_entries"] == 0
        assert stats["by_status"] == {}
        assert stats["by_failure_category"] == {}
        assert stats["max_age_days"] == 30

    @pytest.mark.asyncio
    async def test_get_statistics_with_entries(self, dlq, sample_job, sample_error):
        """Test getting statistics with entries."""
        # Create entries with different categories
        job1 = sample_job.model_copy()
        job1.id = uuid4()
        entry1 = await dlq.enqueue(job1, sample_error, [])
        entry1.failure_category = DLQFailureCategory.PARSING_ERROR
        entry1.status = DLQEntryStatus.PENDING

        job2 = sample_job.model_copy()
        job2.id = uuid4()
        entry2 = await dlq.enqueue(job2, sample_error, [])
        entry2.failure_category = DLQFailureCategory.TIMEOUT
        entry2.status = DLQEntryStatus.RESOLVED

        stats = await dlq.get_statistics()

        assert stats["total_entries"] == 2
        assert stats["by_status"]["pending"] == 1
        assert stats["by_status"]["resolved"] == 1
        assert stats["by_failure_category"]["parsing_error"] == 1
        assert stats["by_failure_category"]["timeout"] == 1
        assert "age_distribution" in stats


@pytest.mark.unit
class TestGlobalDLQ:
    """Tests for global DLQ functions."""

    def test_get_dlq_singleton(self):
        """Test that get_dlq returns a singleton."""
        dlq1 = get_dlq()
        dlq2 = get_dlq()

        assert dlq1 is dlq2

    def test_get_dlq_custom_age(self):
        """Test getting DLQ with custom max age."""
        # Reset global instance for this test
        import src.core.dlq as dlq_module
        dlq_module._dlq = None

        dlq = get_dlq(max_age_days=60)

        assert dlq.max_age_days == 60

    def test_set_dlq(self):
        """Test setting the global DLQ instance."""
        new_dlq = DeadLetterQueue(max_age_days=45)

        set_dlq(new_dlq)

        assert get_dlq() is new_dlq

    def teardown_method(self):
        """Reset global DLQ after each test."""
        import src.core.dlq as dlq_module
        dlq_module._dlq = None
