"""Dead Letter Queue (DLQ) for failed pipeline jobs.

The DLQ stores jobs that have exhausted all retry attempts
and require manual inspection or intervention.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID

from src.api.models import Job, JobError, PipelineConfig, RetryRecord

logger = logging.getLogger(__name__)


class DLQEntryStatus(str, Enum):
    """Status of a DLQ entry."""
    PENDING = "pending"           # Awaiting manual review
    UNDER_REVIEW = "under_review" # Being reviewed by operator
    RETRY_SCHEDULED = "retry_scheduled"  # Retry has been scheduled
    RETRYING = "retrying"         # Currently being retried
    RESOLVED = "resolved"         # Successfully resolved
    DISCARDED = "discarded"       # Discarded after review
    ARCHIVED = "archived"         # Archived for long-term storage


class DLQFailureCategory(str, Enum):
    """Categories of failures that lead to DLQ."""
    PARSING_ERROR = "parsing_error"
    QUALITY_THRESHOLD = "quality_threshold"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class DLQFilters:
    """Filters for listing DLQ entries.
    
    Attributes:
        status: Filter by entry status
        failure_category: Filter by failure category
        source_type: Filter by source type
        date_from: Filter entries from this date
        date_to: Filter entries to this date
        file_name_pattern: Pattern to match file names
        search: Free text search
    """
    status: DLQEntryStatus | None = None
    failure_category: DLQFailureCategory | None = None
    source_type: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    file_name_pattern: str | None = None
    search: str | None = None


@dataclass
class DLQEntry:
    """Single entry in the Dead Letter Queue.
    
    Attributes:
        id: Unique entry ID
        job_id: ID of the failed job
        job: The job data at time of failure
        error: Error information
        failure_category: Categorized failure type
        retry_history: Complete retry history
        status: Current DLQ status
        created_at: When the entry was created
        updated_at: When the entry was last updated
        reviewed_at: When the entry was reviewed
        reviewed_by: Who reviewed the entry
        retry_count: Number of retry attempts made
        max_retries_reached: Whether max retries was reached
        resolution_notes: Notes about resolution
        archived_at: When the entry was archived
        metadata: Additional metadata
    """
    id: UUID
    job_id: UUID
    job: Job
    error: JobError
    failure_category: DLQFailureCategory
    retry_history: list[RetryRecord]
    status: DLQEntryStatus = DLQEntryStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    retry_count: int = 0
    max_retries_reached: bool = False
    resolution_notes: str | None = None
    archived_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "id": str(self.id),
            "job_id": str(self.job_id),
            "job": self.job.model_dump() if hasattr(self.job, "model_dump") else str(self.job),
            "error": self.error.model_dump() if hasattr(self.error, "model_dump") else str(self.error),
            "failure_category": self.failure_category.value,
            "retry_history": [
                r.model_dump() if hasattr(r, "model_dump") else str(r)
                for r in self.retry_history
            ],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewed_by": self.reviewed_by,
            "retry_count": self.retry_count,
            "max_retries_reached": self.max_retries_reached,
            "resolution_notes": self.resolution_notes,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "metadata": self.metadata,
        }


@dataclass
class DLQRetryResult:
    """Result of a DLQ retry operation.
    
    Attributes:
        success: Whether the retry was initiated successfully
        entry_id: ID of the DLQ entry
        new_job_id: ID of the new retry job (if created)
        message: Human-readable result message
        error: Error message if retry failed
    """
    success: bool
    entry_id: UUID
    new_job_id: UUID | None = None
    message: str = ""
    error: str | None = None


class DeadLetterQueue:
    """Dead Letter Queue for managing failed jobs.
    
    The DLQ provides:
    - Storage of failed jobs with full context
    - Manual retry capability with configuration changes
    - Browsing and filtering of failed jobs
    - Analytics on failure patterns
    """

    def __init__(self, max_age_days: int = 30) -> None:
        """Initialize the DLQ.
        
        Args:
            max_age_days: Maximum age of entries before auto-archiving
        """
        self.logger = logger
        self.max_age_days = max_age_days
        self._entries: dict[UUID, DLQEntry] = {}
        self._job_id_to_entry: dict[UUID, UUID] = {}  # job_id -> entry_id

    def _categorize_failure(self, error: JobError, retry_history: list[RetryRecord]) -> DLQFailureCategory:
        """Categorize a failure based on error and history.
        
        Args:
            error: The error that caused failure
            retry_history: History of retry attempts
            
        Returns:
            Failure category
        """
        error_code = (error.code or "").lower()
        error_message = (error.message or "").lower()

        # Check error code patterns
        if any(code in error_code for code in ["parsing", "parse", "format", "corrupt"]):
            return DLQFailureCategory.PARSING_ERROR

        if any(code in error_code for code in ["quality", "threshold", "score"]):
            return DLQFailureCategory.QUALITY_THRESHOLD

        if any(code in error_code for code in ["timeout", "timed out"]):
            return DLQFailureCategory.TIMEOUT

        if any(code in error_code for code in ["memory", "resource", "oom", "quota"]):
            return DLQFailureCategory.RESOURCE_EXHAUSTED

        if any(code in error_code for code in ["network", "connection", "unreachable", "dns"]):
            return DLQFailureCategory.NETWORK_ERROR

        if any(code in error_code for code in ["auth", "unauthorized", "forbidden", "credential"]):
            return DLQFailureCategory.AUTHENTICATION_ERROR

        if any(code in error_code for code in ["validation", "invalid", "schema"]):
            return DLQFailureCategory.VALIDATION_ERROR

        # Check error message patterns as fallback
        if "quality" in error_message or "score" in error_message:
            return DLQFailureCategory.QUALITY_THRESHOLD

        if "timeout" in error_message:
            return DLQFailureCategory.TIMEOUT

        return DLQFailureCategory.UNKNOWN_ERROR

    async def enqueue(
        self,
        job: Job,
        error: Exception,
        retry_history: list[RetryRecord],
        metadata: dict[str, Any] | None = None,
    ) -> DLQEntry:
        """Add a failed job to the DLQ.
        
        Args:
            job: The failed job
            error: The exception that caused failure
            retry_history: Complete retry history
            metadata: Additional metadata
            
        Returns:
            Created DLQ entry
        """
        from uuid import uuid4

        entry_id = uuid4()

        # Convert exception to JobError
        job_error: JobError
        if isinstance(error, JobError):
            job_error = error
        else:
            job_error = JobError(
                code=type(error).__name__,
                message=str(error),
                details={"traceback": getattr(error, "__traceback__", None)},
                failed_stage=job.current_stage,
            )

        # Categorize the failure
        failure_category = self._categorize_failure(job_error, retry_history)

        # Create entry
        entry = DLQEntry(
            id=entry_id,
            job_id=job.id,
            job=job,
            error=job_error,
            failure_category=failure_category,
            retry_history=retry_history.copy(),
            retry_count=job.retry_count,
            max_retries_reached=job.retry_count >= (job.pipeline_config.quality.max_retries if job.pipeline_config else 3),
            status=DLQEntryStatus.PENDING,
            metadata=metadata or {},
        )

        # Store entry
        self._entries[entry_id] = entry
        self._job_id_to_entry[job.id] = entry_id

        self.logger.warning(  # type: ignore[call-arg]
            "job_moved_to_dlq",
            entry_id=str(entry_id),
            job_id=str(job.id),
            failure_category=failure_category.value,
            retry_count=job.retry_count,
        )

        return entry

    async def dequeue(self, entry_id: UUID) -> DLQEntry | None:
        """Remove an entry from the DLQ.
        
        Args:
            entry_id: ID of the entry to remove
            
        Returns:
            The removed entry or None if not found
        """
        entry = self._entries.pop(entry_id, None)
        if entry:
            del self._job_id_to_entry[entry.job_id]
            self.logger.info("entry_removed_from_dlq", entry_id=str(entry_id))  # type: ignore[call-arg]
        return entry

    async def get_entry(self, entry_id: UUID) -> DLQEntry | None:
        """Get a DLQ entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            DLQ entry or None
        """
        return self._entries.get(entry_id)

    async def get_entry_by_job(self, job_id: UUID) -> DLQEntry | None:
        """Get a DLQ entry by job ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            DLQ entry or None
        """
        entry_id = self._job_id_to_entry.get(job_id)
        if entry_id:
            return self._entries.get(entry_id)
        return None

    async def list_entries(
        self,
        filters: DLQFilters | None = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> list[DLQEntry]:
        """List DLQ entries with filtering.
        
        Args:
            filters: Optional filters
            limit: Maximum number of entries
            offset: Offset for pagination
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            List of DLQ entries
        """
        filters = filters or DLQFilters()

        entries = list(self._entries.values())

        # Apply filters
        if filters.status:
            entries = [e for e in entries if e.status == filters.status]

        if filters.failure_category:
            entries = [e for e in entries if e.failure_category == filters.failure_category]

        if filters.source_type:
            entries = [e for e in entries if e.job.source_type.value == filters.source_type]

        if filters.date_from:
            entries = [e for e in entries if e.created_at >= filters.date_from]

        if filters.date_to:
            entries = [e for e in entries if e.created_at <= filters.date_to]

        if filters.file_name_pattern:
            import fnmatch
            entries = [
                e for e in entries
                if fnmatch.fnmatch(e.job.file_name.lower(), filters.file_name_pattern.lower())
            ]

        if filters.search:
            search_lower = filters.search.lower()
            entries = [
                e for e in entries
                if (search_lower in e.job.file_name.lower() or
                    search_lower in e.error.message.lower() or
                    search_lower in str(e.error.code).lower())
            ]

        # Sort entries
        reverse = sort_order.lower() == "desc"
        if sort_by == "created_at":
            entries.sort(key=lambda e: e.created_at, reverse=reverse)
        elif sort_by == "updated_at":
            entries.sort(key=lambda e: e.updated_at, reverse=reverse)
        elif sort_by == "retry_count":
            entries.sort(key=lambda e: e.retry_count, reverse=reverse)
        elif sort_by == "file_name":
            entries.sort(key=lambda e: e.job.file_name, reverse=reverse)

        # Apply pagination
        return entries[offset:offset + limit]

    async def count_entries(self, filters: DLQFilters | None = None) -> int:
        """Count DLQ entries matching filters.
        
        Args:
            filters: Optional filters
            
        Returns:
            Count of matching entries
        """
        entries = await self.list_entries(filters, limit=10000)
        return len(entries)

    async def retry_from_dlq(
        self,
        entry_id: UUID,
        new_config: PipelineConfig | None = None,
        reviewed_by: str | None = None,
        resolution_notes: str | None = None,
    ) -> DLQRetryResult:
        """Retry a job from the DLQ.
        
        Args:
            entry_id: ID of the DLQ entry to retry
            new_config: Optional new pipeline configuration
            reviewed_by: User initiating the retry
            resolution_notes: Notes about the retry
            
        Returns:
            DLQRetryResult with retry status
        """
        entry = self._entries.get(entry_id)
        if not entry:
            return DLQRetryResult(
                success=False,
                entry_id=entry_id,
                error=f"DLQ entry not found: {entry_id}",
            )

        # Update entry status
        entry.status = DLQEntryStatus.RETRY_SCHEDULED
        entry.updated_at = datetime.utcnow()
        entry.reviewed_by = reviewed_by
        entry.resolution_notes = resolution_notes

        from uuid import uuid4
        new_job_id = uuid4()

        self.logger.info(  # type: ignore[call-arg]
            "dlq_retry_initiated",
            entry_id=str(entry_id),
            job_id=str(entry.job_id),
            new_job_id=str(new_job_id),
            reviewed_by=reviewed_by,
        )

        # Create result
        result = DLQRetryResult(
            success=True,
            entry_id=entry_id,
            new_job_id=new_job_id,
            message=f"Job retry scheduled with new ID: {new_job_id}",
        )

        # Note: The actual job creation and queueing would be handled
        # by the orchestration engine using the new_config

        return result

    async def mark_reviewed(
        self,
        entry_id: UUID,
        reviewed_by: str,
        notes: str | None = None,
    ) -> DLQEntry | None:
        """Mark a DLQ entry as reviewed.
        
        Args:
            entry_id: Entry ID
            reviewed_by: User who reviewed
            notes: Review notes
            
        Returns:
            Updated entry or None
        """
        entry = self._entries.get(entry_id)
        if not entry:
            return None

        entry.status = DLQEntryStatus.UNDER_REVIEW
        entry.reviewed_at = datetime.utcnow()
        entry.reviewed_by = reviewed_by
        entry.updated_at = datetime.utcnow()
        if notes:
            entry.resolution_notes = notes

        self.logger.info(  # type: ignore[call-arg]
            "dlq_entry_reviewed",
            entry_id=str(entry_id),
            reviewed_by=reviewed_by,
        )

        return entry

    async def mark_resolved(
        self,
        entry_id: UUID,
        resolution_notes: str | None = None,
    ) -> DLQEntry | None:
        """Mark a DLQ entry as resolved.
        
        Args:
            entry_id: Entry ID
            resolution_notes: Resolution notes
            
        Returns:
            Updated entry or None
        """
        entry = self._entries.get(entry_id)
        if not entry:
            return None

        entry.status = DLQEntryStatus.RESOLVED
        entry.updated_at = datetime.utcnow()
        if resolution_notes:
            entry.resolution_notes = resolution_notes

        self.logger.info("dlq_entry_resolved", entry_id=str(entry_id))  # type: ignore[call-arg]

        return entry

    async def discard_entry(
        self,
        entry_id: UUID,
        reason: str,
        discarded_by: str | None = None,
    ) -> DLQEntry | None:
        """Discard a DLQ entry (mark as permanently failed).
        
        Args:
            entry_id: Entry ID
            reason: Reason for discarding
            discarded_by: User who discarded
            
        Returns:
            Updated entry or None
        """
        entry = self._entries.get(entry_id)
        if not entry:
            return None

        entry.status = DLQEntryStatus.DISCARDED
        entry.updated_at = datetime.utcnow()
        entry.reviewed_by = discarded_by
        entry.resolution_notes = f"Discarded: {reason}"

        self.logger.info(  # type: ignore[call-arg]
            "dlq_entry_discarded",
            entry_id=str(entry_id),
            reason=reason,
            discarded_by=discarded_by,
        )

        return entry

    async def archive_old_entries(self) -> int:
        """Archive entries older than max_age_days.
        
        Returns:
            Number of entries archived
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.max_age_days)
        archived_count = 0

        for entry in self._entries.values():
            if entry.created_at < cutoff_date and entry.status != DLQEntryStatus.ARCHIVED:
                entry.status = DLQEntryStatus.ARCHIVED
                entry.archived_at = datetime.utcnow()
                entry.updated_at = datetime.utcnow()
                archived_count += 1

        if archived_count > 0:
            self.logger.info("dlq_entries_archived", count=archived_count, cutoff=cutoff_date)  # type: ignore[call-arg]

        return archived_count

    async def get_statistics(self) -> dict[str, Any]:
        """Get DLQ statistics.
        
        Returns:
            Dictionary with statistics
        """
        entries = list(self._entries.values())

        total = len(entries)
        by_status: dict[str, int] = {}
        by_category: dict[str, int] = {}

        for entry in entries:
            # Count by status
            status = entry.status.value
            by_status[status] = by_status.get(status, 0) + 1

            # Count by category
            category = entry.failure_category.value
            by_category[category] = by_category.get(category, 0) + 1

        # Calculate age distribution
        now = datetime.utcnow()
        age_distribution = {
            "less_than_1_day": 0,
            "1_to_7_days": 0,
            "7_to_30_days": 0,
            "more_than_30_days": 0,
        }

        for entry in entries:
            age_days = (now - entry.created_at).days
            if age_days < 1:
                age_distribution["less_than_1_day"] += 1
            elif age_days < 7:
                age_distribution["1_to_7_days"] += 1
            elif age_days < 30:
                age_distribution["7_to_30_days"] += 1
            else:
                age_distribution["more_than_30_days"] += 1

        return {
            "total_entries": total,
            "by_status": by_status,
            "by_failure_category": by_category,
            "age_distribution": age_distribution,
            "max_age_days": self.max_age_days,
        }


# Global DLQ instance
_dlq: DeadLetterQueue | None = None


def get_dlq(max_age_days: int = 30) -> DeadLetterQueue:
    """Get the global DLQ instance.
    
    Args:
        max_age_days: Maximum age of entries
        
    Returns:
        DeadLetterQueue instance
    """
    global _dlq
    if _dlq is None:
        _dlq = DeadLetterQueue(max_age_days=max_age_days)
    return _dlq


def set_dlq(dlq: DeadLetterQueue) -> None:
    """Set the global DLQ instance.
    
    Args:
        dlq: DLQ instance to set
    """
    global _dlq
    _dlq = dlq
