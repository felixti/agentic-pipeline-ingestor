"""Repository for job data access."""

from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import asc, desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import JobModel, JobStatus


class JobRepository:
    """Repository for job CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def create(
        self,
        source_type: str,
        source_uri: str | None = None,
        file_name: str | None = None,
        file_size: int | None = None,
        mime_type: str | None = None,
        priority: str = "normal",
        mode: str = "async",
        external_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        pipeline_id: str | None = None,
        pipeline_config: dict[str, Any] | None = None,
    ) -> JobModel:
        """Create a new job.
        
        Args:
            source_type: Type of source (upload, s3, url, etc.)
            source_uri: URI to the source file
            file_name: Name of the file
            file_size: Size of the file in bytes
            mime_type: MIME type of the file
            priority: Job priority (low, normal, high)
            mode: Processing mode (sync, async)
            external_id: External reference ID
            metadata: Additional metadata
            pipeline_id: Pipeline configuration ID
            pipeline_config: Pipeline configuration snapshot
            
        Returns:
            Created JobModel instance
        """
        job = JobModel(
            status=JobStatus.CREATED,
            source_type=source_type,
            source_uri=source_uri,
            file_name=file_name,
            file_size=file_size,
            mime_type=mime_type,
            priority=priority,
            mode=mode,
            external_id=external_id,
            metadata_json=metadata or {},
            retry_count=0,
            max_retries=3,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
        if pipeline_id:
            from uuid import uuid4
            try:
                job.pipeline_id = UUID(pipeline_id) if isinstance(pipeline_id, str) else pipeline_id  # type: ignore[assignment]
            except ValueError:
                pass
        
        if pipeline_config:
            job.pipeline_config = pipeline_config  # type: ignore[assignment]
        
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        
        return job
    
    async def get_by_id(self, job_id: str | UUID) -> JobModel | None:
        """Get job by ID.
        
        Args:
            job_id: Job ID (string or UUID)
            
        Returns:
            JobModel if found, None otherwise
        """
        if isinstance(job_id, str):
            try:
                job_id = UUID(job_id)
            except ValueError:
                return None
        
        result = await self.session.execute(
            select(JobModel).where(JobModel.id == job_id)
        )
        return result.scalar_one_or_none()
    
    async def list_jobs(
        self,
        page: int = 1,
        limit: int = 20,
        status: str | None = None,
        source_type: str | None = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[JobModel], int | None]:
        """List jobs with filtering and pagination.
        
        Args:
            page: Page number (1-indexed)
            limit: Items per page
            status: Filter by status
            source_type: Filter by source type
            sort_by: Field to sort by
            sort_order: Sort order (asc or desc)
            
        Returns:
            Tuple of (jobs list, total count)
        """
        # Build base query
        query = select(JobModel)
        count_query = select(func.count(JobModel.id))
        
        # Apply filters
        if status:
            query = query.where(JobModel.status == status)
            count_query = count_query.where(JobModel.status == status)
        
        if source_type:
            query = query.where(JobModel.source_type == source_type)
            count_query = count_query.where(JobModel.source_type == source_type)
        
        # Get total count
        count_result = await self.session.execute(count_query)
        total = count_result.scalar()
        
        # Apply sorting
        sort_column = getattr(JobModel, sort_by, JobModel.created_at)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        
        # Execute query
        result = await self.session.execute(query)
        jobs = result.scalars().all()
        
        return list(jobs), total
    
    async def poll_pending_job(
        self,
        worker_id: str,
        timeout_seconds: int = 300,
    ) -> JobModel | None:
        """Poll for a pending job with row-level locking.
        
        Uses SELECT FOR UPDATE SKIP LOCKED to prevent duplicate processing
        when multiple workers poll simultaneously.
        
        Args:
            worker_id: Unique identifier for this worker
            timeout_seconds: Lock timeout in seconds
            
        Returns:
            Locked JobModel if found, None otherwise
        """
        # Use SELECT FOR UPDATE SKIP LOCKED for concurrent access
        query = (
            select(JobModel)
            .where(
                JobModel.status.in_([JobStatus.CREATED, JobStatus.PENDING]),
            )
            .where(
                (JobModel.locked_by.is_(None)) | 
                (JobModel.locked_at < datetime.utcnow() - timedelta(seconds=timeout_seconds))
            )
            .order_by(
                desc(JobModel.priority == "high"),  # High priority first
                asc(JobModel.created_at),  # Oldest first
            )
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        
        result = await self.session.execute(query)
        job = result.scalar_one_or_none()
        
        if job:
            # Lock the job
            job.locked_by = worker_id  # type: ignore[assignment]
            job.locked_at = datetime.utcnow()  # type: ignore[assignment]
            job.status = JobStatus.PROCESSING  # type: ignore[assignment]
            job.started_at = datetime.utcnow()  # type: ignore[assignment]
            job.updated_at = datetime.utcnow()  # type: ignore[assignment]
            await self.session.commit()
            await self.session.refresh(job)
        
        return job
    
    async def update_status(
        self,
        job_id: str | UUID,
        status: str,
        error_message: str | None = None,
        error_code: str | None = None,
    ) -> JobModel | None:
        """Update job status.
        
        Args:
            job_id: Job ID
            status: New status
            error_message: Error message (for failed jobs)
            error_code: Error code (for failed jobs)
            
        Returns:
            Updated JobModel if found, None otherwise
        """
        job = await self.get_by_id(job_id)
        if not job:
            return None
        
        job.status = status  # type: ignore[assignment]
        job.updated_at = datetime.utcnow()  # type: ignore[assignment]
        
        if status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = datetime.utcnow()  # type: ignore[assignment]
        
        if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job.completed_at = datetime.utcnow()  # type: ignore[assignment]
            job.locked_by = None  # type: ignore[assignment]
            job.locked_at = None  # type: ignore[assignment]
        
        if error_message:
            job.error_message = error_message  # type: ignore[assignment]
        
        if error_code:
            job.error_code = error_code  # type: ignore[assignment]
        
        await self.session.commit()
        await self.session.refresh(job)
        
        return job
    
    async def update_heartbeat(
        self,
        job_id: str | UUID,
    ) -> JobModel | None:
        """Update job heartbeat timestamp.
        
        Args:
            job_id: Job ID
            
        Returns:
            Updated JobModel if found, None otherwise
        """
        job = await self.get_by_id(job_id)
        if not job:
            return None
        
        job.heartbeat_at = datetime.utcnow()  # type: ignore[assignment]
        job.updated_at = datetime.utcnow()  # type: ignore[assignment]
        await self.session.commit()
        await self.session.refresh(job)
        
        return job
    
    async def release_lock(
        self,
        job_id: str | UUID,
    ) -> JobModel | None:
        """Release job lock.
        
        Args:
            job_id: Job ID
            
        Returns:
            Updated JobModel if found, None otherwise
        """
        job = await self.get_by_id(job_id)
        if not job:
            return None
        
        job.locked_by = None  # type: ignore[assignment]
        job.locked_at = None  # type: ignore[assignment]
        job.updated_at = datetime.utcnow()  # type: ignore[assignment]
        await self.session.commit()
        await self.session.refresh(job)
        
        return job
    
    async def find_stalled_jobs(
        self,
        timeout_seconds: int = 300,
    ) -> list[JobModel]:
        """Find jobs that have been locked but not updated recently.
        
        Args:
            timeout_seconds: Time since last heartbeat to consider stalled
            
        Returns:
            List of stalled jobs
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=timeout_seconds)
        
        query = (
            select(JobModel)
            .where(JobModel.status == JobStatus.PROCESSING)
            .where(JobModel.locked_by.isnot(None))
            .where(
                (JobModel.heartbeat_at.is_(None)) |
                (JobModel.heartbeat_at < cutoff_time)
            )
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def delete(self, job_id: str | UUID) -> bool:
        """Delete a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if deleted, False if not found
        """
        job = await self.get_by_id(job_id)
        if not job:
            return False
        
        await self.session.delete(job)
        await self.session.commit()
        
        return True
