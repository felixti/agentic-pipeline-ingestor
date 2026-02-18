"""Repository for job result data access."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import JobResultModel


class JobResultRepository:
    """Repository for job result CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def save(
        self,
        job_id: str | UUID,
        extracted_text: str | None = None,
        output_data: dict | None = None,
        result_metadata: dict | None = None,
        quality_score: float | None = None,
        processing_time_ms: int | None = None,
        output_uri: str | None = None,
        expires_at: datetime | None = None,
    ) -> JobResultModel:
        """Save a job result.
        
        Args:
            job_id: Job ID
            extracted_text: Extracted text content
            output_data: Output data dictionary
            metadata: Metadata dictionary
            quality_score: Quality score (0-1)
            processing_time_ms: Processing time in milliseconds
            output_uri: URI for large results stored externally
            expires_at: Expiration timestamp
            
        Returns:
            Created JobResultModel instance
        """
        if isinstance(job_id, str):
            job_id = UUID(job_id)
        
        # Set default expiration (30 days)
        if expires_at is None:
            expires_at = datetime.utcnow() + timedelta(days=30)
        
        # Check if result already exists
        existing = await self.get_by_job_id(job_id)
        if existing:
            # Update existing result
            existing.extracted_text = extracted_text
            existing.output_data = output_data or {}
            existing.result_metadata = result_metadata or {}
            existing.quality_score = quality_score
            existing.processing_time_ms = processing_time_ms
            existing.output_uri = output_uri
            existing.expires_at = expires_at
            existing.created_at = datetime.utcnow()
            await self.session.commit()
            await self.session.refresh(existing)
            return existing
        
        # Create new result
        result = JobResultModel(
            job_id=job_id,
            extracted_text=extracted_text,
            output_data=output_data or {},
            result_metadata=result_metadata or {},
            quality_score=quality_score,
            processing_time_ms=processing_time_ms,
            output_uri=output_uri,
            expires_at=expires_at,
            created_at=datetime.utcnow(),
        )
        
        self.session.add(result)
        await self.session.commit()
        await self.session.refresh(result)
        
        return result
    
    async def get_by_job_id(
        self,
        job_id: str | UUID,
    ) -> Optional[JobResultModel]:
        """Get result by job ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            JobResultModel if found, None otherwise
        """
        if isinstance(job_id, str):
            try:
                job_id = UUID(job_id)
            except ValueError:
                return None
        
        result = await self.session.execute(
            select(JobResultModel).where(JobResultModel.job_id == job_id)
        )
        return result.scalar_one_or_none()
    
    async def delete_expired(
        self,
        batch_size: int = 100,
    ) -> int:
        """Delete expired results.
        
        Args:
            batch_size: Maximum number of results to delete
            
        Returns:
            Number of deleted results
        """
        now = datetime.utcnow()
        
        query = (
            select(JobResultModel)
            .where(JobResultModel.expires_at < now)
            .limit(batch_size)
        )
        
        result = await self.session.execute(query)
        expired = result.scalars().all()
        
        for item in expired:
            await self.session.delete(item)
        
        await self.session.commit()
        
        return len(expired)
    
    async def delete_by_job_id(
        self,
        job_id: str | UUID,
    ) -> bool:
        """Delete result by job ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.get_by_job_id(job_id)
        if not result:
            return False
        
        await self.session.delete(result)
        await self.session.commit()
        
        return True
