"""Repository for content detection results."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.content_detection.models import (
    ContentAnalysisResult,
    ContentDetectionRecord,
    ImageStatistics,
    PageAnalysis,
    TextStatistics,
)


class DetectionResultRepository:
    """Repository for content detection results."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def get_by_hash(self, file_hash: str) -> Optional[ContentDetectionRecord]:
        """Get detection result by file hash.
        
        Args:
            file_hash: SHA-256 hash of file content
            
        Returns:
            Detection record if found, None otherwise
        """
        from src.db.models import ContentDetectionResultModel
        
        result = await self.session.execute(
            select(ContentDetectionResultModel)
            .where(ContentDetectionResultModel.file_hash == file_hash)
            .where(
                (ContentDetectionResultModel.expires_at.is_(None)) |
                (ContentDetectionResultModel.expires_at > datetime.utcnow())
            )
        )
        
        db_record = result.scalar_one_or_none()
        if db_record is None:
            return None
        
        return self._db_to_model(db_record)
    
    async def save(self, result: ContentAnalysisResult) -> ContentDetectionRecord:
        """Save detection result to database.
        
        Args:
            result: Content analysis result
            
        Returns:
            Saved detection record
        """
        from src.db.models import ContentDetectionResultModel
        
        # Check if record already exists
        existing = await self.get_by_hash(result.file_hash)
        if existing:
            # Update access count and last accessed
            await self.increment_access(existing.id)
            return existing
        
        # Create new record
        db_record = ContentDetectionResultModel(
            file_hash=result.file_hash,
            file_size=result.file_size,
            content_type=result.content_type,
            confidence=result.confidence,
            recommended_parser=result.recommended_parser,
            alternative_parsers=result.alternative_parsers,
            text_statistics=result.text_statistics.model_dump(),
            image_statistics=result.image_statistics.model_dump(),
            page_results=[p.model_dump() for p in result.page_results],
            processing_time_ms=result.processing_time_ms,
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
        
        self.session.add(db_record)
        await self.session.commit()
        await self.session.refresh(db_record)
        
        return self._db_to_model(db_record)
    
    async def increment_access(self, record_id: UUID) -> None:
        """Increment access count for a record.
        
        Args:
            record_id: Record ID
        """
        from src.db.models import ContentDetectionResultModel
        
        await self.session.execute(
            update(ContentDetectionResultModel)
            .where(ContentDetectionResultModel.id == record_id)
            .values(
                access_count=ContentDetectionResultModel.access_count + 1,
                last_accessed_at=datetime.utcnow()
            )
        )
        await self.session.commit()
    
    async def link_to_job(self, job_id: UUID, detection_result_id: UUID) -> None:
        """Link detection result to a job.
        
        Args:
            job_id: Job ID
            detection_result_id: Detection result ID
        """
        from src.db.models import JobDetectionResultModel
        
        link = JobDetectionResultModel(
            job_id=job_id,
            detection_result_id=detection_result_id
        )
        self.session.add(link)
        await self.session.commit()
    
    async def get_by_job_id(self, job_id: UUID) -> Optional[ContentDetectionRecord]:
        """Get detection result linked to a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Detection record if found, None otherwise
        """
        from src.db.models import ContentDetectionResultModel, JobDetectionResultModel
        
        result = await self.session.execute(
            select(ContentDetectionResultModel)
            .join(
                JobDetectionResultModel,
                JobDetectionResultModel.detection_result_id == ContentDetectionResultModel.id
            )
            .where(JobDetectionResultModel.job_id == job_id)
        )
        
        db_record = result.scalar_one_or_none()
        if db_record is None:
            return None
        
        return self._db_to_model(db_record)
    
    async def delete_expired(self) -> int:
        """Delete expired detection results.
        
        Returns:
            Number of records deleted
        """
        from src.db.models import ContentDetectionResultModel
        
        result = await self.session.execute(
            select(ContentDetectionResultModel)
            .where(ContentDetectionResultModel.expires_at < datetime.utcnow())
        )
        
        expired_records = result.scalars().all()
        count = len(expired_records)
        
        for record in expired_records:
            await self.session.delete(record)
        
        await self.session.commit()
        return count
    
    def _db_to_model(self, db_record) -> ContentDetectionRecord:
        """Convert database record to model.
        
        Args:
            db_record: Database record
            
        Returns:
            Content detection record model
        """
        return ContentDetectionRecord(
            id=db_record.id,
            file_hash=db_record.file_hash,
            file_size=db_record.file_size,
            content_type=db_record.content_type,
            confidence=float(db_record.confidence),
            recommended_parser=db_record.recommended_parser,
            alternative_parsers=db_record.alternative_parsers or [],
            text_statistics=TextStatistics(**db_record.text_statistics),
            image_statistics=ImageStatistics(**db_record.image_statistics),
            page_results=[PageAnalysis(**p) for p in db_record.page_results],
            processing_time_ms=db_record.processing_time_ms,
            created_at=db_record.created_at,
            expires_at=db_record.expires_at,
            access_count=db_record.access_count,
            last_accessed_at=db_record.last_accessed_at,
        )
