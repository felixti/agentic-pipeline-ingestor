"""Repository for audit log data access."""

from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import AuditLogModel


class AuditLogRepository:
    """Repository for audit log operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def log(
        self,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        user_id: str | None = None,
        api_key_id: str | None = None,
        request_method: str | None = None,
        request_path: str | None = None,
        request_details: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        duration_ms: int | None = None,
    ) -> AuditLogModel:
        """Create an audit log entry.
        
        Args:
            action: Action performed (create, read, update, delete)
            resource_type: Type of resource (job, pipeline, etc.)
            resource_id: Resource identifier
            user_id: User who performed the action
            api_key_id: API key used (if applicable)
            request_method: HTTP method
            request_path: Request path
            request_details: Additional request details
            success: Whether the action succeeded
            error_message: Error message if failed
            ip_address: Client IP address
            user_agent: Client user agent
            duration_ms: Request duration in milliseconds
            
        Returns:
            Created AuditLogModel
        """
        log = AuditLogModel(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            api_key_id=api_key_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            request_method=request_method,
            request_path=request_path,
            request_details=request_details or {},
            success=1 if success else 0,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent,
            duration_ms=duration_ms,
        )
        
        self.session.add(log)
        await self.session.commit()
        await self.session.refresh(log)
        
        return log
    
    async def query_logs(
        self,
        page: int = 1,
        limit: int = 20,
        user_id: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        success: bool | None = None,
    ) -> tuple[list[AuditLogModel], int | None]:
        """Query audit logs with filters.
        
        Args:
            page: Page number (1-indexed)
            limit: Items per page
            user_id: Filter by user
            action: Filter by action
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            success: Filter by success status
            
        Returns:
            Tuple of (logs list, total count)
        """
        query = select(AuditLogModel)
        count_query = select(func.count(AuditLogModel.id))
        
        # Apply filters
        if user_id:
            query = query.where(AuditLogModel.user_id == user_id)
            count_query = count_query.where(AuditLogModel.user_id == user_id)
        
        if action:
            query = query.where(AuditLogModel.action == action)
            count_query = count_query.where(AuditLogModel.action == action)
        
        if resource_type:
            query = query.where(AuditLogModel.resource_type == resource_type)
            count_query = count_query.where(AuditLogModel.resource_type == resource_type)
        
        if resource_id:
            query = query.where(AuditLogModel.resource_id == resource_id)
            count_query = count_query.where(AuditLogModel.resource_id == resource_id)
        
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            query = query.where(AuditLogModel.timestamp >= start_dt)
            count_query = count_query.where(AuditLogModel.timestamp >= start_dt)
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            query = query.where(AuditLogModel.timestamp <= end_dt)
            count_query = count_query.where(AuditLogModel.timestamp <= end_dt)
        
        if success is not None:
            query = query.where(AuditLogModel.success == (1 if success else 0))
            count_query = count_query.where(AuditLogModel.success == (1 if success else 0))
        
        # Get total count
        count_result = await self.session.execute(count_query)
        total = count_result.scalar()
        
        # Apply sorting and pagination
        query = query.order_by(desc(AuditLogModel.timestamp))
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        logs = result.scalars().all()
        
        return list(logs), total
    
    async def delete_old_logs(
        self,
        retention_days: int = 90,
        batch_size: int = 1000,
    ) -> int:
        """Delete old audit logs.
        
        Args:
            retention_days: Number of days to retain
            batch_size: Maximum number to delete in one batch
            
        Returns:
            Number of deleted logs
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        query = (
            select(AuditLogModel)
            .where(AuditLogModel.timestamp < cutoff_date)
            .limit(batch_size)
        )
        
        result = await self.session.execute(query)
        old_logs = result.scalars().all()
        
        for log in old_logs:
            await self.session.delete(log)
        
        await self.session.commit()
        
        return len(old_logs)
