"""Repository for pipeline configuration data access."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import PipelineModel


class PipelineRepository:
    """Repository for pipeline CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def create(
        self,
        name: str,
        config: dict,
        description: str | None = None,
        created_by: str | None = None,
    ) -> PipelineModel:
        """Create a new pipeline configuration.
        
        Args:
            name: Pipeline name
            config: Pipeline configuration dictionary
            description: Pipeline description
            created_by: User who created the pipeline
            
        Returns:
            Created PipelineModel instance
        """
        pipeline = PipelineModel(
            name=name,
            description=description,
            config=config,
            version=1,
            is_active=1,
            created_by=created_by,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
        self.session.add(pipeline)
        await self.session.commit()
        await self.session.refresh(pipeline)
        
        return pipeline
    
    async def get_by_id(
        self,
        pipeline_id: str | UUID,
        include_inactive: bool = False,
    ) -> PipelineModel | None:
        """Get pipeline by ID.
        
        Args:
            pipeline_id: Pipeline ID
            include_inactive: Whether to include deleted pipelines
            
        Returns:
            PipelineModel if found, None otherwise
        """
        if isinstance(pipeline_id, str):
            try:
                pipeline_id = UUID(pipeline_id)
            except ValueError:
                return None
        
        query = select(PipelineModel).where(PipelineModel.id == pipeline_id)
        
        if not include_inactive:
            query = query.where(PipelineModel.is_active == 1)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_name(
        self,
        name: str,
        include_inactive: bool = False,
    ) -> PipelineModel | None:
        """Get pipeline by name.
        
        Args:
            name: Pipeline name
            include_inactive: Whether to include deleted pipelines
            
        Returns:
            PipelineModel if found, None otherwise
        """
        query = select(PipelineModel).where(PipelineModel.name == name)
        
        if not include_inactive:
            query = query.where(PipelineModel.is_active == 1)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def list_pipelines(
        self,
        page: int = 1,
        limit: int = 20,
        include_inactive: bool = False,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[PipelineModel], int]:
        """List pipelines with pagination.
        
        Args:
            page: Page number (1-indexed)
            limit: Items per page
            include_inactive: Whether to include deleted pipelines
            sort_by: Field to sort by
            sort_order: Sort order (asc or desc)
            
        Returns:
            Tuple of (pipelines list, total count)
        """
        # Build base query
        query = select(PipelineModel)
        count_query = select(func.count(PipelineModel.id))
        
        if not include_inactive:
            query = query.where(PipelineModel.is_active == 1)
            count_query = count_query.where(PipelineModel.is_active == 1)
        
        # Get total count
        count_result = await self.session.execute(count_query)
        total = count_result.scalar()
        
        # Apply sorting
        sort_column = getattr(PipelineModel, sort_by, PipelineModel.created_at)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        
        # Execute query
        result = await self.session.execute(query)
        pipelines = result.scalars().all()
        
        return list(pipelines), total
    
    async def update(
        self,
        pipeline_id: str | UUID,
        name: str | None = None,
        config: dict | None = None,
        description: str | None = None,
    ) -> PipelineModel | None:
        """Update pipeline configuration.
        
        Args:
            pipeline_id: Pipeline ID
            name: New name
            config: New configuration
            description: New description
            
        Returns:
            Updated PipelineModel if found, None otherwise
        """
        pipeline = await self.get_by_id(pipeline_id)
        if not pipeline:
            return None
        
        if name is not None:
            pipeline.name = name
        
        if config is not None:
            pipeline.config = config
        
        if description is not None:
            pipeline.description = description
        
        pipeline.version += 1
        pipeline.updated_at = datetime.utcnow()
        
        await self.session.commit()
        await self.session.refresh(pipeline)
        
        return pipeline
    
    async def delete(
        self,
        pipeline_id: str | UUID,
        soft_delete: bool = True,
    ) -> bool:
        """Delete a pipeline.
        
        Args:
            pipeline_id: Pipeline ID
            soft_delete: If True, marks as inactive; if False, hard delete
            
        Returns:
            True if deleted, False if not found
        """
        pipeline = await self.get_by_id(pipeline_id, include_inactive=True)
        if not pipeline:
            return False
        
        if soft_delete:
            pipeline.is_active = 0
            pipeline.updated_at = datetime.utcnow()
            await self.session.commit()
        else:
            await self.session.delete(pipeline)
            await self.session.commit()
        
        return True
    
    def validate_config(self, config: dict) -> tuple[bool, list[str]]:
        """Validate pipeline configuration.
        
        Args:
            config: Pipeline configuration dictionary
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not isinstance(config, dict):
            errors.append("Config must be a dictionary")
            return False, errors
        
        # Check for required fields
        if "enabled_stages" in config:
            if not isinstance(config["enabled_stages"], list):
                errors.append("enabled_stages must be a list")
            else:
                valid_stages = ["ingest", "detect", "parse", "enrich", "quality", "transform", "output"]
                for stage in config["enabled_stages"]:
                    if stage not in valid_stages:
                        errors.append(f"Invalid stage: {stage}. Must be one of {valid_stages}")
        
        # Validate parser config
        if "parser" in config:
            if not isinstance(config["parser"], dict):
                errors.append("parser must be a dictionary")
        
        # Validate output config
        if "output" in config:
            if not isinstance(config["output"], dict):
                errors.append("output must be a dictionary")
        
        return len(errors) == 0, errors
