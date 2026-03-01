"""Job processor for the worker service.

This module handles the actual processing of jobs from the queue,
including pipeline execution and error handling.
"""

import asyncio
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.config import settings
from src.core.engine import OrchestrationEngine
from src.db.models import JobModel, JobStatus, get_async_engine
from src.db.repositories.job import JobRepository
from src.db.repositories.job_result import JobResultRepository
from src.llm.config import load_llm_config
from src.llm.provider import LLMProvider
from src.observability.logging import get_logger
from src.plugins.registry import PluginRegistry

logger = get_logger(__name__)


class JobProcessor:
    """Processor for pipeline jobs.
    
    Handles job retrieval, pipeline execution, and result handling.
    
    Example:
        >>> processor = JobProcessor()
        >>> await processor.start()
        >>> await processor.process_job(job_id)
    """

    def __init__(
        self,
        engine: OrchestrationEngine | None = None,
        plugin_registry: PluginRegistry | None = None,
        worker_id: str | None = None,
        db_engine: AsyncEngine | None = None,
    ) -> None:
        """Initialize the job processor.
        
        Args:
            engine: Orchestration engine
            plugin_registry: Plugin registry
            worker_id: Unique identifier for the worker
            db_engine: SQLAlchemy async engine for database operations
        """
        self.engine = engine
        self.registry = plugin_registry or PluginRegistry()
        self.worker_id = worker_id or "unknown"
        self.llm: LLMProvider | None = None
        self._db_engine = db_engine
        self._running = False
        self._current_job: UUID | None = None
        self._heartbeat_task: asyncio.Task[Any] | None = None

    def _get_db_engine(self) -> AsyncEngine:
        """Get the database engine (lazy initialization)."""
        if self._db_engine is None:
            self._db_engine = get_async_engine()
        return self._db_engine

    async def initialize(self) -> None:
        """Initialize the processor with dependencies."""
        logger.info("initializing_job_processor", worker_id=self.worker_id)

        # Initialize plugins
        await self._initialize_plugins()

        # Initialize LLM if configured
        await self._initialize_llm()

        # Initialize engine if not provided
        if self.engine is None:
            self.engine = OrchestrationEngine(
                plugin_registry=self.registry,
                llm_provider=self.llm,
            )

        logger.info("job_processor_initialized", worker_id=self.worker_id)

    async def _initialize_plugins(self) -> None:
        """Initialize all plugins from registry."""
        # Register built-in parsers
        try:
            from src.plugins.parsers.azure_ocr_parser import AzureOCRParser
            from src.plugins.parsers.docling_parser import DoclingParser

            docling = DoclingParser()
            await docling.initialize({})
            self.registry.register_parser(docling)

            azure_ocr = AzureOCRParser()
            await azure_ocr.initialize({
                "endpoint": getattr(settings, "AZURE_AI_VISION_ENDPOINT", None),
                "api_key": getattr(settings, "AZURE_AI_VISION_API_KEY", None),
            })
            self.registry.register_parser(azure_ocr)

            logger.info("registered_parsers", worker_id=self.worker_id)

        except Exception as e:
            logger.warning("failed_to_register_parsers", worker_id=self.worker_id, error=str(e))

        # Register built-in destinations
        await self._initialize_destinations()

    async def _initialize_destinations(self) -> None:
        """Initialize destination plugins."""
        # Register Cognee Local Destination
        try:
            from src.plugins.destinations.cognee_local import CogneeLocalDestination

            cognee_local = CogneeLocalDestination()
            await cognee_local.initialize({
                "dataset_id": "default",
                "graph_name": "default",
                "extract_entities": True,
                "extract_relationships": True,
            })
            self.registry.register_destination(cognee_local)

            logger.info("registered_cognee_local_destination", worker_id=self.worker_id)

        except Exception as e:
            logger.warning("failed_to_register_cognee_local_destination", worker_id=self.worker_id, error=str(e))

        # Register HippoRAG Destination
        try:
            from src.plugins.destinations.hipporag import HippoRAGDestination

            hipporag = HippoRAGDestination()
            await hipporag.initialize({
                "save_dir": getattr(settings, "HIPPO_SAVE_DIR", "/data/hipporag"),
            })
            self.registry.register_destination(hipporag)

            logger.info("registered_hipporag_destination", worker_id=self.worker_id)

        except Exception as e:
            logger.warning("failed_to_register_hipporag_destination", worker_id=self.worker_id, error=str(e))

    async def _initialize_llm(self) -> None:
        """Initialize LLM provider if configured."""
        try:
            config = load_llm_config()
            if config.routers:
                self.llm = LLMProvider(config)
                logger.info("llm_provider_initialized", worker_id=self.worker_id)
            else:
                logger.warning("no_llm_routers_configured", worker_id=self.worker_id)
        except Exception as e:
            logger.warning("failed_to_initialize_llm_provider", worker_id=self.worker_id, error=str(e))

    async def process_job_with_retry(self, job_id: UUID) -> dict[str, Any]:
        """Process a single job with retry logic.
        
        This method provides backward compatibility with the worker main.py
        which calls process_job_with_retry. The retry logic is handled at
        the orchestration level.
        
        Args:
            job_id: ID of the job to process
            
        Returns:
            Processing result
        """
        return await self.process_job(job_id)

    async def process_job(self, job_id: UUID) -> dict[str, Any]:
        """Process a single job.
        
        Args:
            job_id: ID of the job to process
            
        Returns:
            Processing result
        """
        if not self.engine:
            raise RuntimeError("Processor not initialized")

        self._current_job = job_id
        
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(job_id))

        job_status = None
        job_started_at = None
        job_completed_at = None
        job_source_type = None
        job_file_name = None
        job_file_size = None
        job_mime_type = None

        try:
            logger.info("processing_job", worker_id=self.worker_id, job_id=str(job_id))

            # Execute pipeline
            context = await self.engine.process_job(job_id)

            # Get job from database to check final status
            # Use async_sessionmaker for proper async session management
            engine = self._get_db_engine()
            async_session = async_sessionmaker(
                engine, 
                expire_on_commit=False,
                class_=AsyncSession,
            )
            
            async with async_session() as session:
                # Use session.begin() to properly manage the transaction
                # This ensures greenlet context is correctly set up
                async with session.begin():
                    repo = JobRepository(session)
                    job = await repo.get_by_id(job_id)

                    if job is None:
                        return {
                            "job_id": str(job_id),
                            "status": "unknown",
                            "error": "Job not found after processing",
                            "success": False,
                        }

                    # Extract job data before closing session to avoid lazy-loading issues
                    job_status = job.status
                    job_started_at = job.started_at
                    job_completed_at = job.completed_at
                    job_source_type = job.source_type
                    job_file_name = job.file_name
                    job_file_size = job.file_size
                    job_mime_type = job.mime_type
                    
                    result = {
                        "job_id": str(job_id),
                        "status": job_status,
                        "stages_completed": list(context.stage_results.keys()),
                        "success": job_status == JobStatus.COMPLETED,
                    }

                    # Add quality score if available
                    quality_result = context.get_stage_result("quality")
                    if quality_result:
                        result["quality_score"] = quality_result.get("overall_score")

                    logger.info(
                        "job_processing_finished",
                        worker_id=self.worker_id,
                        job_id=str(job_id),
                        status=result["status"],
                        success=result["success"],
                    )

            # Store results outside the session context to avoid transaction issues
            if job_status == JobStatus.COMPLETED:
                await self._store_results_safe(
                    job_id=job_id,
                    job_started_at=job_started_at,
                    job_completed_at=job_completed_at,
                    job_source_type=job_source_type,
                    job_file_name=job_file_name,
                    job_file_size=job_file_size,
                    job_mime_type=job_mime_type,
                    context=context,
                )

            return result

        except Exception as e:
            logger.error(
                "job_processing_failed",
                worker_id=self.worker_id,
                job_id=str(job_id),
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            
            return {
                "job_id": str(job_id),
                "status": "failed",
                "error": str(e),
                "success": False,
            }

        finally:
            self._current_job = None
            # Stop heartbeat task
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

    async def _store_results_safe(
        self,
        job_id: UUID,
        job_started_at: Any,
        job_completed_at: Any,
        job_source_type: str,
        job_file_name: str,
        job_file_size: int | None,
        job_mime_type: str | None,
        context: Any,
    ) -> None:
        """Store job processing results.
        
        Args:
            job_id: Job ID
            job_started_at: Job start timestamp
            job_completed_at: Job completion timestamp
            job_source_type: Source type of the job
            job_file_name: File name
            job_file_size: File size in bytes
            job_mime_type: MIME type
            context: Pipeline context
        """
        # Create new session for storing results (separate from processing transaction)
        engine = self._get_db_engine()
        async_session = async_sessionmaker(
            engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        
        extracted_text = None
        quality_score = None
        
        try:
            async with async_session() as session:
                async with session.begin():
                    repo = JobResultRepository(session)
                    
                    # Calculate processing time
                    processing_time_ms = None
                    if job_started_at and job_completed_at:
                        processing_time_ms = int((job_completed_at - job_started_at).total_seconds() * 1000)
                    
                    # Extract output data from context
                    output_data = {}
                    
                    # Get parse result
                    parse_result = context.get_stage_result("parse")
                    if parse_result:
                        extracted_text = parse_result.get("text", "")
                        output_data["parse_result"] = parse_result
                    
                    # Get quality result
                    quality_result = context.get_stage_result("quality")
                    if quality_result:
                        quality_score = quality_result.get("overall_score")
                        output_data["quality_result"] = quality_result
                    
                    # Get other stage results
                    for stage in ["ingest", "detect", "enrich", "transform", "output"]:
                        stage_result = context.get_stage_result(stage)
                        if stage_result:
                            output_data[f"{stage}_result"] = stage_result
                    
                    # Build metadata
                    result_metadata = {
                        "job_id": str(job_id),
                        "source_type": job_source_type,
                        "file_name": job_file_name,
                        "file_size": job_file_size,
                        "mime_type": job_mime_type,
                        "stages_completed": list(context.stage_results.keys()),
                    }
                    
                    # Save result within the transaction
                    await repo.save(
                        job_id=str(job_id),
                        extracted_text=extracted_text,
                        output_data=output_data,
                        result_metadata=result_metadata,
                        quality_score=quality_score,
                        processing_time_ms=processing_time_ms,
                    )
            
            logger.info(
                "stored_job_results",
                worker_id=self.worker_id,
                job_id=str(job_id),
                has_text=bool(extracted_text),
                quality_score=quality_score,
            )
            
        except Exception as e:
            logger.error(
                "failed_to_store_results",
                worker_id=self.worker_id,
                job_id=str(job_id),
                error=str(e),
            )

    async def _heartbeat_loop(self, job_id: UUID) -> None:
        """Send periodic heartbeats while processing a job.
        
        Args:
            job_id: Job ID being processed
        """
        while self._current_job == job_id:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
                if self._current_job != job_id:
                    break
                    
                # Update job heartbeat in database
                engine = self._get_db_engine()
                async_session = async_sessionmaker(
                    engine,
                    expire_on_commit=False,
                    class_=AsyncSession,
                )
                
                async with async_session() as session:
                    async with session.begin():
                        repo = JobRepository(session)
                        await repo.update_heartbeat(job_id)
                        
                logger.debug("job_heartbeat_sent", worker_id=self.worker_id, job_id=str(job_id))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("heartbeat_failed", worker_id=self.worker_id, job_id=str(job_id), error=str(e))
                await asyncio.sleep(5)  # Short delay before retry

    async def start(self) -> None:
        """Start the processor."""
        self._running = True
        logger.info("job_processor_started", worker_id=self.worker_id)

    async def stop(self) -> None:
        """Stop the processor."""
        self._running = False
        
        # Cancel any ongoing heartbeat
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
                
        logger.info("job_processor_stopped", worker_id=self.worker_id)

    @property
    def is_running(self) -> bool:
        """Check if processor is running."""
        return self._running

    @property
    def current_job(self) -> UUID | None:
        """Get currently processing job ID."""
        return self._current_job
