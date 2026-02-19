"""Job processor for the worker service.

This module handles the actual processing of jobs from the queue,
including pipeline execution and error handling.
"""

import asyncio
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

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
    ) -> None:
        """Initialize the job processor.
        
        Args:
            engine: Orchestration engine
            plugin_registry: Plugin registry
            worker_id: Unique identifier for the worker
        """
        self.engine = engine
        self.registry = plugin_registry or PluginRegistry()
        self.worker_id = worker_id or "unknown"
        self.llm: LLMProvider | None = None
        self._running = False
        self._current_job: UUID | None = None
        self._heartbeat_task: asyncio.Task[Any] | None = None

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

            logger.info("registered_parser_plugins", worker_id=self.worker_id)

        except Exception as e:
            logger.warning("failed_to_register_parser_plugins", worker_id=self.worker_id, error=str(e))

        # Register built-in destinations
        try:
            from src.plugins.destinations.cognee import CogneeDestination

            cognee = CogneeDestination()
            await cognee.initialize({
                "api_url": getattr(settings, "COGNEE_API_URL", None),
                "api_key": getattr(settings, "COGNEE_API_KEY", None),
            })
            self.registry.register_destination(cognee)

            logger.info("registered_destination_plugins", worker_id=self.worker_id)

        except Exception as e:
            logger.warning("failed_to_register_destination_plugins", worker_id=self.worker_id, error=str(e))

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

        try:
            logger.info("processing_job", worker_id=self.worker_id, job_id=str(job_id))

            # Execute pipeline
            context = await self.engine.process_job(job_id)

            # Get job from database to check final status
            engine = get_async_engine()
            async with AsyncSession(engine) as session:
                repo = JobRepository(session)
                job = await repo.get_by_id(job_id)

                if job is None:
                    return {
                        "job_id": str(job_id),
                        "status": "unknown",
                        "error": "Job not found after processing",
                        "success": False,
                    }

                # Store results if completed
                if job.status == JobStatus.COMPLETED:
                    await self._store_results(job, context, session)

                result = {
                    "job_id": str(job_id),
                    "status": job.status,
                    "stages_completed": list(context.stage_results.keys()),
                    "success": job.status == JobStatus.COMPLETED,
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

                return result

        except Exception as e:
            logger.error(
                "job_processing_failed",
                worker_id=self.worker_id,
                job_id=str(job_id),
                error=str(e),
            )

            # Update job status to failed
            try:
                engine = get_async_engine()
                async with AsyncSession(engine) as session:
                    repo = JobRepository(session)
                    await repo.update_status(
                        job_id,
                        JobStatus.FAILED,
                        error_message=str(e),
                        error_code="PROCESSING_ERROR",
                    )
            except Exception as update_error:
                logger.error(
                    "failed_to_update_job_status",
                    worker_id=self.worker_id,
                    job_id=str(job_id),
                    error=str(update_error),
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

    async def _store_results(
        self,
        job: JobModel,
        context: Any,
        session: AsyncSession,
    ) -> None:
        """Store job processing results.
        
        Args:
            job: Job model
            context: Pipeline context
            session: Database session
        """
        try:
            repo = JobResultRepository(session)
            
            # Calculate processing time
            processing_time_ms = None
            if job.started_at and job.completed_at:
                processing_time_ms = int((job.completed_at - job.started_at).total_seconds() * 1000)
            
            # Extract output data from context
            output_data = {}
            extracted_text = None
            quality_score = None
            
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
                "job_id": str(job.id),
                "source_type": job.source_type,
                "file_name": job.file_name,
                "file_size": job.file_size,
                "mime_type": job.mime_type,
                "stages_completed": list(context.stage_results.keys()),
            }
            
            # Save result
            await repo.save(
                job_id=str(job.id),
                extracted_text=extracted_text,
                output_data=output_data,
                result_metadata=result_metadata,
                quality_score=quality_score,
                processing_time_ms=processing_time_ms,
            )
            
            logger.info(
                "stored_job_results",
                worker_id=self.worker_id,
                job_id=str(job.id),
                has_text=bool(extracted_text),
                quality_score=quality_score,
            )
            
        except Exception as e:
            logger.error(
                "failed_to_store_results",
                worker_id=self.worker_id,
                job_id=str(job.id),
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
                
                engine = get_async_engine()
                async with AsyncSession(engine) as session:
                    repo = JobRepository(session)
                    await repo.update_heartbeat(job_id)
                    logger.debug("sent_heartbeat", worker_id=self.worker_id, job_id=str(job_id))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("heartbeat_failed", worker_id=self.worker_id, job_id=str(job_id), error=str(e))

    async def process_job_with_retry(
        self,
        job_id: UUID,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Process a job with automatic retry.
        
        Args:
            job_id: Job ID
            max_retries: Maximum retry attempts
            
        Returns:
            Processing result
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                result = await self.process_job(job_id)

                if result.get("success"):
                    return result

                # Check if we should retry
                if attempt < max_retries:
                    engine = get_async_engine()
                    async with AsyncSession(engine) as session:
                        repo = JobRepository(session)
                        job = await repo.get_by_id(job_id)
                        if job and job.retry_count < job.max_retries:
                            logger.info(
                                "retrying_job",
                                worker_id=self.worker_id,
                                job_id=str(job_id),
                                attempt=attempt + 1,
                            )
                            job.retry_count = job.retry_count + 1  # type: ignore[assignment]
                            job.status = JobStatus.PENDING  # type: ignore[assignment]
                            job.error_message = None  # type: ignore[assignment]
                            job.error_code = None  # type: ignore[assignment]
                            await session.commit()
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue

                return result

            except Exception as e:
                last_error = e
                logger.error(
                    "job_attempt_failed",
                    worker_id=self.worker_id,
                    job_id=str(job_id),
                    attempt=attempt,
                    error=str(e),
                )

                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)

        # All retries exhausted
        return {
            "job_id": str(job_id),
            "status": "failed",
            "error": f"All retries failed: {last_error}",
            "success": False,
        }

    async def shutdown(self) -> None:
        """Shutdown the processor and cleanup resources."""
        logger.info("shutting_down_job_processor", worker_id=self.worker_id)

        self._running = False

        # Cancel heartbeat task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Shutdown plugins
        for plugin in self.registry._parsers.values():
            try:
                await plugin.shutdown()
            except Exception as e:
                logger.warning("error_shutting_down_parser", worker_id=self.worker_id, error=str(e))

        for plugin in self.registry._destinations.values():
            try:
                await plugin.shutdown()
            except Exception as e:
                logger.warning("error_shutting_down_destination", worker_id=self.worker_id, error=str(e))

        logger.info("job_processor_shutdown_complete", worker_id=self.worker_id)

    @property
    def is_processing(self) -> bool:
        """Check if processor is currently processing a job."""
        return self._current_job is not None

    @property
    def current_job_id(self) -> UUID | None:
        """Get the ID of the currently processing job."""
        return self._current_job
