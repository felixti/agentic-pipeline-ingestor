"""Core orchestration engine for the Agentic Data Pipeline Ingestor.

This module provides the orchestration engine that manages job lifecycle,
pipeline execution, and coordination between plugins.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from src.api.models import (
    Job,
    JobCreateRequest,
    JobError,
    JobRetryRequest,
    JobStatus,
    PipelineConfig,
    StageProgress,
)
from src.core.dlq import DeadLetterQueue, get_dlq
from src.core.job_context import JobContext as PipelineContext
from src.core.pipeline import Pipeline as PipelineExecutor
from src.core.retry import RetryContext, RetryStrategyType, get_retry_registry
from src.core.routing import DestinationRouter, get_router
from src.llm.provider import LLMProvider
from src.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class OrchestrationEngine:
    """Main orchestration engine for job processing.
    
    The orchestration engine is responsible for:
    - Job lifecycle management
    - Pipeline execution coordination
    - Stage progress tracking
    - Error handling and retries
    - Plugin coordination
    - DLQ management
    - Retry strategy execution
    
    Example:
        >>> engine = OrchestrationEngine()
        >>> job = await engine.create_job(job_request)
        >>> result = await engine.process_job(job.id)
    """

    def __init__(
        self,
        plugin_registry: PluginRegistry | None = None,
        llm_provider: LLMProvider | None = None,
        dlq: DeadLetterQueue | None = None,
        router: DestinationRouter | None = None,
    ) -> None:
        """Initialize the orchestration engine.
        
        Args:
            plugin_registry: Plugin registry for accessing plugins
            llm_provider: LLM provider for agentic decisions
            dlq: Dead letter queue for failed jobs
            router: Destination router for multi-destination output
        """
        self.logger = logger
        self.registry = plugin_registry or PluginRegistry()
        self.llm = llm_provider
        self.dlq = dlq or get_dlq()
        self.router = router or get_router(self.registry)
        self.retry_registry = get_retry_registry()
        self._active_jobs: dict[UUID, Job] = {}
        self._pipeline_executor: PipelineExecutor | None = None

    def _get_executor(self, pipeline_config: PipelineConfig | None = None) -> PipelineExecutor:
        """Get or create pipeline executor.
        
        Args:
            pipeline_config: Optional pipeline configuration
            
        Returns:
            PipelineExecutor instance
        """
        if self._pipeline_executor is None:
            self._pipeline_executor = PipelineExecutor(  # type: ignore[call-arg]
                config=pipeline_config,
                plugin_registry=self.registry,
                llm_provider=self.llm,
            )
        return self._pipeline_executor

    async def create_job(self, job_data: dict[str, Any]) -> Job:
        """Create a new job.
        
        Args:
            job_data: Job creation data
            
        Returns:
            Created Job instance
        """
        request = JobCreateRequest(**job_data)

        job_id = job_data.get("id")
        job = Job(
            id=UUID(job_id) if job_id else uuid4(),
            source_type=request.source_type,
            source_uri=request.source_uri,
            file_name=request.file_name or "unknown",
            file_size=request.file_size,
            mime_type=request.mime_type,
            mode=request.mode,
            priority=request.priority,
            external_id=request.external_id,
            status=JobStatus.CREATED,
            created_at=datetime.utcnow(),
        )

        self._active_jobs[job.id] = job

        self.logger.info(  # type: ignore[call-arg]
            "job_created",
            job_id=str(job.id),
            source_type=job.source_type.value,
            file_name=job.file_name,
            mode=job.mode.value,
        )

        return job

    async def process_job(
        self,
        job_id: UUID,
        enabled_stages: list[str] | None = None,
    ) -> PipelineContext:
        """Process a job through the pipeline.
        
        Args:
            job_id: ID of the job to process
            enabled_stages: Optional list of stage names to execute
            
        Returns:
            Pipeline context with results
            
        Raises:
            ValueError: If job not found
        """
        job = self._active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        self.logger.info(  # type: ignore[call-arg]
            "processing_job",
            job_id=str(job_id),
            stages=enabled_stages,
        )

        # Create executor and run pipeline
        executor = self._get_executor(job.pipeline_config)

        try:
            context = await executor.execute(job, enabled_stages)  # type: ignore[arg-type]

            self.logger.info(  # type: ignore[call-arg]
                "job_processing_completed",
                job_id=str(job_id),
                status=job.status.value,
                stages_executed=list(context.stage_results.keys()),
            )

            return context

        except Exception as e:
            self.logger.error(  # type: ignore[call-arg]
                "job_processing_failed",
                job_id=str(job_id),
                error=str(e),
                stage=job.current_stage,
            )
            raise

    async def update_job_status(
        self,
        job_id: UUID,
        status: JobStatus,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Update job status.
        
        Args:
            job_id: Job ID
            status: New status
            error: Optional error details
        """
        job = self._active_jobs.get(job_id)
        if job:
            job.status = status
            if error:
                job.error = JobError(**error)

        self.logger.info(  # type: ignore[call-arg]
            "job_status_updated",
            job_id=str(job_id),
            status=status.value,
        )

    async def update_stage_progress(
        self,
        job_id: UUID,
        stage: str,
        progress: StageProgress,
    ) -> None:
        """Update stage progress for a job.
        
        Args:
            job_id: Job ID
            stage: Stage name
            progress: Progress information
        """
        job = self._active_jobs.get(job_id)
        if job:
            job.stage_progress[stage] = progress

        self.logger.info(  # type: ignore[call-arg]
            "stage_progress_updated",
            job_id=str(job_id),
            stage=stage,
            status=progress.status.value,
            progress_percent=progress.progress_percent,
        )

    async def retry_job(
        self,
        job_id: UUID,
        retry_request: JobRetryRequest | None = None,
        strategy: str | None = None,
    ) -> Job:
        """Retry a failed job.
        
        Args:
            job_id: Job ID to retry
            retry_request: Optional retry configuration
            strategy: Optional retry strategy to use
            
        Returns:
            Updated Job instance
            
        Raises:
            ValueError: If job not found or not retryable
        """
        job = self._active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status not in (JobStatus.FAILED, JobStatus.DEAD_LETTER):
            raise ValueError(f"Job cannot be retried in status: {job.status}")

        self.logger.info(  # type: ignore[call-arg]
            "job_retry_initiated",
            job_id=str(job_id),
            previous_status=job.status.value,
            attempt=job.retry_count + 1,
            strategy=strategy,
        )

        # If strategy specified, apply it
        if strategy:
            try:
                strategy_type = RetryStrategyType(strategy)
                retry_strategy = self.retry_registry.get_strategy(strategy_type)

                if retry_strategy:

                    context = RetryContext(
                        job=job,
                        attempt_number=job.retry_count + 1,
                        retry_history=job.retry_history,
                        pipeline_config=job.pipeline_config,
                    )

                    result = await retry_strategy.execute(context)

                    if result.updated_config:
                        job.pipeline_config = result.updated_config

                    self.logger.info(  # type: ignore[call-arg]
                        "retry_strategy_applied",
                        job_id=str(job_id),
                        strategy=strategy,
                        success=result.success,
                    )
            except ValueError:
                self.logger.warning(f"Unknown retry strategy: {strategy}")

        # Reset job state
        job.status = JobStatus.QUEUED
        job.error = None
        job.retry_count += 1

        # Apply updated config if provided
        if retry_request and retry_request.updated_config:
            job.pipeline_config = retry_request.updated_config

        # Force specific parser if requested
        if retry_request and retry_request.force_parser and job.pipeline_config:
            job.pipeline_config.parser.primary_parser = retry_request.force_parser

        return job

    async def move_job_to_dlq(
        self,
        job_id: UUID,
        error: Exception,
    ) -> None:
        """Move a failed job to the Dead Letter Queue.
        
        Args:
            job_id: Job ID to move
            error: The error that caused the failure
        """
        job = self._active_jobs.get(job_id)
        if not job:
            self.logger.warning("cannot_move_to_dlq_job_not_found", job_id=str(job_id))  # type: ignore[call-arg]
            return

        job.status = JobStatus.DEAD_LETTER


        await self.dlq.enqueue(
            job=job,
            error=error,
            retry_history=job.retry_history,
        )

        self.logger.info(  # type: ignore[call-arg]
            "job_moved_to_dlq",
            job_id=str(job_id),
            error_type=type(error).__name__,
        )

    async def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a pending or running job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        job = self._active_jobs.get(job_id)
        if not job:
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            self.logger.warning(  # type: ignore[call-arg]
                "job_cannot_be_cancelled",
                job_id=str(job_id),
                status=job.status.value,
            )
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()

        self.logger.info("job_cancelled", job_id=str(job_id))  # type: ignore[call-arg]
        return True

    async def get_job_result(self, job_id: UUID) -> dict[str, Any] | None:
        """Get processing result for a completed job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job result or None if not available
        """
        job = self._active_jobs.get(job_id)
        if not job:
            return None

        if job.status != JobStatus.COMPLETED or not job.result:
            return None

        return {
            "job_id": str(job_id),
            "status": job.status.value,
            "result": job.result.model_dump() if job.result else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }

    async def get_job(self, job_id: UUID) -> Job | None:
        """Get a job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job or None if not found
        """
        return self._active_jobs.get(job_id)

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs with optional filtering.
        
        Args:
            status: Filter by status
            limit: Maximum number of jobs to return
            offset: Offset for pagination
            
        Returns:
            List of jobs
        """
        jobs = list(self._active_jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[offset:offset + limit]

    async def delete_job(self, job_id: UUID) -> bool:
        """Delete a job from memory.
        
        Args:
            job_id: Job ID to delete
            
        Returns:
            True if deleted
        """
        if job_id in self._active_jobs:
            del self._active_jobs[job_id]
            self.logger.info("job_deleted", job_id=str(job_id))  # type: ignore[call-arg]
            return True
        return False


# Global engine instance
_engine: OrchestrationEngine | None = None


def get_engine() -> OrchestrationEngine:
    """Get the global orchestration engine instance.
    
    Returns:
        OrchestrationEngine instance
    """
    global _engine
    if _engine is None:
        _engine = OrchestrationEngine()
    return _engine


def set_engine(engine: OrchestrationEngine) -> None:
    """Set the global orchestration engine instance.
    
    Args:
        engine: Engine to set
    """
    global _engine
    _engine = engine
