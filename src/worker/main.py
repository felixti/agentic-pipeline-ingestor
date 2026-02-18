"""Worker service main entry point.

This module provides the worker service that processes jobs from
the queue. It can run as a standalone service or be imported for
testing.
"""

import asyncio
import os
import signal
import sys
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

from src.core.engine import OrchestrationEngine
from src.db.models import JobStatus, get_async_engine
from src.db.repositories.job import JobRepository
from src.worker.processor import JobProcessor


class WorkerService:
    """Worker service for processing pipeline jobs.
    
    The worker service polls for jobs from the database and processes
them through the pipeline.
    
    Example:
        >>> worker = WorkerService()
        >>> await worker.start()
        >>> # Run until stopped
        >>> await worker.stop()
    """

    def __init__(
        self,
        poll_interval: float = 5.0,
        max_concurrent_jobs: int = 3,
        worker_id: str | None = None,
        lock_timeout: int = 300,
    ) -> None:
        """Initialize the worker service.
        
        Args:
            poll_interval: Seconds between queue polls
            max_concurrent_jobs: Maximum concurrent jobs to process
            worker_id: Unique identifier for this worker (auto-generated if None)
            lock_timeout: Seconds before considering a job stalled
        """
        self.poll_interval = poll_interval
        self.max_concurrent_jobs = max_concurrent_jobs
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
        self.lock_timeout = lock_timeout
        
        self.processor: JobProcessor | None = None
        self.engine: OrchestrationEngine | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._active_tasks: set = set()
        self._stalled_job_checker: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize the worker service."""
        logger.info(
            "initializing_worker_service",
            worker_id=self.worker_id,
            poll_interval=self.poll_interval,
            max_concurrent=self.max_concurrent_jobs,
        )

        # Initialize database engine
        engine = get_async_engine()
        
        # Initialize orchestration engine
        self.engine = OrchestrationEngine()

        # Initialize processor
        self.processor = JobProcessor(engine=self.engine, worker_id=self.worker_id)
        await self.processor.initialize()

        logger.info("worker_service_initialized", worker_id=self.worker_id)

    async def start(self) -> None:
        """Start the worker service."""
        if self._running:
            logger.warning("worker_service_already_running", worker_id=self.worker_id)
            return

        await self.initialize()

        self._running = True
        logger.info(
            "worker_service_started",
            worker_id=self.worker_id,
            poll_interval=self.poll_interval,
            max_concurrent=self.max_concurrent_jobs,
        )

        # Start stalled job checker
        self._stalled_job_checker = asyncio.create_task(self._check_stalled_jobs())

        # Start job processing loop
        try:
            await self._processing_loop()
        except asyncio.CancelledError:
            logger.info("worker_service_cancelled", worker_id=self.worker_id)
        except Exception as e:
            logger.error("worker_service_error", worker_id=self.worker_id, error=str(e))
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the worker service gracefully."""
        if not self._running:
            return

        logger.info("stopping_worker_service", worker_id=self.worker_id)
        self._running = False
        self._shutdown_event.set()

        # Cancel stalled job checker
        if self._stalled_job_checker and not self._stalled_job_checker.done():
            self._stalled_job_checker.cancel()
            try:
                await self._stalled_job_checker
            except asyncio.CancelledError:
                pass

        # Wait for active tasks to complete
        if self._active_tasks:
            logger.info(
                "waiting_for_active_tasks",
                worker_id=self.worker_id,
                count=len(self._active_tasks),
            )
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        # Shutdown processor
        if self.processor:
            await self.processor.shutdown()

        logger.info("worker_service_stopped", worker_id=self.worker_id)

    async def _processing_loop(self) -> None:
        """Main processing loop."""
        from sqlalchemy.ext.asyncio import AsyncSession
        
        while self._running:
            try:
                # Check if we can process more jobs
                active_count = len(self._active_tasks)
                if active_count >= self.max_concurrent_jobs:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=0.5,
                    )
                    continue

                # Poll for jobs from database
                job = await self._poll_for_job()

                if job:
                    # Start processing job
                    task = asyncio.create_task(self._process_job_safe(job.id))
                    self._active_tasks.add(task)
                    task.add_done_callback(self._active_tasks.discard)
                else:
                    # No jobs available, wait
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.poll_interval,
                    )

            except TimeoutError:
                continue
            except Exception as e:
                logger.error("error_in_processing_loop", worker_id=self.worker_id, error=str(e))
                await asyncio.sleep(1)

    async def _poll_for_job(self) -> Any | None:
        """Poll for a job from the database.
        
        Returns:
            Job model if available, None otherwise
        """
        from sqlalchemy.ext.asyncio import AsyncSession
        
        engine = get_async_engine()
        
        async with AsyncSession(engine) as session:
            try:
                repo = JobRepository(session)
                job = await repo.poll_pending_job(
                    worker_id=self.worker_id,
                    timeout_seconds=self.lock_timeout,
                )
                
                if job:
                    logger.info(
                        "job_claimed",
                        worker_id=self.worker_id,
                        job_id=str(job.id),
                        source_type=job.source_type,
                    )
                
                return job
                
            except Exception as e:
                logger.error("error_polling_for_job", worker_id=self.worker_id, error=str(e))
                await session.rollback()
                return None

    async def _process_job_safe(self, job_id: UUID) -> None:
        """Process a job with error handling.
        
        Args:
            job_id: Job ID to process
        """
        try:
            logger.info(
                "starting_job_processing",
                worker_id=self.worker_id,
                job_id=str(job_id),
            )

            if not self.processor:
                raise RuntimeError("Processor not initialized")

            result = await self.processor.process_job_with_retry(job_id)

            logger.info(
                "job_processing_complete",
                worker_id=self.worker_id,
                job_id=str(job_id),
                status=result.get("status"),
                success=result.get("success"),
            )

        except Exception as e:
            logger.error(
                "job_processing_failed",
                worker_id=self.worker_id,
                job_id=str(job_id),
                error=str(e),
            )

    async def _check_stalled_jobs(self) -> None:
        """Periodically check for and recover stalled jobs."""
        from sqlalchemy.ext.asyncio import AsyncSession
        
        while self._running:
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=60,  # Check every minute
                )
                continue
            except TimeoutError:
                pass
            
            if not self._running:
                break
            
            try:
                engine = get_async_engine()
                async with AsyncSession(engine) as session:
                    repo = JobRepository(session)
                    stalled_jobs = await repo.find_stalled_jobs(timeout_seconds=self.lock_timeout)
                    
                    if stalled_jobs:
                        logger.warning(
                            "found_stalled_jobs",
                            worker_id=self.worker_id,
                            count=len(stalled_jobs),
                            job_ids=[str(j.id) for j in stalled_jobs],
                        )
                        
                        for job in stalled_jobs:
                            # Release the lock and reset to pending
                            job.locked_by = None
                            job.locked_at = None
                            job.status = JobStatus.PENDING
                            job.updated_at = datetime.utcnow()
                        
                        await session.commit()
                        
                        logger.info(
                            "released_stalled_jobs",
                            worker_id=self.worker_id,
                            count=len(stalled_jobs),
                        )
                        
            except Exception as e:
                logger.error("error_checking_stalled_jobs", worker_id=self.worker_id, error=str(e))

    def signal_handler(self, sig: int) -> None:
        """Handle shutdown signals.
        
        Args:
            sig: Signal number
        """
        logger.info("received_shutdown_signal", worker_id=self.worker_id, signal=sig)
        asyncio.create_task(self.stop())


async def run_single_job(job_id: str) -> dict[str, Any]:
    """Run a single job and exit.
    
    Args:
        job_id: Job ID to process
        
    Returns:
        Processing result
    """
    worker = WorkerService()
    await worker.initialize()

    try:
        result = await worker.processor.process_job(UUID(job_id))
        return result
    finally:
        await worker.stop()


async def main() -> None:
    """Main entry point for the worker service."""
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Worker Service")
    parser.add_argument(
        "--single-job",
        help="Process a single job and exit",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Queue poll interval in seconds",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent jobs",
    )
    parser.add_argument(
        "--worker-id",
        help="Unique worker identifier",
    )
    parser.add_argument(
        "--lock-timeout",
        type=int,
        default=300,
        help="Job lock timeout in seconds",
    )

    args = parser.parse_args()

    if args.single_job:
        # Process single job
        result = await run_single_job(args.single_job)
        print(f"Result: {result}")
        sys.exit(0 if result.get("success") else 1)

    # Start worker service
    worker = WorkerService(
        poll_interval=args.poll_interval,
        max_concurrent_jobs=args.max_concurrent,
        worker_id=args.worker_id,
        lock_timeout=args.lock_timeout,
    )

    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.signal_handler, sig)

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("interrupted_by_user", worker_id=worker.worker_id)
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
