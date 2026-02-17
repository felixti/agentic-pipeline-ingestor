"""Worker service main entry point.

This module provides the worker service that processes jobs from
the queue. It can run as a standalone service or be imported for
testing.
"""

import asyncio
import signal
import sys
from typing import Any
from uuid import UUID

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
from src.worker.processor import JobProcessor


class WorkerService:
    """Worker service for processing pipeline jobs.
    
    The worker service polls for jobs from the queue and processes
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
    ) -> None:
        """Initialize the worker service.
        
        Args:
            poll_interval: Seconds between queue polls
            max_concurrent_jobs: Maximum concurrent jobs to process
        """
        self.poll_interval = poll_interval
        self.max_concurrent_jobs = max_concurrent_jobs
        self.processor: JobProcessor | None = None
        self.engine: OrchestrationEngine | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._active_tasks: set = set()

    async def initialize(self) -> None:
        """Initialize the worker service."""
        logger.info("Initializing worker service...")

        # Initialize orchestration engine
        self.engine = OrchestrationEngine()

        # Initialize processor
        self.processor = JobProcessor(engine=self.engine)
        await self.processor.initialize()

        logger.info("Worker service initialized")

    async def start(self) -> None:
        """Start the worker service."""
        if self._running:
            logger.warning("Worker service already running")
            return

        await self.initialize()

        self._running = True
        logger.info(
            "Worker service started",
            poll_interval=self.poll_interval,
            max_concurrent=self.max_concurrent_jobs,
        )

        # Start job processing loop
        try:
            await self._processing_loop()
        except asyncio.CancelledError:
            logger.info("Worker service cancelled")
        except Exception as e:
            logger.error("Worker service error", error=str(e))
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the worker service gracefully."""
        if not self._running:
            return

        logger.info("Stopping worker service...")
        self._running = False
        self._shutdown_event.set()

        # Wait for active tasks to complete
        if self._active_tasks:
            logger.info(
                "Waiting for active tasks to complete",
                count=len(self._active_tasks),
            )
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        # Shutdown processor
        if self.processor:
            await self.processor.shutdown()

        logger.info("Worker service stopped")

    async def _processing_loop(self) -> None:
        """Main processing loop."""
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

                # Poll for jobs (simulated - in real implementation would use queue)
                job = await self._poll_for_job()

                if job:
                    # Start processing job
                    task = asyncio.create_task(self._process_job_safe(job))
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
                logger.error("Error in processing loop", error=str(e))
                await asyncio.sleep(1)

    async def _poll_for_job(self) -> UUID | None:
        """Poll for a job from the queue.
        
        Returns:
            Job ID if available, None otherwise
        """
        # In a real implementation, this would poll from Redis/RQ, Azure Queue, etc.
        # For now, we check the orchestration engine for queued jobs
        if not self.engine:
            return None

        from src.api.models import JobStatus

        # Find queued jobs
        jobs = await self.engine.list_jobs(status=JobStatus.QUEUED, limit=1)

        if jobs:
            job = jobs[0]
            # Mark as processing
            await self.engine.update_job_status(job.id, JobStatus.PROCESSING)
            return job.id

        return None

    async def _process_job_safe(self, job_id: UUID) -> None:
        """Process a job with error handling.
        
        Args:
            job_id: Job ID to process
        """
        try:
            logger.info("Starting job processing", job_id=str(job_id))

            if not self.processor:
                raise RuntimeError("Processor not initialized")

            result = await self.processor.process_job_with_retry(job_id)

            logger.info(
                "Job processing complete",
                job_id=str(job_id),
                status=result.get("status"),
                success=result.get("success"),
            )

        except Exception as e:
            logger.error(
                "Job processing failed",
                job_id=str(job_id),
                error=str(e),
            )

    def signal_handler(self, sig: int) -> None:
        """Handle shutdown signals.
        
        Args:
            sig: Signal number
        """
        logger.info("Received shutdown signal", signal=sig)
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
    )

    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.signal_handler, sig)

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
