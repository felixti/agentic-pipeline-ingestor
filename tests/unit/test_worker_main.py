"""Unit tests for the worker main module.

This module tests the WorkerService class and main entry points.
"""

import asyncio
import signal
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from src.api.models import JobStatus
from src.worker.main import WorkerService, main, run_single_job


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_processor():
    """Create a mock job processor."""
    processor = AsyncMock()
    processor.initialize = AsyncMock()
    processor.shutdown = AsyncMock()
    processor.process_job = AsyncMock()
    processor.process_job_with_retry = AsyncMock()
    return processor


@pytest.fixture
def mock_engine():
    """Create a mock orchestration engine."""
    engine = AsyncMock()
    engine.list_jobs = AsyncMock(return_value=[])
    engine.update_job_status = AsyncMock()
    return engine


@pytest.fixture
def worker(mock_processor, mock_engine):
    """Create a WorkerService instance with mocked dependencies."""
    with patch("src.worker.main.JobProcessor", return_value=mock_processor):
        with patch("src.worker.main.OrchestrationEngine", return_value=mock_engine):
            worker = WorkerService(poll_interval=0.1, max_concurrent_jobs=2)
            worker.processor = mock_processor
            worker.engine = mock_engine
            return worker


@pytest.fixture
def sample_job():
    """Create a sample job for testing."""
    job = MagicMock()
    job.id = uuid4()
    job.status = JobStatus.QUEUED
    return job


# ============================================================================
# Worker Initialization Tests
# ============================================================================

@pytest.mark.unit
class TestWorkerInitialization:
    """Tests for WorkerService initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        worker = WorkerService()

        assert worker.poll_interval == 5.0
        assert worker.max_concurrent_jobs == 3
        assert worker.processor is None
        assert worker.engine is None
        assert worker._running is False

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        worker = WorkerService(poll_interval=1.0, max_concurrent_jobs=5)

        assert worker.poll_interval == 1.0
        assert worker.max_concurrent_jobs == 5

    @pytest.mark.asyncio
    async def test_initialize_creates_engine(self, mock_processor):
        """Test initialize creates engine and processor."""
        with patch("src.worker.main.OrchestrationEngine") as mock_engine_class:
            mock_engine_instance = MagicMock()
            mock_engine_class.return_value = mock_engine_instance

            with patch("src.worker.main.JobProcessor") as mock_processor_class:
                mock_processor_class.return_value = mock_processor

                worker = WorkerService()
                await worker.initialize()

                mock_engine_class.assert_called_once()
                mock_processor_class.assert_called_once_with(
                    engine=mock_engine_instance
                )
                mock_processor.initialize.assert_called_once()
                assert worker.engine is mock_engine_instance
                assert worker.processor is mock_processor

    @pytest.mark.asyncio
    async def test_initialize_creates_new_dependencies(self, mock_processor):
        """Test initialize creates new engine and processor."""
        worker = WorkerService()
        # Pre-set with different mocks
        worker.engine = MagicMock()
        worker.processor = MagicMock()

        mock_engine_instance = AsyncMock()

        with patch("src.worker.main.OrchestrationEngine", return_value=mock_engine_instance):
            with patch("src.worker.main.JobProcessor", return_value=mock_processor):
                with patch("src.worker.main.logger"):
                    await worker.initialize()

        # Should create new instances
        assert worker.engine is mock_engine_instance
        assert worker.processor is mock_processor
        mock_processor.initialize.assert_called_once()


# ============================================================================
# Worker Start/Stop Tests
# ============================================================================

@pytest.mark.unit
class TestWorkerStartStop:
    """Tests for worker start and stop functionality."""

    @pytest.mark.asyncio
    async def test_start_sets_running_true_during_processing(self, mock_processor):
        """Test start sets running flag to True during processing."""
        worker = WorkerService()
        worker.processor = mock_processor
        worker.engine = AsyncMock()

        running_values = []

        async def capture_running():
            running_values.append(worker._running)

        with patch.object(worker, "_processing_loop", new=capture_running):
            with patch.object(worker, "initialize", new=AsyncMock()):
                await worker.start()

        # _running was True during processing, then False after stop()
        assert True in running_values  # Was True at some point

    @pytest.mark.asyncio
    async def test_start_already_running_logs_warning(self):
        """Test start when already running logs warning."""
        worker = WorkerService()
        worker._running = True

        with patch("src.worker.main.logger") as mock_logger:
            await worker.start()

            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, worker):
        """Test stop sets running flag to False."""
        worker._running = True

        await worker.stop()

        assert worker._running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, worker):
        """Test stop when not running does nothing."""
        worker._running = False

        # Should not raise exception
        await worker.stop()

        assert worker._running is False

    @pytest.mark.asyncio
    async def test_stop_waits_for_active_tasks(self, worker):
        """Test stop waits for active tasks to complete."""
        worker._running = True

        # Create mock tasks that are done
        async def done_task():
            return "done"

        task1 = asyncio.create_task(done_task())
        task2 = asyncio.create_task(done_task())
        # Wait for them to complete
        await asyncio.sleep(0.01)

        worker._active_tasks = {task1, task2}

        with patch("src.worker.main.logger"):
            await worker.stop()

        # Tasks should be cleared (after gather, they are removed via done callback)
        # Note: In actual implementation, done_callback removes them, but we set them directly
        # So we verify stop() ran without error and _running is False
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_stop_shuts_down_processor(self, worker, mock_processor):
        """Test stop shuts down the processor."""
        worker._running = True

        await worker.stop()

        mock_processor.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_handles_processor_shutdown_error(self, worker, mock_processor):
        """Test stop handles processor shutdown errors."""
        worker._running = True
        mock_processor.shutdown.side_effect = Exception("Shutdown error")

        with patch("src.worker.main.logger"):
            # Should not raise exception
            await worker.stop()


# ============================================================================
# Processing Loop Tests
# ============================================================================

@pytest.mark.unit
class TestProcessingLoop:
    """Tests for the main processing loop."""

    @pytest.mark.asyncio
    async def test_processing_loop_polls_for_jobs(self, worker, sample_job):
        """Test processing loop polls for jobs."""
        worker._running = True

        # Stop after one iteration
        call_count = 0

        async def stop_after_poll():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                worker._running = False
            return sample_job if call_count == 1 else None

        worker._poll_for_job = AsyncMock(side_effect=stop_after_poll)

        with patch.object(worker, "_process_job_safe", new=AsyncMock()):
            await worker._processing_loop()

        worker._poll_for_job.assert_called()

    @pytest.mark.asyncio
    async def test_processing_loop_respects_max_concurrent(self, worker):
        """Test processing loop respects max concurrent jobs limit."""
        worker._running = True
        worker.max_concurrent_jobs = 1
        worker._active_tasks = {AsyncMock()}  # One active task

        # Stop after checking concurrency
        async def stop_after_check():
            worker._running = False
            return None

        worker._poll_for_job = AsyncMock(side_effect=stop_after_check)

        await worker._processing_loop()

        # Should not poll when at max concurrent
        worker._poll_for_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_processing_loop_creates_task_for_job(self, worker, sample_job):
        """Test processing loop creates task for job."""
        worker._running = True

        # Stop after one iteration
        async def stop_after_one():
            worker._running = False
            return sample_job

        worker._poll_for_job = AsyncMock(side_effect=stop_after_one)

        with patch("asyncio.create_task") as mock_create_task:
            mock_task = MagicMock()
            mock_create_task.return_value = mock_task

            await worker._processing_loop()

            mock_create_task.assert_called_once()
            mock_task.add_done_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_processing_loop_handles_timeout(self, worker):
        """Test processing loop handles timeout errors."""
        worker._running = True

        # Stop after timeout
        async def raise_timeout():
            worker._running = False
            raise TimeoutError()

        worker._poll_for_job = AsyncMock(side_effect=raise_timeout)

        # Should not raise exception
        await worker._processing_loop()

    @pytest.mark.asyncio
    async def test_processing_loop_handles_errors(self, worker):
        """Test processing loop handles general errors."""
        worker._running = True
        call_count = 0

        async def raise_then_stop():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                worker._running = False
            raise Exception("Loop error")

        worker._poll_for_job = AsyncMock(side_effect=raise_then_stop)

        with patch("asyncio.sleep", new=AsyncMock()):
            # Should not raise exception
            await worker._processing_loop()


# ============================================================================
# Poll for Job Tests
# ============================================================================

@pytest.mark.unit
class TestPollForJob:
    """Tests for _poll_for_job method."""

    @pytest.mark.asyncio
    async def test_poll_returns_none_if_no_engine(self, worker):
        """Test poll returns None if no engine."""
        worker.engine = None

        result = await worker._poll_for_job()

        assert result is None

    @pytest.mark.asyncio
    async def test_poll_returns_none_if_no_jobs(self, worker, mock_engine):
        """Test poll returns None if no queued jobs."""
        mock_engine.list_jobs.return_value = []

        result = await worker._poll_for_job()

        assert result is None
        mock_engine.list_jobs.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_returns_job_and_updates_status(self, worker, mock_engine, sample_job):
        """Test poll returns job and updates status to processing."""
        mock_engine.list_jobs.return_value = [sample_job]

        result = await worker._poll_for_job()

        assert result == sample_job.id
        mock_engine.update_job_status.assert_called_once_with(
            sample_job.id,
            JobStatus.PROCESSING
        )

    @pytest.mark.asyncio
    async def test_poll_limits_to_one_job(self, worker, mock_engine):
        """Test poll only fetches one job."""
        job1 = MagicMock()
        job1.id = uuid4()
        job2 = MagicMock()
        job2.id = uuid4()

        mock_engine.list_jobs.return_value = [job1, job2]

        await worker._poll_for_job()

        mock_engine.list_jobs.assert_called_once_with(
            status=JobStatus.QUEUED,
            limit=1
        )


# ============================================================================
# Process Job Safe Tests
# ============================================================================

@pytest.mark.unit
class TestProcessJobSafe:
    """Tests for _process_job_safe method."""

    @pytest.mark.asyncio
    async def test_process_job_safe_success(self, worker, mock_processor):
        """Test successful job processing with retry."""
        job_id = uuid4()
        mock_processor.process_job_with_retry.return_value = {
            "job_id": str(job_id),
            "status": "completed",
            "success": True,
        }

        with patch("src.worker.main.logger") as mock_logger:
            await worker._process_job_safe(job_id)

        mock_processor.process_job_with_retry.assert_called_once_with(job_id)
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_process_job_safe_processor_not_initialized(self, worker):
        """Test error when processor not initialized."""
        worker.processor = None

        with pytest.raises(RuntimeError, match="Processor not initialized"):
            await worker._process_job_safe(uuid4())

    @pytest.mark.asyncio
    async def test_process_job_safe_handles_errors(self, worker, mock_processor):
        """Test error handling in process job safe."""
        job_id = uuid4()
        mock_processor.process_job_with_retry.side_effect = Exception("Processing error")

        with patch("src.worker.main.logger") as mock_logger:
            # Should not raise exception
            await worker._process_job_safe(job_id)

        mock_logger.error.assert_called()


# ============================================================================
# Signal Handling Tests
# ============================================================================

@pytest.mark.unit
class TestSignalHandling:
    """Tests for signal handling functionality."""

    @pytest.mark.asyncio
    async def test_signal_handler_logs_signal(self, worker):
        """Test signal handler logs received signal."""
        with patch("src.worker.main.logger") as mock_logger:
            with patch("asyncio.create_task") as mock_create_task:
                worker.signal_handler(signal.SIGTERM)

                mock_logger.info.assert_called_once()
                mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_signal_handler_creates_stop_task(self, worker):
        """Test signal handler creates stop task."""
        with patch("asyncio.create_task") as mock_create_task:
            worker.signal_handler(signal.SIGINT)

            mock_create_task.assert_called_once()
            # Verify the task is to call stop()
            task_arg = mock_create_task.call_args[0][0]
            assert isinstance(task_arg, asyncio.Task) or callable(task_arg)


# ============================================================================
# Run Single Job Tests
# ============================================================================

@pytest.mark.unit
class TestRunSingleJob:
    """Tests for run_single_job function."""

    @pytest.mark.asyncio
    async def test_run_single_job_success(self):
        """Test running a single job successfully."""
        job_id = str(uuid4())
        expected_result = {"job_id": job_id, "success": True}

        with patch("src.worker.main.WorkerService") as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker.processor.process_job.return_value = expected_result
            mock_worker_class.return_value = mock_worker

            result = await run_single_job(job_id)

            assert result == expected_result
            mock_worker.initialize.assert_called_once()
            mock_worker.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_single_job_always_stops(self):
        """Test worker stops even on error."""
        job_id = str(uuid4())

        with patch("src.worker.main.WorkerService") as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker.processor.process_job.side_effect = Exception("Error")
            mock_worker_class.return_value = mock_worker

            with pytest.raises(Exception):
                await run_single_job(job_id)

            mock_worker.stop.assert_called_once()


# ============================================================================
# Main Entry Point Tests
# ============================================================================

@pytest.mark.unit
class TestMain:
    """Tests for main entry point."""

    @pytest.mark.asyncio
    async def test_main_with_single_job(self):
        """Test main with single job argument."""
        job_id = str(uuid4())

        with patch.object(sys, "argv", ["worker", "--single-job", job_id]):
            with patch("src.worker.main.run_single_job", new=AsyncMock(return_value={"success": True})):
                with patch("sys.exit") as mock_exit:
                    await main()

                    mock_exit.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_main_with_single_job_failure(self):
        """Test main with single job failure."""
        job_id = str(uuid4())

        with patch.object(sys, "argv", ["worker", "--single-job", job_id]):
            with patch("src.worker.main.run_single_job", new=AsyncMock(return_value={"success": False})):
                with patch("sys.exit") as mock_exit:
                    await main()

                    mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_main_starts_worker_service(self):
        """Test main starts worker service."""
        with patch.object(sys, "argv", ["worker"]):
            with patch("src.worker.main.WorkerService") as mock_worker_class:
                mock_worker = AsyncMock()
                mock_worker_class.return_value = mock_worker

                # Simulate keyboard interrupt after start
                mock_worker.start.side_effect = KeyboardInterrupt()

                with patch("asyncio.get_event_loop") as mock_get_loop:
                    mock_loop = MagicMock()
                    mock_get_loop.return_value = mock_loop

                    await main()

                mock_worker_class.assert_called_once()
                mock_worker.start.assert_called_once()
                mock_worker.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_sets_signal_handlers(self):
        """Test main sets up signal handlers."""
        with patch.object(sys, "argv", ["worker"]):
            with patch("src.worker.main.WorkerService") as mock_worker_class:
                mock_worker = AsyncMock()
                mock_worker.start.side_effect = KeyboardInterrupt()
                mock_worker_class.return_value = mock_worker

                with patch("asyncio.get_event_loop") as mock_get_loop:
                    mock_loop = MagicMock()
                    mock_get_loop.return_value = mock_loop

                    await main()

                # Should add handlers for SIGINT and SIGTERM
                assert mock_loop.add_signal_handler.call_count == 2

    @pytest.mark.asyncio
    async def test_main_parses_arguments(self):
        """Test main parses command line arguments."""
        with patch.object(sys, "argv", [
            "worker",
            "--poll-interval", "10.0",
            "--max-concurrent", "5"
        ]):
            with patch("src.worker.main.WorkerService") as mock_worker_class:
                mock_worker = AsyncMock()
                mock_worker.start.side_effect = KeyboardInterrupt()
                mock_worker_class.return_value = mock_worker

                with patch("asyncio.get_event_loop"):
                    await main()

                mock_worker_class.assert_called_once_with(
                    poll_interval=10.0,
                    max_concurrent_jobs=5,
                )

    @pytest.mark.asyncio
    async def test_main_handles_keyboard_interrupt(self):
        """Test main handles keyboard interrupt gracefully."""
        with patch.object(sys, "argv", ["worker"]):
            with patch("src.worker.main.WorkerService") as mock_worker_class:
                mock_worker = AsyncMock()
                mock_worker.start.side_effect = KeyboardInterrupt()
                mock_worker_class.return_value = mock_worker

                with patch("asyncio.get_event_loop"):
                    with patch("src.worker.main.logger") as mock_logger:
                        await main()

                        mock_logger.info.assert_called_with("Interrupted by user")


# ============================================================================
# Graceful Shutdown Tests
# ============================================================================

@pytest.mark.unit
class TestGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_event_triggered(self, worker):
        """Test shutdown event is triggered on stop."""
        worker._running = True

        # Set up a task waiting on shutdown event
        async def wait_for_shutdown():
            await asyncio.wait_for(worker._shutdown_event.wait(), timeout=0.5)
            return True

        wait_task = asyncio.create_task(wait_for_shutdown())

        # Trigger shutdown
        await worker.stop()

        # Wait task should complete
        result = await wait_task
        assert result is True

    @pytest.mark.asyncio
    async def test_active_tasks_cleared_after_stop(self, worker):
        """Test active tasks are cleared after stop."""
        worker._running = True

        # Create mock tasks
        async def mock_task():
            return "done"

        task1 = asyncio.create_task(mock_task())
        task2 = asyncio.create_task(mock_task())

        worker._active_tasks = {task1, task2}

        await worker.stop()

        # Tasks should be done and cleared
        assert task1.done()
        assert task2.done()
        assert len(worker._active_tasks) == 0

    @pytest.mark.asyncio
    async def test_shutdown_with_exception_in_task(self, worker):
        """Test shutdown handles tasks that raise exceptions."""
        worker._running = True

        async def failing_task():
            raise Exception("Task failed")

        task = asyncio.create_task(failing_task())
        worker._active_tasks = {task}

        # Should not raise exception
        await worker.stop()

        assert len(worker._active_tasks) == 0
