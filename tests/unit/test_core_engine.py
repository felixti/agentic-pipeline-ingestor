"""Unit tests for the core orchestration engine module.

Tests cover:
- OrchestrationEngine class
- Engine initialization
- Job submission
- Job execution flow
- Error handling
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from src.api.models import (
    Job,
    JobCreateRequest,
    JobError,
    JobResult,
    JobRetryRequest,
    JobStatus,
    ParserConfig,
    PipelineConfig,
    ProcessingMode,
    QualityConfig,
    RetryRecord,
    SourceType,
    StageProgress,
    StageStatus,
)
from src.core.dlq import DeadLetterQueue
from src.core.engine import (
    OrchestrationEngine,
    get_engine,
    set_engine,
)
from src.core.retry import RetryStrategyType
from src.core.routing import DestinationRouter
from src.llm.provider import LLMProvider
from src.plugins.registry import PluginRegistry


def utcnow():
    """Return UTC datetime (naive for compatibility with source code)."""
    return datetime.utcnow()


@pytest.mark.unit
class TestOrchestrationEngine:
    """Tests for OrchestrationEngine class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock plugin registry."""
        return MagicMock(spec=PluginRegistry)

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        return MagicMock(spec=LLMProvider)

    @pytest.fixture
    def mock_dlq(self):
        """Create a mock DLQ."""
        dlq = MagicMock(spec=DeadLetterQueue)
        dlq.enqueue = AsyncMock()
        return dlq

    @pytest.fixture
    def mock_router(self):
        """Create a mock destination router."""
        return MagicMock(spec=DestinationRouter)

    @pytest.fixture
    def engine(self, mock_registry, mock_llm, mock_dlq, mock_router):
        """Create an OrchestrationEngine instance with patched logger."""
        engine = OrchestrationEngine(
            plugin_registry=mock_registry,
            llm_provider=mock_llm,
            dlq=mock_dlq,
            router=mock_router,
        )
        # Patch the instance logger to avoid structlog issues
        engine.logger = MagicMock()
        return engine

    @pytest.fixture
    def sample_job_data(self):
        """Create sample job creation data."""
        return {
            "source_type": SourceType.UPLOAD,
            "source_uri": "/test/file.pdf",
            "file_name": "test.pdf",
            "file_size": 1024,
            "mime_type": "application/pdf",
            "mode": ProcessingMode.ASYNC,
            "priority": 5,
        }

    @pytest.fixture
    def sample_pipeline_config(self):
        """Create a sample pipeline configuration."""
        return PipelineConfig(
            name="test_pipeline",
            parser=ParserConfig(primary_parser="docling"),
            quality=QualityConfig(max_retries=3),
        )

    def test_engine_initialization_with_dependencies(self, mock_registry, mock_llm, mock_dlq, mock_router):
        """Test engine initialization with all dependencies."""
        engine = OrchestrationEngine(
            plugin_registry=mock_registry,
            llm_provider=mock_llm,
            dlq=mock_dlq,
            router=mock_router,
        )

        assert engine.registry == mock_registry
        assert engine.llm == mock_llm
        assert engine.dlq == mock_dlq
        assert engine.router == mock_router
        assert engine._active_jobs == {}
        assert engine._pipeline_executor is None

    def test_engine_initialization_defaults(self):
        """Test engine initialization with default dependencies."""
        with patch("src.core.engine.PluginRegistry") as mock_reg_class, \
             patch("src.core.engine.get_dlq") as mock_get_dlq, \
             patch("src.core.engine.get_router") as mock_get_router:

            mock_reg_class.return_value = MagicMock(spec=PluginRegistry)
            mock_get_dlq.return_value = MagicMock(spec=DeadLetterQueue)
            mock_get_router.return_value = MagicMock(spec=DestinationRouter)

            engine = OrchestrationEngine()

            assert engine.registry is not None
            assert engine.dlq is not None
            assert engine.router is not None

    @pytest.mark.asyncio
    async def test_create_job(self, engine, sample_job_data):
        """Test creating a new job."""
        job = await engine.create_job(sample_job_data)

        assert isinstance(job, Job)
        assert job.source_type == SourceType.UPLOAD
        assert job.source_uri == "/test/file.pdf"
        assert job.file_name == "test.pdf"
        assert job.status == JobStatus.CREATED
        assert job.id in engine._active_jobs
        assert engine._active_jobs[job.id] == job

    @pytest.mark.asyncio
    async def test_create_job_with_custom_id(self, engine, sample_job_data):
        """Test creating a job with custom ID."""
        custom_id = uuid4()
        sample_job_data["id"] = str(custom_id)

        job = await engine.create_job(sample_job_data)

        assert job.id == custom_id

    @pytest.mark.asyncio
    async def test_create_job_defaults(self, engine):
        """Test creating a job with minimal data."""
        data = {
            "source_type": SourceType.URL,
            "source_uri": "https://example.com/doc.pdf",
        }

        job = await engine.create_job(data)

        assert job.file_name == "unknown"
        assert job.priority == 5
        assert job.mode == ProcessingMode.ASYNC

    @pytest.mark.asyncio
    async def test_process_job_success(self, engine, sample_job_data, sample_pipeline_config):
        """Test successful job processing."""
        # Create job
        job = await engine.create_job(sample_job_data)
        job.pipeline_config = sample_pipeline_config
        job.status = JobStatus.QUEUED

        # Mock pipeline executor
        mock_executor = MagicMock()
        mock_context = MagicMock()
        mock_context.stage_results = {"parse": {"success": True}}
        mock_executor.execute = AsyncMock(return_value=mock_context)
        engine._pipeline_executor = mock_executor

        result = await engine.process_job(job.id)

        assert result == mock_context
        mock_executor.execute.assert_called_once_with(job, None)

    @pytest.mark.asyncio
    async def test_process_job_with_enabled_stages(self, engine, sample_job_data, sample_pipeline_config):
        """Test job processing with specific stages."""
        job = await engine.create_job(sample_job_data)
        job.pipeline_config = sample_pipeline_config
        job.status = JobStatus.QUEUED

        mock_executor = MagicMock()
        mock_context = MagicMock()
        mock_context.stage_results = {}
        mock_executor.execute = AsyncMock(return_value=mock_context)
        engine._pipeline_executor = mock_executor

        enabled_stages = ["ingest", "parse"]
        await engine.process_job(job.id, enabled_stages=enabled_stages)

        mock_executor.execute.assert_called_once_with(job, enabled_stages)

    @pytest.mark.asyncio
    async def test_process_job_not_found(self, engine):
        """Test processing a non-existent job."""
        with pytest.raises(ValueError, match="Job not found"):
            await engine.process_job(uuid4())

    @pytest.mark.asyncio
    async def test_process_job_failure(self, engine, sample_job_data, sample_pipeline_config):
        """Test job processing failure."""
        job = await engine.create_job(sample_job_data)
        job.pipeline_config = sample_pipeline_config
        job.current_stage = "parse"

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(side_effect=Exception("Processing failed"))
        engine._pipeline_executor = mock_executor

        with pytest.raises(Exception, match="Processing failed"):
            await engine.process_job(job.id)

    @pytest.mark.asyncio
    async def test_update_job_status(self, engine, sample_job_data):
        """Test updating job status."""
        job = await engine.create_job(sample_job_data)

        await engine.update_job_status(job.id, JobStatus.PROCESSING)

        assert job.status == JobStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_update_job_status_with_error(self, engine, sample_job_data):
        """Test updating job status with error."""
        job = await engine.create_job(sample_job_data)
        error = {
            "code": "PARSING_ERROR",
            "message": "Failed to parse",
        }

        await engine.update_job_status(job.id, JobStatus.FAILED, error)

        assert job.status == JobStatus.FAILED
        assert job.error is not None
        assert job.error.code == "PARSING_ERROR"

    @pytest.mark.asyncio
    async def test_update_job_status_not_found(self, engine):
        """Test updating status for non-existent job."""
        # Should not raise, just log
        await engine.update_job_status(uuid4(), JobStatus.FAILED)

    @pytest.mark.asyncio
    async def test_update_stage_progress(self, engine, sample_job_data):
        """Test updating stage progress for a job."""
        job = await engine.create_job(sample_job_data)
        progress = StageProgress(
            stage="parse",
            status=StageStatus.RUNNING,
            progress_percent=50,
        )

        await engine.update_stage_progress(job.id, "parse", progress)

        assert job.stage_progress["parse"] == progress

    @pytest.mark.asyncio
    async def test_update_stage_progress_not_found(self, engine):
        """Test updating progress for non-existent job."""
        progress = StageProgress(
            stage="parse",
            status=StageStatus.RUNNING,
            progress_percent=50,
        )

        # Should not raise, just log
        await engine.update_stage_progress(uuid4(), "parse", progress)

    @pytest.mark.asyncio
    async def test_retry_job_success(self, engine, sample_job_data):
        """Test retrying a failed job."""
        job = await engine.create_job(sample_job_data)
        job.status = JobStatus.FAILED
        job.retry_count = 1

        result = await engine.retry_job(job.id)

        assert result.status == JobStatus.QUEUED
        assert result.error is None
        assert result.retry_count == 2

    @pytest.mark.asyncio
    async def test_retry_job_not_found(self, engine):
        """Test retrying a non-existent job."""
        with pytest.raises(ValueError, match="Job not found"):
            await engine.retry_job(uuid4())

    @pytest.mark.asyncio
    async def test_retry_job_not_retryable(self, engine, sample_job_data):
        """Test retrying a job that cannot be retried."""
        job = await engine.create_job(sample_job_data)
        job.status = JobStatus.COMPLETED

        with pytest.raises(ValueError, match="cannot be retried"):
            await engine.retry_job(job.id)

    @pytest.mark.asyncio
    async def test_retry_job_with_strategy(self, engine, sample_job_data, sample_pipeline_config):
        """Test retrying with a specific strategy."""
        job = await engine.create_job(sample_job_data)
        job.status = JobStatus.FAILED
        job.pipeline_config = sample_pipeline_config

        # Mock the retry registry and strategy
        mock_strategy = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.updated_config = sample_pipeline_config
        mock_strategy.execute = AsyncMock(return_value=mock_result)

        with patch.object(engine.retry_registry, "get_strategy", return_value=mock_strategy):
            result = await engine.retry_job(job.id, strategy="fallback_parser")

        assert result.status == JobStatus.QUEUED

    @pytest.mark.asyncio
    async def test_retry_job_with_invalid_strategy(self, engine, sample_job_data):
        """Test retrying with an invalid strategy."""
        job = await engine.create_job(sample_job_data)
        job.status = JobStatus.FAILED

        # Should not raise, just log warning
        result = await engine.retry_job(job.id, strategy="invalid_strategy")

        assert result.status == JobStatus.QUEUED

    @pytest.mark.asyncio
    async def test_retry_job_with_request(self, engine, sample_job_data, sample_pipeline_config):
        """Test retrying with a retry request."""
        job = await engine.create_job(sample_job_data)
        job.status = JobStatus.FAILED
        job.pipeline_config = sample_pipeline_config

        retry_request = JobRetryRequest(
            force_parser="azure_ocr",
            updated_config=sample_pipeline_config,
        )

        result = await engine.retry_job(job.id, retry_request=retry_request)

        assert result.status == JobStatus.QUEUED
        assert result.pipeline_config.parser.primary_parser == "azure_ocr"

    @pytest.mark.asyncio
    async def test_move_job_to_dlq(self, engine, sample_job_data):
        """Test moving a job to DLQ."""
        job = await engine.create_job(sample_job_data)
        error = Exception("Processing failed")

        await engine.move_job_to_dlq(job.id, error)

        assert job.status == JobStatus.DEAD_LETTER
        engine.dlq.enqueue.assert_called_once()

    @pytest.mark.asyncio
    async def test_move_job_to_dlq_not_found(self, engine):
        """Test moving non-existent job to DLQ."""
        # Should not raise, just log warning
        await engine.move_job_to_dlq(uuid4(), Exception("Error"))

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, engine, sample_job_data):
        """Test cancelling a job."""
        job = await engine.create_job(sample_job_data)
        job.status = JobStatus.QUEUED

        result = await engine.cancel_job(job.id)

        assert result is True
        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self, engine):
        """Test cancelling a non-existent job."""
        result = await engine.cancel_job(uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_job_already_completed(self, engine, sample_job_data):
        """Test cancelling an already completed job."""
        job = await engine.create_job(sample_job_data)
        job.status = JobStatus.COMPLETED

        result = await engine.cancel_job(job.id)

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_job_already_cancelled(self, engine, sample_job_data):
        """Test cancelling an already cancelled job."""
        job = await engine.create_job(sample_job_data)
        job.status = JobStatus.CANCELLED

        result = await engine.cancel_job(job.id)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_job_result_completed(self, engine, sample_job_data):
        """Test getting result for completed job."""
        job = await engine.create_job(sample_job_data)
        job.status = JobStatus.COMPLETED
        job.result = JobResult(
            success=True,
            output_uri="s3://bucket/output.json",
            quality_score=0.85,
        )
        job.completed_at = utcnow()

        result = await engine.get_job_result(job.id)

        assert result is not None
        assert result["status"] == "completed"
        assert result["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_get_job_result_not_completed(self, engine, sample_job_data):
        """Test getting result for non-completed job."""
        job = await engine.create_job(sample_job_data)
        job.status = JobStatus.PROCESSING

        result = await engine.get_job_result(job.id)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_job_result_not_found(self, engine):
        """Test getting result for non-existent job."""
        result = await engine.get_job_result(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_job(self, engine, sample_job_data):
        """Test getting a job by ID."""
        job = await engine.create_job(sample_job_data)

        result = await engine.get_job(job.id)

        assert result == job

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, engine):
        """Test getting a non-existent job."""
        result = await engine.get_job(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_list_jobs(self, engine, sample_job_data):
        """Test listing all jobs."""
        job1 = await engine.create_job(sample_job_data)
        job2 = await engine.create_job({**sample_job_data, "source_uri": "/test/file2.pdf"})

        jobs = await engine.list_jobs()

        assert len(jobs) == 2
        assert job1 in jobs
        assert job2 in jobs

    @pytest.mark.asyncio
    async def test_list_jobs_with_status_filter(self, engine, sample_job_data):
        """Test listing jobs with status filter."""
        job1 = await engine.create_job(sample_job_data)
        job1.status = JobStatus.COMPLETED
        job2 = await engine.create_job({**sample_job_data, "source_uri": "/test/file2.pdf"})
        job2.status = JobStatus.FAILED

        jobs = await engine.list_jobs(status=JobStatus.COMPLETED)

        assert len(jobs) == 1
        assert jobs[0] == job1

    @pytest.mark.asyncio
    async def test_list_jobs_sorted(self, engine, sample_job_data):
        """Test that jobs are sorted by created_at descending."""
        job1 = await engine.create_job(sample_job_data)
        job1.created_at = utcnow() - timedelta(hours=1)
        job2 = await engine.create_job({**sample_job_data, "source_uri": "/test/file2.pdf"})

        jobs = await engine.list_jobs()

        assert jobs[0] == job2  # More recent first
        assert jobs[1] == job1

    @pytest.mark.asyncio
    async def test_list_jobs_pagination(self, engine, sample_job_data):
        """Test job listing pagination."""
        for i in range(5):
            await engine.create_job({**sample_job_data, "source_uri": f"/test/file{i}.pdf"})

        jobs = await engine.list_jobs(limit=2, offset=0)
        assert len(jobs) == 2

        jobs = await engine.list_jobs(limit=2, offset=2)
        assert len(jobs) == 2

        jobs = await engine.list_jobs(limit=2, offset=4)
        assert len(jobs) == 1

    @pytest.mark.asyncio
    async def test_delete_job(self, engine, sample_job_data):
        """Test deleting a job."""
        job = await engine.create_job(sample_job_data)

        result = await engine.delete_job(job.id)

        assert result is True
        assert job.id not in engine._active_jobs

    @pytest.mark.asyncio
    async def test_delete_job_not_found(self, engine):
        """Test deleting a non-existent job."""
        result = await engine.delete_job(uuid4())

        assert result is False

    def test_get_executor_creates_new(self, engine, sample_pipeline_config):
        """Test that _get_executor creates a new executor."""
        with patch("src.core.engine.PipelineExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor

            executor = engine._get_executor(sample_pipeline_config)

            assert executor == mock_executor
            assert engine._pipeline_executor == mock_executor
            mock_executor_class.assert_called_once()

    def test_get_executor_reuses_existing(self, engine, sample_pipeline_config):
        """Test that _get_executor reuses existing executor."""
        mock_executor = MagicMock()
        engine._pipeline_executor = mock_executor

        executor = engine._get_executor(sample_pipeline_config)

        assert executor == mock_executor


@pytest.mark.unit
class TestGlobalEngine:
    """Tests for global engine functions."""

    def test_get_engine_singleton(self):
        """Test that get_engine returns a singleton."""
        # Reset global instance
        import src.core.engine as engine_module
        engine_module._engine = None

        engine1 = get_engine()
        engine2 = get_engine()

        assert engine1 is engine2
        assert isinstance(engine1, OrchestrationEngine)

    def test_set_engine(self):
        """Test setting the global engine instance."""
        new_engine = OrchestrationEngine()

        set_engine(new_engine)

        assert get_engine() is new_engine

    def teardown_method(self):
        """Reset global engine after each test."""
        import src.core.engine as engine_module
        engine_module._engine = None
