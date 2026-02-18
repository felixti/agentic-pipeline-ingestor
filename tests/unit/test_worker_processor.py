"""Unit tests for the worker processor module.

This module tests the JobProcessor class which handles pipeline job execution.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from src.api.models import Job, JobStatus
from src.core.engine import PipelineContext
from src.worker.processor import JobProcessor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_engine():
    """Create a mock orchestration engine."""
    engine = AsyncMock()
    engine.process_job = AsyncMock()
    engine.get_job = AsyncMock()
    engine.update_job_status = AsyncMock()
    return engine


@pytest.fixture
def mock_registry():
    """Create a mock plugin registry."""
    registry = MagicMock()
    registry._parsers = {}
    registry._destinations = {}
    return registry


@pytest.fixture
def sample_job():
    """Create a sample job for testing."""
    job = MagicMock(spec=Job)
    job.id = uuid4()
    job.status = JobStatus.COMPLETED
    job.created_at = datetime.utcnow()
    return job


@pytest.fixture
def sample_context():
    """Create a sample pipeline context."""
    context = MagicMock(spec=PipelineContext)
    context.stage_results = {"ingest": {}, "parse": {}}
    context.get_stage_result = Mock(return_value=None)
    return context


@pytest.fixture
def processor(mock_engine, mock_registry):
    """Create a JobProcessor instance with mocked dependencies."""
    return JobProcessor(engine=mock_engine, plugin_registry=mock_registry)


# ============================================================================
# JobProcessor Initialization Tests
# ============================================================================

@pytest.mark.unit
class TestJobProcessorInitialization:
    """Tests for JobProcessor initialization."""

    @pytest.mark.asyncio
    async def test_init_with_dependencies(self, mock_engine, mock_registry):
        """Test initialization with provided dependencies."""
        processor = JobProcessor(engine=mock_engine, plugin_registry=mock_registry)

        assert processor.engine is mock_engine
        assert processor.registry is mock_registry
        assert processor.llm is None
        assert processor._running is False
        assert processor._current_job is None

    @pytest.mark.asyncio
    async def test_init_without_dependencies(self):
        """Test initialization without dependencies creates defaults."""
        processor = JobProcessor()

        assert processor.engine is None
        assert processor.registry is not None
        assert processor.llm is None

    @pytest.mark.asyncio
    async def test_initialize_creates_engine_if_not_provided(self, mock_registry):
        """Test initialize creates engine when not provided."""
        with patch("src.worker.processor.OrchestrationEngine") as mock_engine_class:
            mock_engine_instance = AsyncMock()
            mock_engine_class.return_value = mock_engine_instance

            processor = JobProcessor(plugin_registry=mock_registry)

            with patch.object(processor, "_initialize_plugins", new=AsyncMock()):
                with patch.object(processor, "_initialize_llm", new=AsyncMock()):
                    await processor.initialize()

            mock_engine_class.assert_called_once()
            assert processor.engine is mock_engine_instance

    @pytest.mark.asyncio
    async def test_initialize_skips_engine_creation_if_provided(self, mock_engine, mock_registry):
        """Test initialize skips engine creation when provided."""
        with patch("src.worker.processor.OrchestrationEngine") as mock_engine_class:
            processor = JobProcessor(engine=mock_engine, plugin_registry=mock_registry)

            with patch.object(processor, "_initialize_plugins", new=AsyncMock()):
                with patch.object(processor, "_initialize_llm", new=AsyncMock()):
                    await processor.initialize()

            mock_engine_class.assert_not_called()


# ============================================================================
# Process Job Tests
# ============================================================================

@pytest.mark.unit
class TestProcessJob:
    """Tests for the process_job method."""

    @pytest.mark.asyncio
    async def test_process_job_success(self, processor, mock_engine, sample_job, sample_context):
        """Test successful job processing."""
        job_id = sample_job.id
        mock_engine.process_job.return_value = sample_context
        mock_engine.get_job.return_value = sample_job

        result = await processor.process_job(job_id)

        assert result["job_id"] == str(job_id)
        assert result["status"] == "completed"
        assert result["success"] is True
        assert "stages_completed" in result
        mock_engine.process_job.assert_called_once_with(job_id)
        mock_engine.get_job.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_process_job_not_initialized(self, processor):
        """Test processing job without initialization raises error."""
        processor.engine = None

        with pytest.raises(RuntimeError, match="Processor not initialized"):
            await processor.process_job(uuid4())

    @pytest.mark.asyncio
    async def test_process_job_sets_current_job(self, processor, mock_engine, sample_context):
        """Test that current_job is set during processing."""
        job_id = uuid4()
        mock_engine.process_job.return_value = sample_context
        mock_engine.get_job.return_value = None

        assert processor.current_job_id is None

        # Create a side effect to check current_job during processing
        async def check_current_job(*args):
            assert processor.current_job_id == job_id
            return sample_context

        mock_engine.process_job.side_effect = check_current_job

        await processor.process_job(job_id)

        assert processor.current_job_id is None  # Should be cleared after

    @pytest.mark.asyncio
    async def test_process_job_with_quality_score(self, processor, mock_engine, sample_job, sample_context):
        """Test processing job with quality score in result."""
        sample_context.get_stage_result.return_value = {"overall_score": 0.95}
        mock_engine.process_job.return_value = sample_context
        mock_engine.get_job.return_value = sample_job

        result = await processor.process_job(uuid4())

        assert result["quality_score"] == 0.95

    @pytest.mark.asyncio
    async def test_process_job_failed_status(self, processor, mock_engine, sample_context):
        """Test processing job that ends with failed status."""
        job_id = uuid4()
        failed_job = MagicMock(spec=Job)
        failed_job.id = job_id
        failed_job.status = JobStatus.FAILED

        mock_engine.process_job.return_value = sample_context
        mock_engine.get_job.return_value = failed_job

        result = await processor.process_job(job_id)

        assert result["status"] == "failed"
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_process_job_unknown_status(self, processor, mock_engine, sample_context):
        """Test processing job with unknown status (job not found)."""
        mock_engine.process_job.return_value = sample_context
        mock_engine.get_job.return_value = None

        result = await processor.process_job(uuid4())

        assert result["status"] == "unknown"
        assert result["success"] is False


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestErrorHandling:
    """Tests for error handling in job processing."""

    @pytest.mark.asyncio
    async def test_process_job_error_updates_status(self, processor, mock_engine):
        """Test that errors update job status to failed."""
        job_id = uuid4()
        error_message = "Pipeline execution failed"
        mock_engine.process_job.side_effect = Exception(error_message)

        with patch("src.worker.processor.logger"):
            result = await processor.process_job(job_id)

        assert result["status"] == "failed"
        assert result["success"] is False
        assert result["error"] == error_message
        mock_engine.update_job_status.assert_called_once()
        call_args = mock_engine.update_job_status.call_args
        assert call_args[0][0] == job_id
        assert call_args[0][1] == JobStatus.FAILED
        assert call_args[1]["error"]["code"] == "PROCESSING_ERROR"

    @pytest.mark.asyncio
    async def test_process_job_error_status_update_fails(self, processor, mock_engine):
        """Test handling when both processing and status update fail."""
        job_id = uuid4()
        mock_engine.process_job.side_effect = Exception("Processing error")
        mock_engine.update_job_status.side_effect = Exception("Update failed")

        with patch("src.worker.processor.logger"):
            result = await processor.process_job(job_id)

        # Should still return error result even if status update fails
        assert result["status"] == "failed"
        assert result["success"] is False
        assert "Processing error" in result["error"]

    @pytest.mark.asyncio
    async def test_process_job_clears_current_job_on_error(self, processor, mock_engine):
        """Test that current_job is cleared even on error."""
        job_id = uuid4()
        mock_engine.process_job.side_effect = Exception("Processing error")

        with patch("src.worker.processor.logger"):
            await processor.process_job(job_id)

        assert processor.current_job_id is None


# ============================================================================
# Process Job With Retry Tests
# ============================================================================

@pytest.mark.unit
class TestProcessJobWithRetry:
    """Tests for process_job_with_retry method."""

    @pytest.mark.asyncio
    async def test_retry_success_on_first_attempt(self, processor):
        """Test successful processing on first attempt."""
        job_id = uuid4()
        expected_result = {"job_id": str(job_id), "status": "completed", "success": True}

        with patch.object(processor, "process_job", new=AsyncMock(return_value=expected_result)):
            result = await processor.process_job_with_retry(job_id)

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, processor, mock_engine):
        """Test retry with exponential backoff."""
        job_id = uuid4()

        # First attempt fails, second succeeds
        failing_result = {"job_id": str(job_id), "status": "retrying", "success": False}
        success_result = {"job_id": str(job_id), "status": "completed", "success": True}

        with patch.object(processor, "process_job", new=AsyncMock(side_effect=[
            failing_result,
            success_result,
        ])):
            # Mock job status to indicate retrying
            mock_job = MagicMock()
            mock_job.status = JobStatus.RETRYING
            mock_engine.get_job.return_value = mock_job

            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                result = await processor.process_job_with_retry(job_id, max_retries=2)

        assert result["success"] is True
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_exhausted_returns_failure(self, processor, mock_engine):
        """Test that exhausted retries return failure result."""
        job_id = uuid4()
        failing_result = {"job_id": str(job_id), "status": "failed", "success": False}

        with patch.object(processor, "process_job", new=AsyncMock(return_value=failing_result)):
            mock_job = MagicMock()
            mock_job.status = JobStatus.FAILED
            mock_engine.get_job.return_value = mock_job

            with patch("asyncio.sleep", new=AsyncMock()):
                result = await processor.process_job_with_retry(job_id, max_retries=1)

        assert result["success"] is False
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_retry_with_exception(self, processor):
        """Test retry when process_job raises exception."""
        job_id = uuid4()

        with patch("src.worker.processor.logger"):
            with patch.object(processor, "process_job", new=AsyncMock(side_effect=Exception("Error"))):
                with patch("asyncio.sleep", new=AsyncMock()):
                    result = await processor.process_job_with_retry(job_id, max_retries=1)

        assert result["success"] is False
        assert "All retries failed" in result["error"]

    @pytest.mark.asyncio
    async def test_retry_all_attempts_fail(self, processor):
        """Test when all retry attempts fail."""
        job_id = uuid4()

        with patch("src.worker.processor.logger"):
            with patch.object(processor, "process_job", new=AsyncMock(side_effect=Exception("Error"))):
                with patch("asyncio.sleep", new=AsyncMock()):
                    result = await processor.process_job_with_retry(job_id, max_retries=2)

        assert result["success"] is False
        assert result["job_id"] == str(job_id)
        assert "All retries failed" in result["error"]


# ============================================================================
# Shutdown Tests
# ============================================================================

@pytest.mark.unit
class TestShutdown:
    """Tests for shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_running_false(self, processor):
        """Test shutdown sets running flag to False."""
        processor._running = True

        await processor.shutdown()

        assert processor._running is False

    @pytest.mark.asyncio
    async def test_shutdown_calls_plugin_shutdown(self, processor, mock_registry):
        """Test shutdown calls shutdown on all plugins."""
        parser_mock = AsyncMock()
        dest_mock = AsyncMock()

        mock_registry._parsers = {"parser1": parser_mock}
        mock_registry._destinations = {"dest1": dest_mock}

        await processor.shutdown()

        parser_mock.shutdown.assert_called_once()
        dest_mock.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_plugin_errors(self, processor, mock_registry):
        """Test shutdown handles errors from plugin shutdown."""
        parser_mock = AsyncMock()
        parser_mock.shutdown.side_effect = Exception("Shutdown error")

        mock_registry._parsers = {"parser1": parser_mock}
        mock_registry._destinations = {}

        # Should not raise exception
        await processor.shutdown()

        parser_mock.shutdown.assert_called_once()


# ============================================================================
# Property Tests
# ============================================================================

@pytest.mark.unit
class TestProperties:
    """Tests for JobProcessor properties."""

    def test_is_processing_true(self, processor):
        """Test is_processing returns True when processing."""
        processor._current_job = uuid4()
        assert processor.is_processing is True

    def test_is_processing_false(self, processor):
        """Test is_processing returns False when not processing."""
        processor._current_job = None
        assert processor.is_processing is False

    def test_current_job_id(self, processor):
        """Test current_job_id property."""
        job_id = uuid4()
        processor._current_job = job_id
        assert processor.current_job_id == job_id

    def test_current_job_id_none(self, processor):
        """Test current_job_id returns None when not processing."""
        processor._current_job = None
        assert processor.current_job_id is None


# ============================================================================
# Plugin Initialization Tests
# ============================================================================

@pytest.mark.unit
class TestPluginInitialization:
    """Tests for plugin initialization."""

    @pytest.mark.asyncio
    async def test_initialize_plugins_registers_parsers(self, processor):
        """Test plugin initialization registers parsers."""
        mock_docling = AsyncMock()
        mock_azure = AsyncMock()

        with patch("src.worker.processor.logger"):
            # Patch the imports where they are used (inside the method)
            with patch("src.plugins.parsers.docling_parser.DoclingParser", return_value=mock_docling):
                with patch("src.plugins.parsers.azure_ocr_parser.AzureOCRParser", return_value=mock_azure):
                    with patch("src.worker.processor.settings") as mock_settings:
                        mock_settings.AZURE_AI_VISION_ENDPOINT = "https://test"
                        mock_settings.AZURE_AI_VISION_API_KEY = "key"

                        await processor._initialize_plugins()

        mock_docling.initialize.assert_called_once_with({})
        mock_azure.initialize.assert_called_once()
        processor.registry.register_parser.assert_any_call(mock_docling)
        processor.registry.register_parser.assert_any_call(mock_azure)

    @pytest.mark.asyncio
    async def test_initialize_plugins_handles_parser_errors(self, processor):
        """Test plugin initialization handles parser registration errors."""
        with patch("src.worker.processor.logger"):
            # Patch the import where it's used (inside the method)
            with patch("src.plugins.parsers.docling_parser.DoclingParser", side_effect=Exception("Import error")):
                # Should not raise exception
                await processor._initialize_plugins()

    @pytest.mark.asyncio
    async def test_initialize_llm_success(self, processor):
        """Test successful LLM initialization."""
        mock_config = MagicMock()
        mock_config.routers = [{"model": "gpt-4"}]

        with patch("src.worker.processor.load_llm_config", return_value=mock_config):
            with patch("src.worker.processor.LLMProvider") as mock_llm_class:
                mock_llm_instance = MagicMock()
                mock_llm_class.return_value = mock_llm_instance

                await processor._initialize_llm()

                assert processor.llm is mock_llm_instance

    @pytest.mark.asyncio
    async def test_initialize_llm_no_routers(self, processor):
        """Test LLM initialization with no routers configured."""
        mock_config = MagicMock()
        mock_config.routers = []

        with patch("src.worker.processor.load_llm_config", return_value=mock_config):
            await processor._initialize_llm()

            assert processor.llm is None

    @pytest.mark.asyncio
    async def test_initialize_llm_error(self, processor):
        """Test LLM initialization handles errors gracefully."""
        with patch("src.worker.processor.load_llm_config", side_effect=Exception("Config error")):
            # Should not raise exception
            await processor._initialize_llm()

            assert processor.llm is None
