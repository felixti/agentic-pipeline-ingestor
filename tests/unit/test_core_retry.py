"""Unit tests for retry strategies module."""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from src.api.models import Job, JobStatus, ParserConfig, PipelineConfig, ProcessingMode, RetryRecord, SourceType
from src.core.retry import (
    FallbackParserRetry,
    PreprocessRetry,
    RetryContext,
    RetryResult,
    RetryStatus,
    RetryStrategy,
    RetryStrategyRegistry,
    RetryStrategyType,
    SameParserRetry,
    SplitProcessingRetry,
    get_retry_registry,
)


@pytest.mark.unit
class TestRetryStrategyType:
    """Tests for RetryStrategyType enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert RetryStrategyType.SAME_PARSER.value == "same_parser"
        assert RetryStrategyType.FALLBACK_PARSER.value == "fallback_parser"
        assert RetryStrategyType.PREPROCESS_THEN_RETRY.value == "preprocess_then_retry"
        assert RetryStrategyType.SPLIT_PROCESSING.value == "split_processing"


@pytest.mark.unit
class TestRetryStatus:
    """Tests for RetryStatus enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert RetryStatus.PENDING.value == "pending"
        assert RetryStatus.IN_PROGRESS.value == "in_progress"
        assert RetryStatus.SUCCESS.value == "success"
        assert RetryStatus.FAILED.value == "failed"
        assert RetryStatus.CANCELLED.value == "cancelled"


@pytest.mark.unit
class TestRetryContext:
    """Tests for RetryContext dataclass."""

    def test_retry_context_creation(self):
        """Test creating a RetryContext instance."""
        job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/file.pdf",
            file_name="file.pdf",
            status=JobStatus.FAILED,
            created_at=datetime.utcnow(),
        )

        context = RetryContext(
            job=job,
            attempt_number=1,
            previous_error=Exception("Test error"),
            previous_strategy="same_parser",
        )

        assert context.job == job
        assert context.attempt_number == 1
        assert isinstance(context.previous_error, Exception)
        assert context.previous_strategy == "same_parser"
        assert context.retry_history == []
        assert context.pipeline_config is None

    def test_retry_context_defaults(self):
        """Test RetryContext with default values."""
        job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/file.pdf",
            file_name="file.pdf",
            status=JobStatus.FAILED,
            created_at=datetime.utcnow(),
        )

        context = RetryContext(job=job, attempt_number=2)

        assert context.previous_error is None
        assert context.previous_strategy is None
        assert context.retry_history == []


@pytest.mark.unit
class TestRetryResult:
    """Tests for RetryResult dataclass."""

    def test_retry_result_creation(self):
        """Test creating a RetryResult instance."""
        result = RetryResult(
            success=True,
            status=RetryStatus.SUCCESS,
            message="Retry successful",
            strategy_used="same_parser",
        )

        assert result.success is True
        assert result.status == RetryStatus.SUCCESS
        assert result.message == "Retry successful"
        assert result.strategy_used == "same_parser"
        assert result.updated_config is None
        assert result.preprocessing_applied is False
        assert result.error is None
        assert result.metadata == {}

    def test_retry_result_with_metadata(self):
        """Test RetryResult with metadata."""
        result = RetryResult(
            success=False,
            status=RetryStatus.FAILED,
            message="Retry failed",
            strategy_used="fallback_parser",
            error="Parser error",
            metadata={"attempt": 1, "reason": "timeout"},
        )

        assert result.success is False
        assert result.error == "Parser error"
        assert result.metadata["attempt"] == 1


@pytest.mark.unit
class TestSameParserRetry:
    """Tests for SameParserRetry strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a SameParserRetry instance."""
        return SameParserRetry()

    @pytest.fixture
    def base_job(self):
        """Create a base job fixture."""
        return Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/file.pdf",
            file_name="file.pdf",
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )

    def test_strategy_type(self, strategy):
        """Test that strategy type is correct."""
        assert strategy.strategy_type == RetryStrategyType.SAME_PARSER

    @pytest.mark.asyncio
    async def test_can_apply_first_attempt(self, strategy, base_job):
        """Test can_apply returns True on first attempt."""
        context = RetryContext(job=base_job, attempt_number=1, retry_history=[])
        assert await strategy.can_apply(context) is True

    @pytest.mark.asyncio
    async def test_can_apply_after_same_strategy_twice(self, strategy, base_job):
        """Test can_apply returns False after same strategy used twice."""
        history = [
            RetryRecord(attempt=1, timestamp=datetime.utcnow(), strategy="same_parser"),
            RetryRecord(attempt=2, timestamp=datetime.utcnow(), strategy="same_parser"),
        ]
        context = RetryContext(job=base_job, attempt_number=3, retry_history=history)
        assert await strategy.can_apply(context) is False

    @pytest.mark.asyncio
    async def test_can_apply_after_one_same_strategy(self, strategy, base_job):
        """Test can_apply returns True after same strategy used once."""
        history = [
            RetryRecord(attempt=1, timestamp=datetime.utcnow(), strategy="same_parser"),
        ]
        context = RetryContext(job=base_job, attempt_number=2, retry_history=history)
        assert await strategy.can_apply(context) is True

    @pytest.mark.asyncio
    async def test_execute_calculates_delay(self, strategy, base_job):
        """Test that execute calculates exponential backoff delay."""
        context = RetryContext(job=base_job, attempt_number=1)
        result = await strategy.execute(context)

        assert result.success is True
        assert result.status == RetryStatus.PENDING
        # Delay is in metadata or in parser options
        assert "delay_seconds" in result.metadata or "retry_delay" in str(result.updated_config)
        assert result.metadata["reason"] == "transient_failure_recovery"

    @pytest.mark.asyncio
    async def test_execute_delay_increases_with_attempt(self, strategy, base_job):
        """Test that delay increases with attempt number."""
        context1 = RetryContext(job=base_job, attempt_number=1)
        context2 = RetryContext(job=base_job, attempt_number=2)

        result1 = await strategy.execute(context1)
        result2 = await strategy.execute(context2)

        # Delay should roughly double (with some jitter)
        delay1 = result1.metadata["delay_seconds"]
        delay2 = result2.metadata["delay_seconds"]
        assert delay2 > delay1

    @pytest.mark.asyncio
    async def test_execute_updates_parser_options(self, strategy, base_job):
        """Test that parser options are updated with retry info."""
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            parser=ParserConfig(primary_parser="docling"),
        )
        context = RetryContext(
            job=base_job, attempt_number=2, pipeline_config=pipeline_config
        )
        result = await strategy.execute(context)

        assert result.updated_config is not None
        assert result.updated_config.parser.parser_options["retry_attempt"] == 2
        assert "retry_delay" in result.updated_config.parser.parser_options


@pytest.mark.unit
class TestFallbackParserRetry:
    """Tests for FallbackParserRetry strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a FallbackParserRetry instance."""
        return FallbackParserRetry()

    @pytest.fixture
    def base_job(self):
        """Create a base job fixture."""
        return Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/file.pdf",
            file_name="file.pdf",
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )

    def test_strategy_type(self, strategy):
        """Test that strategy type is correct."""
        assert strategy.strategy_type == RetryStrategyType.FALLBACK_PARSER

    @pytest.mark.asyncio
    async def test_can_apply_no_pipeline_config(self, strategy, base_job):
        """Test can_apply returns False when no pipeline config."""
        context = RetryContext(job=base_job, attempt_number=1)
        assert await strategy.can_apply(context) is False

    @pytest.mark.asyncio
    async def test_can_apply_no_fallback_parser(self, strategy, base_job):
        """Test can_apply returns False when no fallback parser configured."""
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            parser=ParserConfig(primary_parser="docling", fallback_parser=None),
        )
        context = RetryContext(
            job=base_job, attempt_number=1, pipeline_config=pipeline_config
        )
        assert await strategy.can_apply(context) is False

    @pytest.mark.asyncio
    async def test_can_apply_with_fallback_parser(self, strategy, base_job):
        """Test can_apply returns True when fallback parser configured."""
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            parser=ParserConfig(primary_parser="docling", fallback_parser="azure_ocr"),
        )
        context = RetryContext(
            job=base_job, attempt_number=1, pipeline_config=pipeline_config
        )
        assert await strategy.can_apply(context) is True

    @pytest.mark.asyncio
    async def test_can_apply_after_fallback_already_tried(self, strategy, base_job):
        """Test can_apply returns False after fallback already tried."""
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            parser=ParserConfig(primary_parser="docling", fallback_parser="azure_ocr"),
        )
        history = [
            RetryRecord(
                attempt=1, timestamp=datetime.utcnow(), strategy="fallback_parser"
            ),
        ]
        context = RetryContext(
            job=base_job,
            attempt_number=2,
            pipeline_config=pipeline_config,
            retry_history=history,
        )
        assert await strategy.can_apply(context) is False

    @pytest.mark.asyncio
    async def test_execute_swaps_parsers(self, strategy, base_job):
        """Test that execute swaps primary and fallback parsers."""
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            parser=ParserConfig(
                primary_parser="docling", fallback_parser="azure_ocr"
            ),
        )
        context = RetryContext(
            job=base_job, attempt_number=1, pipeline_config=pipeline_config
        )
        result = await strategy.execute(context)

        assert result.success is True
        assert result.updated_config is not None
        assert result.updated_config.parser.primary_parser == "azure_ocr"
        assert result.updated_config.parser.fallback_parser == "docling"
        assert result.updated_config.parser.parser_options["using_fallback"] is True

    @pytest.mark.asyncio
    async def test_execute_no_pipeline_config(self, strategy, base_job):
        """Test execute fails when no pipeline config."""
        context = RetryContext(job=base_job, attempt_number=1)
        result = await strategy.execute(context)

        assert result.success is False
        assert result.status == RetryStatus.FAILED
        assert "No pipeline configuration" in result.message

    @pytest.mark.asyncio
    async def test_execute_no_fallback_configured(self, strategy, base_job):
        """Test execute fails when no fallback configured."""
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            parser=ParserConfig(primary_parser="docling", fallback_parser=None),
        )
        context = RetryContext(
            job=base_job, attempt_number=1, pipeline_config=pipeline_config
        )
        result = await strategy.execute(context)

        assert result.success is False
        assert result.status == RetryStatus.FAILED
        assert "No fallback parser configured" in result.message


@pytest.mark.unit
class TestPreprocessRetry:
    """Tests for PreprocessRetry strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a PreprocessRetry instance."""
        return PreprocessRetry()

    @pytest.fixture
    def base_job(self):
        """Create a base job fixture."""
        return Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/file.pdf",
            file_name="file.pdf",
            mime_type="application/pdf",
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )

    def test_strategy_type(self, strategy):
        """Test that strategy type is correct."""
        assert strategy.strategy_type == RetryStrategyType.PREPROCESS_THEN_RETRY

    @pytest.mark.asyncio
    async def test_can_apply_image_related_error(self, strategy, base_job):
        """Test can_apply returns True for image-related errors."""
        context = RetryContext(
            job=base_job,
            attempt_number=1,
            previous_error=Exception("OCR quality too low"),
        )
        assert await strategy.can_apply(context) is True

    @pytest.mark.asyncio
    async def test_can_apply_image_mime_type(self, strategy, base_job):
        """Test can_apply returns True for image mime types."""
        job_with_image = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/image.png",
            file_name="image.png",
            mime_type="image/png",
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )
        context = RetryContext(job=job_with_image, attempt_number=1)
        assert await strategy.can_apply(context) is True

    @pytest.mark.asyncio
    async def test_can_apply_non_image_mime_type(self, strategy, base_job):
        """Test can_apply returns False for non-image mime types."""
        job_with_text = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/file.txt",
            file_name="file.txt",
            mime_type="text/plain",
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )
        context = RetryContext(job=job_with_text, attempt_number=1)
        assert await strategy.can_apply(context) is False

    @pytest.mark.asyncio
    async def test_execute_adds_preprocessing_steps(self, strategy, base_job):
        """Test that execute adds preprocessing steps to config."""
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            parser=ParserConfig(),
        )
        context = RetryContext(
            job=base_job, attempt_number=1, pipeline_config=pipeline_config
        )
        result = await strategy.execute(context)

        assert result.success is True
        assert result.preprocessing_applied is True
        assert result.updated_config is not None
        assert "preprocessing_steps" in result.updated_config.parser.parser_options
        assert result.updated_config.parser.parser_options["preprocessing_enabled"] is True
        assert result.updated_config.parser.parser_options["target_dpi"] == 300

    @pytest.mark.asyncio
    async def test_execute_creates_default_config(self, strategy, base_job):
        """Test that execute creates default config if none provided."""
        context = RetryContext(job=base_job, attempt_number=1)
        result = await strategy.execute(context)

        assert result.success is True
        assert result.updated_config is not None
        assert result.updated_config.name == f"preprocess_config_{base_job.id}"


@pytest.mark.unit
class TestSplitProcessingRetry:
    """Tests for SplitProcessingRetry strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a SplitProcessingRetry instance."""
        return SplitProcessingRetry()

    @pytest.fixture
    def base_job(self):
        """Create a base job fixture."""
        return Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/large_file.pdf",
            file_name="large_file.pdf",
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )

    def test_strategy_type(self, strategy):
        """Test that strategy type is correct."""
        assert strategy.strategy_type == RetryStrategyType.SPLIT_PROCESSING

    def test_size_threshold_constant(self, strategy):
        """Test that size threshold is 100MB."""
        assert strategy.SIZE_THRESHOLD == 100 * 1024 * 1024

    def test_page_threshold_constant(self, strategy):
        """Test that page threshold is 100."""
        assert strategy.PAGE_THRESHOLD == 100

    @pytest.mark.asyncio
    async def test_can_apply_memory_error(self, strategy, base_job):
        """Test can_apply returns True for memory errors."""
        context = RetryContext(
            job=base_job,
            attempt_number=1,
            previous_error=Exception("Memory error: out of memory"),
        )
        assert await strategy.can_apply(context) is True

    @pytest.mark.asyncio
    async def test_can_apply_timeout_error(self, strategy, base_job):
        """Test can_apply returns True for timeout errors."""
        context = RetryContext(
            job=base_job,
            attempt_number=1,
            previous_error=Exception("Timeout: processing took too long"),
        )
        assert await strategy.can_apply(context) is True

    @pytest.mark.asyncio
    async def test_can_apply_large_file(self, strategy, base_job):
        """Test can_apply returns True for large files."""
        large_job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/huge.pdf",
            file_name="huge.pdf",
            file_size=200 * 1024 * 1024,  # 200 MB
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )
        context = RetryContext(job=large_job, attempt_number=1)
        assert await strategy.can_apply(context) is True

    @pytest.mark.asyncio
    async def test_can_apply_small_file_no_error(self, strategy, base_job):
        """Test can_apply returns False for small files without errors."""
        small_job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/small.pdf",
            file_name="small.pdf",
            file_size=1024,  # 1 KB
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )
        context = RetryContext(job=small_job, attempt_number=1)
        assert await strategy.can_apply(context) is False

    @pytest.mark.asyncio
    async def test_execute_configures_split_processing(self, strategy, base_job):
        """Test that execute configures split processing options."""
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            parser=ParserConfig(),
        )
        context = RetryContext(
            job=base_job, attempt_number=1, pipeline_config=pipeline_config
        )
        result = await strategy.execute(context)

        assert result.success is True
        assert result.updated_config is not None
        options = result.updated_config.parser.parser_options
        assert options["split_processing"] is True
        assert "chunk_size" in options
        assert options["chunk_overlap"] == 100
        assert options["max_pages_per_chunk"] == 10
        assert options["parallel_chunks"] == 2

    def test_calculate_chunk_size_large_file(self, strategy):
        """Test chunk size calculation for large files."""
        # > 500 MB
        assert strategy._calculate_chunk_size(600 * 1024 * 1024) == 50 * 1024 * 1024

    def test_calculate_chunk_size_medium_file(self, strategy):
        """Test chunk size calculation for medium files."""
        # 100-500 MB
        assert strategy._calculate_chunk_size(200 * 1024 * 1024) == 20 * 1024 * 1024

    def test_calculate_chunk_size_small_file(self, strategy):
        """Test chunk size calculation for small files."""
        # < 100 MB
        assert strategy._calculate_chunk_size(50 * 1024 * 1024) == 10 * 1024 * 1024

    def test_calculate_chunk_size_none(self, strategy):
        """Test chunk size calculation for None file size."""
        assert strategy._calculate_chunk_size(None) == 10 * 1024 * 1024


@pytest.mark.unit
class TestRetryStrategyRegistry:
    """Tests for RetryStrategyRegistry."""

    def test_registry_initialization(self):
        """Test that registry initializes with default strategies."""
        registry = RetryStrategyRegistry()

        strategies = registry.list_strategies()
        assert RetryStrategyType.SAME_PARSER in strategies
        assert RetryStrategyType.FALLBACK_PARSER in strategies
        assert RetryStrategyType.PREPROCESS_THEN_RETRY in strategies
        assert RetryStrategyType.SPLIT_PROCESSING in strategies

    def test_get_strategy(self):
        """Test getting a strategy by type."""
        registry = RetryStrategyRegistry()

        strategy = registry.get_strategy(RetryStrategyType.SAME_PARSER)
        assert strategy is not None
        assert isinstance(strategy, SameParserRetry)

    def test_get_strategy_not_found(self):
        """Test getting a non-existent strategy returns None."""
        registry = RetryStrategyRegistry()

        # Create a new strategy type dynamically
        class FakeType:
            value = "fake"

        strategy = registry.get_strategy(FakeType())  # type: ignore
        assert strategy is None

    def test_register_strategy(self):
        """Test registering a custom strategy."""
        registry = RetryStrategyRegistry()

        # Create a mock strategy
        mock_strategy = MagicMock(spec=RetryStrategy)
        mock_strategy.strategy_type = RetryStrategyType.SAME_PARSER

        registry.register(mock_strategy)

        retrieved = registry.get_strategy(RetryStrategyType.SAME_PARSER)
        assert retrieved == mock_strategy

    @pytest.mark.asyncio
    async def test_select_strategy_default_order(self):
        """Test selecting strategy with default order."""
        registry = RetryStrategyRegistry()

        job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/file.pdf",
            file_name="file.pdf",
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )
        context = RetryContext(job=job, attempt_number=1)

        strategy = await registry.select_strategy(context)

        assert strategy is not None
        assert strategy.strategy_type == RetryStrategyType.SAME_PARSER

    @pytest.mark.asyncio
    async def test_select_strategy_preferred_order(self):
        """Test selecting strategy with custom preferred order."""
        registry = RetryStrategyRegistry()

        pipeline_config = PipelineConfig(
            name="test_pipeline",
            parser=ParserConfig(primary_parser="docling", fallback_parser="azure_ocr"),
        )
        job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/file.pdf",
            file_name="file.pdf",
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )
        context = RetryContext(
            job=job, attempt_number=1, pipeline_config=pipeline_config
        )

        strategy = await registry.select_strategy(
            context, preferred_order=[RetryStrategyType.FALLBACK_PARSER]
        )

        assert strategy is not None
        assert strategy.strategy_type == RetryStrategyType.FALLBACK_PARSER

    @pytest.mark.asyncio
    async def test_select_strategy_none_applicable(self):
        """Test selecting strategy when none are applicable."""
        registry = RetryStrategyRegistry()

        # Create a job with a small file that won't trigger split processing
        job = Job(
            id=uuid4(),
            source_type=SourceType.UPLOAD,
            source_uri="/test/small.txt",
            file_name="small.txt",
            file_size=100,
            mime_type="text/plain",
            status=JobStatus.RETRYING,
            created_at=datetime.utcnow(),
        )

        # Mock history to block same_parser (limit is 2 attempts)
        history = [
            RetryRecord(attempt=1, timestamp=datetime.utcnow(), strategy="same_parser"),
            RetryRecord(attempt=2, timestamp=datetime.utcnow(), strategy="same_parser"),
        ]
        context = RetryContext(job=job, attempt_number=3, retry_history=history)

        # This should return None since no strategy can be applied
        # (same_parser at limit, no fallback configured, preprocess doesn't apply to text, split doesn't apply to small files)
        # Patch the logger to avoid structlog issues with kwargs
        with patch('src.core.retry.logger'):
            strategy = await registry.select_strategy(context)

        # Should return None as preprocess won't apply to text files
        # and split_processing won't apply to small files
        assert strategy is None


@pytest.mark.unit
class TestGetRetryRegistry:
    """Tests for get_retry_registry function."""

    def test_get_retry_registry_singleton(self):
        """Test that get_retry_registry returns a singleton."""
        registry1 = get_retry_registry()
        registry2 = get_retry_registry()

        assert registry1 is registry2

    def test_get_retry_registry_returns_registry(self):
        """Test that get_retry_registry returns a RetryStrategyRegistry."""
        registry = get_retry_registry()

        assert isinstance(registry, RetryStrategyRegistry)
        assert len(registry.list_strategies()) == 4
