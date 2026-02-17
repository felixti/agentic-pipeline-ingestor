"""Retry strategies for failed pipeline jobs.

This module implements the 4 retry strategies from the specification:
1. SameParserRetry - Retry with same parser (transient failures)
2. FallbackParserRetry - Switch to fallback parser
3. PreprocessRetry - Enhance image quality then retry
4. SplitProcessingRetry - Process large documents in chunks
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.api.models import Job, PipelineConfig, RetryRecord

logger = logging.getLogger(__name__)


class RetryStrategyType(str, Enum):
    """Types of retry strategies."""
    SAME_PARSER = "same_parser"
    FALLBACK_PARSER = "fallback_parser"
    PREPROCESS_THEN_RETRY = "preprocess_then_retry"
    SPLIT_PROCESSING = "split_processing"


class RetryStatus(str, Enum):
    """Status of a retry attempt."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetryContext:
    """Context for retry execution.
    
    Attributes:
        job: The job being retried
        attempt_number: Current retry attempt number (1-based)
        previous_error: The error that caused the previous failure
        previous_strategy: Strategy used in previous attempt
        retry_history: History of all retry attempts
        pipeline_config: Current pipeline configuration
    """
    job: Job
    attempt_number: int
    previous_error: Exception | None = None
    previous_strategy: str | None = None
    retry_history: list[RetryRecord] = field(default_factory=list)
    pipeline_config: PipelineConfig | None = None


@dataclass
class RetryResult:
    """Result of a retry attempt.
    
    Attributes:
        success: Whether the retry was successful
        status: Retry status
        message: Human-readable status message
        updated_config: Updated pipeline configuration for retry
        preprocessing_applied: Whether preprocessing was applied
        strategy_used: The retry strategy that was used
        error: Error message if retry failed
        metadata: Additional metadata about the retry
    """
    success: bool
    status: RetryStatus
    message: str
    updated_config: PipelineConfig | None = None
    preprocessing_applied: bool = False
    strategy_used: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RetryStrategy(ABC):
    """Abstract base class for retry strategies.
    
    All retry strategies must inherit from this class and implement
    the execute method.
    """

    def __init__(self) -> None:
        """Initialize the retry strategy."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def strategy_type(self) -> RetryStrategyType:
        """Return the strategy type."""
        ...

    @abstractmethod
    async def execute(self, context: RetryContext) -> RetryResult:
        """Execute the retry strategy.
        
        Args:
            context: Retry context containing job and history
            
        Returns:
            RetryResult indicating success/failure and any config changes
        """
        ...

    async def can_apply(self, context: RetryContext) -> bool:
        """Check if this strategy can be applied to the given context.
        
        Args:
            context: Retry context
            
        Returns:
            True if strategy can be applied
        """
        return True

    def _create_retry_record(
        self,
        context: RetryContext,
        success: bool,
        error: str | None = None,
    ) -> RetryRecord:
        """Create a retry record.
        
        Args:
            context: Retry context
            success: Whether the retry succeeded
            error: Optional error message
            
        Returns:
            RetryRecord
        """
        return RetryRecord(
            attempt=context.attempt_number,
            timestamp=datetime.utcnow(),
            strategy=self.strategy_type.value,
            error_code=type(context.previous_error).__name__ if context.previous_error else None,
            error_message=error or (str(context.previous_error) if context.previous_error else None),
        )


class SameParserRetry(RetryStrategy):
    """Retry with the same parser.
    
    Use this strategy for transient failures such as:
    - Network timeouts
    - Temporary service unavailability
    - Rate limiting (with backoff)
    - Memory pressure that has been resolved
    """

    @property
    def strategy_type(self) -> RetryStrategyType:
        return RetryStrategyType.SAME_PARSER

    async def can_apply(self, context: RetryContext) -> bool:
        """Check if same parser retry can be applied.
        
        Don't apply if we've already tried this strategy multiple times.
        """
        same_strategy_count = sum(
            1 for r in context.retry_history
            if r.strategy == self.strategy_type.value
        )
        # Limit same-parser retries to 2 attempts
        return same_strategy_count < 2

    async def execute(self, context: RetryContext) -> RetryResult:
        """Execute same parser retry.
        
        This strategy keeps the same configuration but adds a delay
        to allow transient issues to resolve.
        
        Args:
            context: Retry context
            
        Returns:
            RetryResult with updated config including delay
        """
        self.logger.info(
            "executing_same_parser_retry",
            job_id=str(context.job.id),
            attempt=context.attempt_number,
        )

        # Calculate exponential backoff delay
        base_delay = 5  # seconds
        delay = base_delay * (2 ** (context.attempt_number - 1))

        # Add jitter to prevent thundering herd
        import random
        jitter = random.uniform(0, 1)
        total_delay = delay + jitter

        # Get current config or create default
        updated_config = context.pipeline_config
        if updated_config is None:
            from src.api.models import PipelineConfig
            updated_config = PipelineConfig(name=f"retry_config_{context.job.id}")

        # Add retry delay to parser options
        if updated_config.parser:
            updated_config.parser.parser_options["retry_delay"] = total_delay
            updated_config.parser.parser_options["retry_attempt"] = context.attempt_number

        return RetryResult(
            success=True,
            status=RetryStatus.PENDING,
            message=f"Retrying with same parser after {total_delay:.1f}s delay",
            updated_config=updated_config,
            strategy_used=self.strategy_type.value,
            metadata={
                "delay_seconds": total_delay,
                "reason": "transient_failure_recovery",
            },
        )


class FallbackParserRetry(RetryStrategy):
    """Retry with a fallback parser.
    
    Use this strategy when:
    - Primary parser fails with parsing errors
    - Quality score is too low with primary parser
    - Document format is not well supported by primary parser
    """

    @property
    def strategy_type(self) -> RetryStrategyType:
        return RetryStrategyType.FALLBACK_PARSER

    async def can_apply(self, context: RetryContext) -> bool:
        """Check if fallback parser retry can be applied.
        
        Requires a fallback parser to be configured and not already tried.
        """
        if context.pipeline_config is None:
            return False

        # Check if fallback parser is configured
        fallback = context.pipeline_config.parser.fallback_parser
        if not fallback:
            return False

        # Check if we've already tried fallback
        fallback_attempts = [
            r for r in context.retry_history
            if r.strategy == self.strategy_type.value
        ]

        return len(fallback_attempts) == 0

    async def execute(self, context: RetryContext) -> RetryResult:
        """Execute fallback parser retry.
        
        Switches the primary parser to the fallback parser.
        
        Args:
            context: Retry context
            
        Returns:
            RetryResult with swapped parser configuration
        """
        self.logger.info(
            "executing_fallback_parser_retry",
            job_id=str(context.job.id),
            attempt=context.attempt_number,
        )

        if context.pipeline_config is None:
            return RetryResult(
                success=False,
                status=RetryStatus.FAILED,
                message="No pipeline configuration available",
                strategy_used=self.strategy_type.value,
                error="Missing pipeline configuration",
            )

        current_primary = context.pipeline_config.parser.primary_parser
        fallback = context.pipeline_config.parser.fallback_parser

        if not fallback:
            return RetryResult(
                success=False,
                status=RetryStatus.FAILED,
                message="No fallback parser configured",
                strategy_used=self.strategy_type.value,
                error="Fallback parser not configured",
            )

        # Create updated config with swapped parsers
        from copy import deepcopy
        updated_config = deepcopy(context.pipeline_config)

        # Swap primary and fallback
        updated_config.parser.primary_parser = fallback
        updated_config.parser.fallback_parser = current_primary

        # Add fallback flag to options
        updated_config.parser.parser_options["using_fallback"] = True
        updated_config.parser.parser_options["original_parser"] = current_primary

        return RetryResult(
            success=True,
            status=RetryStatus.PENDING,
            message=f"Switching from {current_primary} to fallback parser {fallback}",
            updated_config=updated_config,
            strategy_used=self.strategy_type.value,
            metadata={
                "original_parser": current_primary,
                "fallback_parser": fallback,
                "reason": "primary_parser_failed",
            },
        )


class PreprocessRetry(RetryStrategy):
    """Retry with preprocessing.
    
    Use this strategy when:
    - Document has poor image quality
    - OCR confidence is low
    - Document needs deskewing or denoising
    - Images need enhancement before parsing
    """

    @property
    def strategy_type(self) -> RetryStrategyType:
        return RetryStrategyType.PREPROCESS_THEN_RETRY

    async def can_apply(self, context: RetryContext) -> bool:
        """Check if preprocess retry can be applied.
        
        Only applies to image-based documents or PDFs with images.
        """
        # Check if previous error suggests image quality issues
        if context.previous_error:
            error_str = str(context.previous_error).lower()
            image_related = [
                "image", "ocr", "quality", "resolution",
                "blurry", "skew", "noise", "contrast",
            ]
            if any(keyword in error_str for keyword in image_related):
                return True

        # Check if document type suggests preprocessing might help
        mime_type = (context.job.mime_type or "").lower()
        return any(t in mime_type for t in ["pdf", "image", "tiff", "png", "jpg"])

    async def execute(self, context: RetryContext) -> RetryResult:
        """Execute preprocess and retry.
        
        Adds preprocessing steps to the pipeline configuration.
        
        Args:
            context: Retry context
            
        Returns:
            RetryResult with preprocessing configuration
        """
        self.logger.info(
            "executing_preprocess_retry",
            job_id=str(context.job.id),
            attempt=context.attempt_number,
        )

        from copy import deepcopy
        updated_config = deepcopy(context.pipeline_config) if context.pipeline_config else None

        if updated_config is None:
            from src.api.models import ParserConfig, PipelineConfig
            updated_config = PipelineConfig(
                name=f"preprocess_config_{context.job.id}",
                parser=ParserConfig(),
            )

        # Define preprocessing steps
        preprocessing_steps = [
            "deskew",           # Correct page rotation
            "denoise",          # Remove noise
            "contrast_enhance", # Enhance contrast
            "binarize",         # Convert to black and white
        ]

        # Add preprocessing to parser options
        updated_config.parser.parser_options["preprocessing_steps"] = preprocessing_steps
        updated_config.parser.parser_options["preprocessing_enabled"] = True
        updated_config.parser.parser_options["target_dpi"] = 300  # Ensure high resolution

        return RetryResult(
            success=True,
            status=RetryStatus.PENDING,
            message=f"Applying preprocessing: {', '.join(preprocessing_steps)}",
            updated_config=updated_config,
            preprocessing_applied=True,
            strategy_used=self.strategy_type.value,
            metadata={
                "preprocessing_steps": preprocessing_steps,
                "target_dpi": 300,
                "reason": "image_quality_enhancement",
            },
        )


class SplitProcessingRetry(RetryStrategy):
    """Retry with split processing.
    
    Use this strategy when:
    - Document is too large for single-pass processing
    - Memory errors occurred
    - Timeout due to document size
    - Processing fails on specific pages only
    """

    # Size threshold for considering split processing (100 MB)
    SIZE_THRESHOLD = 100 * 1024 * 1024
    # Page threshold for PDFs
    PAGE_THRESHOLD = 100

    @property
    def strategy_type(self) -> RetryStrategyType:
        return RetryStrategyType.SPLIT_PROCESSING

    async def can_apply(self, context: RetryContext) -> bool:
        """Check if split processing can be applied.
        
        Applies to large files or when memory/resource errors occurred.
        """
        # Check for resource-related errors
        if context.previous_error:
            error_str = str(context.previous_error).lower()
            resource_errors = [
                "memory", "timeout", "too large", "size exceeded",
                "resource", "overflow", "chunk", "split",
            ]
            if any(keyword in error_str for keyword in resource_errors):
                return True

        # Check file size
        if context.job.file_size and context.job.file_size > self.SIZE_THRESHOLD:
            return True

        return False

    async def execute(self, context: RetryContext) -> RetryResult:
        """Execute split processing retry.
        
        Configures the parser to process the document in chunks/pages.
        
        Args:
            context: Retry context
            
        Returns:
            RetryResult with split processing configuration
        """
        self.logger.info(
            "executing_split_processing_retry",
            job_id=str(context.job.id),
            attempt=context.attempt_number,
            file_size=context.job.file_size,
        )

        from copy import deepcopy
        updated_config = deepcopy(context.pipeline_config) if context.pipeline_config else None

        if updated_config is None:
            from src.api.models import ParserConfig, PipelineConfig
            updated_config = PipelineConfig(
                name=f"split_config_{context.job.id}",
                parser=ParserConfig(),
            )

        # Determine chunk size based on file size
        chunk_size = self._calculate_chunk_size(context.job.file_size)

        # Configure split processing
        updated_config.parser.parser_options["split_processing"] = True
        updated_config.parser.parser_options["chunk_size"] = chunk_size
        updated_config.parser.parser_options["chunk_overlap"] = 100  # Characters overlap
        updated_config.parser.parser_options["max_pages_per_chunk"] = 10
        updated_config.parser.parser_options["parallel_chunks"] = 2  # Process 2 chunks at a time
        updated_config.parser.parser_options["merge_results"] = True

        return RetryResult(
            success=True,
            status=RetryStatus.PENDING,
            message=f"Processing in chunks of {chunk_size} bytes/pages",
            updated_config=updated_config,
            strategy_used=self.strategy_type.value,
            metadata={
                "chunk_size": chunk_size,
                "chunk_overlap": 100,
                "max_pages_per_chunk": 10,
                "parallel_chunks": 2,
                "reason": "large_document_handling",
                "file_size": context.job.file_size,
            },
        )

    def _calculate_chunk_size(self, file_size: int | None) -> int:
        """Calculate appropriate chunk size based on file size.
        
        Args:
            file_size: File size in bytes
            
        Returns:
            Chunk size in bytes
        """
        if file_size is None:
            return 10 * 1024 * 1024  # 10 MB default

        # For files > 500 MB, use 50 MB chunks
        if file_size > 500 * 1024 * 1024:
            return 50 * 1024 * 1024
        # For files > 100 MB, use 20 MB chunks
        elif file_size > 100 * 1024 * 1024:
            return 20 * 1024 * 1024
        # For smaller files, use 10 MB chunks
        else:
            return 10 * 1024 * 1024


class RetryStrategyRegistry:
    """Registry for retry strategies.
    
    Manages available retry strategies and selects appropriate
    strategies based on context.
    """

    def __init__(self) -> None:
        """Initialize the retry strategy registry."""
        self._strategies: dict[RetryStrategyType, RetryStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register the default retry strategies."""
        self.register(SameParserRetry())
        self.register(FallbackParserRetry())
        self.register(PreprocessRetry())
        self.register(SplitProcessingRetry())

    def register(self, strategy: RetryStrategy) -> None:
        """Register a retry strategy.
        
        Args:
            strategy: Strategy to register
        """
        self._strategies[strategy.strategy_type] = strategy
        logger.debug(f"Registered retry strategy: {strategy.strategy_type.value}")

    def get_strategy(self, strategy_type: RetryStrategyType) -> RetryStrategy | None:
        """Get a retry strategy by type.
        
        Args:
            strategy_type: Type of strategy to get
            
        Returns:
            RetryStrategy or None if not found
        """
        return self._strategies.get(strategy_type)

    async def select_strategy(
        self,
        context: RetryContext,
        preferred_order: list[RetryStrategyType] | None = None,
    ) -> RetryStrategy | None:
        """Select the best retry strategy for the context.
        
        Args:
            context: Retry context
            preferred_order: Optional preferred order of strategies
            
        Returns:
            Best applicable strategy or None
        """
        order = preferred_order or [
            RetryStrategyType.SAME_PARSER,
            RetryStrategyType.FALLBACK_PARSER,
            RetryStrategyType.PREPROCESS_THEN_RETRY,
            RetryStrategyType.SPLIT_PROCESSING,
        ]

        for strategy_type in order:
            strategy = self._strategies.get(strategy_type)
            if strategy and await strategy.can_apply(context):
                logger.info(
                    "selected_retry_strategy",
                    job_id=str(context.job.id),
                    strategy=strategy_type.value,
                    attempt=context.attempt_number,
                )
                return strategy

        logger.warning(
            "no_applicable_retry_strategy",
            job_id=str(context.job.id),
            attempt=context.attempt_number,
        )
        return None

    def list_strategies(self) -> list[RetryStrategyType]:
        """List all registered strategy types.
        
        Returns:
            List of strategy types
        """
        return list(self._strategies.keys())


# Global registry instance
_retry_registry: RetryStrategyRegistry | None = None


def get_retry_registry() -> RetryStrategyRegistry:
    """Get the global retry strategy registry.
    
    Returns:
        RetryStrategyRegistry instance
    """
    global _retry_registry
    if _retry_registry is None:
        _retry_registry = RetryStrategyRegistry()
    return _retry_registry
