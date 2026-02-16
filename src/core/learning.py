"""Historical learning system for processing optimization.

This module tracks processing patterns, success/failure rates,
and provides ML-based recommendations for optimal configurations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from src.api.models import Job, PipelineConfig
from src.core.quality import QualityScore

logger = logging.getLogger(__name__)


class ProcessingOutcome(str, Enum):
    """Outcome of a processing attempt."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RETRY_SUCCESS = "retry_success"
    FAILED = "failed"
    DLQ = "dlq"


@dataclass
class ProcessingRecord:
    """Record of a processing attempt.
    
    Attributes:
        id: Unique record ID
        job_id: Job ID
        timestamp: When processing occurred
        file_type: File type/extension
        content_type: Detected content type
        parser_used: Parser that was used
        fallback_used: Whether fallback parser was used
        quality_score: Quality assessment score
        processing_time_ms: Time taken to process
        outcome: Processing outcome
        retry_count: Number of retries required
        file_size_bytes: Size of the file
        page_count: Number of pages (for documents)
        config_used: Configuration used
        error_type: Type of error if failed
        metadata: Additional metadata
    """
    id: UUID
    job_id: UUID
    timestamp: datetime
    file_type: str
    content_type: str
    parser_used: str
    fallback_used: bool
    quality_score: float
    processing_time_ms: int
    outcome: ProcessingOutcome
    retry_count: int
    file_size_bytes: int
    page_count: Optional[int] = None
    config_used: Optional[PipelineConfig] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "id": str(self.id),
            "job_id": str(self.job_id),
            "timestamp": self.timestamp.isoformat(),
            "file_type": self.file_type,
            "content_type": self.content_type,
            "parser_used": self.parser_used,
            "fallback_used": self.fallback_used,
            "quality_score": self.quality_score,
            "processing_time_ms": self.processing_time_ms,
            "outcome": self.outcome.value,
            "retry_count": self.retry_count,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "error_type": self.error_type,
        }


@dataclass
class ParserPerformance:
    """Performance metrics for a parser.
    
    Attributes:
        parser_id: Parser identifier
        total_attempts: Total processing attempts
        success_count: Successful processing count
        failure_count: Failed processing count
        avg_quality_score: Average quality score
        avg_processing_time_ms: Average processing time
        success_rate: Success rate (0-1)
        best_for_types: Content types where parser performs best
    """
    parser_id: str
    total_attempts: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_quality_score: float = 0.0
    avg_processing_time_ms: float = 0.0
    success_rate: float = 0.0
    best_for_types: List[str] = field(default_factory=list)
    
    def update(self, success: bool, quality_score: float, processing_time_ms: int) -> None:
        """Update metrics with new data point.
        
        Args:
            success: Whether processing was successful
            quality_score: Quality score achieved
            processing_time_ms: Time taken
        """
        self.total_attempts += 1
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Update rolling averages
        self.avg_quality_score = (
            (self.avg_quality_score * (self.total_attempts - 1) + quality_score)
            / self.total_attempts
        )
        self.avg_processing_time_ms = (
            (self.avg_processing_time_ms * (self.total_attempts - 1) + processing_time_ms)
            / self.total_attempts
        )
        
        self.success_rate = self.success_count / self.total_attempts if self.total_attempts > 0 else 0.0


@dataclass
class ContentTypePattern:
    """Processing patterns for a content type.
    
    Attributes:
        content_type: Content type identifier
        total_count: Total jobs processed
        optimal_parser: Parser with best success rate
        optimal_config: Recommended configuration
        common_issues: Common issues encountered
        avg_quality_score: Average quality score
        avg_processing_time_ms: Average processing time
    """
    content_type: str
    total_count: int = 0
    optimal_parser: str = ""
    optimal_config: Optional[PipelineConfig] = None
    common_issues: List[str] = field(default_factory=list)
    avg_quality_score: float = 0.0
    avg_processing_time_ms: float = 0.0


class ProcessingHistory:
    """Historical processing data store.
    
    Records processing outcomes and provides analytics
    for optimizing future processing decisions.
    """
    
    def __init__(self, max_records: int = 10000) -> None:
        """Initialize the processing history.
        
        Args:
            max_records: Maximum number of records to keep
        """
        self.logger = logger
        self.max_records = max_records
        self._records: List[ProcessingRecord] = []
        self._parser_performance: Dict[str, ParserPerformance] = {}
        self._content_patterns: Dict[str, ContentTypePattern] = {}
    
    async def record_success(
        self,
        job: Job,
        result: Any,
        parser_used: str,
        quality_score: float,
        processing_time_ms: int,
        fallback_used: bool = False,
    ) -> ProcessingRecord:
        """Record a successful processing.
        
        Args:
            job: The job that was processed
            result: Processing result
            parser_used: Parser that was used
            quality_score: Quality score achieved
            processing_time_ms: Time taken
            fallback_used: Whether fallback parser was used
            
        Returns:
            Created processing record
        """
        from uuid import uuid4
        
        # Determine outcome
        outcome = ProcessingOutcome.RETRY_SUCCESS if job.retry_count > 0 else ProcessingOutcome.SUCCESS
        
        record = ProcessingRecord(
            id=uuid4(),
            job_id=job.id,
            timestamp=datetime.utcnow(),
            file_type=self._get_file_extension(job.file_name),
            content_type=result.get("detected_type", "unknown") if isinstance(result, dict) else "unknown",
            parser_used=parser_used,
            fallback_used=fallback_used,
            quality_score=quality_score,
            processing_time_ms=processing_time_ms,
            outcome=outcome,
            retry_count=job.retry_count,
            file_size_bytes=job.file_size or 0,
            config_used=job.pipeline_config,
        )
        
        await self._add_record(record)
        
        self.logger.debug(
            "recorded_success",
            job_id=str(job.id),
            parser=parser_used,
            quality_score=quality_score,
        )
        
        return record
    
    async def record_failure(
        self,
        job: Job,
        error: Exception,
        parser_used: str,
        quality_score: float = 0.0,
        processing_time_ms: int = 0,
        is_dlq: bool = False,
    ) -> ProcessingRecord:
        """Record a failed processing.
        
        Args:
            job: The job that failed
            error: The error that occurred
            parser_used: Parser that was used
            quality_score: Quality score (may be 0 for failures)
            processing_time_ms: Time taken before failure
            is_dlq: Whether job went to DLQ
            
        Returns:
            Created processing record
        """
        from uuid import uuid4
        
        outcome = ProcessingOutcome.DLQ if is_dlq else ProcessingOutcome.FAILED
        
        record = ProcessingRecord(
            id=uuid4(),
            job_id=job.id,
            timestamp=datetime.utcnow(),
            file_type=self._get_file_extension(job.file_name),
            content_type="unknown",
            parser_used=parser_used,
            fallback_used=False,
            quality_score=quality_score,
            processing_time_ms=processing_time_ms,
            outcome=outcome,
            retry_count=job.retry_count,
            file_size_bytes=job.file_size or 0,
            error_type=type(error).__name__,
        )
        
        await self._add_record(record)
        
        self.logger.debug(
            "recorded_failure",
            job_id=str(job.id),
            parser=parser_used,
            error_type=type(error).__name__,
        )
        
        return record
    
    async def _add_record(self, record: ProcessingRecord) -> None:
        """Add a record and update derived metrics.
        
        Args:
            record: Record to add
        """
        self._records.append(record)
        
        # Trim if needed
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]
        
        # Update parser performance
        await self._update_parser_performance(record)
        
        # Update content patterns
        await self._update_content_patterns(record)
    
    async def _update_parser_performance(self, record: ProcessingRecord) -> None:
        """Update performance metrics for a parser.
        
        Args:
            record: Processing record
        """
        parser_id = record.parser_used
        
        if parser_id not in self._parser_performance:
            self._parser_performance[parser_id] = ParserPerformance(parser_id=parser_id)
        
        perf = self._parser_performance[parser_id]
        success = record.outcome in (ProcessingOutcome.SUCCESS, ProcessingOutcome.RETRY_SUCCESS)
        
        perf.update(
            success=success,
            quality_score=record.quality_score,
            processing_time_ms=record.processing_time_ms,
        )
    
    async def _update_content_patterns(self, record: ProcessingRecord) -> None:
        """Update patterns for a content type.
        
        Args:
            record: Processing record
        """
        content_type = record.content_type
        
        if content_type not in self._content_patterns:
            self._content_patterns[content_type] = ContentTypePattern(content_type=content_type)
        
        pattern = self._content_patterns[content_type]
        pattern.total_count += 1
        
        # Update averages
        n = pattern.total_count
        pattern.avg_quality_score = (
            (pattern.avg_quality_score * (n - 1) + record.quality_score) / n
        )
        pattern.avg_processing_time_ms = (
            (pattern.avg_processing_time_ms * (n - 1) + record.processing_time_ms) / n
        )
    
    def _get_file_extension(self, file_name: str) -> str:
        """Get file extension from file name.
        
        Args:
            file_name: File name
            
        Returns:
            File extension (lowercase)
        """
        if "." in file_name:
            return file_name.rsplit(".", 1)[-1].lower()
        return "unknown"
    
    async def get_optimal_config(
        self,
        file_type: str,
        content_type: str,
    ) -> Optional[PipelineConfig]:
        """Get ML-based optimal configuration recommendation.
        
        Args:
            file_type: File extension/type
            content_type: Detected or expected content type
            
        Returns:
            Recommended pipeline configuration or None
        """
        # Find patterns for this content type
        pattern = self._content_patterns.get(content_type)
        
        if not pattern or pattern.total_count < 5:
            # Not enough data, return default config
            return None
        
        # Find best parser for this content type
        best_parser = ""
        best_success_rate = 0.0
        
        for record in self._records:
            if record.content_type == content_type:
                parser_perf = self._parser_performance.get(record.parser_used)
                if parser_perf and parser_perf.success_rate > best_success_rate:
                    best_success_rate = parser_perf.success_rate
                    best_parser = record.parser_used
        
        if not best_parser:
            return None
        
        # Build recommended config
        from src.api.models import ParserConfig, QualityConfig
        
        # Adjust quality threshold based on historical performance
        recommended_quality_threshold = max(0.5, min(0.9, pattern.avg_quality_score * 0.9))
        
        config = PipelineConfig(
            name=f"ml_optimized_{content_type}",
            parser=ParserConfig(
                primary_parser=best_parser,
                fallback_parser="azure_ocr" if best_parser != "azure_ocr" else "docling",
            ),
            quality=QualityConfig(
                min_quality_score=round(recommended_quality_threshold, 2),
                max_retries=3 if best_success_rate < 0.95 else 2,
            ),
        )
        
        self.logger.info(
            "generated_optimal_config",
            content_type=content_type,
            parser=best_parser,
            quality_threshold=recommended_quality_threshold,
        )
        
        return config
    
    async def get_parser_recommendation(
        self,
        content_type: str,
        file_type: str,
    ) -> Tuple[str, float]:
        """Get recommended parser for content type.
        
        Args:
            content_type: Content type
            file_type: File extension
            
        Returns:
            Tuple of (recommended_parser, confidence)
        """
        # Find records for this content type
        relevant_records = [
            r for r in self._records
            if r.content_type == content_type or r.file_type == file_type
        ]
        
        if not relevant_records:
            # Default recommendation based on content type rules
            if content_type in ("scanned_pdf", "image"):
                return "azure_ocr", 0.7
            return "docling", 0.7
        
        # Calculate success rate per parser
        parser_stats: Dict[str, Dict[str, Any]] = {}
        
        for record in relevant_records:
            parser = record.parser_used
            if parser not in parser_stats:
                parser_stats[parser] = {"success": 0, "total": 0, "quality_sum": 0}
            
            parser_stats[parser]["total"] += 1
            parser_stats[parser]["quality_sum"] += record.quality_score
            
            if record.outcome in (ProcessingOutcome.SUCCESS, ProcessingOutcome.RETRY_SUCCESS):
                parser_stats[parser]["success"] += 1
        
        # Find best parser
        best_parser = ""
        best_score = 0.0
        
        for parser, stats in parser_stats.items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            avg_quality = stats["quality_sum"] / stats["total"] if stats["total"] > 0 else 0
            
            # Combined score: 70% success rate, 30% quality
            score = (success_rate * 0.7) + (avg_quality * 0.3)
            
            if score > best_score:
                best_score = score
                best_parser = parser
        
        # Calculate confidence based on sample size
        total_samples = sum(s["total"] for s in parser_stats.values())
        confidence = min(0.95, 0.5 + (total_samples / 100) * 0.45)
        
        return best_parser, confidence
    
    async def get_parser_performance_summary(self) -> Dict[str, ParserPerformance]:
        """Get performance summary for all parsers.
        
        Returns:
            Dictionary mapping parser IDs to performance metrics
        """
        return self._parser_performance.copy()
    
    async def get_content_type_stats(self) -> Dict[str, ContentTypePattern]:
        """Get statistics for all content types.
        
        Returns:
            Dictionary mapping content types to patterns
        """
        return self._content_patterns.copy()
    
    async def get_recent_failures(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> List[ProcessingRecord]:
        """Get recent failure records.
        
        Args:
            hours: Time window in hours
            limit: Maximum records to return
            
        Returns:
            List of failure records
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        failures = [
            r for r in self._records
            if r.timestamp > cutoff
            and r.outcome in (ProcessingOutcome.FAILED, ProcessingOutcome.DLQ)
        ]
        
        return sorted(failures, key=lambda r: r.timestamp, reverse=True)[:limit]
    
    async def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in failures.
        
        Returns:
            Analysis results
        """
        failures = [r for r in self._records if r.outcome == ProcessingOutcome.DLQ]
        
        if not failures:
            return {"message": "No failures to analyze"}
        
        # Analyze by parser
        parser_failures: Dict[str, int] = {}
        for r in failures:
            parser_failures[r.parser_used] = parser_failures.get(r.parser_used, 0) + 1
        
        # Analyze by content type
        content_failures: Dict[str, int] = {}
        for r in failures:
            content_failures[r.content_type] = content_failures.get(r.content_type, 0) + 1
        
        # Analyze by error type
        error_types: Dict[str, int] = {}
        for r in failures:
            if r.error_type:
                error_types[r.error_type] = error_types.get(r.error_type, 0) + 1
        
        return {
            "total_failures": len(failures),
            "parser_failure_distribution": parser_failures,
            "content_type_failure_distribution": content_failures,
            "error_type_distribution": error_types,
            "common_error_types": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5],
        }


# Global processing history instance
_history: Optional[ProcessingHistory] = None


def get_processing_history(max_records: int = 10000) -> ProcessingHistory:
    """Get the global processing history instance.
    
    Args:
        max_records: Maximum records to keep
        
    Returns:
        ProcessingHistory instance
    """
    global _history
    if _history is None:
        _history = ProcessingHistory(max_records=max_records)
    return _history
