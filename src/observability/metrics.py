"""Prometheus metrics for the Agentic Data Pipeline Ingestor.

This module provides comprehensive metrics collection for monitoring
pipeline performance, throughput, and health.
"""

import time
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Default registry
DEFAULT_REGISTRY = CollectorRegistry()

# =============================================================================
# Job Metrics
# =============================================================================

JOBS_CREATED = Counter(
    'ingestion_jobs_created_total',
    'Total number of ingestion jobs created',
    ['source_type', 'priority'],
    registry=DEFAULT_REGISTRY,
)

JOBS_COMPLETED = Counter(
    'ingestion_jobs_completed_total',
    'Total number of ingestion jobs completed',
    ['source_type', 'status'],
    registry=DEFAULT_REGISTRY,
)

JOBS_FAILED = Counter(
    'ingestion_jobs_failed_total',
    'Total number of ingestion jobs that failed',
    ['source_type', 'stage', 'error_type'],
    registry=DEFAULT_REGISTRY,
)

JOB_DURATION = Histogram(
    'ingestion_job_duration_seconds',
    'Time spent processing a job from start to completion',
    ['source_type', 'priority'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
    registry=DEFAULT_REGISTRY,
)

JOB_RETRIES = Counter(
    'ingestion_job_retries_total',
    'Total number of job retry attempts',
    ['source_type', 'stage'],
    registry=DEFAULT_REGISTRY,
)

JOBS_IN_PROGRESS = Gauge(
    'ingestion_jobs_in_progress',
    'Number of jobs currently being processed',
    ['source_type'],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# Stage Metrics
# =============================================================================

STAGE_DURATION = Histogram(
    'ingestion_stage_duration_seconds',
    'Time spent in each pipeline stage',
    ['stage_name'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
    registry=DEFAULT_REGISTRY,
)

STAGE_ERRORS = Counter(
    'ingestion_stage_errors_total',
    'Total number of errors in each pipeline stage',
    ['stage_name', 'error_type'],
    registry=DEFAULT_REGISTRY,
)

STAGE_ITEMS_PROCESSED = Counter(
    'ingestion_stage_items_processed_total',
    'Total number of items processed by each stage',
    ['stage_name'],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# Quality Metrics
# =============================================================================

QUALITY_SCORE = Gauge(
    'ingestion_quality_score',
    'Quality score for processed documents (0-1)',
    ['job_id', 'parser_used'],
    registry=DEFAULT_REGISTRY,
)

QUALITY_CHECKS_PASSED = Counter(
    'ingestion_quality_checks_passed_total',
    'Total number of quality checks passed',
    ['parser_used'],
    registry=DEFAULT_REGISTRY,
)

QUALITY_CHECKS_FAILED = Counter(
    'ingestion_quality_checks_failed_total',
    'Total number of quality checks failed',
    ['parser_used', 'failure_reason'],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# Parser Metrics
# =============================================================================

PARSER_ERRORS = Counter(
    'ingestion_parser_errors_total',
    'Total number of parser errors',
    ['parser_name', 'file_type', 'error_type'],
    registry=DEFAULT_REGISTRY,
)

PARSER_DURATION = Histogram(
    'ingestion_parser_duration_seconds',
    'Time spent parsing documents',
    ['parser_name', 'file_type'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
    registry=DEFAULT_REGISTRY,
)

PARSER_FALLBACK_USED = Counter(
    'ingestion_parser_fallback_used_total',
    'Total number of times fallback parser was used',
    ['primary_parser', 'fallback_parser', 'reason'],
    registry=DEFAULT_REGISTRY,
)

PARSER_SUCCESS = Counter(
    'ingestion_parser_success_total',
    'Total number of successful parsing operations',
    ['parser_name', 'file_type'],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# Destination Metrics
# =============================================================================

DESTINATION_ERRORS = Counter(
    'ingestion_destination_errors_total',
    'Total number of destination errors',
    ['destination_type', 'error_type'],
    registry=DEFAULT_REGISTRY,
)

DESTINATION_WRITES = Counter(
    'ingestion_destination_writes_total',
    'Total number of write operations to destinations',
    ['destination_type', 'status'],
    registry=DEFAULT_REGISTRY,
)

DESTINATION_DURATION = Histogram(
    'ingestion_destination_duration_seconds',
    'Time spent writing to destinations',
    ['destination_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30],
    registry=DEFAULT_REGISTRY,
)

DESTINATION_RECORDS_WRITTEN = Counter(
    'ingestion_destination_records_written_total',
    'Total number of records written to destinations',
    ['destination_type'],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# LLM Metrics
# =============================================================================

LLM_REQUESTS = Counter(
    'ingestion_llm_requests_total',
    'Total number of LLM requests made',
    ['model', 'operation', 'status'],
    registry=DEFAULT_REGISTRY,
)

LLM_LATENCY = Histogram(
    'ingestion_llm_latency_seconds',
    'LLM request latency',
    ['model', 'operation'],
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60],
    registry=DEFAULT_REGISTRY,
)

LLM_TOKENS_USED = Counter(
    'ingestion_llm_tokens_used_total',
    'Total number of tokens used in LLM requests',
    ['model', 'token_type'],
    registry=DEFAULT_REGISTRY,
)

LLM_FALLBACK_USED = Counter(
    'ingestion_llm_fallback_used_total',
    'Total number of times LLM fallback was used',
    ['primary_model', 'fallback_model', 'reason'],
    registry=DEFAULT_REGISTRY,
)

LLM_RATE_LIMITS_HIT = Counter(
    'ingestion_llm_rate_limits_hit_total',
    'Total number of rate limit errors from LLM providers',
    ['model', 'provider'],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# Queue Metrics
# =============================================================================

QUEUE_DEPTH = Gauge(
    'ingestion_queue_depth',
    'Current depth of processing queues',
    ['queue_name'],
    registry=DEFAULT_REGISTRY,
)

QUEUE_MESSAGES_PROCESSED = Counter(
    'ingestion_queue_messages_processed_total',
    'Total number of queue messages processed',
    ['queue_name', 'status'],
    registry=DEFAULT_REGISTRY,
)

QUEUE_PROCESSING_TIME = Histogram(
    'ingestion_queue_processing_duration_seconds',
    'Time spent processing queue messages',
    ['queue_name'],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# API Metrics
# =============================================================================

API_REQUESTS = Counter(
    'ingestion_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code'],
    registry=DEFAULT_REGISTRY,
)

API_REQUEST_DURATION = Histogram(
    'ingestion_api_request_duration_seconds',
    'API request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
    registry=DEFAULT_REGISTRY,
)

API_REQUEST_SIZE = Histogram(
    'ingestion_api_request_size_bytes',
    'API request body size',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000, 10000000, 100000000],
    registry=DEFAULT_REGISTRY,
)

API_RESPONSE_SIZE = Histogram(
    'ingestion_api_response_size_bytes',
    'API response body size',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
    registry=DEFAULT_REGISTRY,
)

API_RATE_LIMITS_HIT = Counter(
    'ingestion_api_rate_limits_hit_total',
    'Total number of API rate limits hit',
    ['endpoint'],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# Database Metrics
# =============================================================================

DB_CONNECTIONS = Gauge(
    'ingestion_db_connections',
    'Current number of database connections',
    ['state'],
    registry=DEFAULT_REGISTRY,
)

DB_QUERY_DURATION = Histogram(
    'ingestion_db_query_duration_seconds',
    'Database query latency',
    ['operation', 'table'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5],
    registry=DEFAULT_REGISTRY,
)

DB_QUERY_ERRORS = Counter(
    'ingestion_db_query_errors_total',
    'Total number of database query errors',
    ['operation', 'table', 'error_type'],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# Storage Metrics
# =============================================================================

STORAGE_OPERATIONS = Counter(
    'ingestion_storage_operations_total',
    'Total number of storage operations',
    ['operation', 'storage_type', 'status'],
    registry=DEFAULT_REGISTRY,
)

STORAGE_OPERATION_DURATION = Histogram(
    'ingestion_storage_operation_duration_seconds',
    'Storage operation latency',
    ['operation', 'storage_type'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5],
    registry=DEFAULT_REGISTRY,
)

STORAGE_BYTES_TRANSFERRED = Counter(
    'ingestion_storage_bytes_transferred_total',
    'Total bytes transferred to/from storage',
    ['operation', 'storage_type'],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# System Metrics
# =============================================================================

SYSTEM_INFO = Info(
    'ingestion_system',
    'System information',
    registry=DEFAULT_REGISTRY,
)

CACHE_HITS = Counter(
    'ingestion_cache_hits_total',
    'Total number of cache hits',
    ['cache_name'],
    registry=DEFAULT_REGISTRY,
)

CACHE_MISSES = Counter(
    'ingestion_cache_misses_total',
    'Total number of cache misses',
    ['cache_name'],
    registry=DEFAULT_REGISTRY,
)

CACHE_OPERATION_DURATION = Histogram(
    'ingestion_cache_operation_duration_seconds',
    'Cache operation latency',
    ['operation', 'cache_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
    registry=DEFAULT_REGISTRY,
)

# =============================================================================
# Metrics Manager
# =============================================================================

class MetricsManager:
    """Manages metrics collection and reporting.
    
    This class provides a centralized way to track and report metrics
    across the pipeline system.
    
    Example:
        >>> manager = MetricsManager()
        >>> manager.record_job_created("upload", "high")
        >>> with manager.time_stage("parse"):
        ...     parse_document(file)
    """
    
    def __init__(self, registry: CollectorRegistry = DEFAULT_REGISTRY):
        """Initialize the metrics manager.
        
        Args:
            registry: Prometheus registry to use
        """
        self.registry = registry
        self._job_start_times: Dict[str, float] = {}
        self._stage_start_times: Dict[str, float] = {}
    
    def record_job_created(self, source_type: str, priority: str = "normal") -> None:
        """Record a job creation.
        
        Args:
            source_type: Type of source (upload, s3, url, etc.)
            priority: Job priority level
        """
        JOBS_CREATED.labels(source_type=source_type, priority=priority).inc()
        JOBS_IN_PROGRESS.labels(source_type=source_type).inc()
    
    def record_job_completed(
        self,
        source_type: str,
        status: str,
        job_id: Optional[str] = None,
    ) -> None:
        """Record a job completion.
        
        Args:
            source_type: Type of source
            status: Final job status (completed, failed, cancelled)
            job_id: Optional job ID for tracking duration
        """
        JOBS_COMPLETED.labels(source_type=source_type, status=status).inc()
        JOBS_IN_PROGRESS.labels(source_type=source_type).dec()
        
        # Record duration if we have a start time
        if job_id and job_id in self._job_start_times:
            duration = time.time() - self._job_start_times.pop(job_id)
            JOB_DURATION.labels(source_type=source_type, priority="normal").observe(duration)
    
    def record_job_failed(
        self,
        source_type: str,
        stage: str,
        error_type: str,
    ) -> None:
        """Record a job failure.
        
        Args:
            source_type: Type of source
            stage: Pipeline stage where failure occurred
            error_type: Type of error
        """
        JOBS_FAILED.labels(
            source_type=source_type,
            stage=stage,
            error_type=error_type,
        ).inc()
        JOBS_IN_PROGRESS.labels(source_type=source_type).dec()
    
    def record_job_start(self, job_id: str) -> None:
        """Record the start time of a job.
        
        Args:
            job_id: Unique job identifier
        """
        self._job_start_times[job_id] = time.time()
    
    @contextmanager
    def time_stage(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager to time a pipeline stage.
        
        Args:
            stage_name: Name of the stage being timed
            
        Example:
            >>> with manager.time_stage("parse"):
            ...     result = parser.parse(file)
        """
        start = time.time()
        try:
            yield
            STAGE_ITEMS_PROCESSED.labels(stage_name=stage_name).inc()
        except Exception as e:
            STAGE_ERRORS.labels(
                stage_name=stage_name,
                error_type=type(e).__name__,
            ).inc()
            raise
        finally:
            duration = time.time() - start
            STAGE_DURATION.labels(stage_name=stage_name).observe(duration)
    
    def record_parser_success(self, parser_name: str, file_type: str) -> None:
        """Record a successful parse operation.
        
        Args:
            parser_name: Name of the parser used
            file_type: Type of file parsed
        """
        PARSER_SUCCESS.labels(parser_name=parser_name, file_type=file_type).inc()
    
    def record_parser_error(
        self,
        parser_name: str,
        file_type: str,
        error_type: str,
    ) -> None:
        """Record a parser error.
        
        Args:
            parser_name: Name of the parser
            file_type: Type of file
            error_type: Type of error
        """
        PARSER_ERRORS.labels(
            parser_name=parser_name,
            file_type=file_type,
            error_type=error_type,
        ).inc()
    
    @contextmanager
    def time_parser(self, parser_name: str, file_type: str) -> Generator[None, None, None]:
        """Context manager to time a parser operation.
        
        Args:
            parser_name: Name of the parser
            file_type: Type of file
            
        Yields:
            None
        """
        start = time.time()
        try:
            yield
            self.record_parser_success(parser_name, file_type)
        except Exception as e:
            self.record_parser_error(parser_name, file_type, type(e).__name__)
            raise
        finally:
            duration = time.time() - start
            PARSER_DURATION.labels(
                parser_name=parser_name,
                file_type=file_type,
            ).observe(duration)
    
    def record_quality_score(
        self,
        job_id: str,
        parser_used: str,
        score: float,
    ) -> None:
        """Record a quality score.
        
        Args:
            job_id: Job identifier
            parser_used: Parser that processed the document
            score: Quality score (0-1)
        """
        QUALITY_SCORE.labels(job_id=job_id, parser_used=parser_used).set(score)
    
    def record_llm_request(
        self,
        model: str,
        operation: str,
        status: str,
        latency: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record an LLM request.
        
        Args:
            model: Model used
            operation: Operation type (completion, embedding, etc.)
            status: Request status (success, error, rate_limited)
            latency: Request latency in seconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        LLM_REQUESTS.labels(model=model, operation=operation, status=status).inc()
        LLM_LATENCY.labels(model=model, operation=operation).observe(latency)
        
        if input_tokens:
            LLM_TOKENS_USED.labels(model=model, token_type="input").inc(input_tokens)
        if output_tokens:
            LLM_TOKENS_USED.labels(model=model, token_type="output").inc(output_tokens)
        
        if status == "rate_limited":
            LLM_RATE_LIMITS_HIT.labels(model=model, provider="unknown").inc()
    
    def record_destination_write(
        self,
        destination_type: str,
        status: str,
        records_written: int = 0,
    ) -> None:
        """Record a destination write operation.
        
        Args:
            destination_type: Type of destination
            status: Write status (success, error)
            records_written: Number of records written
        """
        DESTINATION_WRITES.labels(
            destination_type=destination_type,
            status=status,
        ).inc()
        
        if records_written:
            DESTINATION_RECORDS_WRITTEN.labels(
                destination_type=destination_type,
            ).inc(records_written)
    
    def record_destination_error(
        self,
        destination_type: str,
        error_type: str,
    ) -> None:
        """Record a destination error.
        
        Args:
            destination_type: Type of destination
            error_type: Type of error
        """
        DESTINATION_ERRORS.labels(
            destination_type=destination_type,
            error_type=error_type,
        ).inc()
    
    def set_queue_depth(self, queue_name: str, depth: int) -> None:
        """Set the current queue depth.
        
        Args:
            queue_name: Name of the queue
            depth: Current number of messages
        """
        QUEUE_DEPTH.labels(queue_name=queue_name).set(depth)
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: int = 0,
        response_size: int = 0,
    ) -> None:
        """Record an API request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
            request_size: Request body size in bytes
            response_size: Response body size in bytes
        """
        API_REQUESTS.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        
        API_REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)
        
        if request_size:
            API_REQUEST_SIZE.labels(
                method=method,
                endpoint=endpoint,
            ).observe(request_size)
        
        if response_size:
            API_RESPONSE_SIZE.labels(
                method=method,
                endpoint=endpoint,
            ).observe(response_size)
    
    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus exposition format.
        
        Returns:
            Metrics as bytes in Prometheus format
        """
        return generate_latest(self.registry)


# Global metrics manager instance
_metrics_manager: Optional[MetricsManager] = None


def get_metrics_manager() -> MetricsManager:
    """Get or create the global metrics manager.
    
    Returns:
        MetricsManager singleton instance
    """
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    return _metrics_manager
