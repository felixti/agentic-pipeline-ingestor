"""Prometheus metrics for content detection."""

from prometheus_client import Counter, Gauge, Histogram, Info

# System information
SYSTEM_INFO = Info(
    "pipeline_system",
    "System information about the pipeline ingestor"
)

# Content detection counters
CONTENT_DETECTION_TOTAL = Counter(
    "content_detection_total",
    "Total number of content detection operations",
    ["content_type"]
)

CONTENT_DETECTION_CACHE_HITS = Counter(
    "content_detection_cache_hits_total",
    "Total number of cache hits"
)

CONTENT_DETECTION_CACHE_MISSES = Counter(
    "content_detection_cache_misses_total",
    "Total number of cache misses"
)

CONTENT_DETECTION_ERRORS = Counter(
    "content_detection_errors_total",
    "Total number of detection errors",
    ["error_type"]
)

# Histograms
CONTENT_DETECTION_DURATION = Histogram(
    "content_detection_duration_seconds",
    "Time spent on content detection",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

CONTENT_DETECTION_FILE_SIZE = Histogram(
    "content_detection_file_size_bytes",
    "Size of analyzed files",
    buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600]  # 1KB to 100MB
)

# Gauges
CONTENT_DETECTION_CACHE_HIT_RATIO = Gauge(
    "content_detection_cache_hit_ratio",
    "Cache hit ratio (0.0-1.0)"
)

# Pipeline metrics
PIPELINE_STAGE_DURATION = Histogram(
    "pipeline_stage_duration_seconds",
    "Time spent in each pipeline stage",
    ["stage"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PIPELINE_JOBS_TOTAL = Counter(
    "pipeline_jobs_total",
    "Total number of pipeline jobs",
    ["status"]
)


class DetectionMetrics:
    """Helper class for recording detection metrics."""
    
    _cache_hits = 0
    _cache_misses = 0
    
    @classmethod
    def record_detection(
        cls,
        content_type: str,
        duration_seconds: float,
        file_size_bytes: int
    ) -> None:
        """Record a detection operation.
        
        Args:
            content_type: Detected content type
            duration_seconds: Detection duration
            file_size_bytes: File size
        """
        CONTENT_DETECTION_TOTAL.labels(content_type=content_type).inc()
        CONTENT_DETECTION_DURATION.observe(duration_seconds)
        CONTENT_DETECTION_FILE_SIZE.observe(file_size_bytes)
    
    @classmethod
    def record_cache_hit(cls) -> None:
        """Record a cache hit."""
        CONTENT_DETECTION_CACHE_HITS.inc()
        cls._cache_hits += 1
        cls._update_hit_ratio()
    
    @classmethod
    def record_cache_miss(cls) -> None:
        """Record a cache miss."""
        CONTENT_DETECTION_CACHE_MISSES.inc()
        cls._cache_misses += 1
        cls._update_hit_ratio()
    
    @classmethod
    def record_error(cls, error_type: str) -> None:
        """Record an error.
        
        Args:
            error_type: Type of error
        """
        CONTENT_DETECTION_ERRORS.labels(error_type=error_type).inc()
    
    @classmethod
    def _update_hit_ratio(cls) -> None:
        """Update cache hit ratio gauge."""
        total = cls._cache_hits + cls._cache_misses
        if total > 0:
            ratio = cls._cache_hits / total
            CONTENT_DETECTION_CACHE_HIT_RATIO.set(ratio)
    
    @classmethod
    def get_stats(cls) -> dict:
        """Get current metrics stats.
        
        Returns:
            Dictionary with stats
        """
        total = cls._cache_hits + cls._cache_misses
        return {
            "cache_hits": cls._cache_hits,
            "cache_misses": cls._cache_misses,
            "hit_ratio": cls._cache_hits / total if total > 0 else 0.0,
        }


class PipelineMetrics:
    """Helper class for recording pipeline metrics."""
    
    @staticmethod
    def record_stage_duration(stage: str, duration_seconds: float) -> None:
        """Record stage duration.
        
        Args:
            stage: Stage name
            duration_seconds: Duration
        """
        PIPELINE_STAGE_DURATION.labels(stage=stage).observe(duration_seconds)
    
    @staticmethod
    def record_job_completed() -> None:
        """Record a completed job."""
        PIPELINE_JOBS_TOTAL.labels(status="completed").inc()
    
    @staticmethod
    def record_job_failed() -> None:
        """Record a failed job."""
        PIPELINE_JOBS_TOTAL.labels(status="failed").inc()


class MetricsManager:
    """Manager for all metrics."""
    
    def __init__(self) -> None:
        self.detection = DetectionMetrics()
        self.pipeline = PipelineMetrics()
    
    def record_llm_request(
        self,
        model: str,
        operation: str,
        status: str,
        latency: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record an LLM request metric.
        
        Args:
            model: Model name
            operation: Operation type (e.g., chat_completion)
            status: Status (success, error, rate_limited)
            latency: Request latency in seconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        # For now, just pass - this can be enhanced with actual Prometheus counters
        pass
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        user_agent: str = "",
    ) -> None:
        """Record an API request metric.
        
        Args:
            method: HTTP method
            endpoint: Request endpoint/path
            status_code: HTTP status code
            duration: Request duration in seconds
            user_agent: User agent string
        """
        # For now, just pass - this can be enhanced with actual Prometheus counters
        pass


# Global metrics manager instance
_metrics_manager: MetricsManager | None = None


def get_metrics_manager() -> MetricsManager:
    """Get the global metrics manager.
    
    Returns:
        MetricsManager instance
    """
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    return _metrics_manager
