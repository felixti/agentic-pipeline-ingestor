# Models package
# Import base models from base_models.py (which was models.py)
from src.api.base_models import (
    # API Response
    ApiLinks,
    ApiResponse,
    # Enums
    ContentType,
    FilterOperator,
    HealthStatus,
    JobStatus,
    SourceType,
    # Detection
    ComponentHealth,
    ContentDetectionResult,
    # Pipeline
    DestinationConfig,
    DestinationFilter,
    ParserConfig,
    PipelineConfig,
    QualityConfig,
    StageProgress,
    # Job
    Job,
    JobCreateRequest,
    JobError,
    JobListResponse,
    JobResponse,
    JobRetryRequest,
    RetryRecord,
    # Upload
    UploadMultipleResponse,
    UploadResponse,
    # Chunks
    DocumentChunkListItem,
    DocumentChunkListResponse,
    DocumentChunkResponse,
    # Health
    HealthAlive,
    HealthReady,
    HealthStatusResponse,
)

# RAG API models
from .rag import (
    RAGBenchmarkRequest,
    RAGBenchmarkResponse,
    RAGMetricsResponse,
    RAGMetricsSummary,
    RAGQueryErrorResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGSource,
    RAGStrategyEvaluateRequest,
    RAGStrategyEvaluateResponse,
    RAGStrategyInfo,
    RAGStrategiesResponse,
    RAGValidationErrorResponse,
    StrategyPreset,
)

__all__ = [
    # API Response
    "ApiLinks",
    "ApiResponse",
    # Enums
    "ContentType",
    "FilterOperator",
    "HealthStatus",
    "JobStatus",
    "SourceType",
    # Detection
    "ComponentHealth",
    "ContentDetectionResult",
    # Pipeline
    "DestinationConfig",
    "DestinationFilter",
    "ParserConfig",
    "PipelineConfig",
    "QualityConfig",
    "StageProgress",
    # Job
    "Job",
    "JobCreateRequest",
    "JobError",
    "JobListResponse",
    "JobResponse",
    "JobRetryRequest",
    "RetryRecord",
    # Upload
    "UploadMultipleResponse",
    "UploadResponse",
    # Chunks
    "DocumentChunkListItem",
    "DocumentChunkListResponse",
    "DocumentChunkResponse",
    # Health
    "HealthAlive",
    "HealthReady",
    "HealthStatusResponse",
    # RAG models
    "RAGQueryRequest",
    "RAGQueryResponse",
    "RAGStrategyInfo",
    "RAGStrategiesResponse",
    "RAGSource",
    "StrategyPreset",
    "RAGBenchmarkRequest",
    "RAGBenchmarkResponse",
    "RAGMetricsResponse",
    "RAGMetricsSummary",
    "RAGStrategyEvaluateRequest",
    "RAGStrategyEvaluateResponse",
    "RAGQueryErrorResponse",
    "RAGValidationErrorResponse",
]
