"""Pydantic models for API requests and responses.

These models are derived from the OpenAPI 3.1 specification and provide
validation and serialization for all API interactions.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Enums
# ============================================================================

class JobStatus(str, Enum):
    """Job processing status."""
    CREATED = "created"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


class ProcessingMode(str, Enum):
    """Job processing mode."""
    SYNC = "sync"
    ASYNC = "async"


class ContentType(str, Enum):
    """Detected content type of a document."""
    TEXT_BASED_PDF = "text_based_pdf"
    SCANNED_PDF = "scanned_pdf"
    MIXED_PDF = "mixed_pdf"
    OFFICE_DOC = "office_doc"
    OFFICE_SPREADSHEET = "office_spreadsheet"
    OFFICE_PRESENTATION = "office_presentation"
    IMAGE = "image"
    ARCHIVE = "archive"
    CSV = "csv"
    UNKNOWN = "unknown"


class DetectionMethod(str, Enum):
    """Content detection method."""
    HEURISTIC = "heuristic"
    ML = "ml"
    HYBRID = "hybrid"


class StageStatus(str, Enum):
    """Pipeline stage status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChunkingStrategy(str, Enum):
    """Text chunking strategy."""
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


class OutputFormat(str, Enum):
    """Output format for processed documents."""
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"
    COGNEE = "cognee"


class SourceType(str, Enum):
    """Document source type."""
    UPLOAD = "upload"
    URL = "url"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    SHAREPOINT = "sharepoint"


class DestinationType(str, Enum):
    """Output destination type."""
    COGNEE = "cognee"
    WEBHOOK = "webhook"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    NEO4J = "neo4j"
    VECTOR_STORE = "vector_store"


class EventType(str, Enum):
    """Job event types for SSE."""
    CREATED = "created"
    QUEUED = "queued"
    STAGE_STARTED = "stage_started"
    STAGE_PROGRESS = "stage_progress"
    STAGE_COMPLETED = "stage_completed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PluginType(str, Enum):
    """Plugin type classification."""
    SOURCE = "source"
    PARSER = "parser"
    DESTINATION = "destination"


class HealthStatus(str, Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class FilterOperator(str, Enum):
    """Destination filter operators."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"


class RetryStrategy(str, Enum):
    """Retry strategies for failed jobs."""
    SAME_PARSER = "same_parser"
    FALLBACK_PARSER = "fallback_parser"
    PREPROCESS_THEN_RETRY = "preprocess_then_retry"
    SPLIT_PROCESSING = "split_processing"


# ============================================================================
# Configuration Models
# ============================================================================

class ContentDetectionConfig(BaseModel):
    """Configuration for content type detection."""
    model_config = ConfigDict(populate_by_name=True)

    auto_detect: bool = Field(default=True, description="Automatically detect content type")
    detection_method: DetectionMethod = Field(default=DetectionMethod.HYBRID)
    text_ratio_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Threshold for text-based vs scanned detection"
    )


class ParserConfig(BaseModel):
    """Configuration for document parsing."""
    model_config = ConfigDict(populate_by_name=True)

    primary_parser: str = Field(default="docling")
    fallback_parser: str | None = Field(default="azure_ocr")
    parser_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Parser-specific options"
    )
    ocr_language: str = Field(default="eng", description="Language for OCR")


class EnrichmentConfig(BaseModel):
    """Configuration for document enrichment."""
    model_config = ConfigDict(populate_by_name=True)

    extract_entities: bool = Field(default=False)
    entity_types: list[str] = Field(default_factory=lambda: ["person", "organization", "location"])
    classify_document: bool = Field(default=False)
    classification_model: str | None = Field(default=None)
    add_metadata: bool = Field(default=True)


class QualityConfig(BaseModel):
    """Configuration for quality assessment."""
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = Field(default=True)
    min_quality_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum quality score to accept result"
    )
    auto_retry: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0)
    retry_strategies: list[str] = Field(
        default_factory=lambda: ["same_parser", "fallback_parser"]
    )


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = Field(default=True)
    strategy: ChunkingStrategy = Field(default=ChunkingStrategy.SEMANTIC)
    chunk_size: int = Field(default=1000, ge=100, description="Maximum chunk size in tokens/characters")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")


class TransformationConfig(BaseModel):
    """Configuration for document transformation."""
    model_config = ConfigDict(populate_by_name=True)

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    generate_embeddings: bool = Field(default=False)
    embedding_model: str | None = Field(default="text-embedding-3-small")
    output_format: OutputFormat = Field(default=OutputFormat.JSON)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    model_config = ConfigDict(populate_by_name=True)

    id: UUID | None = Field(default=None)
    name: str = Field(..., description="Unique pipeline name")
    description: str | None = Field(default=None)
    content_detection: ContentDetectionConfig = Field(default_factory=ContentDetectionConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    transformation: TransformationConfig = Field(default_factory=TransformationConfig)
    enabled_stages: list[str] = Field(
        default_factory=lambda: [
            "ingest", "detect", "parse", "enrich", "quality", "transform", "output"
        ]
    )
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)


class DestinationFilter(BaseModel):
    """Filter for selective destination routing."""
    model_config = ConfigDict(populate_by_name=True)

    field: str = Field(..., description="Field to filter on")
    operator: FilterOperator = Field(...)
    value: str = Field(..., description="Value to match")


class DestinationConfig(BaseModel):
    """Configuration for an output destination."""
    model_config = ConfigDict(populate_by_name=True)

    id: UUID | None = Field(default=None)
    type: DestinationType = Field(...)
    name: str | None = Field(default=None)
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Destination-specific configuration"
    )
    filters: list[DestinationFilter] = Field(
        default_factory=list,
        description="Optional filters for selective routing"
    )
    enabled: bool = Field(default=True)


class SourceConfig(BaseModel):
    """Configuration for a data source."""
    model_config = ConfigDict(populate_by_name=True)

    id: UUID | None = Field(default=None)
    type: SourceType = Field(...)
    name: str = Field(...)
    description: str | None = Field(default=None)
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific configuration"
    )
    enabled: bool = Field(default=True)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)


# ============================================================================
# Job Models
# ============================================================================

class StageProgress(BaseModel):
    """Progress information for a pipeline stage."""
    model_config = ConfigDict(populate_by_name=True)

    stage: str = Field(...)
    status: StageStatus = Field(...)
    progress_percent: int = Field(default=0, ge=0, le=100)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    message: str | None = Field(default=None)


class JobResult(BaseModel):
    """Result of a completed job."""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(...)
    output_uri: str | None = Field(default=None)
    output_format: str | None = Field(default=None)
    extracted_text: str | None = Field(
        default=None,
        description="Extracted text content (for small documents)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    processing_time_ms: int | None = Field(default=None)
    pages_processed: int | None = Field(default=None)


class JobError(BaseModel):
    """Error information for a failed job."""
    model_config = ConfigDict(populate_by_name=True)

    code: str = Field(...)
    message: str = Field(...)
    details: dict[str, Any] | None = Field(default=None)
    stack_trace: str | None = Field(default=None)
    failed_stage: str | None = Field(default=None)


class RetryRecord(BaseModel):
    """Record of a retry attempt."""
    model_config = ConfigDict(populate_by_name=True)

    attempt: int = Field(..., ge=1)
    timestamp: datetime = Field(...)
    strategy: str = Field(..., description="Retry strategy used")
    error_code: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class Job(BaseModel):
    """Complete job representation."""
    model_config = ConfigDict(populate_by_name=True)

    id: UUID = Field(...)
    external_id: str | None = Field(default=None)
    source_type: SourceType = Field(...)
    source_uri: str = Field(...)
    file_name: str = Field(...)
    file_size: int | None = Field(default=None)
    file_hash: str | None = Field(default=None)
    mime_type: str | None = Field(default=None)
    mode: ProcessingMode = Field(default=ProcessingMode.ASYNC)
    priority: int = Field(default=5, ge=1, le=10)
    pipeline_config: PipelineConfig | None = Field(default=None)
    destinations: list[DestinationConfig] = Field(default_factory=list)
    status: JobStatus = Field(...)
    current_stage: str | None = Field(default=None)
    stage_progress: dict[str, StageProgress] = Field(default_factory=dict)
    created_at: datetime = Field(...)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)
    result: JobResult | None = Field(default=None)
    error: JobError | None = Field(default=None)
    retry_count: int = Field(default=0)
    retry_history: list[RetryRecord] = Field(default_factory=list)
    created_by: str | None = Field(default=None)
    source_ip: str | None = Field(default=None)


class JobCreateRequest(BaseModel):
    """Request to create a new job."""
    model_config = ConfigDict(populate_by_name=True)

    source_type: SourceType = Field(...)
    source_uri: str = Field(...)
    file_name: str | None = Field(default=None)
    file_size: int | None = Field(default=None)
    mime_type: str | None = Field(default=None)
    pipeline_id: UUID | None = Field(default=None)
    destination_ids: list[UUID] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)
    external_id: str | None = Field(default=None)
    mode: ProcessingMode = Field(default=ProcessingMode.ASYNC)


class JobRetryRequest(BaseModel):
    """Request to retry a failed job."""
    model_config = ConfigDict(populate_by_name=True)

    force_parser: str | None = Field(
        default=None,
        description="Override parser for retry"
    )
    updated_config: PipelineConfig | None = Field(
        default=None,
        description="Updated pipeline configuration"
    )


class JobEvent(BaseModel):
    """Server-sent event for job updates."""
    model_config = ConfigDict(populate_by_name=True)

    event: EventType = Field(...)
    job_id: UUID = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = Field(default_factory=dict)


class JobResponse(BaseModel):
    """Response model for job data."""
    model_config = ConfigDict(populate_by_name=True)

    id: UUID = Field(...)
    status: JobStatus = Field(...)
    source_type: SourceType = Field(...)
    source_uri: str = Field(...)
    file_name: str | None = Field(default=None)
    file_size: int | None = Field(default=None)
    mime_type: str | None = Field(default=None)
    priority: int = Field(default=5)
    mode: ProcessingMode = Field(default=ProcessingMode.ASYNC)
    external_id: str | None = Field(default=None)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    created_at: datetime = Field(...)
    updated_at: datetime = Field(...)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    error: JobError | None = Field(default=None)


class JobListResponse(BaseModel):
    """Response model for job list."""
    model_config = ConfigDict(populate_by_name=True)

    items: list[JobResponse] = Field(default_factory=list)
    total: int = Field(...)
    page: int = Field(...)
    page_size: int = Field(...)


class UploadResponse(BaseModel):
    """Response model for file upload."""
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(...)
    job_id: UUID = Field(...)
    file_name: str = Field(...)
    file_size: int = Field(...)


class UploadMultipleResponse(BaseModel):
    """Response model for multiple file upload."""
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(...)
    job_ids: list[UUID] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)


# ============================================================================
# Content Detection Models
# ============================================================================

class TextStatistics(BaseModel):
    """Statistics about text content in a document."""
    model_config = ConfigDict(populate_by_name=True)

    total_characters: int = Field(default=0)
    total_words: int = Field(default=0)
    text_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_chars_per_page: int | None = Field(default=None)


class ContentDetectionResult(BaseModel):
    """Result of content type detection."""
    model_config = ConfigDict(populate_by_name=True)

    detected_type: ContentType = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    detection_method: str = Field(default="hybrid")
    page_count: int | None = Field(default=None)
    has_text_layer: bool = Field(default=False)
    has_images: bool = Field(default=False)
    image_count: int = Field(default=0)
    text_statistics: TextStatistics = Field(default_factory=TextStatistics)
    recommended_parser: str = Field(...)
    alternative_parsers: list[str] = Field(default_factory=list)
    preprocessing_required: bool = Field(default=False)


# ============================================================================
# Upload Models
# ============================================================================

class FileUploadRequest(BaseModel):
    """Request for file upload."""
    model_config = ConfigDict(populate_by_name=True)

    pipeline_id: UUID | None = Field(default=None)
    destination_ids: list[UUID] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)
    external_id: str | None = Field(default=None)
    mode: ProcessingMode = Field(default=ProcessingMode.ASYNC)


class UrlIngestRequest(BaseModel):
    """Request for URL-based ingestion."""
    model_config = ConfigDict(populate_by_name=True)

    url: str = Field(...)
    pipeline_id: UUID | None = Field(default=None)
    destination_ids: list[UUID] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)
    external_id: str | None = Field(default=None)
    headers: dict[str, str] = Field(default_factory=dict)
    mode: ProcessingMode = Field(default=ProcessingMode.ASYNC)


# ============================================================================
# Audit Models
# ============================================================================

class AuditLogEntry(BaseModel):
    """Single audit log entry."""
    model_config = ConfigDict(populate_by_name=True)

    id: UUID = Field(...)
    timestamp: datetime = Field(...)
    event_type: str = Field(...)
    user: str = Field(...)
    resource_type: str | None = Field(default=None)
    resource_id: str | None = Field(default=None)
    action: str = Field(...)
    details: dict[str, Any] = Field(default_factory=dict)
    source_ip: str | None = Field(default=None)
    user_agent: str | None = Field(default=None)


class LineageStep(BaseModel):
    """Single step in data lineage."""
    model_config = ConfigDict(populate_by_name=True)

    stage: str = Field(...)
    input_hash: str | None = Field(default=None)
    output_hash: str | None = Field(default=None)
    transformation: str = Field(...)
    timestamp: datetime = Field(...)


class DataLineage(BaseModel):
    """Complete data lineage for a job."""
    model_config = ConfigDict(populate_by_name=True)

    job_id: UUID = Field(...)
    input_hash: str | None = Field(default=None)
    output_hash: str | None = Field(default=None)
    steps: list[LineageStep] = Field(default_factory=list)


class AuditExportRequest(BaseModel):
    """Request to export audit data."""
    model_config = ConfigDict(populate_by_name=True)

    format: str = Field(default="json", pattern="^(json|csv)$")
    date_from: datetime | None = Field(default=None)
    date_to: datetime | None = Field(default=None)
    event_types: list[str] = Field(default_factory=list)


class AuditExportResult(BaseModel):
    """Result of audit export."""
    model_config = ConfigDict(populate_by_name=True)

    export_id: UUID = Field(...)
    download_url: str = Field(...)
    expires_at: datetime = Field(...)
    record_count: int = Field(...)


# ============================================================================
# Plugin Models
# ============================================================================

class PluginInfo(BaseModel):
    """Information about a plugin."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    name: str = Field(...)
    version: str = Field(...)
    type: PluginType = Field(...)
    description: str | None = Field(default=None)
    supported_formats: list[str] = Field(default_factory=list)
    config_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema for plugin configuration"
    )


class ConnectionTestResult(BaseModel):
    """Result of a connection test."""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(...)
    message: str = Field(...)
    latency_ms: int | None = Field(default=None)
    details: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Validation Models
# ============================================================================

class ValidationError(BaseModel):
    """Single validation error."""
    model_config = ConfigDict(populate_by_name=True)

    field: str = Field(...)
    code: str = Field(...)
    message: str = Field(...)


class ValidationWarning(BaseModel):
    """Single validation warning."""
    model_config = ConfigDict(populate_by_name=True)

    field: str = Field(...)
    message: str = Field(...)


class ValidationResult(BaseModel):
    """Result of validation."""
    model_config = ConfigDict(populate_by_name=True)

    valid: bool = Field(...)
    errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[ValidationWarning] = Field(default_factory=list)


# ============================================================================
# Health Models
# ============================================================================

class ComponentHealth(BaseModel):
    """Health status of a system component."""
    model_config = ConfigDict(populate_by_name=True)

    status: HealthStatus = Field(...)
    latency_ms: int | None = Field(default=None)
    message: str | None = Field(default=None)


class HealthStatusResponse(BaseModel):
    """Complete health status response."""
    model_config = ConfigDict(populate_by_name=True)

    status: HealthStatus = Field(...)
    version: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, ComponentHealth] = Field(default_factory=dict)


class HealthReady(BaseModel):
    """Readiness probe response."""
    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(default="ready")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthNotReady(BaseModel):
    """Not ready response."""
    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(default="not_ready")
    reason: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthAlive(BaseModel):
    """Liveness probe response."""
    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(default="alive")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Response Wrapper Models
# ============================================================================

class ApiMeta(BaseModel):
    """Metadata for API responses."""
    model_config = ConfigDict(populate_by_name=True)

    request_id: UUID = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    api_version: str = Field(default="v1")
    total_count: int | None = Field(default=None)


class ApiLinks(BaseModel):
    """Pagination and navigation links."""
    model_config = ConfigDict(populate_by_name=True)

    self: str = Field(...)
    next: str | None = Field(default=None)
    prev: str | None = Field(default=None)
    first: str | None = Field(default=None)
    last: str | None = Field(default=None)


class ApiResponse(BaseModel):
    """Standard API response wrapper."""
    model_config = ConfigDict(populate_by_name=True)

    data: Any | None = Field(default=None)
    meta: ApiMeta = Field(...)
    links: ApiLinks | None = Field(default=None)

    @classmethod
    def create(
        cls,
        data: Any,
        request_id: UUID,
        total_count: int | None = None,
        links: ApiLinks | None = None,
    ) -> "ApiResponse":
        """Create a standardized API response.
        
        Args:
            data: Response data
            request_id: Unique request identifier
            total_count: Total count for paginated responses
            links: Pagination links
            
        Returns:
            Standardized API response
        """
        return cls(
            data=data,
            meta=ApiMeta(request_id=request_id, total_count=total_count),
            links=links,
        )


# ============================================================================
# Document Chunk Models
# ============================================================================

class DocumentChunkResponse(BaseModel):
    """Response model for a single document chunk.
    
    Used for both list items and detailed responses. The embedding field
    is only included when explicitly requested via include_embedding parameter.
    """
    model_config = ConfigDict(populate_by_name=True)

    id: UUID = Field(..., description="Unique chunk identifier")
    job_id: UUID = Field(..., description="Parent job identifier")
    chunk_index: int = Field(..., ge=0, description="Position within document (0-indexed)")
    content: str = Field(..., description="Text content of the chunk")
    content_hash: str | None = Field(default=None, description="SHA-256 hash for deduplication")
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding for semantic search (only when requested)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata (page numbers, etc.)")
    created_at: datetime = Field(..., description="Timestamp of record creation")


class DocumentChunkListItem(BaseModel):
    """List item response for document chunks (excludes embedding by default).
    
    Used in list endpoints where embedding data is excluded for performance.
    """
    model_config = ConfigDict(populate_by_name=True)

    id: UUID = Field(..., description="Unique chunk identifier")
    job_id: UUID = Field(..., description="Parent job identifier")
    chunk_index: int = Field(..., ge=0, description="Position within document (0-indexed)")
    content: str = Field(..., description="Text content of the chunk")
    content_hash: str | None = Field(default=None, description="SHA-256 hash for deduplication")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata (page numbers, etc.)")
    created_at: datetime = Field(..., description="Timestamp of record creation")


class DocumentChunkListResponse(BaseModel):
    """Response model for paginated chunk list."""
    model_config = ConfigDict(populate_by_name=True)

    items: list[DocumentChunkListItem] = Field(default_factory=list, description="List of chunks")
    total: int = Field(..., ge=0, description="Total number of chunks for this job")
    limit: int = Field(..., ge=1, le=1000, description="Number of items returned")
    offset: int = Field(..., ge=0, description="Offset from start of results")


# ============================================================================
# Error Models
# ============================================================================

class FieldError(BaseModel):
    """Error for a specific field."""
    model_config = ConfigDict(populate_by_name=True)

    field: str = Field(...)
    code: str = Field(...)
    message: str = Field(...)


class ErrorDetail(BaseModel):
    """Detailed error information."""
    model_config = ConfigDict(populate_by_name=True)

    code: str = Field(...)
    message: str = Field(...)
    details: list[FieldError] | None = Field(default=None)
    documentation_url: str | None = Field(default=None)


class ErrorResponse(BaseModel):
    """Standard error response."""
    model_config = ConfigDict(populate_by_name=True)

    error: ErrorDetail = Field(...)
    meta: ApiMeta = Field(...)

    @classmethod
    def create(
        cls,
        code: str,
        message: str,
        request_id: UUID,
        details: list[FieldError] | None = None,
        documentation_url: str | None = None,
    ) -> "ErrorResponse":
        """Create a standardized error response.
        
        Args:
            code: Error code
            message: Error message
            request_id: Unique request identifier
            details: Field-specific error details
            documentation_url: Link to error documentation
            
        Returns:
            Standardized error response
        """
        return cls(
            error=ErrorDetail(
                code=code,
                message=message,
                details=details,
                documentation_url=documentation_url,
            ),
            meta=ApiMeta(request_id=request_id),
        )
