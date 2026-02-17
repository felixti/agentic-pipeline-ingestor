"""SQLAlchemy database models for the Agentic Data Pipeline Ingestor.

This module defines the database schema for jobs, audit logs, pipeline
configurations, and plugin configurations.
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    
    # Enable use of mapped_column with default=None
    pass


# ============================================================================
# Enums
# ============================================================================

class JobStatus(str, PyEnum):
    """Job processing status."""
    CREATED = "created"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


class ProcessingMode(str, PyEnum):
    """Job processing mode."""
    SYNC = "sync"
    ASYNC = "async"


class SourceType(str, PyEnum):
    """Document source type."""
    UPLOAD = "upload"
    URL = "url"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    SHAREPOINT = "sharepoint"


class DestinationType(str, PyEnum):
    """Output destination type."""
    COGNEE = "cognee"
    WEBHOOK = "webhook"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    NEO4J = "neo4j"
    VECTOR_STORE = "vector_store"


class AuditEventType(str, PyEnum):
    """Audit event types."""
    # Job lifecycle events
    JOB_CREATED = "job.created"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"
    JOB_RETRY = "job.retry"
    JOB_DELETED = "job.deleted"
    
    # Stage events
    STAGE_STARTED = "stage.started"
    STAGE_COMPLETED = "stage.completed"
    STAGE_FAILED = "stage.failed"
    
    # Config events
    CONFIG_CREATED = "config.created"
    CONFIG_UPDATED = "config.updated"
    CONFIG_DELETED = "config.deleted"
    
    # Source/Destination events
    SOURCE_CREATED = "source.created"
    SOURCE_UPDATED = "source.updated"
    SOURCE_DELETED = "source.deleted"
    SOURCE_ACCESSED = "source.accessed"
    DESTINATION_CREATED = "destination.created"
    DESTINATION_UPDATED = "destination.updated"
    DESTINATION_DELETED = "destination.deleted"
    
    # Authentication events
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    AUTH_TOKEN_REFRESH = "auth.token_refresh"
    
    # API key events
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"
    API_KEY_USED = "api_key.used"
    
    # User management events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    
    # Audit and compliance events
    AUDIT_EXPORTED = "audit.exported"
    LINEAGE_ACCESSED = "lineage.accessed"
    
    # DLQ events
    DLQ_ENTRY_CREATED = "dlq.entry_created"
    DLQ_ENTRY_RETRIED = "dlq.entry_retried"
    DLQ_ENTRY_ARCHIVED = "dlq.entry_archived"
    
    # System events
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_METRICS_ACCESSED = "system.metrics_accessed"


# ============================================================================
# Job Models
# ============================================================================

class Job(Base):
    """Job model for document ingestion jobs.
    
    This is the core entity representing a document ingestion job
    through the entire processing pipeline.
    """
    
    __tablename__ = "jobs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    external_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    
    # Source information
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    
    # File information
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Processing configuration
    mode: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default=ProcessingMode.ASYNC.value,
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    
    # Pipeline configuration (embedded or reference)
    pipeline_config_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("pipeline_configs.id"),
        nullable=True,
    )
    pipeline_config_snapshot: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Snapshot of pipeline config at job creation time",
    )
    
    # Status tracking
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=JobStatus.CREATED.value,
        index=True,
    )
    current_stage: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    stage_progress: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Results and errors
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Retry tracking
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_history: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    
    # Audit fields
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    
    # Relationships
    destinations: Mapped[List["JobDestination"]] = relationship(
        "JobDestination",
        back_populates="job",
        cascade="all, delete-orphan",
    )
    lineage: Mapped[List["DataLineage"]] = relationship(
        "DataLineage",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="DataLineage.step_order",
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="job",
        cascade="all, delete-orphan",
    )
    pipeline_config_ref: Mapped[Optional["PipelineConfig"]] = relationship(
        "PipelineConfig",
        back_populates="jobs",
    )
    
    def __repr__(self) -> str:
        return f"<Job(id={self.id}, status={self.status}, file={self.file_name})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "external_id": self.external_id,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "mime_type": self.mime_type,
            "mode": self.mode,
            "priority": self.priority,
            "status": self.status,
            "current_stage": self.current_stage,
            "stage_progress": self.stage_progress,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "created_by": self.created_by,
        }


class JobDestination(Base):
    """Association between jobs and destinations.
    
    Tracks which destinations a job was configured to output to
    and the status of each output operation.
    """
    
    __tablename__ = "job_destinations"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id"),
        nullable=False,
    )
    destination_config_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("destination_configs.id"),
        nullable=True,
    )
    destination_type: Mapped[str] = mapped_column(String(50), nullable=False)
    destination_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    config_snapshot: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    # Output status
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    output_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    records_written: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    job: Mapped[Job] = relationship("Job", back_populates="destinations")
    destination_config: Mapped[Optional["DestinationConfig"]] = relationship(
        "DestinationConfig",
        back_populates="job_destinations",
    )


# ============================================================================
# Pipeline Configuration Models
# ============================================================================

class PipelineConfig(Base):
    """Pipeline configuration model.
    
    Stores configurable pipeline settings for document processing.
    """
    
    __tablename__ = "pipeline_configs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Configuration sections
    content_detection: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "auto_detect": True,
            "detection_method": "hybrid",
            "text_ratio_threshold": 0.95,
        },
    )
    parser: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "primary_parser": "docling",
            "fallback_parser": "azure_ocr",
            "ocr_language": "eng",
        },
    )
    enrichment: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "extract_entities": False,
            "classify_document": False,
            "add_metadata": True,
        },
    )
    quality: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "enabled": True,
            "min_quality_score": 0.7,
            "auto_retry": True,
            "max_retries": 3,
        },
    )
    transformation: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "chunking": {
                "enabled": True,
                "strategy": "semantic",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
            "generate_embeddings": False,
            "output_format": "json",
        },
    )
    enabled_stages: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: [
            "ingest", "detect", "parse", "enrich", "quality", "transform", "output"
        ],
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    
    # Audit
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    updated_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Relationships
    jobs: Mapped[List[Job]] = relationship("Job", back_populates="pipeline_config_ref")


# ============================================================================
# Source and Destination Configuration Models
# ============================================================================

class SourceConfig(Base):
    """Source configuration model.
    
    Stores configuration for data sources like S3, Azure Blob, SharePoint.
    """
    
    __tablename__ = "source_configs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Configuration (encrypted at application level for sensitive fields)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    
    # Audit
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    updated_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)


class DestinationConfig(Base):
    """Destination configuration model.
    
    Stores configuration for output destinations like Cognee, webhooks.
    """
    
    __tablename__ = "destination_configs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )
    
    # Configuration
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    filters: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    
    # Audit
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    updated_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Relationships
    job_destinations: Mapped[List[JobDestination]] = relationship(
        "JobDestination",
        back_populates="destination_config",
    )


# ============================================================================
# Data Lineage Model
# ============================================================================

class DataLineage(Base):
    """Data lineage model.
    
    Tracks the transformation lineage of documents through the pipeline.
    """
    
    __tablename__ = "data_lineage"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id"),
        nullable=False,
        index=True,
    )
    
    # Step information
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    transformation: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Hash tracking for data integrity
    input_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    output_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    
    # Additional metadata
    extra_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    
    # Relationships
    job: Mapped[Job] = relationship("Job", back_populates="lineage")
    
    __table_args__ = (
        # Ensure unique step order per job
        {"sqlite_autoincrement": True},
    )


# ============================================================================
# Audit Log Model
# ============================================================================

class AuditLog(Base):
    """Audit log model.
    
    Stores comprehensive audit trail for compliance and debugging.
    """
    
    __tablename__ = "audit_logs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Event information
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    
    # Actor information
    user: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    source_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Resource information
    resource_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Event details
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    # Request tracking
    request_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    
    # Optional job reference
    job_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id"),
        nullable=True,
        index=True,
    )
    
    # Relationships
    job: Mapped[Optional[Job]] = relationship("Job", back_populates="audit_logs")
    
    def __repr__(self) -> str:
        return f"<AuditLog(event={self.event_type}, user={self.user}, timestamp={self.timestamp})>"


# ============================================================================
# Dead Letter Queue Model
# ============================================================================

class DLQEntry(Base):
    """Dead Letter Queue entry model.
    
    Stores failed jobs that have exhausted all retry attempts
    and require manual inspection or intervention.
    """
    
    __tablename__ = "dlq_entries"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id"),
        nullable=False,
        index=True,
    )
    
    # Failure information
    failure_category: Mapped[str] = mapped_column(String(50), nullable=False)
    error_code: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Retry history
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_history: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    max_retries_reached: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    
    # Status tracking
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
    )
    
    # Review information
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    archived_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Additional metadata
    extra_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    # Relationships
    job: Mapped[Job] = relationship("Job", backref="dlq_entry")


# ============================================================================
# Processing History Model
# ============================================================================

class ProcessingHistory(Base):
    """Processing history model.
    
    Tracks processing outcomes for machine learning-based optimization.
    """
    
    __tablename__ = "processing_history"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id"),
        nullable=False,
        index=True,
    )
    
    # File information
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Processing information
    parser_used: Mapped[str] = mapped_column(String(100), nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quality_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    outcome: Mapped[str] = mapped_column(String(20), nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Error information (if failed)
    error_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Configuration snapshot
    config_snapshot: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )
    
    # Additional metadata
    extra_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    # Relationships
    job: Mapped[Job] = relationship("Job", backref="processing_history")
    
    __table_args__ = (
        # Index for ML queries
        {"sqlite_autoincrement": True},
    )


# ============================================================================
# Circuit Breaker State Model
# ============================================================================

class CircuitBreakerState(Base):
    """Circuit breaker state model.
    
    Tracks the state of circuit breakers for external services.
    """
    
    __tablename__ = "circuit_breakers"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Circuit identification
    service_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )
    service_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )  # 'parser', 'destination', 'source'
    
    # Circuit state
    state: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="closed",
    )  # 'closed', 'open', 'half_open'
    
    # Failure tracking
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_threshold: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    success_threshold: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    
    # Timestamps
    last_failure_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_success_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_state_change: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Timeout configuration
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


# ============================================================================
# Source Sync State Model
# ============================================================================

class SourceSyncState(Base):
    """Source synchronization state model.
    
    Tracks synchronization state for external sources (S3, Azure Blob, etc.)
    to enable incremental/delta syncing.
    """
    
    __tablename__ = "source_sync_states"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Source identification
    source_config_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("source_configs.id"),
        nullable=False,
        index=True,
    )
    
    # Sync tracking
    last_sync_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_sync_marker: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )  # S3 continuation token, Azure marker, etc.
    
    # Sync statistics
    files_synced: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    files_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bytes_synced: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    
    # Sync status
    sync_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="idle",
    )  # 'idle', 'running', 'failed', 'completed'
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Incremental sync cursor
    cursor_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cursor_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )  # 'timestamp', 'token', 'etag', etc.
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    
    # Relationships
    source_config: Mapped[SourceConfig] = relationship(
        "SourceConfig",
        backref="sync_states",
    )


# ============================================================================
# User and Authentication Models
# ============================================================================

class User(Base):
    """User model for authentication and authorization.
    
    Stores user information for all authentication methods including
    OAuth2, Azure AD, and API key service accounts.
    """
    
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
    )
    username: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        unique=True,
        index=True,
    )
    
    # Role and permissions
    role: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="viewer",
    )
    permissions: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    
    # Authentication
    auth_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="jwt",
    )  # api_key, oauth2, azure_ad, jwt
    external_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )  # ID from external provider (Azure AD OID, etc.)
    
    # Profile
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    given_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    family_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    department: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    job_title: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Account status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    is_service_account: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Metadata
    extra_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    
    # Relationships
    api_keys: Mapped[List["APIKey"]] = relationship(
        "APIKey",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "email": self.email,
            "username": self.username,
            "role": self.role,
            "permissions": self.permissions,
            "auth_provider": self.auth_provider,
            "display_name": self.display_name,
            "is_active": self.is_active,
            "is_service_account": self.is_service_account,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


class APIKey(Base):
    """API key model for service account authentication.
    
    Stores API keys for programmatic access to the system.
    Keys are hashed before storage.
    """
    
    __tablename__ = "api_keys"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )
    
    # Key information (hash only, never store plain key)
    key_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )
    key_prefix: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )  # First few characters of key for identification
    
    # Key metadata
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Permissions (override user permissions if specified)
    permissions: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    scopes: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    
    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Usage tracking
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    use_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    
    # Rate limiting
    rate_limit_per_minute: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    revoked_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Metadata
    extra_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    
    # Relationships
    user: Mapped[User] = relationship("User", back_populates="api_keys")
    
    def is_expired(self) -> bool:
        """Check if API key has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if API key is valid (active and not expired)."""
        return self.is_active and not self.is_expired()
    
    def to_dict(self, include_hash: bool = False) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Args:
            include_hash: Whether to include key hash (usually False)
        """
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "key_prefix": self.key_prefix,
            "name": self.name,
            "description": self.description,
            "permissions": self.permissions,
            "scopes": self.scopes,
            "is_active": self.is_active,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "use_count": self.use_count,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
        }
        
        if include_hash:
            result["key_hash"] = self.key_hash
        
        return result


# ============================================================================
# Database Engine and Session Management
# ============================================================================

def create_async_engine_from_url(database_url: str, **kwargs: Any) -> Any:
    """Create an async database engine.
    
    Args:
        database_url: Database connection URL
        **kwargs: Additional engine arguments
        
    Returns:
        AsyncEngine instance
    """
    # Convert postgresql:// to postgresql+asyncpg://
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    return create_async_engine(
        database_url,
        echo=kwargs.get("echo", False),
        pool_size=kwargs.get("pool_size", 10),
        max_overflow=kwargs.get("max_overflow", 20),
    )


async def init_db(engine: Any) -> None:
    """Initialize the database schema.
    
    Args:
        engine: Database engine
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db(engine: Any) -> None:
    """Drop all database tables.
    
    Args:
        engine: Database engine
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# Global engine and session factory (initialized at startup)
_engine: Any = None
_async_session_factory: Any = None


async def get_session() -> AsyncSession:
    """Get a database session.
    
    Yields:
        AsyncSession for database operations
    """
    global _async_session_factory
    
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_engine() first.")
    
    async with _async_session_factory() as session:
        yield session


def init_engine(database_url: str, **kwargs: Any) -> Any:
    """Initialize the database engine and session factory.
    
    Args:
        database_url: Database connection URL
        **kwargs: Additional engine arguments
        
    Returns:
        AsyncEngine instance
    """
    global _engine, _async_session_factory
    
    _engine = create_async_engine_from_url(database_url, **kwargs)
    _async_session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    return _engine
