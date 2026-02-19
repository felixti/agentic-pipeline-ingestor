"""SQLAlchemy models for database tables."""

from collections.abc import AsyncGenerator
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import BigInteger, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import UserDefinedType


class Vector(UserDefinedType[Any]):
    """SQLAlchemy type for pgvector VECTOR.
    
    This is a custom type that maps Python lists to PostgreSQL VECTOR columns.
    Supports configurable dimensions (default 1536 for OpenAI embeddings).
    
    Example:
        embedding = Column(Vector(1536), nullable=True)
    """
    
    cache_ok = True
    
    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions
    
    def get_col_spec(self, **kw: Any) -> str:
        return f"VECTOR({self.dimensions})"
    
    def bind_processor(self, dialect: Any) -> Any:
        def process(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, list):
                return f"[{','.join(str(x) for x in value)}]"
            return value
        return process
    
    def result_processor(self, dialect: Any, coltype: Any) -> Any:
        def process(value: Any) -> Any:
            if value is None:
                return None
            # Parse vector string representation into list of floats
            if isinstance(value, str):
                value = value.strip("[]")
                return [float(x) for x in value.split(",")]
            return value
        return process


Base: Any = declarative_base()

# Async engine (initialized on demand)
_async_engine: Any = None
_engine: Any = None  # Alias for backward compatibility


def init_engine(
    database_url: str | None = None,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> Any:
    """Initialize the async engine (for backward compatibility).
    
    Args:
        database_url: Database URL
        echo: Whether to echo SQL statements
        pool_size: Connection pool size
        max_overflow: Max overflow connections
        
    Returns:
        AsyncEngine instance
    """
    global _engine, _async_engine
    if database_url is None:
        import os
        database_url = os.getenv(
            "DB_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline"
        )
    
    _engine = create_async_engine(
        database_url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
    )
    _async_engine = _engine
    return _engine


async def init_db(engine: Any = None) -> None:
    """Initialize the database (create tables).
    
    This is a placeholder - actual migrations should be run via Alembic.
    
    Args:
        engine: Optional engine instance (uses global engine if not provided)
    """
    if engine is None:
        engine = get_async_engine()
    async with engine.begin() as conn:
        # In production, use Alembic migrations instead
        # await conn.run_sync(Base.metadata.create_all)
        pass


def get_async_engine(database_url: str | None = None) -> Any:
    """Get or create async engine.
    
    Args:
        database_url: Database URL (uses env var or default if not provided)
        
    Returns:
        AsyncEngine instance
    """
    global _async_engine, _engine
    if _async_engine is None:
        # Check if init_engine was already called
        if _engine is not None:
            _async_engine = _engine
        else:
            if database_url is None:
                import os
                database_url = os.getenv(
                    "DB_URL",
                    "postgresql+asyncpg://postgres:postgres@localhost:5432/pipeline"
                )
            _async_engine = create_async_engine(database_url, echo=False)
    return _async_engine


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session.
    
    Yields:
        AsyncSession for database operations
    """
    engine = get_async_engine()
    async with AsyncSession(engine) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


class ContentDetectionResultModel(Base):  # type: ignore[misc]
    """Database model for content detection results."""
    
    __tablename__ = "content_detection_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    file_size = Column(BigInteger, nullable=False)
    content_type = Column(String(20), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    recommended_parser = Column(String(50), nullable=False)
    alternative_parsers: Any = Column(ARRAY(String), nullable=False, default=[])
    text_statistics = Column(JSONB, nullable=False)
    image_statistics = Column(JSONB, nullable=False)
    page_results = Column(JSONB, nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    access_count = Column(Integer, default=1)
    last_accessed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    jobs = relationship("JobDetectionResultModel", back_populates="detection_result")


class JobDetectionResultModel(Base):  # type: ignore[misc]
    """Link table between jobs and detection results."""
    
    __tablename__ = "job_detection_results"
    
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    detection_result_id = Column(UUID(as_uuid=True), ForeignKey("content_detection_results.id", ondelete="CASCADE"), primary_key=True)
    
    # Relationships
    job = relationship("JobModel", back_populates="detection_results")
    detection_result = relationship("ContentDetectionResultModel", back_populates="jobs")


class JobStatus(str, Enum):
    """Job status enumeration."""
    
    CREATED = "created"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCELLING = "cancelling"


class JobModel(Base):  # type: ignore[misc]
    """Job model for pipeline processing jobs."""
    
    __tablename__ = "jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    status = Column(String(20), nullable=False, default=JobStatus.CREATED, index=True)
    source_type = Column(String(50), nullable=False, index=True)
    source_uri = Column(String(500), nullable=True)
    file_name = Column(String(255), nullable=True)
    file_size = Column(BigInteger, nullable=True)
    mime_type = Column(String(100), nullable=True)
    priority = Column(String(20), nullable=False, default="normal")
    mode = Column(String(20), nullable=False, default="async")
    external_id = Column(String(255), nullable=True, index=True)
    metadata_json = Column(JSONB, nullable=False, default={})
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Job locking for concurrent workers
    locked_by = Column(String(255), nullable=True, index=True)
    locked_at = Column(DateTime(timezone=True), nullable=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    
    # Pipeline reference
    pipeline_id = Column(UUID(as_uuid=True), ForeignKey("pipelines.id", ondelete="SET NULL"), nullable=True)
    pipeline_config = Column(JSONB, nullable=True)
    
    # Relationships
    detection_results = relationship("JobDetectionResultModel", back_populates="job", cascade="all, delete-orphan")
    result = relationship("JobResultModel", back_populates="job", uselist=False, cascade="all, delete-orphan")
    pipeline = relationship("PipelineModel", back_populates="jobs")
    chunks = relationship(
        "DocumentChunkModel",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="DocumentChunkModel.chunk_index",
    )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "id": str(self.id),
            "status": self.status,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "priority": self.priority,
            "mode": self.mode,
            "external_id": self.external_id,
            "metadata": self.metadata_json,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": {
                "message": self.error_message,
                "code": self.error_code,
            } if self.error_message or self.error_code else None,
        }


class PipelineModel(Base):  # type: ignore[misc]
    """Pipeline configuration model."""
    
    __tablename__ = "pipelines"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    config = Column(JSONB, nullable=False, default={})
    version = Column(Integer, nullable=False, default=1)
    is_active = Column(Integer, nullable=False, default=1)  # 1 = active, 0 = deleted
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    jobs = relationship("JobModel", back_populates="pipeline", foreign_keys="JobModel.pipeline_id")


class JobResultModel(Base):  # type: ignore[misc]
    """Job processing result model."""
    
    __tablename__ = "job_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True, unique=True)
    extracted_text = Column(Text, nullable=True)
    output_data = Column(JSONB, nullable=True)
    result_metadata = Column("metadata", JSONB, nullable=False, default={})
    quality_score = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    output_uri = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Relationships
    job = relationship("JobModel", back_populates="result")


class DocumentChunkModel(Base):  # type: ignore[misc]
    """Document chunk model with vector embedding for semantic search.
    
    Stores chunks of text extracted from documents with their vector embeddings
    for similarity search. Each chunk belongs to a specific job and is ordered
    by chunk_index within that job.
    
    Attributes:
        id: Unique identifier (UUID)
        job_id: Reference to parent job (cascade delete)
        chunk_index: Position of chunk within document (0-indexed, non-negative)
        content: Text content of the chunk
        content_hash: SHA-256 hash for deduplication
        embedding: Vector embedding for semantic search (nullable initially)
        chunk_metadata: JSONB metadata (source, page numbers, etc.)
        created_at: Timestamp of record creation
    """
    
    __tablename__ = "document_chunks"
    
    # Table constraints and indexes
    __table_args__ = (
        # Unique constraint: one chunk_index per job
        Index("uq_document_chunks_job_chunk", "job_id", "chunk_index", unique=True),
        # Composite index for efficient job + chunk_index queries
        Index("idx_document_chunks_job_chunk", "job_id", "chunk_index"),
    )
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to jobs table with CASCADE delete
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Chunk ordering within document (must be non-negative)
    chunk_index = Column(Integer, nullable=False)
    
    # Content storage
    content = Column(Text, nullable=False)
    
    # SHA-256 hash for deduplication
    content_hash = Column(String(64), nullable=True, index=True)
    
    # Vector embedding for semantic search (nullable during initial creation)
    embedding: Any = Column(Vector(dimensions=1536), nullable=True)
    
    # Flexible metadata storage
    chunk_metadata = Column(JSONB, nullable=False, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    # Relationships
    job = relationship("JobModel", back_populates="chunks")
    
    def __repr__(self) -> str:
        return (
            f"<DocumentChunkModel(id={self.id}, "
            f"job_id={self.job_id}, "
            f"chunk_index={self.chunk_index}, "
            f"has_embedding={self.embedding is not None})>"
        )
    
    @property
    def has_embedding(self) -> bool:
        """Check if chunk has an embedding vector."""
        return self.embedding is not None
    
    def set_embedding(self, embedding: list[float]) -> None:
        """Set the embedding vector with dimension validation.
        
        Args:
            embedding: List of float values
            
        Raises:
            ValueError: If embedding dimensions don't match expected size
        """
        expected_dims = 1536  # From Vector type definition
        if len(embedding) != expected_dims:
            raise ValueError(
                f"Embedding must have {expected_dims} dimensions, "
                f"got {len(embedding)}"
            )
        self.embedding = embedding


class AuditLogModel(Base):  # type: ignore[misc]
    """Audit log entry model."""
    
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    api_key_id = Column(String(255), nullable=True, index=True)
    action = Column(String(50), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False, index=True)
    resource_id = Column(String(255), nullable=True, index=True)
    request_method = Column(String(10), nullable=True)
    request_path = Column(String(500), nullable=True)
    request_details = Column(JSONB, nullable=True)
    success = Column(Integer, nullable=False, default=1)  # 1 = success, 0 = failure
    error_message = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    duration_ms = Column(Integer, nullable=True)


class ApiKeyModel(Base):  # type: ignore[misc]
    """API key model for service-to-service authentication."""
    
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    key_hash = Column(String(64), nullable=False, index=True, unique=True)
    name = Column(String(255), nullable=False)
    permissions: Any = Column(ARRAY(String), nullable=False, default=[])
    is_active = Column(Integer, nullable=False, default=1)
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)


class WebhookSubscriptionModel(Base):  # type: ignore[misc]
    """Webhook subscription model."""
    
    __tablename__ = "webhook_subscriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    url = Column(String(500), nullable=False)
    events: Any = Column(ARRAY(String), nullable=False, default=[])
    secret = Column(String(255), nullable=True)
    is_active = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class WebhookDeliveryModel(Base):  # type: ignore[misc]
    """Webhook delivery attempt model."""
    
    __tablename__ = "webhook_deliveries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("webhook_subscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    payload = Column(JSONB, nullable=False, default={})
    status = Column(String(20), nullable=False, default="pending")  # pending, delivered, failed
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=5)
    http_status = Column(Integer, nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)
