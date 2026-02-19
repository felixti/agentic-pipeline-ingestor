"""FastAPI application entry point for the Agentic Data Pipeline Ingestor.

This module initializes the FastAPI application with all routes, middleware,
and lifecycle management for the document processing pipeline.
"""

import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.api.routes import chunks, health, search
from src.config import settings
from src.db.models import _engine as db_engine
from src.db.models import init_db, init_engine
from src.observability.logging import get_logger, setup_logging
from src.observability.metrics import SYSTEM_INFO, get_metrics_manager

# Observability imports
from src.observability.tracing import setup_tracing_from_settings
from src.plugins.loaders import AutoDiscoveryPluginLoader
from src.plugins.registry import get_registry

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.
    
    Handles startup and shutdown events:
    - Initialize database connection
    - Load plugins
    - Initialize LLM provider
    - Cleanup resources on shutdown
    
    Args:
        app: FastAPI application instance
        
    Yields:
        Control to the application
    """
    # Startup
    # Setup structured logging
    setup_logging(json_format=settings.env == "production")

    # Setup OpenTelemetry tracing
    setup_tracing_from_settings()

    # Record system info in metrics
    SYSTEM_INFO.info({
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.env,
    })

    logger.info(
        "starting_application",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.env,
    )

    try:
        # Initialize database
        logger.info("initializing_database", url=str(settings.database.url))
        init_engine(
            str(settings.database.url),
            echo=settings.database.echo,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
        )

        # Create tables (in production, use Alembic migrations)
        if db_engine:
            await init_db(db_engine)
            logger.info("database_initialized")

        # Load plugins
        logger.info("loading_plugins")
        loader = AutoDiscoveryPluginLoader()
        plugin_counts = loader.load_all()
        logger.info(
            "plugins_loaded",
            **plugin_counts,
            total=sum(plugin_counts.values()),
        )

        # Initialize plugins
        registry = get_registry()
        init_failures = await registry.initialize_all()
        if init_failures:
            logger.warning(
                "plugin_initialization_failures",
                failures={k: str(v) for k, v in init_failures.items()},
            )

        logger.info("application_startup_complete")

    except Exception as e:
        logger.error("startup_failed", error=str(e), exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("shutting_down_application")

    try:
        # Shutdown plugins
        registry = get_registry()
        await registry.shutdown_all()
        logger.info("plugins_shutdown")

        # Close database connections
        if db_engine:
            await db_engine.dispose()
            logger.info("database_connections_closed")

        logger.info("application_shutdown_complete")

    except Exception as e:
        logger.error("shutdown_error", error=str(e), exc_info=True)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Enterprise-grade agentic data pipeline for document ingestion",
        docs_url="/docs" if settings.env != "production" else None,
        redoc_url="/redoc" if settings.env != "production" else None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add middleware
    _add_middleware(app)

    # Add routes
    _add_routes(app)

    # Include health check router
    app.include_router(health.router)

    # Include chunks router
    app.include_router(chunks.router, prefix="/api/v1")

    # Include search router
    app.include_router(search.router, prefix="/api/v1")

    return app


def _add_middleware(app: FastAPI) -> None:
    """Add middleware to the application.
    
    Args:
        app: FastAPI application
    """
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=settings.security.cors_allow_credentials,
        allow_methods=settings.security.cors_allow_methods,
        allow_headers=settings.security.cors_allow_headers,
    )

    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Observability middleware for tracing and metrics
    @app.middleware("http")
    async def observability_middleware(request: Request, call_next: "Callable[[Request], Awaitable[Response]]") -> Response:
        """Middleware for tracing, metrics, and logging."""
        from opentelemetry.trace import SpanKind, Status, StatusCode

        from src.observability.tracing import get_tracer

        start_time = time.time()
        tracer = get_tracer("fastapi")

        # Extract route info
        route = request.url.path
        method = request.method

        # Start span
        with tracer.start_as_current_span(
            name=f"{method} {route}",
            kind=SpanKind.SERVER,
            attributes={
                "http.method": method,
                "http.route": route,
                "http.target": str(request.url.path),
                "http.scheme": request.url.scheme,
                "http.host": request.headers.get("host", "unknown"),
                "http.user_agent": request.headers.get("user-agent", "unknown"),
            },
        ) as span:
            try:
                response = await call_next(request)

                duration = time.time() - start_time
                status_code = response.status_code

                # Update span
                span.set_attribute("http.status_code", status_code)

                if status_code >= 400:
                    span.set_status(Status(StatusCode.ERROR))

                # Record metrics
                metrics = get_metrics_manager()
                metrics.record_api_request(
                    method=method,
                    endpoint=route,
                    status_code=status_code,
                    duration=duration,
                )

                return response

            except Exception as e:
                duration = time.time() - start_time

                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)

                # Record error metrics
                metrics = get_metrics_manager()
                metrics.record_api_request(
                    method=method,
                    endpoint=route,
                    status_code=500,
                    duration=duration,
                )

                raise

    # Request ID and timing middleware
    @app.middleware("http")
    async def add_request_metadata(request: Request, call_next: "Callable[[Request], Awaitable[Response]]") -> Response:
        """Add request ID and track timing."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Store request ID in request state
        request.state.request_id = request_id

        # Add request ID to logger context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-API-Version"] = "v1"

            # Log request completion
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "request_failed",
                error=str(e),
                duration_ms=round(duration_ms, 2),
                exc_info=True,
            )
            raise


def _add_routes(app: FastAPI) -> None:
    """Add routes to the application.
    
    Args:
        app: FastAPI application
    """
    # API version prefix
    api_prefix = "/api/v1"

    # Health checks
    @app.get("/health", tags=["System"])
    async def get_health(request: Request) -> dict[str, Any]:  # type: ignore[return]
        """Comprehensive health check endpoint."""
        from src.api.models import (
            ComponentHealth,
            HealthStatus,
            HealthStatusResponse,
        )

        components: dict[str, ComponentHealth] = {
            "api": ComponentHealth(status=HealthStatus.HEALTHY),
        }

        # Check database
        try:
            from sqlalchemy import text

            from src.db.models import get_session
            async for session in get_session():
                await session.execute(text("SELECT 1"))
                components["database"] = ComponentHealth(status=HealthStatus.HEALTHY)
                break
        except Exception as e:
            components["database"] = ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

        # Check plugins
        try:
            registry = get_registry()
            plugin_health = await registry.health_check_all()
            healthy_plugins = sum(1 for h in plugin_health.values() if h)
            total_plugins = len(plugin_health)

            if healthy_plugins == total_plugins:
                components["plugins"] = ComponentHealth(status=HealthStatus.HEALTHY)
            elif healthy_plugins > 0:
                components["plugins"] = ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    message=f"{healthy_plugins}/{total_plugins} plugins healthy",
                )
            else:
                components["plugins"] = ComponentHealth(
                    status=HealthStatus.UNHEALTHY,
                    message="No plugins initialized",
                )
        except Exception as e:
            components["plugins"] = ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

        # Overall status
        overall_status = HealthStatus.HEALTHY
        if any(c.status == HealthStatus.UNHEALTHY for c in components.values()):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components.values()):
            overall_status = HealthStatus.DEGRADED

        return HealthStatusResponse(
            status=overall_status,
            version=settings.app_version,
            components=components,
        ).model_dump()

    @app.get("/health/ready", tags=["System"])
    async def get_readiness() -> dict:
        """Kubernetes readiness probe."""
        from src.api.models import HealthReady
        return HealthReady().model_dump()

    @app.get("/health/live", tags=["System"])
    async def get_liveness() -> dict:
        """Kubernetes liveness probe."""
        from src.api.models import HealthAlive
        return HealthAlive().model_dump()

    @app.get("/health/queue", tags=["System"])
    async def get_queue_health(request: Request) -> dict:
        """Queue health and status."""
        from src.api.models import ApiResponse
        from src.core.queue import get_queue

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        try:
            queue = get_queue()
            depths = await queue.get_queue_depths()
            processing = await queue.get_processing_count()
            
            return ApiResponse.create(
                data={
                    "status": "healthy",
                    "queue_depths": depths,
                    "processing": processing,
                    "total_pending": sum(depths.values()),
                    "total_processing": sum(processing.values()) if isinstance(processing, dict) else 0,
                },
                request_id=request_id,
            ).model_dump()
        except Exception as e:
            return ApiResponse.create(
                data={
                    "status": "unhealthy",
                    "error": str(e),
                },
                request_id=request_id,
            ).model_dump()

    @app.get("/metrics", tags=["System"])
    async def get_metrics() -> PlainTextResponse:
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get("/api/v1/openapi.yaml", tags=["System"])
    async def get_openapi_yaml() -> PlainTextResponse:
        """Get OpenAPI specification as YAML."""
        openapi_path = Path(__file__).parent.parent / "api" / "openapi.yaml"
        if openapi_path.exists():
            content = openapi_path.read_text()
            return PlainTextResponse(content=content, media_type="text/yaml")
        else:
            return PlainTextResponse(
                content="# OpenAPI spec not found",
                status_code=status.HTTP_404_NOT_FOUND,
            )

    # Job Management Routes
    @app.post(f"{api_prefix}/jobs", status_code=status.HTTP_202_ACCEPTED, tags=["Jobs"])
    async def create_job(
        request: Request,
        job_data: dict[str, Any],
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Submit a new ingestion job."""
        from src.api.models import ApiResponse, JobResponse, SourceType
        from src.core.queue import JobQueue, QueuePriority, get_queue
        from src.db.repositories.job import JobRepository
        from src.db.repositories.pipeline import PipelineRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Validate required fields
        if not job_data.get("source_type"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "MISSING_SOURCE_TYPE", "message": "source_type is required"}}
            )

        # Validate pipeline if provided
        pipeline_id = job_data.get("pipeline_id")
        pipeline_config = None
        if pipeline_id:
            pipeline_repo = PipelineRepository(db)
            pipeline = await pipeline_repo.get_by_id(pipeline_id)
            if pipeline is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": {"code": "INVALID_PIPELINE", "message": f"Pipeline '{pipeline_id}' not found"}},
                )
            pipeline_config = pipeline.config

        # Create job via repository
        repo = JobRepository(db)
        job = await repo.create(
            source_type=job_data.get("source_type"),  # type: ignore[arg-type]
            source_uri=job_data.get("source_uri"),
            file_name=job_data.get("file_name"),
            file_size=job_data.get("file_size"),
            mime_type=job_data.get("mime_type"),
            priority=job_data.get("priority", "normal"),
            mode=job_data.get("mode", "async"),
            external_id=job_data.get("external_id"),
            metadata=job_data.get("metadata", {}),
            pipeline_id=pipeline_id,
            pipeline_config=pipeline_config,  # type: ignore[arg-type]
        )

        # Enqueue job to Redis for processing
        try:
            queue = get_queue()
            priority = job_data.get("priority", "normal")
            queue_priority = QueuePriority.NORMAL
            if priority == "high":
                queue_priority = QueuePriority.HIGH
            elif priority == "low":
                queue_priority = QueuePriority.LOW
            
            await queue.enqueue(str(job.id), priority=queue_priority)
            logger.info("job_enqueued", job_id=str(job.id), priority=priority)
        except Exception as e:
            # Log error but don't fail the request - job will be picked up by DB polling
            logger.warning("failed_to_enqueue_job", job_id=str(job.id), error=str(e))

        # Build response
        response_data = JobResponse(
            id=job.id,  # type: ignore[arg-type]
            status=job.status,  # type: ignore[arg-type]
            source_type=SourceType(job.source_type),  # type: ignore[arg-type]
            source_uri=job.source_uri or "",  # type: ignore[arg-type]
            file_name=job.file_name,  # type: ignore[arg-type]
            file_size=job.file_size,  # type: ignore[arg-type]
            mime_type=job.mime_type,  # type: ignore[arg-type]
            priority=5 if job.priority == "normal" else 10 if job.priority == "high" else 1,
            mode=job.mode,  # type: ignore[arg-type]
            external_id=job.external_id,  # type: ignore[arg-type]
            retry_count=job.retry_count,  # type: ignore[arg-type]
            max_retries=job.max_retries,  # type: ignore[arg-type]
            created_at=job.created_at,  # type: ignore[arg-type]
            updated_at=job.updated_at,  # type: ignore[arg-type]
            started_at=job.started_at,  # type: ignore[arg-type]
            completed_at=job.completed_at,  # type: ignore[arg-type]
        )

        return ApiResponse.create(
            data=response_data.model_dump(),
            request_id=request_id,  # type: ignore[arg-type]
        ).model_dump()

    @app.get(f"{api_prefix}/jobs", tags=["Jobs"])
    async def list_jobs(
        request: Request,
        page: int = 1,
        limit: int = 20,
        status: str | None = None,
        source_type: str | None = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """List jobs with filtering."""
        from src.api.models import ApiLinks, ApiResponse, JobListResponse, JobResponse, SourceType
        from src.db.repositories.job import JobRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Get jobs from repository
        repo = JobRepository(db)
        jobs, total = await repo.list_jobs(
            page=page,
            limit=limit,
            status=status,
            source_type=source_type,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        # Convert to response models
        job_responses = []
        for job in jobs:
            job_responses.append(JobResponse(
                id=job.id,  # type: ignore[arg-type]
                status=job.status,  # type: ignore[arg-type]
                source_type=SourceType(job.source_type) if job.source_type in [e.value for e in SourceType] else SourceType.UPLOAD,  # type: ignore[arg-type]
                source_uri=job.source_uri or "",  # type: ignore[arg-type]
                file_name=job.file_name,  # type: ignore[arg-type]
                file_size=job.file_size,  # type: ignore[arg-type]
                mime_type=job.mime_type,  # type: ignore[arg-type]
                priority=5 if job.priority == "normal" else 10 if job.priority == "high" else 1,
                mode=job.mode,  # type: ignore[arg-type]
                external_id=job.external_id,  # type: ignore[arg-type]
                retry_count=job.retry_count,  # type: ignore[arg-type]
                max_retries=job.max_retries,  # type: ignore[arg-type]
                created_at=job.created_at,  # type: ignore[arg-type]
                updated_at=job.updated_at,  # type: ignore[arg-type]
                started_at=job.started_at,  # type: ignore[arg-type]
                completed_at=job.completed_at,  # type: ignore[arg-type]
            ))

        # Build list response
        list_response = JobListResponse(
            items=job_responses,
            total=total,
            page=page,
            page_size=limit,
        )

        # Build pagination links
        links = ApiLinks(self=str(request.url))
        base_url = str(request.url).split("?")[0]
        
        if page > 1:
            links.prev = f"{base_url}?page={page-1}&limit={limit}"
        if (page * limit) < total:
            links.next = f"{base_url}?page={page+1}&limit={limit}"

        return ApiResponse.create(
            data=list_response.model_dump(),
            request_id=request_id,  # type: ignore[arg-type]
            total_count=total,
            links=links,
        ).model_dump()

    @app.get(f"{api_prefix}/jobs/{{job_id}}", tags=["Jobs"])
    async def get_job(
        job_id: str,
        request: Request,
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Get job details."""
        from src.api.models import ApiResponse, JobResponse, SourceType
        from src.db.repositories.job import JobRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Get job from repository
        repo = JobRepository(db)
        job = await repo.get_by_id(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job with ID '{job_id}' not found",
            )

        # Build response
        response_data = JobResponse(
            id=job.id,  # type: ignore[arg-type]
            status=job.status,  # type: ignore[arg-type]
            source_type=SourceType(job.source_type) if job.source_type in [e.value for e in SourceType] else SourceType.UPLOAD,  # type: ignore[arg-type]
            source_uri=job.source_uri or "",  # type: ignore[arg-type]
            file_name=job.file_name,  # type: ignore[arg-type]
            file_size=job.file_size,  # type: ignore[arg-type]
            mime_type=job.mime_type,  # type: ignore[arg-type]
            priority=5 if job.priority == "normal" else 10 if job.priority == "high" else 1,
            mode=job.mode,  # type: ignore[arg-type]
            external_id=job.external_id,  # type: ignore[arg-type]
            retry_count=job.retry_count,  # type: ignore[arg-type]
            max_retries=job.max_retries,  # type: ignore[arg-type]
            created_at=job.created_at,  # type: ignore[arg-type]
            updated_at=job.updated_at,  # type: ignore[arg-type]
            started_at=job.started_at,  # type: ignore[arg-type]
            completed_at=job.completed_at,  # type: ignore[arg-type]
            error={"message": job.error_message, "code": job.error_code} if job.error_message or job.error_code else None,  # type: ignore[arg-type]
        )

        return ApiResponse.create(
            data=response_data.model_dump(),
            request_id=request_id,
        ).model_dump()

    @app.delete(f"{api_prefix}/jobs/{{job_id}}", status_code=status.HTTP_204_NO_CONTENT, tags=["Jobs"])
    async def cancel_job(
        job_id: str,
        db: AsyncSession = Depends(get_db),
    ) -> None:
        """Cancel a job."""
        from src.db.models import JobStatus
        from src.db.repositories.job import JobRepository

        repo = JobRepository(db)
        job = await repo.get_by_id(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job with ID '{job_id}' not found",
            )

        # Only allow cancellation of jobs that can be cancelled
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job with status '{job.status}'",
            )

        # Update status to cancelled
        await repo.update_status(job_id, JobStatus.CANCELLED)

    @app.post(f"{api_prefix}/jobs/{{job_id}}/retry", status_code=status.HTTP_202_ACCEPTED, tags=["Jobs"])
    async def retry_job(
        job_id: str,
        request: Request,
        retry_data: dict[str, Any] | None = None,
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Retry a failed job."""
        from src.api.models import ApiResponse, JobResponse, SourceType
        from src.db.models import JobStatus
        from src.db.repositories.job import JobRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        repo = JobRepository(db)
        
        # Get original job
        original_job = await repo.get_by_id(job_id)
        if original_job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job with ID '{job_id}' not found",
            )

        # Only allow retry of failed jobs
        if original_job.status != JobStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot retry job with status '{original_job.status}'. Only failed jobs can be retried.",
            )

        # Create new job with same configuration
        new_job = await repo.create(
            source_type=original_job.source_type,  # type: ignore[arg-type]
            source_uri=original_job.source_uri,  # type: ignore[arg-type]
            file_name=original_job.file_name,  # type: ignore[arg-type]
            file_size=original_job.file_size,  # type: ignore[arg-type]
            mime_type=original_job.mime_type,  # type: ignore[arg-type]
            priority=original_job.priority,  # type: ignore[arg-type]
            mode=original_job.mode,  # type: ignore[arg-type]
            external_id=original_job.external_id,  # type: ignore[arg-type]
            metadata=original_job.metadata_json,  # type: ignore[arg-type]
        )

        # Increment retry count on original job
        # Increment retry count on original job
        original_job.retry_count = original_job.retry_count + 1  # type: ignore[assignment]
        await db.commit()

        # Build response
        response_data = JobResponse(
            id=new_job.id,  # type: ignore[arg-type]
            status=new_job.status,  # type: ignore[arg-type]
            source_type=SourceType(new_job.source_type) if new_job.source_type in [e.value for e in SourceType] else SourceType.UPLOAD,  # type: ignore[arg-type]
            source_uri=new_job.source_uri or "",  # type: ignore[arg-type]
            file_name=new_job.file_name,  # type: ignore[arg-type]
            file_size=new_job.file_size,  # type: ignore[arg-type]
            mime_type=new_job.mime_type,  # type: ignore[arg-type]
            priority=5 if new_job.priority == "normal" else 10 if new_job.priority == "high" else 1,
            mode=new_job.mode,  # type: ignore[arg-type]
            external_id=new_job.external_id,  # type: ignore[arg-type]
            retry_count=new_job.retry_count,  # type: ignore[arg-type]
            max_retries=new_job.max_retries,  # type: ignore[arg-type]
            created_at=new_job.created_at,  # type: ignore[arg-type]
            updated_at=new_job.updated_at,  # type: ignore[arg-type]
            started_at=new_job.started_at,  # type: ignore[arg-type]
            completed_at=new_job.completed_at,  # type: ignore[arg-type]
        )

        return ApiResponse.create(
            data={
                "original_job_id": str(job_id),
                "new_job_id": str(new_job.id),
                "status": new_job.status,
                "message": "Job retry initiated",
                "job": response_data.model_dump(),
            },
            request_id=request_id,
        ).model_dump()

    @app.get(f"{api_prefix}/jobs/{{job_id}}/result", tags=["Jobs"])
    async def get_job_result(
        job_id: str,
        request: Request,
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Get job processing result."""
        from src.api.models import ApiResponse
        from src.db.models import JobStatus
        from src.db.repositories.job import JobRepository
        from src.db.repositories.job_result import JobResultRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Get job to check status
        job_repo = JobRepository(db)
        job = await job_repo.get_by_id(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job with ID '{job_id}' not found",
            )

        # Only return results for completed jobs
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "JOB_NOT_COMPLETE",
                        "message": f"Job is not complete. Current status: {job.status}",
                        "status": job.status,
                    }
                },
            )

        # Get result
        result_repo = JobResultRepository(db)
        result = await result_repo.get_by_job_id(job_id)

        if result is None:
            # Job is completed but no result stored yet
            return ApiResponse.create(
                data={
                    "job_id": job_id,
                    "status": job.status,
                    "success": True,
                    "message": "Job completed but result not yet available",
                },
                request_id=request_id,
            ).model_dump()

        # Build response
        response_data = {
            "job_id": job_id,
            "status": job.status,
            "success": True,
            "extracted_text": result.extracted_text,
            "output_data": result.output_data,
            "metadata": result.result_metadata,
            "quality_score": result.quality_score,
            "processing_time_ms": result.processing_time_ms,
            "output_uri": result.output_uri,
            "created_at": result.created_at.isoformat() if result.created_at else None,
        }

        return ApiResponse.create(
            data=response_data,
            request_id=request_id,
        ).model_dump()

    # File Upload Routes
    @app.post(f"{api_prefix}/upload", status_code=status.HTTP_202_ACCEPTED, tags=["Upload"])
    async def upload_files(
        request: Request,
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Upload file(s) for processing."""
        import shutil
        from pathlib import Path

        from fastapi import UploadFile

        from src.api.models import ApiResponse, UploadMultipleResponse, UploadResponse
        from src.db.repositories.job import JobRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Parse multipart form data manually
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("multipart/form-data"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "INVALID_CONTENT_TYPE", "message": "Content-Type must be multipart/form-data"}}
            )
        
        # Read and parse form data
        form_data = await request.form()
        
        # Extract files from form data
        files: list[UploadFile] = []
        for key, value in form_data.multi_items():
            if hasattr(value, "filename") and value.filename:
                files.append(value)  # type: ignore[arg-type]

        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "NO_FILES", "message": "No files provided"}}
            )

        # Create staging directory
        staging_dir = Path("/tmp/pipeline/uploads")
        staging_dir.mkdir(parents=True, exist_ok=True)

        repo = JobRepository(db)
        uploaded_jobs = []

        for upload_file in files:
            # Generate unique file path
            file_id = str(uuid.uuid4())
            file_ext = Path(upload_file.filename or "unknown").suffix
            file_path = staging_dir / f"{file_id}{file_ext}"

            try:
                # Save uploaded file
                content = await upload_file.read()
                with open(file_path, "wb") as buffer:
                    buffer.write(content)

                # Get file size
                file_size = file_path.stat().st_size

                # Create job for this file
                job = await repo.create(
                    source_type="upload",
                    source_uri=str(file_path),
                    file_name=upload_file.filename,
                    file_size=file_size,
                    mime_type=upload_file.content_type or "application/octet-stream",
                    priority="normal",
                    mode="async",
                    metadata={"original_filename": upload_file.filename},
                )

                uploaded_jobs.append({
                    "job_id": job.id,
                    "file_name": upload_file.filename,
                    "file_size": file_size,
                })

            except Exception as e:
                # Clean up file on error
                if file_path.exists():
                    file_path.unlink()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"error": {"code": "UPLOAD_FAILED", "message": f"Failed to process {upload_file.filename}: {e!s}"}}
                )

        # Build response
        response_data: UploadResponse | UploadMultipleResponse
        if len(uploaded_jobs) == 1:
            response_data = UploadResponse(
                message="File uploaded successfully",
                job_id=uploaded_jobs[0]["job_id"],  # type: ignore[arg-type]
                file_name=uploaded_jobs[0]["file_name"],  # type: ignore[arg-type]
                file_size=uploaded_jobs[0]["file_size"],  # type: ignore[arg-type]
            )
        else:
            response_data = UploadMultipleResponse(
                message=f"{len(uploaded_jobs)} files uploaded successfully",
                job_ids=[j["job_id"] for j in uploaded_jobs],  # type: ignore[misc]
                files=[j["file_name"] for j in uploaded_jobs],  # type: ignore[misc]
            )

        return ApiResponse.create(
            data=response_data.model_dump(),
            request_id=request_id,
        ).model_dump()

    @app.post(f"{api_prefix}/upload/url", status_code=status.HTTP_202_ACCEPTED, tags=["Upload"])
    async def ingest_from_url(
        request: Request,
        url_data: dict[str, Any],
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Ingest from URL."""
        from pathlib import Path

        import httpx

        from src.api.models import ApiResponse, UploadResponse
        from src.db.repositories.job import JobRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        url = url_data.get("url")
        if not url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "MISSING_URL", "message": "URL is required"}}
            )

        # Create staging directory
        staging_dir = Path("/tmp/pipeline/staging")
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download file from URL
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0, follow_redirects=True)
                response.raise_for_status()
                
                # Determine filename
                filename = url_data.get("filename")
                if not filename:
                    # Try to get from Content-Disposition header
                    content_disp = response.headers.get("content-disposition", "")
                    if "filename=" in content_disp:
                        filename = content_disp.split("filename=")[1].strip('"\'')
                    else:
                        # Extract from URL
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        filename = Path(parsed.path).name or "downloaded_file"
                
                # Save file
                file_id = str(uuid.uuid4())
                file_ext = Path(filename).suffix
                file_path = staging_dir / f"{file_id}{file_ext}"
                
                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                file_size = len(response.content)
                content_type = response.headers.get("content-type", "application/octet-stream")

            # Create job
            repo = JobRepository(db)
            job = await repo.create(
                source_type="url",
                source_uri=url,
                file_name=filename,
                file_size=file_size,
                mime_type=content_type,
                priority=url_data.get("priority", "normal"),
                mode=url_data.get("mode", "async"),
                external_id=url_data.get("external_id"),
                metadata={"source_url": url, "headers": dict(url_data.get("headers", {}))},
            )

            response_data = UploadResponse(
                message="URL ingestion started",
                job_id=job.id,  # type: ignore[arg-type]
                file_name=filename,
                file_size=file_size,
            )

            return ApiResponse.create(
                data=response_data.model_dump(),
                request_id=request_id,
            ).model_dump()

        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "DOWNLOAD_FAILED", "message": f"Failed to download from URL: {e!s}"}}
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {"code": "INGESTION_FAILED", "message": f"Failed to process URL: {e!s}"}}
            )

    # Pipeline Configuration Routes
    @app.get(f"{api_prefix}/pipelines", tags=["Pipelines"])
    async def list_pipelines(
        request: Request,
        db: AsyncSession = Depends(get_db),
        page: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List pipeline configurations."""
        from src.api.models import ApiLinks, ApiResponse
        from src.db.repositories.pipeline import PipelineRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        repo = PipelineRepository(db)
        pipelines, total = await repo.list_pipelines(page=page, limit=limit)

        # Build response
        pipeline_data = []
        for p in pipelines:
            pipeline_data.append({
                "id": str(p.id),
                "name": p.name,
                "description": p.description,
                "config": p.config,
                "version": p.version,
                "is_active": bool(p.is_active),
                "created_by": p.created_by,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None,
            })

        # Build pagination links
        links = ApiLinks(self=str(request.url))
        base_url = str(request.url).split("?")[0]
        
        if page > 1:
            links.prev = f"{base_url}?page={page-1}&limit={limit}"
        if (page * limit) < total:
            links.next = f"{base_url}?page={page+1}&limit={limit}"

        return ApiResponse.create(
            data={
                "items": pipeline_data,
                "total": total,
                "page": page,
                "page_size": limit,
            },
            request_id=request_id,
            total_count=total,
            links=links,
        ).model_dump()

    @app.post(f"{api_prefix}/pipelines", status_code=status.HTTP_201_CREATED, tags=["Pipelines"])
    async def create_pipeline(
        request: Request,
        pipeline_data: dict[str, Any],
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Create pipeline configuration."""
        from src.api.models import ApiResponse
        from src.db.repositories.pipeline import PipelineRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Validate required fields
        name = pipeline_data.get("name")
        if not name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "MISSING_NAME", "message": "Pipeline name is required"}},
            )

        config = pipeline_data.get("config", {})
        
        # Validate config
        repo = PipelineRepository(db)
        is_valid, errors = repo.validate_config(config)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "INVALID_CONFIG", "message": "Invalid configuration", "errors": errors}},
            )

        # Create pipeline
        pipeline = await repo.create(
            name=name,
            description=pipeline_data.get("description"),
            config=config,
            created_by=pipeline_data.get("created_by"),
        )

        return ApiResponse.create(
            data={
                "id": str(pipeline.id),
                "name": pipeline.name,
                "description": pipeline.description,
                "config": pipeline.config,
                "version": pipeline.version,
                "created_at": pipeline.created_at.isoformat() if pipeline.created_at else None,
            },
            request_id=request_id,
        ).model_dump()

    @app.get(f"{api_prefix}/pipelines/{{pipeline_id}}", tags=["Pipelines"])
    async def get_pipeline(
        pipeline_id: str,
        request: Request,
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Get pipeline configuration."""
        from src.api.models import ApiResponse
        from src.db.repositories.pipeline import PipelineRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        repo = PipelineRepository(db)
        pipeline = await repo.get_by_id(pipeline_id)

        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline with ID '{pipeline_id}' not found",
            )

        return ApiResponse.create(
            data={
                "id": str(pipeline.id),
                "name": pipeline.name,
                "description": pipeline.description,
                "config": pipeline.config,
                "version": pipeline.version,
                "is_active": bool(pipeline.is_active),
                "created_by": pipeline.created_by,
                "created_at": pipeline.created_at.isoformat() if pipeline.created_at else None,
                "updated_at": pipeline.updated_at.isoformat() if pipeline.updated_at else None,
            },
            request_id=request_id,
        ).model_dump()

    @app.put(f"{api_prefix}/pipelines/{{pipeline_id}}", tags=["Pipelines"])
    async def update_pipeline(
        pipeline_id: str,
        request: Request,
        pipeline_data: dict[str, Any],
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Update pipeline configuration."""
        from src.api.models import ApiResponse
        from src.db.repositories.pipeline import PipelineRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        repo = PipelineRepository(db)
        
        # Check if pipeline exists
        existing = await repo.get_by_id(pipeline_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline with ID '{pipeline_id}' not found",
            )

        # Validate config if provided
        config = pipeline_data.get("config")
        if config is not None:
            is_valid, errors = repo.validate_config(config)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": {"code": "INVALID_CONFIG", "message": "Invalid configuration", "errors": errors}},
                )

        # Update pipeline
        pipeline = await repo.update(
            pipeline_id=pipeline_id,
            name=pipeline_data.get("name"),
            config=config,
            description=pipeline_data.get("description"),
        )

        return ApiResponse.create(
            data={
                "id": str(pipeline.id),
                "name": pipeline.name,
                "description": pipeline.description,
                "config": pipeline.config,
                "version": pipeline.version,
                "updated_at": pipeline.updated_at.isoformat() if pipeline.updated_at else None,
            },
            request_id=request_id,
        ).model_dump()

    @app.delete(f"{api_prefix}/pipelines/{{pipeline_id}}", status_code=status.HTTP_204_NO_CONTENT, tags=["Pipelines"])
    async def delete_pipeline(
        pipeline_id: str,
        db: AsyncSession = Depends(get_db),
    ) -> None:
        """Delete pipeline configuration."""
        from src.db.repositories.pipeline import PipelineRepository

        repo = PipelineRepository(db)
        
        # Check if pipeline exists
        existing = await repo.get_by_id(pipeline_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline with ID '{pipeline_id}' not found",
            )

        # Delete pipeline (soft delete)
        await repo.delete(pipeline_id, soft_delete=True)

    # Sources & Destinations Routes
    @app.get(f"{api_prefix}/sources", tags=["Sources"])
    async def list_sources(request: Request) -> dict[str, Any]:
        """List source plugins."""
        from src.api.models import ApiResponse
        from src.plugins.registry import get_registry

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        registry = get_registry()

        plugins = registry.list_sources()

        return ApiResponse.create(
            data={
                "plugins": [p.__dict__ for p in plugins],
                "configurations": [],
            },
            request_id=request_id,
        ).model_dump()

    @app.get(f"{api_prefix}/destinations", tags=["Destinations"])
    async def list_destinations(request: Request) -> dict[str, Any]:
        """List destination plugins."""
        from src.api.models import ApiResponse
        from src.plugins.registry import get_registry

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        registry = get_registry()

        plugins = registry.list_destinations()

        return ApiResponse.create(
            data={
                "plugins": [p.__dict__ for p in plugins],
                "configurations": [],
            },
            request_id=request_id,
        ).model_dump()

    # Authentication Routes
    @app.post(f"{api_prefix}/auth/login", tags=["Authentication"])
    async def login(
        request: Request,
        credentials: dict[str, Any],
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Login and get JWT token."""
        from uuid import uuid4
        
        from src.api.models import ApiResponse
        from src.auth.jwt import JWTHandler
        from src.config import settings

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Validate credentials (simplified - in production use proper user auth)
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "MISSING_CREDENTIALS", "message": "Username and password are required"}},
            )

        # Create JWT handler
        jwt_handler = JWTHandler(
            secret_key=settings.security.secret_key,
            algorithm=settings.security.algorithm,
            access_token_expire_minutes=settings.security.access_token_expire_minutes,
        )

        # Generate tokens
        user_id = uuid4()  # In production, look up user from database
        access_token = jwt_handler.create_access_token(
            user_id=user_id,
            username=username,
            roles=credentials.get("roles", ["operator"]),
        )
        refresh_token = jwt_handler.create_refresh_token(user_id=user_id)

        return ApiResponse.create(
            data={
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": settings.security.access_token_expire_minutes * 60,
            },
            request_id=request_id,
        ).model_dump()

    @app.post(f"{api_prefix}/auth/refresh", tags=["Authentication"])
    async def refresh_token(
        request: Request,
        refresh_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Refresh access token."""
        from src.api.models import ApiResponse
        from src.auth.jwt import JWTHandler
        from src.config import settings

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        refresh_token = refresh_data.get("refresh_token")
        if not refresh_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "MISSING_TOKEN", "message": "Refresh token is required"}},
            )

        try:
            jwt_handler = JWTHandler(
                secret_key=settings.security.secret_key,
                algorithm=settings.security.algorithm,
                access_token_expire_minutes=settings.security.access_token_expire_minutes,
            )

            new_token = jwt_handler.refresh_access_token(refresh_token)

            return ApiResponse.create(
                data={
                    "access_token": new_token,
                    "token_type": "bearer",
                    "expires_in": settings.security.access_token_expire_minutes * 60,
                },
                request_id=request_id,
            ).model_dump()

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": {"code": "INVALID_TOKEN", "message": str(e)}},
            )

    @app.post(f"{api_prefix}/auth/api-keys", status_code=status.HTTP_201_CREATED, tags=["Authentication"])
    async def create_api_key(
        request: Request,
        key_data: dict[str, Any],
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Create a new API key."""
        from datetime import datetime, timedelta
        
        from src.api.models import ApiResponse
        from src.db.repositories.api_key import APIKeyRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        repo = APIKeyRepository(db)
        
        # Parse expiration
        expires_at = None
        expires_in_days = key_data.get("expires_in_days")
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create API key
        api_key, raw_key = await repo.create(
            name=key_data.get("name", "API Key"),
            permissions=key_data.get("permissions", []),
            created_by=key_data.get("created_by"),
            expires_at=expires_at,
        )

        return ApiResponse.create(
            data={
                "id": str(api_key.id),
                "name": api_key.name,
                "api_key": raw_key,  # Only returned once
                "permissions": api_key.permissions,
                "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            },
            request_id=request_id,
        ).model_dump()

    @app.get(f"{api_prefix}/auth/api-keys", tags=["Authentication"])
    async def list_api_keys(
        request: Request,
        db: AsyncSession = Depends(get_db),
        page: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List API keys."""
        from src.api.models import ApiLinks, ApiResponse
        from src.db.repositories.api_key import APIKeyRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        repo = APIKeyRepository(db)
        keys, total = await repo.list_keys(page=page, limit=limit)

        # Build response (excluding key_hash)
        key_data = []
        for k in keys:
            key_data.append({
                "id": str(k.id),
                "name": k.name,
                "permissions": k.permissions,
                "is_active": bool(k.is_active),
                "created_by": k.created_by,
                "created_at": k.created_at.isoformat() if k.created_at else None,
                "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            })

        links = ApiLinks(self=str(request.url))
        base_url = str(request.url).split("?")[0]
        if page > 1:
            links.prev = f"{base_url}?page={page-1}&limit={limit}"
        if (page * limit) < total:
            links.next = f"{base_url}?page={page+1}&limit={limit}"

        return ApiResponse.create(
            data={
                "items": key_data,
                "total": total,
                "page": page,
                "page_size": limit,
            },
            request_id=request_id,
            total_count=total,
            links=links,
        ).model_dump()

    @app.delete(f"{api_prefix}/auth/api-keys/{{key_id}}", status_code=status.HTTP_204_NO_CONTENT, tags=["Authentication"])
    async def revoke_api_key(
        key_id: str,
        db: AsyncSession = Depends(get_db),
    ) -> None:
        """Revoke an API key."""
        from src.db.repositories.api_key import APIKeyRepository

        repo = APIKeyRepository(db)
        deactivated = await repo.deactivate(key_id)

        if not deactivated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key '{key_id}' not found",
            )

    # Webhook Routes
    @app.post(f"{api_prefix}/webhooks", status_code=status.HTTP_201_CREATED, tags=["Webhooks"])
    async def create_webhook(
        request: Request,
        webhook_data: dict[str, Any],
        db: AsyncSession = Depends(get_db),
    ) -> dict[str, Any]:
        """Create a webhook subscription."""
        from src.api.models import ApiResponse
        from src.db.repositories.webhook import WebhookRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Validate required fields
        url = webhook_data.get("url")
        events = webhook_data.get("events", [])
        
        if not url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "MISSING_URL", "message": "Webhook URL is required"}},
            )
        
        if not events:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "MISSING_EVENTS", "message": "At least one event type is required"}},
            )

        # Create subscription
        repo = WebhookRepository(db)
        sub = await repo.create_subscription(
            user_id=webhook_data.get("user_id", "anonymous"),
            url=url,
            events=events,
            secret=webhook_data.get("secret"),
        )

        return ApiResponse.create(
            data={
                "id": str(sub.id),
                "url": sub.url,
                "events": sub.events,
                "secret": sub.secret,
                "created_at": sub.created_at.isoformat() if sub.created_at else None,
            },
            request_id=request_id,
        ).model_dump()

    @app.get(f"{api_prefix}/webhooks", tags=["Webhooks"])
    async def list_webhooks(
        request: Request,
        db: AsyncSession = Depends(get_db),
        user_id: str | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List webhook subscriptions."""
        from src.api.models import ApiLinks, ApiResponse
        from src.db.repositories.webhook import WebhookRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        repo = WebhookRepository(db)
        subs, total = await repo.list_subscriptions(
            user_id=user_id,
            page=page,
            limit=limit,
        )

        # Build response (excluding secret)
        sub_data = []
        for s in subs:
            sub_data.append({
                "id": str(s.id),
                "url": s.url,
                "events": s.events,
                "is_active": bool(s.is_active),
                "created_at": s.created_at.isoformat() if s.created_at else None,
            })

        links = ApiLinks(self=str(request.url))
        base_url = str(request.url).split("?")[0]
        if page > 1:
            links.prev = f"{base_url}?page={page-1}&limit={limit}"
        if (page * limit) < total:
            links.next = f"{base_url}?page={page+1}&limit={limit}"

        return ApiResponse.create(
            data={
                "items": sub_data,
                "total": total,
                "page": page,
                "page_size": limit,
            },
            request_id=request_id,
            total_count=total,
            links=links,
        ).model_dump()

    @app.delete(f"{api_prefix}/webhooks/{{webhook_id}}", status_code=status.HTTP_204_NO_CONTENT, tags=["Webhooks"])
    async def delete_webhook(
        webhook_id: str,
        db: AsyncSession = Depends(get_db),
    ) -> None:
        """Delete a webhook subscription."""
        from src.db.repositories.webhook import WebhookRepository

        repo = WebhookRepository(db)
        deleted = await repo.delete_subscription(webhook_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook '{webhook_id}' not found",
            )

    @app.get(f"{api_prefix}/webhooks/{{webhook_id}}/deliveries", tags=["Webhooks"])
    async def list_webhook_deliveries(
        webhook_id: str,
        request: Request,
        db: AsyncSession = Depends(get_db),
        status: str | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List webhook deliveries for a subscription."""
        from uuid import UUID

        from src.api.models import ApiLinks, ApiResponse
        from src.db.repositories.webhook import WebhookRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        repo = WebhookRepository(db)
        
        # Verify subscription exists
        sub = await repo.get_subscription(UUID(webhook_id))
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook '{webhook_id}' not found",
            )

        deliveries, total = await repo.list_deliveries(
            subscription_id=webhook_id,
            status=status,
            page=page,
            limit=limit,
        )

        # Build response
        delivery_data = []
        for d in deliveries:
            delivery_data.append({
                "id": str(d.id),
                "event_type": d.event_type,
                "status": d.status,
                "attempts": d.attempts,
                "max_attempts": d.max_attempts,
                "http_status": d.http_status,
                "last_error": d.last_error,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "delivered_at": d.delivered_at.isoformat() if d.delivered_at else None,
                "next_retry_at": d.next_retry_at.isoformat() if d.next_retry_at else None,
            })

        links = ApiLinks(self=str(request.url))
        base_url = str(request.url).split("?")[0]
        if page > 1:
            links.prev = f"{base_url}?page={page-1}&limit={limit}"
        if (page * limit) < total:
            links.next = f"{base_url}?page={page+1}&limit={limit}"

        return ApiResponse.create(
            data={
                "items": delivery_data,
                "total": total,
                "page": page,
                "page_size": limit,
            },
            request_id=request_id,
            total_count=total,
            links=links,
        ).model_dump()

    # Audit Routes
    @app.get(f"{api_prefix}/audit/logs", tags=["Audit"])
    async def query_audit_logs(
        request: Request,
        db: AsyncSession = Depends(get_db),
        page: int = 1,
        limit: int = 20,
        user_id: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Query audit logs."""
        from src.api.models import ApiLinks, ApiResponse
        from src.db.repositories.audit import AuditLogRepository

        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        repo = AuditLogRepository(db)
        logs, total = await repo.query_logs(
            page=page,
            limit=limit,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            start_date=start_date,
            end_date=end_date,
        )

        # Build response
        log_data = []
        for log in logs:
            log_data.append({
                "id": str(log.id),
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "user_id": log.user_id,
                "api_key_id": log.api_key_id,
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "request_method": log.request_method,
                "request_path": log.request_path,
                "request_details": log.request_details,
                "success": bool(log.success),
                "error_message": log.error_message,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "duration_ms": log.duration_ms,
            })

        links = ApiLinks(self=str(request.url))
        base_url = str(request.url).split("?")[0]
        if page > 1:
            links.prev = f"{base_url}?page={page-1}&limit={limit}"
        if (page * limit) < total:
            links.next = f"{base_url}?page={page+1}&limit={limit}"

        return ApiResponse.create(
            data={
                "items": log_data,
                "total": total,
                "page": page,
                "page_size": limit,
            },
            request_id=request_id,
            total_count=total,
            links=links,
        ).model_dump()


def cli() -> None:
    """Command-line entry point."""
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
    )


# Create the application instance
app = create_app()

if __name__ == "__main__":
    cli()
