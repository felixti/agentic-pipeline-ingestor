"""FastAPI application entry point for the Agentic Data Pipeline Ingestor.

This module initializes the FastAPI application with all routes, middleware,
and lifecycle management for the document processing pipeline.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.config import get_settings, settings
from src.db.models import init_engine, init_db, _engine as db_engine
from src.plugins.registry import get_registry
from src.plugins.loaders import AutoDiscoveryPluginLoader

# Observability imports
from src.observability.tracing import setup_tracing_from_settings
from src.observability.metrics import get_metrics_manager, SYSTEM_INFO
from src.observability.logging import setup_logging, get_logger
from src.api.routes import health

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
    async def observability_middleware(request: Request, call_next):
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
    async def add_request_metadata(request: Request, call_next) -> JSONResponse:
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
    async def get_health(request: Request) -> dict:
        """Comprehensive health check endpoint."""
        from src.api.models import (
            HealthStatusResponse,
            ComponentHealth,
            HealthStatus,
        )
        
        components: dict = {
            "api": ComponentHealth(status=HealthStatus.HEALTHY),
        }
        
        # Check database
        try:
            from src.db.models import get_session
            async for session in get_session():
                await session.execute("SELECT 1")
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
    async def create_job(request: Request) -> dict:
        """Submit a new ingestion job."""
        from src.api.models import ApiResponse, JobCreateRequest, Job, JobStatus
        
        # TODO: Implement actual job creation
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        return ApiResponse.create(
            data={
                "id": str(uuid.uuid4()),
                "status": JobStatus.CREATED.value,
                "message": "Job accepted for processing",
            },
            request_id=request_id,
        ).model_dump()
    
    @app.get(f"{api_prefix}/jobs", tags=["Jobs"])
    async def list_jobs(
        request: Request,
        page: int = 1,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> dict:
        """List jobs with filtering."""
        from src.api.models import ApiResponse, ApiLinks
        
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # TODO: Implement actual job listing
        return ApiResponse.create(
            data=[],
            request_id=request_id,
            links=ApiLinks(self=str(request.url)),
        ).model_dump()
    
    @app.get(f"{api_prefix}/jobs/{{job_id}}", tags=["Jobs"])
    async def get_job(job_id: str, request: Request) -> dict:
        """Get job details."""
        from src.api.models import ApiResponse
        
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # TODO: Implement actual job retrieval
        return ApiResponse.create(
            data={"id": job_id, "status": "created"},
            request_id=request_id,
        ).model_dump()
    
    @app.delete(f"{api_prefix}/jobs/{{job_id}}", status_code=status.HTTP_204_NO_CONTENT, tags=["Jobs"])
    async def cancel_job(job_id: str) -> None:
        """Cancel a job."""
        # TODO: Implement actual job cancellation
        pass
    
    @app.post(f"{api_prefix}/jobs/{{job_id}}/retry", status_code=status.HTTP_202_ACCEPTED, tags=["Jobs"])
    async def retry_job(job_id: str, request: Request) -> dict:
        """Retry a failed job."""
        from src.api.models import ApiResponse
        
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # TODO: Implement actual job retry
        return ApiResponse.create(
            data={"id": job_id, "status": "retrying"},
            request_id=request_id,
        ).model_dump()
    
    @app.get(f"{api_prefix}/jobs/{{job_id}}/result", tags=["Jobs"])
    async def get_job_result(job_id: str, request: Request) -> dict:
        """Get job processing result."""
        from src.api.models import ApiResponse
        
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # TODO: Implement actual result retrieval
        return ApiResponse.create(
            data={"job_id": job_id, "success": True},
            request_id=request_id,
        ).model_dump()
    
    # File Upload Routes
    @app.post(f"{api_prefix}/upload", status_code=status.HTTP_202_ACCEPTED, tags=["Upload"])
    async def upload_files(request: Request) -> dict:
        """Upload file(s) for processing."""
        from src.api.models import ApiResponse
        
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # TODO: Implement actual file upload
        return ApiResponse.create(
            data={"message": "Files accepted", "job_id": str(uuid.uuid4())},
            request_id=request_id,
        ).model_dump()
    
    @app.post(f"{api_prefix}/upload/url", status_code=status.HTTP_202_ACCEPTED, tags=["Upload"])
    async def ingest_from_url(request: Request) -> dict:
        """Ingest from URL."""
        from src.api.models import ApiResponse
        
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # TODO: Implement actual URL ingestion
        return ApiResponse.create(
            data={"message": "URL ingestion started", "job_id": str(uuid.uuid4())},
            request_id=request_id,
        ).model_dump()
    
    # Pipeline Configuration Routes
    @app.get(f"{api_prefix}/pipelines", tags=["Pipelines"])
    async def list_pipelines(request: Request) -> dict:
        """List pipeline configurations."""
        from src.api.models import ApiResponse
        
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # TODO: Implement actual pipeline listing
        return ApiResponse.create(
            data=[],
            request_id=request_id,
        ).model_dump()
    
    @app.post(f"{api_prefix}/pipelines", status_code=status.HTTP_201_CREATED, tags=["Pipelines"])
    async def create_pipeline(request: Request) -> dict:
        """Create pipeline configuration."""
        from src.api.models import ApiResponse
        
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # TODO: Implement actual pipeline creation
        return ApiResponse.create(
            data={"id": str(uuid.uuid4()), "name": "new-pipeline"},
            request_id=request_id,
        ).model_dump()
    
    @app.get(f"{api_prefix}/pipelines/{{pipeline_id}}", tags=["Pipelines"])
    async def get_pipeline(pipeline_id: str, request: Request) -> dict:
        """Get pipeline configuration."""
        from src.api.models import ApiResponse
        
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # TODO: Implement actual pipeline retrieval
        return ApiResponse.create(
            data={"id": pipeline_id, "name": "example"},
            request_id=request_id,
        ).model_dump()
    
    # Sources & Destinations Routes
    @app.get(f"{api_prefix}/sources", tags=["Sources"])
    async def list_sources(request: Request) -> dict:
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
    async def list_destinations(request: Request) -> dict:
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
    
    # Audit Routes
    @app.get(f"{api_prefix}/audit/logs", tags=["Audit"])
    async def query_audit_logs(request: Request) -> dict:
        """Query audit logs."""
        from src.api.models import ApiResponse
        
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # TODO: Implement actual audit log querying
        return ApiResponse.create(
            data=[],
            request_id=request_id,
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
