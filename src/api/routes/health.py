"""Health check routes for the Agentic Data Pipeline Ingestor.

This module provides comprehensive health checks for Kubernetes probes
and system monitoring including database, Redis, LLM providers,
destinations, and other critical dependencies.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.api.models import HealthStatus, ComponentHealth
from src.config import settings

router = APIRouter(prefix="/health", tags=["Health"])


class HealthCheckResult(BaseModel):
    """Result of a health check."""
    healthy: bool
    component: str
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class ComprehensiveHealthResponse(BaseModel):
    """Comprehensive health check response."""
    status: HealthStatus
    timestamp: str
    version: str
    environment: str
    components: Dict[str, ComponentHealth]
    overall_healthy: bool


class ReadinessResponse(BaseModel):
    """Kubernetes readiness probe response."""
    ready: bool
    timestamp: str
    checks: Dict[str, bool]


class LivenessResponse(BaseModel):
    """Kubernetes liveness probe response."""
    alive: bool
    timestamp: str


# Store for tracking health check latencies
_health_check_latencies: Dict[str, List[float]] = {}


async def check_database() -> HealthCheckResult:
    """Check database connectivity.
    
    Returns:
        HealthCheckResult with database status
    """
    start = time.time()
    try:
        from src.db.models import get_session
        
        async for session in get_session():
            result = await session.execute("SELECT 1")
            row = result.scalar()
            
            latency = (time.time() - start) * 1000
            
            return HealthCheckResult(
                healthy=row == 1,
                component="database",
                message="Database connection successful",
                latency_ms=round(latency, 2),
            )
    except Exception as e:
        return HealthCheckResult(
            healthy=False,
            component="database",
            message=f"Database check failed: {str(e)}",
            latency_ms=round((time.time() - start) * 1000, 2),
        )


async def check_redis() -> HealthCheckResult:
    """Check Redis connectivity.
    
    Returns:
        HealthCheckResult with Redis status
    """
    start = time.time()
    try:
        import redis.asyncio as redis
        
        # Parse Redis URL
        redis_url = settings.redis_url
        client = redis.from_url(redis_url, socket_connect_timeout=5)
        
        # Test connection
        pong = await client.ping()
        await client.close()
        
        latency = (time.time() - start) * 1000
        
        return HealthCheckResult(
            healthy=pong,
            component="redis",
            message="Redis connection successful",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        return HealthCheckResult(
            healthy=False,
            component="redis",
            message=f"Redis check failed: {str(e)}",
            latency_ms=round((time.time() - start) * 1000, 2),
        )


async def check_llm_providers() -> HealthCheckResult:
    """Check LLM provider health.
    
    Returns:
        HealthCheckResult with LLM provider status
    """
    start = time.time()
    try:
        from src.llm.provider import LLMProvider
        from src.llm.config import load_llm_config
        
        config = load_llm_config()
        provider = LLMProvider(config)
        
        health = await provider.health_check()
        latency = (time.time() - start) * 1000
        
        if health.get("healthy", False):
            return HealthCheckResult(
                healthy=True,
                component="llm_providers",
                message=f"All LLM providers healthy: {len(health.get('models', {}))} models",
                latency_ms=round(latency, 2),
                details=health,
            )
        else:
            return HealthCheckResult(
                healthy=False,
                component="llm_providers",
                message="Some LLM providers unhealthy",
                latency_ms=round(latency, 2),
                details=health,
            )
    except Exception as e:
        return HealthCheckResult(
            healthy=False,
            component="llm_providers",
            message=f"LLM provider check failed: {str(e)}",
            latency_ms=round((time.time() - start) * 1000, 2),
        )


async def check_destinations() -> HealthCheckResult:
    """Check destination plugin health.
    
    Returns:
        HealthCheckResult with destination status
    """
    start = time.time()
    try:
        from src.plugins.registry import get_registry
        
        registry = get_registry()
        health_status = await registry.health_check_all()
        latency = (time.time() - start) * 1000
        
        healthy_count = sum(1 for h in health_status.values() if h)
        total_count = len(health_status)
        
        if healthy_count == total_count:
            return HealthCheckResult(
                healthy=True,
                component="destinations",
                message=f"All {total_count} destinations healthy",
                latency_ms=round(latency, 2),
                details={"healthy": healthy_count, "total": total_count},
            )
        elif healthy_count > 0:
            return HealthCheckResult(
                healthy=True,  # Degraded but still functional
                component="destinations",
                message=f"{healthy_count}/{total_count} destinations healthy",
                latency_ms=round(latency, 2),
                details={"healthy": healthy_count, "total": total_count, "status": health_status},
            )
        else:
            return HealthCheckResult(
                healthy=False,
                component="destinations",
                message="No destinations healthy",
                latency_ms=round(latency, 2),
                details={"healthy": 0, "total": total_count},
            )
    except Exception as e:
        return HealthCheckResult(
            healthy=False,
            component="destinations",
            message=f"Destination check failed: {str(e)}",
            latency_ms=round((time.time() - start) * 1000, 2),
        )


async def check_storage() -> HealthCheckResult:
    """Check storage connectivity.
    
    Returns:
        HealthCheckResult with storage status
    """
    start = time.time()
    try:
        # Try to check if storage is accessible
        # This could be Azure Blob, S3, or local filesystem
        import os
        
        storage_path = "/tmp/pipeline"
        os.makedirs(storage_path, exist_ok=True)
        
        # Try to write and read a test file
        test_file = os.path.join(storage_path, ".healthcheck")
        with open(test_file, "w") as f:
            f.write("ok")
        
        with open(test_file, "r") as f:
            content = f.read()
        
        os.remove(test_file)
        
        latency = (time.time() - start) * 1000
        
        if content == "ok":
            return HealthCheckResult(
                healthy=True,
                component="storage",
                message="Storage is writable",
                latency_ms=round(latency, 2),
            )
        else:
            return HealthCheckResult(
                healthy=False,
                component="storage",
                message="Storage check failed: content mismatch",
                latency_ms=round(latency, 2),
            )
    except Exception as e:
        return HealthCheckResult(
            healthy=False,
            component="storage",
            message=f"Storage check failed: {str(e)}",
            latency_ms=round((time.time() - start) * 1000, 2),
        )


async def check_opentelemetry() -> HealthCheckResult:
    """Check OpenTelemetry instrumentation status.
    
    Returns:
        HealthCheckResult with OpenTelemetry status
    """
    start = time.time()
    try:
        from src.observability.tracing import get_telemetry_manager
        
        manager = get_telemetry_manager()
        latency = (time.time() - start) * 1000
        
        if manager.is_initialized:
            return HealthCheckResult(
                healthy=True,
                component="opentelemetry",
                message=f"Tracing enabled with {len(manager.active_exporters)} exporters",
                latency_ms=round(latency, 2),
                details={"exporters": manager.active_exporters},
            )
        else:
            return HealthCheckResult(
                healthy=True,  # Not critical, just informational
                component="opentelemetry",
                message="Tracing not initialized",
                latency_ms=round(latency, 2),
            )
    except Exception as e:
        return HealthCheckResult(
            healthy=False,
            component="opentelemetry",
            message=f"OpenTelemetry check failed: {str(e)}",
            latency_ms=round((time.time() - start) * 1000, 2),
        )


@router.get("", response_model=ComprehensiveHealthResponse)
async def health_check() -> ComprehensiveHealthResponse:
    """Comprehensive health check endpoint.
    
    Performs health checks on all critical components:
    - Database
    - Redis
    - LLM Providers
    - Destinations
    - Storage
    - OpenTelemetry
    
    Returns:
        Comprehensive health status of all components
    """
    from datetime import datetime
    
    # Run all health checks concurrently
    checks = await asyncio.gather(
        check_database(),
        check_redis(),
        check_llm_providers(),
        check_destinations(),
        check_storage(),
        check_opentelemetry(),
        return_exceptions=True,
    )
    
    # Process results
    components: Dict[str, ComponentHealth] = {
        "api": ComponentHealth(status=HealthStatus.HEALTHY),
    }
    
    all_healthy = True
    any_degraded = False
    
    for check in checks:
        if isinstance(check, Exception):
            components["unknown"] = ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed with exception: {str(check)}",
            )
            all_healthy = False
            continue
        
        status = HealthStatus.HEALTHY if check.healthy else HealthStatus.UNHEALTHY
        if not check.healthy:
            all_healthy = False
        
        components[check.component] = ComponentHealth(
            status=status,
            message=check.message,
            latency_ms=check.latency_ms,
        )
    
    # Determine overall status
    if all_healthy:
        overall_status = HealthStatus.HEALTHY
    elif any_degraded:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.UNHEALTHY
    
    return ComprehensiveHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version=settings.app_version,
        environment=settings.env,
        components=components,
        overall_healthy=all_healthy,
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_probe() -> ReadinessResponse:
    """Kubernetes readiness probe.
    
    Checks if the service is ready to receive traffic.
    This should return success when all required dependencies are available.
    
    Returns:
        Readiness status
    """
    from datetime import datetime
    
    # Check critical dependencies
    checks = await asyncio.gather(
        check_database(),
        check_redis(),
        return_exceptions=True,
    )
    
    check_results = {}
    all_ready = True
    
    for check in checks:
        if isinstance(check, Exception):
            check_results["unknown"] = False
            all_ready = False
        else:
            check_results[check.component] = check.healthy
            if not check.healthy:
                all_ready = False
    
    if not all_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "ready": False,
                "timestamp": datetime.utcnow().isoformat(),
                "checks": check_results,
            },
        )
    
    return ReadinessResponse(
        ready=True,
        timestamp=datetime.utcnow().isoformat(),
        checks=check_results,
    )


@router.get("/live", response_model=LivenessResponse)
async def liveness_probe() -> LivenessResponse:
    """Kubernetes liveness probe.
    
    Checks if the service is alive and should not be restarted.
    This is a lightweight check that just verifies the process is running.
    
    Returns:
        Liveness status
    """
    from datetime import datetime
    
    # This is a simple liveness check - if the server responds, it's alive
    return LivenessResponse(
        alive=True,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/detailed/{component}")
async def detailed_component_health(component: str) -> Dict[str, Any]:
    """Get detailed health information for a specific component.
    
    Args:
        component: Component name to check
        
    Returns:
        Detailed health information
        
    Raises:
        HTTPException: If component not found
    """
    check_functions = {
        "database": check_database,
        "redis": check_redis,
        "llm_providers": check_llm_providers,
        "destinations": check_destinations,
        "storage": check_storage,
        "opentelemetry": check_opentelemetry,
    }
    
    if component not in check_functions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown component: {component}. Available: {list(check_functions.keys())}",
        )
    
    result = await check_functions[component]()
    
    return {
        "component": result.component,
        "healthy": result.healthy,
        "message": result.message,
        "latency_ms": result.latency_ms,
        "details": result.details,
        "timestamp": datetime.utcnow().isoformat(),
    }
