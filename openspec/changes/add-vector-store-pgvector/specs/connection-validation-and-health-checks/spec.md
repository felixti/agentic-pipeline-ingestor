# Spec: Connection Validation and Health Checks

## Purpose
Verify pgvector extension availability on application startup and provide health check endpoints to monitor vector store readiness.

## Interface

### Health Check Endpoints
```python
# API Endpoints
GET /health          # Basic health check
GET /health/ready    # Readiness probe (includes pgvector check)
GET /health/live     # Liveness probe
GET /health/vector   # Detailed pgvector status
```

### Health Response Schema
```json
// GET /health/ready
{
  "status": "healthy",
  "timestamp": "2024-02-18T10:30:00Z",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 5
    },
    "pgvector": {
      "status": "healthy",
      "version": "0.5.1",
      "min_version_required": "0.5.0",
      "compatible": true
    }
  }
}

// GET /health/vector (detailed)
{
  "status": "healthy",
  "extension": {
    "installed": true,
    "version": "0.5.1",
    "available_versions": ["0.4.4", "0.5.0", "0.5.1"],
    "schema": "public"
  },
  "configuration": {
    "dimensions": 1536,
    "supported_dimensions": [384, 768, 1536, 3072],
    "distance_metrics": ["cosine", "euclidean", "inner_product"]
  },
  "performance": {
    "query_response_ms": 12,
    "index_count": 1
  }
}
```

### Python Service Interface
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class PgvectorHealth:
    status: HealthStatus
    installed: bool
    version: Optional[str]
    min_version_required: str
    compatible: bool
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None

class VectorStoreHealthService:
    async def check_pgvector_health(self) -> PgvectorHealth:
        """Check pgvector extension status."""
        pass
    
    async def validate_connection(self) -> bool:
        """Validate database connection with pgvector support."""
        pass
    
    async def get_detailed_status(self) -> dict:
        """Get comprehensive pgvector status information."""
        pass
```

### Startup Validation
```python
# Application startup hook
async def validate_vector_store_on_startup(app: FastAPI) -> None:
    """
    Validate pgvector on application startup.
    
    Behavior:
    - If vector_store.enabled=true: Fail fast if pgvector unavailable
    - If vector_store.enabled=false: Warn but continue
    - Log detailed version information
    """
    pass
```

## Behavior

1. **Startup Validation**
   - Execute on application startup (FastAPI `lifespan` or `startup` event)
   - Check if pgvector extension is installed and functional
   - Verify version meets minimum requirements (≥ 0.5.0)
   - Validate configured dimensions are supported
   - Log startup status with version information

2. **Fail-Fast Configuration**
   - When `vector_store.required=true`: Raise exception on missing pgvector, preventing app startup
   - When `vector_store.required=false`: Log warning, continue startup, disable vector features
   - Configuration determines if vector store is critical or optional

3. **Health Check Endpoint - /health/ready**
   - Return 200 OK if all checks pass
   - Return 503 Service Unavailable if pgvector is required but unavailable
   - Include response time for database query
   - Cache results for 5 seconds to prevent thundering herd

4. **Detailed Health Check - /health/vector**
   - Return comprehensive pgvector status
   - Include installed version and available versions
   - List supported distance metrics
   - Show configuration parameters
   - Report index statistics if available

5. **Version Checking**
   - Parse semantic version strings (e.g., "0.5.1")
   - Compare against minimum required version
   - Warn if version is outdated but still functional
   - Block startup if version is incompatible

6. **Performance Metrics**
   - Measure query response time for basic vector operations
   - Report index count and sizes
   - Track connection pool statistics
   - Log slow queries for monitoring

7. **Liveness vs Readiness**
   - `/health/live`: Simple check - is the process running? (for Kubernetes liveness probe)
   - `/health/ready`: Full dependency check - can the app serve traffic? (for readiness probe)
   - `/health/vector`: Specific vector store check (for debugging)

## Error Handling

| Error Case | Startup Behavior | Health Check Response |
|------------|------------------|----------------------|
| Extension not installed | If required: raise RuntimeError<br>If optional: log warning | `{"status": "unhealthy", "pgvector": {"installed": false}}` |
| Version too old | If required: raise RuntimeError<br>If optional: log warning | `{"status": "degraded", "pgvector": {"compatible": false}}` |
| Database connection failed | Raise ConnectionError | HTTP 503 with connection error details |
| Permission denied | Raise PermissionError | HTTP 503 with permission error |
| Extension disabled after startup | Log error, mark degraded | `{"status": "degraded", "error": "Extension was disabled"}` |
| Query timeout | Log warning | Include timeout in response_time_ms |

## Implementation Example

```python
# src/services/vector_store_health.py
import time
from typing import Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

MIN_PGVVECTOR_VERSION = "0.5.0"

class VectorStoreHealthError(Exception):
    pass

async def check_pgvector_health(session: AsyncSession) -> dict:
    """Check pgvector extension health."""
    start_time = time.time()
    
    try:
        # Check if extension is installed
        result = await session.execute(text(
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
        ))
        row = result.fetchone()
        
        if not row:
            return {
                "status": "unhealthy",
                "installed": False,
                "error": "pgvector extension not installed"
            }
        
        version = row.extversion
        
        # Check version compatibility (simplified comparison)
        compatible = version >= MIN_PGVVECTOR_VERSION
        
        # Test basic operation
        await session.execute(text("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector"))
        
        response_time = (time.time() - start_time) * 1000
        
        status = "healthy" if compatible else "degraded"
        
        return {
            "status": status,
            "installed": True,
            "version": version,
            "min_version_required": MIN_PGVVECTOR_VERSION,
            "compatible": compatible,
            "response_time_ms": round(response_time, 2)
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "installed": False,
            "error": str(e)
        }

# FastAPI integration
from fastapi import FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

@app.get("/health/ready")
async def health_ready(db: AsyncSession):
    """Readiness probe including pgvector check."""
    pgvector_health = await check_pgvector_health(db)
    
    if pgvector_health["status"] == "unhealthy":
        raise HTTPException(
            status_code=503,
            detail={"status": "not ready", "pgvector": pgvector_health}
        )
    
    return {
        "status": "ready",
        "pgvector": pgvector_health
    }

# Startup validation
@app.on_event("startup")
async def startup_validation():
    """Validate pgvector on startup."""
    from src.config import settings
    
    if not settings.vector_store.enabled:
        logger.info("Vector store disabled, skipping pgvector validation")
        return
    
    # Perform health check
    async with db_session() as session:
        health = await check_pgvector_health(session)
    
    if health["status"] == "unhealthy" and settings.vector_store.required:
        raise RuntimeError(
            f"Vector store is required but pgvector is not available: {health.get('error')}"
        )
    
    if health["status"] == "degraded":
        logger.warning(
            f"pgvector version {health['version']} is below recommended "
            f"{health['min_version_required']}"
        )
    
    logger.info(f"Vector store initialized: pgvector {health['version']}")
```

## Configuration

```yaml
# config/vector_store.yaml
vector_store:
  enabled: true
  required: true  # If true, app fails to start without pgvector
  
  health_check:
    enabled: true
    cache_ttl_seconds: 5
    timeout_seconds: 10
    
  version:
    min_required: "0.5.0"
    check_on_startup: true
```

## Kubernetes Integration

```yaml
# kubernetes deployment probes
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5

startupProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  failureThreshold: 30
```

## Dependencies

- Database connection pool established
- pgvector extension (optionally) installed
- FastAPI or similar web framework for endpoints
- SQLAlchemy for database queries

## Success Criteria

- [ ] Startup validation runs before accepting traffic
- [ ] `/health/ready` returns 200 when pgvector is healthy
- [ ] `/health/ready` returns 503 when pgvector is required but unavailable
- [ ] Version compatibility is checked and reported
- [ ] Response times are measured and logged
- [ ] Kubernetes probes work correctly
- [ ] Graceful degradation when vector store is optional
- [ ] Clear error messages for troubleshooting
