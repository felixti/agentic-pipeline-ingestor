"""API route handlers.

This package contains all API route handlers for the
Agentic Data Pipeline Ingestor.
"""

from src.api.routes.auth import router as auth_router
from src.api.routes.audit import router as audit_router
from src.api.routes.lineage import router as lineage_router
from src.api.routes.dlq import router as dlq_router
from src.api.routes.health import router as health_router

__all__ = [
    "auth_router",
    "audit_router",
    "lineage_router",
    "dlq_router",
    "health_router",
]
