"""FastAPI dependencies for the API layer.

This module provides dependency injection functions for FastAPI routes,
including database sessions, authentication, and service instances.
"""

from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Header, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.config import Settings, get_settings
from src.db.models import AsyncSession, get_session
from src.llm.provider import LLMProvider
from src.plugins.registry import PluginRegistry, get_registry
from src.core.engine import OrchestrationEngine, get_engine

# Re-export auth dependencies from the auth module
from src.auth.dependencies import (
    get_current_user,
    get_optional_user,
    require_admin,
    require_operator,
    require_developer,
    require_role,
    require_permissions,
    PermissionChecker,
    # Permission-based dependencies
    require_read_jobs,
    require_create_jobs,
    require_cancel_jobs,
    require_retry_jobs,
    require_read_sources,
    require_manage_sources,
    require_read_audit,
    require_export_audit,
    require_read_lineage,
    require_manage_users,
    require_manage_api_keys,
)
from src.auth.base import User
from src.audit.logger import get_audit_logger

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency.
    
    Yields:
        AsyncSession for database operations
    """
    async for session in get_session():
        yield session


async def get_config() -> Settings:
    """Get application settings dependency.
    
    Returns:
        Application settings
    """
    return get_settings()


async def get_plugin_registry() -> PluginRegistry:
    """Get plugin registry dependency.
    
    Returns:
        Global plugin registry instance
    """
    return get_registry()


async def get_orchestration_engine() -> OrchestrationEngine:
    """Get orchestration engine dependency.
    
    Returns:
        Global orchestration engine instance
    """
    return get_engine()


def get_llm_provider(config: Settings = Depends(get_config)) -> LLMProvider:
    """Get LLM provider dependency.
    
    Args:
        config: Application settings
        
    Returns:
        LLM provider instance
    """
    from src.llm.config import load_llm_config
    
    llm_config = load_llm_config(str(config.llm_yaml.config_path))
    return LLMProvider(llm_config)


def parse_uuid(uuid_str: str) -> UUID:
    """Parse and validate UUID string.
    
    Args:
        uuid_str: UUID string to parse
        
    Returns:
        Parsed UUID
        
    Raises:
        HTTPException: If UUID is invalid
    """
    try:
        return UUID(uuid_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID format: {uuid_str}",
        )


class CommonQueryParams:
    """Common query parameters for list endpoints."""
    
    def __init__(
        self,
        page: int = 1,
        limit: int = 20,
    ) -> None:
        """Initialize common query parameters.
        
        Args:
            page: Page number (1-indexed)
            limit: Items per page
        """
        self.page = page
        self.limit = limit
    
    @property
    def offset(self) -> int:
        """Calculate database offset."""
        return (self.page - 1) * self.limit


async def get_pagination_params(
    page: int = 1,
    limit: int = 20,
) -> CommonQueryParams:
    """Get pagination parameters.
    
    Args:
        page: Page number
        limit: Items per page
        
    Returns:
        CommonQueryParams instance
    """
    return CommonQueryParams(page=page, limit=limit)


# ============================================================================
# Audit Logging Dependencies
# ============================================================================

async def get_audit_logger_dependency():
    """Get audit logger dependency."""
    return get_audit_logger()


# ============================================================================
# Legacy Compatibility
# ============================================================================

async def get_current_user_legacy(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_config),
) -> dict:
    """Legacy user dependency for backward compatibility.
    
    This function maintains backward compatibility with code expecting
    a dict instead of the new User model. New code should use get_current_user.
    
    Args:
        credentials: JWT Bearer token
        x_api_key: API key header
        settings: Application settings
        
    Returns:
        User information dictionary
    """
    # Try new auth system first
    from src.auth.dependencies import get_auth_manager
    
    try:
        auth_manager = get_auth_manager()
        result = await auth_manager.authenticate_request(
            request=Request(scope={"type": "http"}),
            credentials=credentials,
            api_key=x_api_key,
        )
        
        if result.success and result.user:
            return {
                "id": str(result.user.id),
                "email": result.user.email,
                "username": result.user.username,
                "type": "service_account" if result.user.is_service_account else "user",
                "roles": result.user.roles,
                "permissions": result.user.permissions,
            }
    except Exception:
        pass
    
    # Fallback to legacy mock auth for development
    if x_api_key:
        if x_api_key == "test-api-key":
            return {
                "id": "api-key-user",
                "type": "service_account",
                "roles": ["operator"],
            }
    
    if credentials:
        return {
            "id": "jwt-user",
            "type": "user",
            "roles": ["developer"],
        }
    
    # For development, allow unauthenticated requests
    if settings.env == "development":
        return {
            "id": "anonymous",
            "type": "anonymous",
            "roles": ["viewer"],
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )
