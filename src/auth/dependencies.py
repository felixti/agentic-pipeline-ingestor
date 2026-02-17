"""FastAPI dependencies for authentication and authorization.

This module provides FastAPI dependency injection functions for
protecting routes with authentication and RBAC.
"""


from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.base import AuthProvider, AuthResult, Credentials, User
from src.auth.jwt import JWTHandler
from src.auth.rbac import RBACManager, Role, get_rbac_manager

# Security scheme for FastAPI
security = HTTPBearer(auto_error=False)


class AuthManager:
    """Manager for multiple authentication backends.
    
    This class coordinates multiple authentication methods and provides
    a unified interface for authenticating requests.
    """

    def __init__(self):
        """Initialize authentication manager."""
        self.backends: dict[AuthProvider, any] = {}
        self.jwt_handler: JWTHandler | None = None
        self._rbac: RBACManager | None = None

    def register_backend(
        self,
        provider: AuthProvider,
        backend: any,
    ) -> None:
        """Register an authentication backend.
        
        Args:
            provider: Provider type
            backend: Backend instance
        """
        self.backends[provider] = backend

        # If backend has JWT handler, use it
        if hasattr(backend, "jwt_handler") and backend.jwt_handler:
            self.jwt_handler = backend.jwt_handler

    def set_jwt_handler(self, handler: JWTHandler) -> None:
        """Set the JWT handler for session tokens.
        
        Args:
            handler: JWT handler instance
        """
        self.jwt_handler = handler

    @property
    def rbac(self) -> RBACManager:
        """Get RBAC manager."""
        if self._rbac is None:
            self._rbac = get_rbac_manager()
        return self._rbac

    async def authenticate(
        self,
        credentials: Credentials,
    ) -> AuthResult:
        """Authenticate using credentials.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            Authentication result
        """
        backend = self.backends.get(credentials.provider)

        if not backend:
            return AuthResult.failure_result(
                f"Unknown authentication provider: {credentials.provider}",
                "UNKNOWN_PROVIDER"
            )

        return await backend.authenticate(credentials)

    async def authenticate_request(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = None,
        api_key: str | None = None,
    ) -> AuthResult:
        """Authenticate an incoming request.
        
        Tries multiple authentication methods in order:
        1. API key (X-API-Key header)
        2. JWT Bearer token (Authorization header)
        3. OAuth2/OIDC token
        4. Azure AD token
        
        Args:
            request: FastAPI request
            credentials: Bearer token credentials
            api_key: API key header value
            
        Returns:
            Authentication result
        """
        # Try API key first
        if api_key:
            creds = Credentials(
                provider=AuthProvider.API_KEY,
                api_key=api_key,
            )
            result = await self.authenticate(creds)
            if result.success:
                return result

        # Try JWT/OAuth2/Azure AD token
        if credentials and credentials.credentials:
            token = credentials.credentials

            # Try JWT validation first (our own tokens)
            if self.jwt_handler:
                try:
                    payload = self.jwt_handler.verify_token(token)
                    user = User(
                        id=payload.sub,
                        email=payload.email,
                        username=payload.username,
                        roles=payload.roles,
                        permissions=payload.permissions,
                        auth_provider=AuthProvider.JWT,
                    )
                    return AuthResult.success_result(
                        user=user,
                        metadata={"auth_method": "jwt"},
                    )
                except Exception:
                    pass  # Not our JWT, try other methods

            # Try OAuth2
            if AuthProvider.OAUTH2 in self.backends:
                creds = Credentials(
                    provider=AuthProvider.OAUTH2,
                    token=token,
                )
                result = await self.authenticate(creds)
                if result.success:
                    return result

            # Try Azure AD
            if AuthProvider.AZURE_AD in self.backends:
                creds = Credentials(
                    provider=AuthProvider.AZURE_AD,
                    token=token,
                )
                result = await self.authenticate(creds)
                if result.success:
                    return result

        return AuthResult.failure_result(
            "Authentication required",
            "AUTH_REQUIRED"
        )


# Global auth manager
_auth_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    """Get global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    api_key: str | None = None,  # Would come from Header(...)
) -> User:
    """Get current authenticated user from request.
    
    This is a FastAPI dependency that extracts and validates authentication
    from the incoming request.
    
    Args:
        request: FastAPI request
        credentials: Bearer token from Authorization header
        api_key: API key from X-API-Key header
        
    Returns:
        Authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    # Get API key from header
    api_key = request.headers.get("X-API-Key")

    # Get auth manager
    auth_manager = get_auth_manager()

    # Authenticate request
    result = await auth_manager.authenticate_request(
        request=request,
        credentials=credentials,
        api_key=api_key,
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result.error or "Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return result.user


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User | None:
    """Get current user if authenticated, None otherwise.
    
    This is useful for endpoints that can work with or without authentication.
    
    Args:
        request: FastAPI request
        credentials: Bearer token
        
    Returns:
        User if authenticated, None otherwise
    """
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


def require_permissions(resource: str, action: str):
    """Create dependency to require specific permissions.
    
    Args:
        resource: Resource type
        action: Required action
        
    Returns:
        Dependency function
    """
    async def check(
        user: User = Depends(get_current_user),
    ) -> User:
        rbac = get_rbac_manager()

        if not rbac.check_permission(user, resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {action} on {resource}",
            )

        return user

    return check


def require_role(role: str):
    """Create dependency to require a specific role.
    
    Args:
        role: Required role
        
    Returns:
        Dependency function
    """
    async def check(
        user: User = Depends(get_current_user),
    ) -> User:
        # Admin has all roles
        if role != Role.ADMIN.value and user.has_role(Role.ADMIN.value):
            return user

        if not user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {role}",
            )

        return user

    return check


# Common permission dependencies
require_admin = require_role(Role.ADMIN.value)
require_operator = require_role(Role.OPERATOR.value)
require_developer = require_role(Role.DEVELOPER.value)

# Resource-specific permission dependencies
require_read_jobs = require_permissions("jobs", "read")
require_create_jobs = require_permissions("jobs", "create")
require_cancel_jobs = require_permissions("jobs", "cancel")
require_retry_jobs = require_permissions("jobs", "retry")
require_read_sources = require_permissions("sources", "read")
require_manage_sources = require_permissions("sources", "create")
require_read_audit = require_permissions("audit", "read")
require_export_audit = require_permissions("audit", "export")
require_read_lineage = require_permissions("lineage", "read")
require_manage_users = require_permissions("users", "create")
require_manage_api_keys = require_permissions("api_keys", "create")


class PermissionChecker:
    """Helper class for checking permissions in route handlers.
    
    Usage:
        @router.post("/jobs")
        async def create_job(
            data: JobCreate,
            user: User = Depends(get_current_user),
            checker: PermissionChecker = Depends(),
        ):
            checker.ensure_can_create_jobs(user)
            # ... create job
    """

    def __init__(self):
        """Initialize permission checker."""
        self.rbac = get_rbac_manager()

    def ensure_permission(self, user: User, resource: str, action: str) -> None:
        """Ensure user has permission, raise HTTPException if not.
        
        Args:
            user: User to check
            resource: Resource type
            action: Action to check
            
        Raises:
            HTTPException: If permission check fails
        """
        if not self.rbac.check_permission(user, resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {action} on {resource}",
            )

    def ensure_role(self, user: User, role: str) -> None:
        """Ensure user has role, raise HTTPException if not.
        
        Args:
            user: User to check
            role: Required role
            
        Raises:
            HTTPException: If role check fails
        """
        if not user.has_role(role) and not user.has_role(Role.ADMIN.value):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {role}",
            )

    # Convenience methods for common permissions
    def ensure_can_create_jobs(self, user: User) -> None:
        """Ensure user can create jobs."""
        self.ensure_permission(user, "jobs", "create")

    def ensure_can_read_jobs(self, user: User) -> None:
        """Ensure user can read jobs."""
        self.ensure_permission(user, "jobs", "read")

    def ensure_can_cancel_jobs(self, user: User) -> None:
        """Ensure user can cancel jobs."""
        self.ensure_permission(user, "jobs", "cancel")

    def ensure_can_retry_jobs(self, user: User) -> None:
        """Ensure user can retry jobs."""
        self.ensure_permission(user, "jobs", "retry")

    def ensure_can_read_sources(self, user: User) -> None:
        """Ensure user can read sources."""
        self.ensure_permission(user, "sources", "read")

    def ensure_can_manage_sources(self, user: User) -> None:
        """Ensure user can manage sources."""
        self.ensure_permission(user, "sources", "create")

    def ensure_can_read_audit(self, user: User) -> None:
        """Ensure user can read audit logs."""
        self.ensure_permission(user, "audit", "read")

    def ensure_can_export_audit(self, user: User) -> None:
        """Ensure user can export audit data."""
        self.ensure_permission(user, "audit", "export")

    def ensure_can_manage_users(self, user: User) -> None:
        """Ensure user can manage users."""
        self.ensure_permission(user, "users", "create")

    def ensure_can_manage_api_keys(self, user: User) -> None:
        """Ensure user can manage API keys."""
        self.ensure_permission(user, "api_keys", "create")

    def ensure_is_admin(self, user: User) -> None:
        """Ensure user is an admin."""
        self.ensure_role(user, Role.ADMIN.value)
