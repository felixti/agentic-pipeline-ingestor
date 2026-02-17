"""Base classes for the authentication framework.

This module defines the abstract base classes and data models for
authentication backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


class AuthProvider(str, Enum):
    """Authentication provider types."""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    AZURE_AD = "azure_ad"
    JWT = "jwt"


class Permission(str, Enum):
    """Permission flags for RBAC.
    
    Permissions are combined using bitwise operations to create role permissions.
    """
    NONE = "none"
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    CANCEL = "cancel"
    RETRY = "retry"
    ADMIN = "admin"
    CREATE_JOBS = "create_jobs"
    READ_SOURCES = "read_sources"
    MANAGE_CONFIG = "manage_config"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT = "view_audit"
    EXPORT_DATA = "export_data"


@dataclass
class User:
    """Authenticated user representation.
    
    Attributes:
        id: Unique user identifier (UUID)
        email: User email address
        username: User login name
        role: Primary role assigned to user
        roles: All roles assigned to user (including groups)
        auth_provider: Authentication provider used
        permissions: Explicit permissions granted
        metadata: Additional user attributes
        created_at: Account creation timestamp
        last_login_at: Last successful login timestamp
        is_active: Whether the account is active
        is_service_account: Whether this is a service account (API key)
    """
    id: UUID
    email: str | None = None
    username: str | None = None
    role: str = "viewer"
    roles: list[str] = field(default_factory=list)
    auth_provider: AuthProvider = AuthProvider.JWT
    permissions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    last_login_at: datetime | None = None
    is_active: bool = True
    is_service_account: bool = False

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            True if user has the permission
        """
        # Admin has all permissions
        if "admin" in self.roles or Permission.ADMIN in self.permissions:
            return True
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role.
        
        Args:
            role: Role to check
            
        Returns:
            True if user has the role
        """
        return role in self.roles or self.role == role

    def to_dict(self) -> dict[str, Any]:
        """Convert user to dictionary representation."""
        return {
            "id": str(self.id),
            "email": self.email,
            "username": self.username,
            "role": self.role,
            "roles": self.roles,
            "auth_provider": self.auth_provider.value,
            "permissions": self.permissions,
            "is_active": self.is_active,
            "is_service_account": self.is_service_account,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


@dataclass
class Credentials:
    """Authentication credentials.
    
    Attributes:
        provider: Authentication provider type
        token: JWT or OAuth2 token
        api_key: API key for service authentication
        username: Username for basic auth
        password: Password for basic auth
        extra: Additional provider-specific credentials
    """
    provider: AuthProvider
    token: str | None = None
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthResult:
    """Result of authentication attempt.
    
    Attributes:
        success: Whether authentication succeeded
        user: Authenticated user (if success)
        error: Error message (if failed)
        error_code: Error code for programmatic handling
        metadata: Additional authentication metadata
    """
    success: bool
    user: User | None = None
    error: str | None = None
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(cls, user: User, metadata: dict[str, Any] | None = None) -> "AuthResult":
        """Create a successful authentication result."""
        return cls(success=True, user=user, metadata=metadata or {})

    @classmethod
    def failure_result(cls, error: str, error_code: str = "AUTH_FAILED") -> "AuthResult":
        """Create a failed authentication result."""
        return cls(success=False, error=error, error_code=error_code)


class AuthenticationBackend(ABC):
    """Abstract base class for authentication backends.
    
    All authentication backends must inherit from this class and implement
    the authenticate method.
    """

    @property
    @abstractmethod
    def provider_type(self) -> AuthProvider:
        """Return the authentication provider type."""
        ...

    @abstractmethod
    async def authenticate(self, credentials: Credentials) -> AuthResult:
        """Authenticate using the provided credentials.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            AuthResult containing authentication outcome
        """
        ...

    async def validate_token(self, token: str) -> AuthResult:
        """Validate an existing token.
        
        Args:
            token: Token to validate
            
        Returns:
            AuthResult containing validation outcome
        """
        # Default implementation: create credentials and authenticate
        credentials = Credentials(provider=self.provider_type, token=token)
        return await self.authenticate(credentials)

    async def refresh_token(self, refresh_token: str) -> AuthResult:
        """Refresh an expired token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            AuthResult with new token if successful
        """
        # Default: not implemented
        return AuthResult.failure_result(
            "Token refresh not supported",
            "REFRESH_NOT_SUPPORTED"
        )
