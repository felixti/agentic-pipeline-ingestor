"""Authentication framework for the Agentic Data Pipeline Ingestor.

This module provides comprehensive authentication and authorization capabilities
including API key, OAuth2/OIDC, and Azure AD authentication methods.
"""

from src.auth.base import (
    AuthenticationBackend,
    AuthProvider,
    AuthResult,
    Credentials,
    User,
)
from src.auth.api_key import APIKeyAuth
from src.auth.jwt import JWTHandler, TokenPayload
from src.auth.oauth2 import OAuth2Auth
from src.auth.azure_ad import AzureADAuth
from src.auth.rbac import RBACManager, Permission, Role

__all__ = [
    "AuthenticationBackend",
    "AuthProvider",
    "AuthResult",
    "Credentials",
    "User",
    "APIKeyAuth",
    "JWTHandler",
    "TokenPayload",
    "OAuth2Auth",
    "AzureADAuth",
    "RBACManager",
    "Permission",
    "Role",
]
