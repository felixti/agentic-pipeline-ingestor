"""Authentication API routes.

This module provides endpoints for authentication, token management,
and API key management.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.audit.logger import get_audit_logger
from src.auth.api_key import generate_api_key
from src.auth.base import AuthProvider, Credentials, User
from src.auth.dependencies import (
    get_auth_manager,
    get_current_user,
    require_admin,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============================================================================
# Request/Response Models
# ============================================================================

class LoginRequest(BaseModel):
    """Login request."""
    username: str | None = None
    password: str | None = None
    provider: str = "oauth2"  # oauth2, azure_ad


class LoginResponse(BaseModel):
    """Login response."""
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int
    user: dict[str, Any]


class TokenRefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class TokenRefreshResponse(BaseModel):
    """Token refresh response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class CurrentUserResponse(BaseModel):
    """Current user response."""
    id: str
    email: str | None
    username: str | None
    role: str
    roles: list[str]
    permissions: list[str]
    auth_provider: str
    is_service_account: bool


class APIKeyCreateRequest(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    role: str = "operator"
    expires_in_days: int | None = Field(default=None, ge=1, le=365)
    permissions: list[str] = Field(default_factory=list)


class APIKeyCreateResponse(BaseModel):
    """API key creation response."""
    id: str
    key: str  # The actual API key (shown only once)
    name: str
    role: str
    expires_at: str | None


class APIKeyResponse(BaseModel):
    """API key response (without sensitive data)."""
    id: str
    name: str
    key_prefix: str
    role: str
    is_active: bool
    expires_at: str | None
    last_used_at: str | None
    use_count: int
    created_at: str


class OAuth2AuthorizeResponse(BaseModel):
    """OAuth2 authorization URL response."""
    authorization_url: str
    state: str


class OAuth2CallbackRequest(BaseModel):
    """OAuth2 callback request."""
    code: str
    state: str


# ============================================================================
# Authentication Endpoints
# ============================================================================

@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    auth_manager=Depends(get_auth_manager),
    audit_logger=Depends(get_audit_logger),
):
    """Login with credentials.
    
    Supports OAuth2 and Azure AD login flows.
    """
    # This is a simplified implementation
    # In production, this would integrate with the actual OAuth2/Azure AD flow

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Use /auth/authorize for OAuth2/Azure AD login"
    )


@router.post("/token/refresh", response_model=TokenRefreshResponse)
async def refresh_token(
    refresh_data: TokenRefreshRequest,
    auth_manager=Depends(get_auth_manager),
):
    """Refresh access token using refresh token."""
    if not auth_manager.jwt_handler:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT handler not configured"
        )

    try:
        new_token = auth_manager.jwt_handler.refresh_access_token(
            refresh_data.refresh_token
        )

        return TokenRefreshResponse(
            access_token=new_token,
            expires_in=auth_manager.jwt_handler.access_token_expire_minutes * 60,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {e}",
        )


@router.get("/me", response_model=CurrentUserResponse)
async def get_me(
    user: User = Depends(get_current_user),
):
    """Get current authenticated user information."""
    return CurrentUserResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        role=user.role,
        roles=user.roles,
        permissions=user.permissions,
        auth_provider=user.auth_provider.value,
        is_service_account=user.is_service_account,
    )


@router.post("/logout")
async def logout(
    user: User = Depends(get_current_user),
    audit_logger=Depends(get_audit_logger),
):
    """Logout current user.
    
    Note: With JWT tokens, actual invalidation would require
    a token blacklist or short expiration times.
    """
    await audit_logger.log_auth_login(
        user_id=str(user.id),
        success=True,
    )

    return {"message": "Successfully logged out"}


# ============================================================================
# OAuth2 / Azure AD Endpoints
# ============================================================================

@router.get("/oauth2/authorize", response_model=OAuth2AuthorizeResponse)
async def oauth2_authorize(
    redirect_uri: str,
    provider: str = "oauth2",  # oauth2 or azure_ad
    auth_manager=Depends(get_auth_manager),
):
    """Get OAuth2 authorization URL.
    
    Initiates the OAuth2 authorization flow by returning the
    authorization URL for the configured provider.
    """
    import secrets

    state = secrets.token_urlsafe(32)

    if provider == "azure_ad":
        backend = auth_manager.backends.get(AuthProvider.AZURE_AD)
        if not backend:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Azure AD not configured"
            )
        auth_url = backend.get_authorization_url(
            redirect_uri=redirect_uri,
            state=state,
        )
    elif provider == "oauth2":
        backend = auth_manager.backends.get(AuthProvider.OAUTH2)
        if not backend:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="OAuth2 not configured"
            )
        auth_url = backend.get_authorization_url(
            redirect_uri=redirect_uri,
            state=state,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {provider}"
        )

    return OAuth2AuthorizeResponse(
        authorization_url=auth_url,
        state=state,
    )


@router.post("/oauth2/callback", response_model=LoginResponse)
async def oauth2_callback(
    request: Request,
    callback_data: OAuth2CallbackRequest,
    redirect_uri: str,
    provider: str = "oauth2",
    auth_manager=Depends(get_auth_manager),
    audit_logger=Depends(get_audit_logger),
):
    """Handle OAuth2 callback.
    
    Exchanges authorization code for tokens and creates user session.
    """
    # Exchange code for tokens
    if provider == "azure_ad":
        backend = auth_manager.backends.get(AuthProvider.AZURE_AD)
    elif provider == "oauth2":
        backend = auth_manager.backends.get(AuthProvider.OAUTH2)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {provider}"
        )

    if not backend:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"{provider} not configured"
        )

    try:
        token_response = await backend.exchange_code(
            code=callback_data.code,
            redirect_uri=redirect_uri,
        )

        access_token = token_response.get("access_token")

        # Authenticate with the access token
        credentials = Credentials(
            provider=AuthProvider.AZURE_AD if provider == "azure_ad" else AuthProvider.OAUTH2,
            token=access_token,
        )

        result = await auth_manager.authenticate(credentials)

        if not result.success:
            await audit_logger.log_auth_failed(
                reason=result.error,
                details={"provider": provider},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.error or "Authentication failed",
            )

        user = result.user

        # Create session token
        session_token = None
        refresh_token = None

        if auth_manager.jwt_handler:
            session_token = auth_manager.jwt_handler.create_access_token(
                user_id=user.id,
                email=user.email,
                username=user.username,
                roles=user.roles,
                permissions=user.permissions,
            )
            refresh_token = auth_manager.jwt_handler.create_refresh_token(
                user_id=user.id,
            )

        # Log successful login
        await audit_logger.log_auth_login(
            user_id=str(user.id),
            success=True,
            details={"provider": provider},
        )

        return LoginResponse(
            access_token=session_token or access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=auth_manager.jwt_handler.access_token_expire_minutes * 60 if auth_manager.jwt_handler else 3600,
            user=user.to_dict(),
        )

    except HTTPException:
        raise
    except Exception as e:
        await audit_logger.log_auth_failed(
            reason=str(e),
            details={"provider": provider},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth2 callback failed: {e}",
        )


# ============================================================================
# API Key Management Endpoints (Admin only)
# ============================================================================

@router.get("/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    user: User = Depends(require_admin),
):
    """List all API keys (admin only).
    
    Returns a list of all API keys in the system.
    """
    # In production, this would query the database
    # For now, return empty list
    return []


@router.post("/api-keys", response_model=APIKeyCreateResponse)
async def create_api_key(
    request: APIKeyCreateRequest,
    user: User = Depends(require_admin),
    audit_logger=Depends(get_audit_logger),
):
    """Create a new API key (admin only).
    
    Creates a new API key with specified permissions.
    The actual key is returned only once.
    """
    # Generate API key
    api_key = generate_api_key(prefix="sk_live")

    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        from datetime import datetime, timedelta
        expires_at = (datetime.utcnow() + timedelta(days=request.expires_in_days)).isoformat()

    # In production, this would:
    # 1. Hash the API key
    # 2. Store in database
    # 3. Associate with a service account user

    key_id = str(UUID(int=0))  # Placeholder

    # Log API key creation
    await audit_logger.log_api_key_created(
        key_id=key_id,
        user_id=str(user.id),
        details={
            "name": request.name,
            "role": request.role,
        },
    )

    return APIKeyCreateResponse(
        id=key_id,
        key=api_key,  # Only shown once!
        name=request.name,
        role=request.role,
        expires_at=expires_at,
    )


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: User = Depends(require_admin),
    audit_logger=Depends(get_audit_logger),
):
    """Revoke an API key (admin only).
    
    Permanently deactivates an API key.
    """
    # In production, this would:
    # 1. Mark key as revoked in database
    # 2. Optionally add to blacklist

    # Log revocation
    await audit_logger.log_api_key_revoked(
        key_id=key_id,
        user_id=str(user.id),
    )

    return {"message": "API key revoked successfully"}


@router.get("/api-keys/{key_id}/usage")
async def get_api_key_usage(
    key_id: str,
    user: User = Depends(require_admin),
):
    """Get API key usage statistics (admin only).
    
    Returns usage statistics for a specific API key.
    """
    # In production, this would query usage logs
    return {
        "key_id": key_id,
        "use_count": 0,
        "last_used_at": None,
        "requests_per_day": [],
    }
