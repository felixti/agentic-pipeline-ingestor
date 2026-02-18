"""Unit tests for authentication API routes.

Tests the auth-related API endpoints including login, token refresh,
OAuth2 callbacks, API key management, and logout.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException, Request, status

from src.api.routes.auth import (
    APIKeyCreateRequest,
    LoginRequest,
    OAuth2CallbackRequest,
    TokenRefreshRequest,
    create_api_key,
    get_me,
    list_api_keys,
    login,
    logout,
    oauth2_authorize,
    oauth2_callback,
    refresh_token,
    revoke_api_key,
)
from src.auth.base import AuthProvider, User
from src.auth.jwt import JWTHandler


@pytest.mark.unit
class TestLoginEndpoint:
    """Tests for POST /auth/login - Login endpoint."""

    @pytest.mark.asyncio
    async def test_login_not_implemented(self):
        """Test that login returns 501 Not Implemented."""
        mock_request = MagicMock()
        login_data = LoginRequest(
            username="testuser",
            password="testpass",
            provider="oauth2",
        )
        mock_auth_manager = MagicMock()
        mock_audit_logger = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await login(mock_request, login_data, mock_auth_manager, mock_audit_logger)
        
        assert exc_info.value.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "Use /auth/authorize" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_login_with_azure_ad_provider(self):
        """Test login with Azure AD provider."""
        mock_request = MagicMock()
        login_data = LoginRequest(
            username="testuser",
            password="testpass",
            provider="azure_ad",
        )
        mock_auth_manager = MagicMock()
        mock_audit_logger = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await login(mock_request, login_data, mock_auth_manager, mock_audit_logger)
        
        assert exc_info.value.status_code == status.HTTP_501_NOT_IMPLEMENTED


@pytest.mark.unit
class TestTokenRefresh:
    """Tests for POST /auth/token/refresh - Token refresh endpoint."""

    @pytest.mark.asyncio
    async def test_refresh_token_success(self):
        """Test successful token refresh."""
        refresh_data = TokenRefreshRequest(refresh_token="valid_refresh_token")
        
        mock_auth_manager = MagicMock()
        mock_auth_manager.jwt_handler = MagicMock()
        mock_auth_manager.jwt_handler.refresh_access_token.return_value = "new_access_token"
        mock_auth_manager.jwt_handler.access_token_expire_minutes = 30
        
        response = await refresh_token(refresh_data, mock_auth_manager)
        
        assert response.access_token == "new_access_token"
        assert response.token_type == "bearer"
        assert response.expires_in == 1800  # 30 minutes in seconds

    @pytest.mark.asyncio
    async def test_refresh_token_no_jwt_handler(self):
        """Test token refresh when JWT handler is not configured."""
        refresh_data = TokenRefreshRequest(refresh_token="valid_refresh_token")
        
        mock_auth_manager = MagicMock()
        mock_auth_manager.jwt_handler = None
        
        with pytest.raises(HTTPException) as exc_info:
            await refresh_token(refresh_data, mock_auth_manager)
        
        assert exc_info.value.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "JWT handler not configured" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self):
        """Test token refresh with invalid refresh token."""
        refresh_data = TokenRefreshRequest(refresh_token="invalid_token")
        
        mock_auth_manager = MagicMock()
        mock_auth_manager.jwt_handler = MagicMock()
        mock_auth_manager.jwt_handler.refresh_access_token.side_effect = Exception("Invalid token")
        
        with pytest.raises(HTTPException) as exc_info:
            await refresh_token(refresh_data, mock_auth_manager)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid refresh token" in exc_info.value.detail


@pytest.mark.unit
class TestGetMe:
    """Tests for GET /auth/me - Get current user endpoint."""

    @pytest.mark.asyncio
    async def test_get_me_success(self):
        """Test successful current user retrieval."""
        user = User(
            id=uuid4(),
            email="test@example.com",
            username="testuser",
            role="operator",
            roles=["operator", "viewer"],
            permissions=["jobs:read", "jobs:create"],
            auth_provider=AuthProvider.JWT,
            is_service_account=False,
        )
        
        response = await get_me(user)
        
        assert response.id == str(user.id)
        assert response.email == "test@example.com"
        assert response.username == "testuser"
        assert response.role == "operator"
        assert response.roles == ["operator", "viewer"]
        assert response.permissions == ["jobs:read", "jobs:create"]
        assert response.auth_provider == "jwt"
        assert response.is_service_account is False

    @pytest.mark.asyncio
    async def test_get_me_service_account(self):
        """Test current user retrieval for service account."""
        user = User(
            id=uuid4(),
            email="service@example.com",
            is_service_account=True,
            auth_provider=AuthProvider.API_KEY,
        )
        
        response = await get_me(user)
        
        assert response.is_service_account is True
        assert response.auth_provider == "api_key"


@pytest.mark.unit
class TestLogout:
    """Tests for POST /auth/logout - Logout endpoint."""

    @pytest.mark.asyncio
    async def test_logout_success(self):
        """Test successful logout."""
        user = User(
            id=uuid4(),
            email="test@example.com",
        )
        mock_audit_logger = AsyncMock()
        
        response = await logout(user, mock_audit_logger)
        
        assert response["message"] == "Successfully logged out"
        mock_audit_logger.log_auth_login.assert_called_once_with(
            user_id=str(user.id),
            success=True,
        )


@pytest.mark.unit
class TestOAuth2Authorize:
    """Tests for GET /auth/oauth2/authorize - OAuth2 authorization endpoint."""

    @pytest.mark.asyncio
    async def test_oauth2_authorize_azure_ad(self):
        """Test OAuth2 authorization with Azure AD."""
        mock_auth_manager = MagicMock()
        mock_backend = MagicMock()
        mock_backend.get_authorization_url.return_value = "https://login.microsoftonline.com/auth"
        mock_auth_manager.backends = {AuthProvider.AZURE_AD: mock_backend}
        
        response = await oauth2_authorize(
            redirect_uri="http://localhost/callback",
            provider="azure_ad",
            auth_manager=mock_auth_manager,
        )
        
        assert response.authorization_url == "https://login.microsoftonline.com/auth"
        assert response.state is not None
        assert len(response.state) > 0

    @pytest.mark.asyncio
    async def test_oauth2_authorize_oauth2(self):
        """Test OAuth2 authorization with generic OAuth2."""
        mock_auth_manager = MagicMock()
        mock_backend = MagicMock()
        mock_backend.get_authorization_url.return_value = "https://oauth.example.com/auth"
        mock_auth_manager.backends = {AuthProvider.OAUTH2: mock_backend}
        
        response = await oauth2_authorize(
            redirect_uri="http://localhost/callback",
            provider="oauth2",
            auth_manager=mock_auth_manager,
        )
        
        assert response.authorization_url == "https://oauth.example.com/auth"
        assert response.state is not None

    @pytest.mark.asyncio
    async def test_oauth2_authorize_azure_ad_not_configured(self):
        """Test OAuth2 authorization when Azure AD is not configured."""
        mock_auth_manager = MagicMock()
        mock_auth_manager.backends = {}
        
        with pytest.raises(HTTPException) as exc_info:
            await oauth2_authorize(
                redirect_uri="http://localhost/callback",
                provider="azure_ad",
                auth_manager=mock_auth_manager,
            )
        
        assert exc_info.value.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "Azure AD not configured" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_oauth2_authorize_unknown_provider(self):
        """Test OAuth2 authorization with unknown provider."""
        mock_auth_manager = MagicMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await oauth2_authorize(
                redirect_uri="http://localhost/callback",
                provider="unknown_provider",
                auth_manager=mock_auth_manager,
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unknown provider" in exc_info.value.detail


@pytest.mark.unit
class TestOAuth2Callback:
    """Tests for POST /auth/oauth2/callback - OAuth2 callback endpoint."""

    @pytest.mark.asyncio
    async def test_oauth2_callback_success(self):
        """Test successful OAuth2 callback."""
        mock_request = MagicMock()
        callback_data = OAuth2CallbackRequest(code="auth_code", state="random_state")
        
        mock_auth_manager = MagicMock()
        mock_backend = AsyncMock()
        mock_backend.exchange_code.return_value = {"access_token": "oauth_access_token"}
        
        mock_auth_manager.backends = {AuthProvider.AZURE_AD: mock_backend}
        mock_auth_manager.authenticate.return_value = MagicMock(
            success=True,
            user=User(
                id=uuid4(),
                email="test@example.com",
                username="testuser",
                roles=["operator"],
                permissions=["jobs:read"],
            ),
        )
        mock_auth_manager.jwt_handler = MagicMock()
        mock_auth_manager.jwt_handler.create_access_token.return_value = "new_jwt_token"
        mock_auth_manager.jwt_handler.create_refresh_token.return_value = "new_refresh_token"
        mock_auth_manager.jwt_handler.access_token_expire_minutes = 30
        
        mock_audit_logger = AsyncMock()
        
        response = await oauth2_callback(
            mock_request,
            callback_data,
            redirect_uri="http://localhost/callback",
            provider="azure_ad",
            auth_manager=mock_auth_manager,
            audit_logger=mock_audit_logger,
        )
        
        assert response.access_token is not None
        assert response.refresh_token is not None
        assert response.token_type == "bearer"
        assert "user" in response.model_dump()

    @pytest.mark.asyncio
    async def test_oauth2_callback_authentication_failed(self):
        """Test OAuth2 callback when authentication fails."""
        mock_request = MagicMock()
        callback_data = OAuth2CallbackRequest(code="auth_code", state="random_state")
        
        mock_auth_manager = MagicMock()
        mock_backend = AsyncMock()
        mock_backend.exchange_code.return_value = {"access_token": "oauth_access_token"}
        
        mock_auth_manager.backends = {AuthProvider.OAUTH2: mock_backend}
        mock_auth_manager.authenticate.return_value = MagicMock(
            success=False,
            error="Invalid credentials",
        )
        
        mock_audit_logger = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await oauth2_callback(
                mock_request,
                callback_data,
                redirect_uri="http://localhost/callback",
                provider="oauth2",
                auth_manager=mock_auth_manager,
                audit_logger=mock_audit_logger,
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        mock_audit_logger.log_auth_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_oauth2_callback_unknown_provider(self):
        """Test OAuth2 callback with unknown provider."""
        mock_request = MagicMock()
        callback_data = OAuth2CallbackRequest(code="auth_code", state="random_state")
        mock_auth_manager = MagicMock()
        mock_audit_logger = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await oauth2_callback(
                mock_request,
                callback_data,
                redirect_uri="http://localhost/callback",
                provider="unknown_provider",
                auth_manager=mock_auth_manager,
                audit_logger=mock_audit_logger,
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_oauth2_callback_exchange_error(self):
        """Test OAuth2 callback when code exchange fails."""
        mock_request = MagicMock()
        callback_data = OAuth2CallbackRequest(code="invalid_code", state="random_state")
        
        mock_auth_manager = MagicMock()
        mock_backend = AsyncMock()
        mock_backend.exchange_code.side_effect = Exception("Invalid code")
        mock_auth_manager.backends = {AuthProvider.OAUTH2: mock_backend}
        
        mock_audit_logger = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await oauth2_callback(
                mock_request,
                callback_data,
                redirect_uri="http://localhost/callback",
                provider="oauth2",
                auth_manager=mock_auth_manager,
                audit_logger=mock_audit_logger,
            )
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        mock_audit_logger.log_auth_failed.assert_called_once()


@pytest.mark.unit
class TestAPIKeyList:
    """Tests for GET /auth/api-keys - List API keys endpoint."""

    @pytest.mark.asyncio
    async def test_list_api_keys_success(self):
        """Test successful API key listing (admin only)."""
        admin_user = User(
            id=uuid4(),
            email="admin@example.com",
            roles=["admin"],
        )
        
        response = await list_api_keys(admin_user)
        
        assert response == []  # Currently returns empty list


@pytest.mark.unit
class TestAPIKeyCreate:
    """Tests for POST /auth/api-keys - Create API key endpoint."""

    @pytest.mark.asyncio
    async def test_create_api_key_success(self):
        """Test successful API key creation."""
        request = APIKeyCreateRequest(
            name="Test API Key",
            description="For testing",
            role="operator",
            expires_in_days=30,
            permissions=["jobs:read"],
        )
        
        admin_user = User(
            id=uuid4(),
            email="admin@example.com",
            roles=["admin"],
        )
        mock_audit_logger = AsyncMock()
        
        with patch("src.api.routes.auth.generate_api_key", return_value="sk_live_abc123"):
            response = await create_api_key(request, admin_user, mock_audit_logger)
        
        assert response.key == "sk_live_abc123"
        assert response.name == "Test API Key"
        assert response.role == "operator"
        assert response.expires_at is not None
        mock_audit_logger.log_api_key_created.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_api_key_no_expiration(self):
        """Test API key creation without expiration."""
        request = APIKeyCreateRequest(
            name="Test API Key",
            role="viewer",
        )
        
        admin_user = User(
            id=uuid4(),
            email="admin@example.com",
            roles=["admin"],
        )
        mock_audit_logger = AsyncMock()
        
        with patch("src.api.routes.auth.generate_api_key", return_value="sk_live_xyz789"):
            response = await create_api_key(request, admin_user, mock_audit_logger)
        
        assert response.expires_at is None


@pytest.mark.unit
class TestAPIKeyRevoke:
    """Tests for DELETE /auth/api-keys/{key_id} - Revoke API key endpoint."""

    @pytest.mark.asyncio
    async def test_revoke_api_key_success(self):
        """Test successful API key revocation."""
        key_id = str(uuid4())
        
        admin_user = User(
            id=uuid4(),
            email="admin@example.com",
            roles=["admin"],
        )
        mock_audit_logger = AsyncMock()
        
        response = await revoke_api_key(key_id, admin_user, mock_audit_logger)
        
        assert response["message"] == "API key revoked successfully"
        mock_audit_logger.log_api_key_revoked.assert_called_once_with(
            key_id=key_id,
            user_id=str(admin_user.id),
        )
