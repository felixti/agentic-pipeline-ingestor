"""Unit tests for auth base classes."""

from datetime import datetime
from uuid import uuid4

import pytest

from src.auth.base import (
    AuthProvider,
    AuthResult,
    Credentials,
    Permission,
    User,
)


class TestAuthProvider:
    """Tests for AuthProvider enum."""

    def test_auth_provider_values(self):
        """Test auth provider has expected values."""
        assert AuthProvider.API_KEY == "api_key"
        assert AuthProvider.OAUTH2 == "oauth2"
        assert AuthProvider.AZURE_AD == "azure_ad"
        assert AuthProvider.JWT == "jwt"


class TestPermission:
    """Tests for Permission enum."""

    def test_permission_values(self):
        """Test permission values."""
        assert Permission.READ == "read"
        assert Permission.CREATE == "create"
        assert Permission.UPDATE == "update"
        assert Permission.DELETE == "delete"
        assert Permission.ADMIN == "admin"


class TestUser:
    """Tests for User dataclass."""

    def test_create_basic_user(self):
        """Test creating a basic user."""
        user_id = uuid4()
        user = User(id=user_id, email="test@example.com")

        assert user.id == user_id
        assert user.email == "test@example.com"
        assert user.role == "viewer"  # Default
        assert user.is_active is True

    def test_user_has_permission(self):
        """Test checking user permissions."""
        user = User(
            id=uuid4(),
            permissions=["read", "create"],
        )

        assert user.has_permission("read") is True
        assert user.has_permission("create") is True
        assert user.has_permission("delete") is False

    def test_admin_has_all_permissions(self):
        """Test admin user has all permissions."""
        user = User(
            id=uuid4(),
            roles=["admin"],
        )

        assert user.has_permission("read") is True
        assert user.has_permission("delete") is True
        assert user.has_permission("anything") is True

    def test_user_has_role(self):
        """Test checking user roles."""
        user = User(
            id=uuid4(),
            role="operator",
            roles=["developer", "viewer"],
        )

        assert user.has_role("operator") is True
        assert user.has_role("developer") is True
        assert user.has_role("admin") is False

    def test_user_to_dict(self):
        """Test converting user to dictionary."""
        user_id = uuid4()
        user = User(
            id=user_id,
            email="test@example.com",
            username="testuser",
            role="operator",
            permissions=["read"],
        )

        d = user.to_dict()

        assert d["id"] == str(user_id)
        assert d["email"] == "test@example.com"
        assert d["username"] == "testuser"
        assert d["role"] == "operator"
        assert d["permissions"] == ["read"]

    def test_service_account(self):
        """Test service account user."""
        user = User(
            id=uuid4(),
            is_service_account=True,
        )

        assert user.is_service_account is True


class TestCredentials:
    """Tests for Credentials dataclass."""

    def test_create_credentials(self):
        """Test creating credentials."""
        creds = Credentials(
            provider=AuthProvider.JWT,
            token="test-token",
        )

        assert creds.provider == AuthProvider.JWT
        assert creds.token == "test-token"

    def test_api_key_credentials(self):
        """Test API key credentials."""
        creds = Credentials(
            provider=AuthProvider.API_KEY,
            api_key="secret-key",
        )

        assert creds.api_key == "secret-key"

    def test_basic_auth_credentials(self):
        """Test basic auth credentials."""
        creds = Credentials(
            provider=AuthProvider.OAUTH2,
            username="user",
            password="pass",
        )

        assert creds.username == "user"
        assert creds.password == "pass"


class TestAuthResult:
    """Tests for AuthResult dataclass."""

    def test_success_result(self):
        """Test creating success result."""
        user = User(id=uuid4())
        result = AuthResult.success_result(user)

        assert result.success is True
        assert result.user == user
        assert result.error is None

    def test_failure_result(self):
        """Test creating failure result."""
        result = AuthResult.failure_result("Invalid credentials")

        assert result.success is False
        assert result.user is None
        assert result.error == "Invalid credentials"
        assert result.error_code == "AUTH_FAILED"

    def test_failure_with_custom_code(self):
        """Test failure with custom error code."""
        result = AuthResult.failure_result("Token expired", "TOKEN_EXPIRED")

        assert result.error_code == "TOKEN_EXPIRED"

    def test_failure_with_metadata(self):
        """Test failure with metadata."""
        result = AuthResult.failure_result("Error", error_code="RATE_LIMITED")
        result.metadata["retry_after"] = 60

        assert result.metadata["retry_after"] == 60
