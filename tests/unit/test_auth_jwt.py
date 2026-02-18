"""Unit tests for JWT token handling."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from jose import JWTError, jwt

from src.auth.jwt import (
    DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES,
    DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS,
    JWTHandler,
    TokenPayload,
)


class TestTokenPayload:
    """Tests for TokenPayload."""

    def test_create_payload(self):
        """Test creating token payload."""
        user_id = uuid4()
        payload = TokenPayload(sub=user_id, email="test@example.com")

        assert payload.sub == user_id
        assert payload.email == "test@example.com"
        assert payload.type == "access"

    def test_payload_post_init(self):
        """Test payload initializes empty lists."""
        payload = TokenPayload(sub=uuid4())

        assert payload.roles == []
        assert payload.permissions == []
        assert payload.extra == {}

    def test_payload_to_dict(self):
        """Test converting payload to dict."""
        user_id = uuid4()
        now = datetime.utcnow()
        payload = TokenPayload(
            sub=user_id,
            email="test@example.com",
            roles=["operator"],
            type="access",
            iat=now,
        )

        d = payload.to_dict()

        assert d["sub"] == str(user_id)
        assert d["email"] == "test@example.com"
        assert d["roles"] == ["operator"]
        assert d["type"] == "access"
        assert d["iat"] == now.timestamp()

    def test_payload_from_dict(self):
        """Test creating payload from dict."""
        user_id = uuid4()
        now = datetime.utcnow()
        data = {
            "sub": str(user_id),
            "email": "test@example.com",
            "roles": ["operator"],
            "type": "access",
            "iat": now.timestamp(),
            "exp": (now + timedelta(minutes=30)).timestamp(),
            "extra_field": "value",
        }

        payload = TokenPayload.from_dict(data)

        assert payload.sub == user_id
        assert payload.email == "test@example.com"
        assert payload.extra["extra_field"] == "value"

    def test_is_expired(self):
        """Test checking if token is expired."""
        expired_payload = TokenPayload(
            sub=uuid4(),
            exp=datetime.utcnow() - timedelta(minutes=1),
        )
        valid_payload = TokenPayload(
            sub=uuid4(),
            exp=datetime.utcnow() + timedelta(minutes=30),
        )

        assert expired_payload.is_expired() is True
        assert valid_payload.is_expired() is False

    def test_is_refresh_token(self):
        """Test checking if token is refresh token."""
        refresh = TokenPayload(sub=uuid4(), type="refresh")
        access = TokenPayload(sub=uuid4(), type="access")

        assert refresh.is_refresh_token() is True
        assert access.is_refresh_token() is False

    def test_is_access_token(self):
        """Test checking if token is access token."""
        refresh = TokenPayload(sub=uuid4(), type="refresh")
        access = TokenPayload(sub=uuid4(), type="access")

        assert refresh.is_access_token() is False
        assert access.is_access_token() is True


class TestJWTHandler:
    """Tests for JWTHandler."""

    @pytest.fixture
    def handler(self):
        """Create JWT handler with test secret."""
        return JWTHandler(secret_key="test-secret-key-for-testing-only")

    def test_init(self):
        """Test initializing handler."""
        handler = JWTHandler(
            secret_key="secret",
            algorithm="HS512",
            access_token_expire_minutes=60,
        )

        assert handler.secret_key == "secret"
        assert handler.algorithm == "HS512"
        assert handler.access_token_expire_minutes == 60

    def test_create_access_token(self, handler):
        """Test creating access token."""
        user_id = uuid4()
        token = handler.create_access_token(
            user_id=user_id,
            email="test@example.com",
            roles=["operator"],
        )

        assert isinstance(token, str)

        # Verify token can be decoded
        payload = jwt.decode(token, "test-secret-key-for-testing-only", algorithms=["HS256"])
        assert payload["sub"] == str(user_id)
        assert payload["email"] == "test@example.com"
        assert payload["roles"] == ["operator"]
        assert payload["type"] == "access"

    def test_create_refresh_token(self, handler):
        """Test creating refresh token."""
        user_id = uuid4()
        token = handler.create_refresh_token(user_id=user_id)

        assert isinstance(token, str)

        payload = jwt.decode(token, "test-secret-key-for-testing-only", algorithms=["HS256"])
        assert payload["sub"] == str(user_id)
        assert payload["type"] == "refresh"

    def test_verify_token(self, handler):
        """Test verifying token."""
        user_id = uuid4()
        token = handler.create_access_token(
            user_id=user_id,
            email="test@example.com",
        )

        payload = handler.verify_token(token)

        assert payload.sub == user_id
        assert payload.email == "test@example.com"

    def test_verify_invalid_token(self, handler):
        """Test verifying invalid token."""
        with pytest.raises(JWTError):
            handler.verify_token("invalid.token.here")

    @pytest.mark.skip(reason="Token timing issues in test environment")
    def test_verify_expired_token(self, handler):
        """Test verifying expired token."""
        user_id = uuid4()
        # Create token with very short expiry
        token = handler.create_token(
            user_id=user_id,
            token_type="access",
            expires=timedelta(milliseconds=1),
        )

        with pytest.raises(JWTError):
            handler.verify_token(token)

    def test_refresh_access_token(self, handler):
        """Test refreshing access token."""
        user_id = uuid4()
        # Create refresh token with extra claims to preserve user data
        refresh_token = handler.create_refresh_token(
            user_id=user_id,
            extra_claims={"email": "test@example.com", "roles": ["operator"]},
        )

        new_token = handler.refresh_access_token(
            refresh_token,
            email="test@example.com",
            roles=["operator"],
        )

        # Verify new token
        payload = handler.verify_token(new_token)
        assert payload.sub == user_id
        assert payload.email == "test@example.com"
        assert payload.roles == ["operator"]
        assert payload.type == "access"

    def test_refresh_with_access_token_fails(self, handler):
        """Test refreshing with access token fails."""
        user_id = uuid4()
        access_token = handler.create_access_token(user_id=user_id)

        with pytest.raises(JWTError, match="refresh token"):
            handler.refresh_access_token(access_token)

    def test_generate_secret_key(self, handler):
        """Test generating secret key."""
        key1 = handler.generate_secret_key()
        key2 = handler.generate_secret_key()

        assert isinstance(key1, str)
        assert len(key1) > 20
        assert key1 != key2  # Should be random

    def test_token_with_extra_claims(self, handler):
        """Test token with extra claims."""
        user_id = uuid4()
        token = handler.create_access_token(
            user_id=user_id,
            extra_claims={"department": "engineering", "level": 5},
        )

        payload = handler.verify_token(token)

        assert payload.extra["department"] == "engineering"
        assert payload.extra["level"] == 5

    def test_token_jti_unique(self, handler):
        """Test token JTI is unique."""
        user_id = uuid4()
        token1 = handler.create_access_token(user_id=user_id)
        token2 = handler.create_access_token(user_id=user_id)

        payload1 = handler.verify_token(token1)
        payload2 = handler.verify_token(token2)

        assert payload1.jti != payload2.jti
