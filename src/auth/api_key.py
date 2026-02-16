"""API Key authentication backend.

This module provides API key-based authentication for service accounts
and programmatic access.
"""

import hashlib
import secrets
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from src.auth.base import AuthProvider, AuthResult, AuthenticationBackend, Credentials, User
from src.auth.rbac import Role


class APIKeyAuth(AuthenticationBackend):
    """API Key authentication backend.
    
    This backend authenticates requests using API keys. API keys are
    hashed before storage and comparison for security.
    
    Example:
        auth = APIKeyAuth(api_key_store)
        credentials = Credentials(
            provider=AuthProvider.API_KEY,
            api_key="sk_live_..."
        )
        result = await auth.authenticate(credentials)
    """
    
    def __init__(self, api_key_store: Optional[Any] = None):
        """Initialize API key authentication.
        
        Args:
            api_key_store: Storage backend for API keys (database, cache, etc.)
        """
        self.api_key_store = api_key_store
    
    @property
    def provider_type(self) -> AuthProvider:
        """Return the authentication provider type."""
        return AuthProvider.API_KEY
    
    async def authenticate(self, credentials: Credentials) -> AuthResult:
        """Authenticate using API key.
        
        Args:
            credentials: Credentials containing api_key
            
        Returns:
            AuthResult with user if authenticated
        """
        if not credentials.api_key:
            return AuthResult.failure_result(
                "API key is required",
                "MISSING_API_KEY"
            )
        
        api_key = credentials.api_key
        
        # Validate key format
        if not self._validate_key_format(api_key):
            return AuthResult.failure_result(
                "Invalid API key format",
                "INVALID_KEY_FORMAT"
            )
        
        # Look up key in store
        key_data = await self._lookup_key(api_key)
        
        if not key_data:
            return AuthResult.failure_result(
                "Invalid API key",
                "INVALID_API_KEY"
            )
        
        # Check if key is active
        if not key_data.get("is_active", True):
            return AuthResult.failure_result(
                "API key is deactivated",
                "KEY_DEACTIVATED"
            )
        
        # Check expiration
        if key_data.get("expires_at"):
            expires_at = key_data["expires_at"]
            if isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if datetime.utcnow() > expires_at:
                return AuthResult.failure_result(
                    "API key has expired",
                    "KEY_EXPIRED"
                )
        
        # Update last used timestamp
        await self._update_last_used(key_data["id"])
        
        # Build user from key data
        user = User(
            id=UUID(key_data["user_id"]),
            email=key_data.get("user_email"),
            username=key_data.get("user_name"),
            role=key_data.get("role", Role.OPERATOR.value),
            roles=key_data.get("roles", [key_data.get("role", Role.OPERATOR.value)]),
            auth_provider=AuthProvider.API_KEY,
            permissions=key_data.get("permissions", []),
            metadata={
                "api_key_id": str(key_data["id"]),
                "api_key_name": key_data.get("name"),
            },
            is_service_account=True,
        )
        
        return AuthResult.success_result(
            user=user,
            metadata={
                "api_key_id": str(key_data["id"]),
                "auth_method": "api_key",
            }
        )
    
    def _validate_key_format(self, api_key: str) -> bool:
        """Validate API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if format is valid
        """
        # API keys should be at least 32 characters
        if len(api_key) < 32:
            return False
        
        # Should be alphanumeric with some special characters
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
        if not all(c in allowed_chars for c in api_key):
            return False
        
        return True
    
    async def _lookup_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Look up API key in storage.
        
        Args:
            api_key: Raw API key
            
        Returns:
            Key data if found, None otherwise
        """
        # Hash the key for lookup
        key_hash = self._hash_key(api_key)
        
        if self.api_key_store:
            return await self.api_key_store.get_by_hash(key_hash)
        
        # Fallback: in-memory lookup (development only)
        return self._in_memory_lookup(key_hash)
    
    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for storage/comparison.
        
        Args:
            api_key: Raw API key
            
        Returns:
            Hashed key
        """
        # Use SHA-256 for hashing
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def _update_last_used(self, key_id: UUID) -> None:
        """Update last used timestamp for a key.
        
        Args:
            key_id: API key ID
        """
        if self.api_key_store:
            await self.api_key_store.update_last_used(key_id, datetime.utcnow())
    
    # In-memory store for development (replace with database in production)
    _dev_keys: Dict[str, Dict[str, Any]] = {}
    
    def _in_memory_lookup(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Development-only in-memory key lookup."""
        return self._dev_keys.get(key_hash)
    
    @classmethod
    def register_dev_key(
        cls,
        api_key: str,
        user_id: UUID,
        role: str = "operator",
        name: str = "Development Key",
    ) -> None:
        """Register a key for development/testing.
        
        Args:
            api_key: The API key string
            user_id: Associated user ID
            role: Role for the key
            name: Name/description of the key
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        cls._dev_keys[key_hash] = {
            "id": UUID(secrets.token_hex(16)[:32]),
            "user_id": str(user_id),
            "user_name": "dev_user",
            "role": role,
            "roles": [role],
            "name": name,
            "is_active": True,
            "permissions": [],
        }
    
    async def validate_token(self, token: str) -> AuthResult:
        """Validate an API key token.
        
        Args:
            token: API key string
            
        Returns:
            AuthResult containing validation outcome
        """
        credentials = Credentials(
            provider=AuthProvider.API_KEY,
            api_key=token,
        )
        return await self.authenticate(credentials)


def generate_api_key(prefix: str = "sk", length: int = 48) -> str:
    """Generate a new secure API key.
    
    Args:
        prefix: Key prefix (e.g., "sk" for secret key)
        length: Length of random portion
        
    Returns:
        Generated API key
        
    Example:
        >>> generate_api_key("sk_live", 48)
        'sk_live_aB3x9K...'
    """
    # Generate cryptographically secure random string
    random_part = secrets.token_urlsafe(length)
    
    # Clean up any URL-unsafe characters and truncate to length
    random_part = random_part.replace("-", "").replace("_", "")[:length]
    
    return f"{prefix}_{random_part}"


def generate_key_pair() -> tuple[str, str]:
    """Generate an API key and its hash.
    
    Returns:
        Tuple of (api_key, key_hash)
    """
    api_key = generate_api_key()
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return api_key, key_hash
