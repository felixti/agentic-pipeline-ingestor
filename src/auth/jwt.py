"""JWT token handling for authentication.

This module provides JWT token creation, validation, and refresh functionality.
"""

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError


# Configuration constants - in production, these would come from environment/settings
DEFAULT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS = 7


@dataclass
class TokenPayload:
    """Decoded JWT token payload.
    
    Attributes:
        sub: Subject (user ID)
        email: User email
        username: User login name
        roles: User roles
        permissions: User permissions
        type: Token type (access, refresh)
        iat: Issued at timestamp
        exp: Expiration timestamp
        jti: JWT ID (unique token identifier)
        extra: Additional claims
    """
    sub: UUID
    email: Optional[str] = None
    username: Optional[str] = None
    roles: List[str] = None
    permissions: List[str] = None
    type: str = "access"
    iat: Optional[datetime] = None
    exp: Optional[datetime] = None
    jti: Optional[str] = None
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.permissions is None:
            self.permissions = []
        if self.extra is None:
            self.extra = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary for JWT encoding."""
        return {
            "sub": str(self.sub),
            "email": self.email,
            "username": self.username,
            "roles": self.roles,
            "permissions": self.permissions,
            "type": self.type,
            "iat": self.iat.timestamp() if self.iat else None,
            "exp": self.exp.timestamp() if self.exp else None,
            "jti": self.jti,
            **self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenPayload":
        """Create payload from decoded JWT dictionary."""
        # Handle timestamp conversion
        iat = data.get("iat")
        exp = data.get("exp")
        
        if isinstance(iat, (int, float)):
            iat = datetime.utcfromtimestamp(iat)
        if isinstance(exp, (int, float)):
            exp = datetime.utcfromtimestamp(exp)
        
        # Extract extra fields
        known_fields = {"sub", "email", "username", "roles", "permissions", 
                       "type", "iat", "exp", "jti"}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(
            sub=UUID(data["sub"]) if "sub" in data else None,
            email=data.get("email"),
            username=data.get("username"),
            roles=data.get("roles", []),
            permissions=data.get("permissions", []),
            type=data.get("type", "access"),
            iat=iat,
            exp=exp,
            jti=data.get("jti"),
            extra=extra,
        )
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.exp:
            return False
        return datetime.utcnow() > self.exp
    
    def is_refresh_token(self) -> bool:
        """Check if this is a refresh token."""
        return self.type == "refresh"
    
    def is_access_token(self) -> bool:
        """Check if this is an access token."""
        return self.type == "access"


class JWTHandler:
    """Handler for JWT token operations.
    
    This class provides methods for creating, validating, and refreshing
    JWT tokens for authentication.
    """
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = DEFAULT_ALGORITHM,
        access_token_expire_minutes: int = DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days: int = DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS,
    ):
        """Initialize JWT handler.
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm to use
            access_token_expire_minutes: Access token lifetime in minutes
            refresh_token_expire_days: Refresh token lifetime in days
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
    
    def create_token(
        self,
        user_id: UUID,
        email: Optional[str] = None,
        username: Optional[str] = None,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
        token_type: str = "access",
        expires: Optional[timedelta] = None,
        extra_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a JWT token.
        
        Args:
            user_id: User identifier
            email: User email address
            username: User login name
            roles: User roles
            permissions: User permissions
            token_type: Type of token (access or refresh)
            expires: Custom expiration time
            extra_claims: Additional JWT claims
            
        Returns:
            Encoded JWT token string
        """
        now = datetime.utcnow()
        
        # Determine expiration
        if expires is None:
            if token_type == "refresh":
                expires = timedelta(days=self.refresh_token_expire_days)
            else:
                expires = timedelta(minutes=self.access_token_expire_minutes)
        
        exp = now + expires
        
        # Create payload
        payload = TokenPayload(
            sub=user_id,
            email=email,
            username=username,
            roles=roles or [],
            permissions=permissions or [],
            type=token_type,
            iat=now,
            exp=exp,
            jti=secrets.token_hex(16),
            extra=extra_claims or {},
        )
        
        # Encode token
        return jwt.encode(
            payload.to_dict(),
            self.secret_key,
            algorithm=self.algorithm,
        )
    
    def verify_token(self, token: str) -> TokenPayload:
        """Verify and decode a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            payload_dict = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
            )
            return TokenPayload.from_dict(payload_dict)
        except ExpiredSignatureError as e:
            raise JWTError("Token has expired") from e
        except JWTError:
            raise
    
    def create_access_token(
        self,
        user_id: UUID,
        email: Optional[str] = None,
        username: Optional[str] = None,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
        extra_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create an access token.
        
        Args:
            user_id: User identifier
            email: User email address
            username: User login name
            roles: User roles
            permissions: User permissions
            extra_claims: Additional JWT claims
            
        Returns:
            Encoded access token
        """
        return self.create_token(
            user_id=user_id,
            email=email,
            username=username,
            roles=roles,
            permissions=permissions,
            token_type="access",
            extra_claims=extra_claims,
        )
    
    def create_refresh_token(
        self,
        user_id: UUID,
        extra_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a refresh token.
        
        Args:
            user_id: User identifier
            extra_claims: Additional JWT claims
            
        Returns:
            Encoded refresh token
        """
        return self.create_token(
            user_id=user_id,
            token_type="refresh",
            extra_claims=extra_claims,
        )
    
    def refresh_access_token(
        self,
        refresh_token: str,
        email: Optional[str] = None,
        username: Optional[str] = None,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
    ) -> str:
        """Create a new access token from a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            email: User email (optional, uses token payload if not provided)
            username: Username (optional, uses token payload if not provided)
            roles: User roles (optional, uses token payload if not provided)
            permissions: User permissions (optional, uses token payload if not provided)
            
        Returns:
            New access token
            
        Raises:
            JWTError: If refresh token is invalid or expired
        """
        payload = self.verify_token(refresh_token)
        
        if not payload.is_refresh_token():
            raise JWTError("Invalid token type: expected refresh token")
        
        return self.create_access_token(
            user_id=payload.sub,
            email=email or payload.email,
            username=username or payload.username,
            roles=roles or payload.roles,
            permissions=permissions or payload.permissions,
        )
    
    def generate_secret_key(self) -> str:
        """Generate a new random secret key.
        
        Returns:
            Random secret key suitable for JWT signing
        """
        return secrets.token_urlsafe(32)
