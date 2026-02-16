"""OAuth2/OIDC authentication backend.

This module provides OAuth2 and OpenID Connect authentication
for user authentication via external identity providers.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import httpx

from src.auth.base import AuthProvider, AuthResult, AuthenticationBackend, Credentials, User
from src.auth.jwt import JWTHandler, TokenPayload


class OAuth2Config:
    """Configuration for OAuth2 authentication.
    
    Attributes:
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret
        authorization_endpoint: Authorization endpoint URL
        token_endpoint: Token endpoint URL
        userinfo_endpoint: UserInfo endpoint URL
        jwks_uri: JWKS endpoint for token validation (OIDC)
        scopes: Requested OAuth2 scopes
        issuer: Expected token issuer (OIDC)
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        authorization_endpoint: str,
        token_endpoint: str,
        userinfo_endpoint: str,
        jwks_uri: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        issuer: Optional[str] = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorization_endpoint = authorization_endpoint
        self.token_endpoint = token_endpoint
        self.userinfo_endpoint = userinfo_endpoint
        self.jwks_uri = jwks_uri
        self.scopes = scopes or ["openid", "profile", "email"]
        self.issuer = issuer


class OAuth2Auth(AuthenticationBackend):
    """OAuth2/OIDC authentication backend.
    
    This backend authenticates users via OAuth2/OIDC providers
    like Auth0, Okta, Keycloak, etc.
    
    Example:
        config = OAuth2Config(
            client_id="...",
            client_secret="...",
            authorization_endpoint="...",
            token_endpoint="...",
            userinfo_endpoint="...",
        )
        auth = OAuth2Auth(config)
        credentials = Credentials(
            provider=AuthProvider.OAUTH2,
            token="eyJ..."
        )
        result = await auth.authenticate(credentials)
    """
    
    def __init__(
        self,
        config: OAuth2Config,
        jwt_handler: Optional[JWTHandler] = None,
    ):
        """Initialize OAuth2 authentication.
        
        Args:
            config: OAuth2 configuration
            jwt_handler: Optional JWT handler for session tokens
        """
        self.config = config
        self.jwt_handler = jwt_handler
        self._http_client: Optional[httpx.AsyncClient] = None
    
    @property
    def provider_type(self) -> AuthProvider:
        """Return the authentication provider type."""
        return AuthProvider.OAUTH2
    
    @property
    def http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    async def authenticate(self, credentials: Credentials) -> AuthResult:
        """Authenticate using OAuth2 token.
        
        Args:
            credentials: Credentials containing OAuth2 token
            
        Returns:
            AuthResult with user if authenticated
        """
        if not credentials.token:
            return AuthResult.failure_result(
                "OAuth2 token is required",
                "MISSING_TOKEN"
            )
        
        token = credentials.token
        
        # Validate token by fetching user info
        try:
            userinfo = await self._fetch_userinfo(token)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return AuthResult.failure_result(
                    "Invalid or expired OAuth2 token",
                    "INVALID_TOKEN"
                )
            return AuthResult.failure_result(
                f"OAuth2 provider error: {e}",
                "PROVIDER_ERROR"
            )
        except Exception as e:
            return AuthResult.failure_result(
                f"Failed to validate token: {e}",
                "VALIDATION_ERROR"
            )
        
        # Extract user information from userinfo
        user = self._create_user_from_userinfo(userinfo)
        
        # Generate session token if JWT handler is configured
        session_token = None
        if self.jwt_handler:
            session_token = self.jwt_handler.create_access_token(
                user_id=user.id,
                email=user.email,
                username=user.username,
                roles=user.roles,
                permissions=user.permissions,
                extra_claims={
                    "auth_provider": "oauth2",
                    "external_id": userinfo.get("sub"),
                },
            )
        
        return AuthResult.success_result(
            user=user,
            metadata={
                "auth_method": "oauth2",
                "external_id": userinfo.get("sub"),
                "session_token": session_token,
            }
        )
    
    async def _fetch_userinfo(self, token: str) -> Dict[str, Any]:
        """Fetch user info from OAuth2 provider.
        
        Args:
            token: OAuth2 access token
            
        Returns:
            User info dictionary
        """
        headers = {"Authorization": f"Bearer {token}"}
        response = await self.http_client.get(
            self.config.userinfo_endpoint,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()
    
    def _create_user_from_userinfo(self, userinfo: Dict[str, Any]) -> User:
        """Create User object from OAuth2 userinfo.
        
        Args:
            userinfo: User info from OAuth2 provider
            
        Returns:
            User object
        """
        # Extract standard OIDC claims
        sub = userinfo.get("sub", "")
        email = userinfo.get("email")
        username = userinfo.get("preferred_username") or userinfo.get("nickname") or email
        name = userinfo.get("name", "")
        given_name = userinfo.get("given_name", "")
        family_name = userinfo.get("family_name", "")
        
        # Map roles from token if available
        roles = userinfo.get("roles", [])
        if not roles:
            # Default role based on email domain or other logic
            roles = ["viewer"]
        
        # Determine primary role
        primary_role = roles[0] if roles else "viewer"
        
        # Generate deterministic UUID from subject
        # This ensures the same external user always gets the same internal ID
        user_id = UUID(bytes=bytes.fromhex(sub.replace("-", "")[:32].ljust(32, "0")[:32]))
        
        return User(
            id=user_id,
            email=email,
            username=username,
            role=primary_role,
            roles=roles,
            auth_provider=AuthProvider.OAUTH2,
            permissions=[],  # Will be determined by RBAC based on role
            metadata={
                "name": name,
                "given_name": given_name,
                "family_name": family_name,
                "external_sub": sub,
            },
        )
    
    async def exchange_code(
        self,
        code: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens.
        
        Args:
            code: Authorization code
            redirect_uri: Redirect URI used in authorization request
            code_verifier: PKCE code verifier (if PKCE was used)
            
        Returns:
            Token response dictionary
        """
        data = {
            "grant_type": "authorization_code",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        }
        
        if code_verifier:
            data["code_verifier"] = code_verifier
        
        response = await self.http_client.post(
            self.config.token_endpoint,
            data=data,
        )
        response.raise_for_status()
        return response.json()
    
    def get_authorization_url(
        self,
        redirect_uri: str,
        state: str,
        code_challenge: Optional[str] = None,
        code_challenge_method: Optional[str] = None,
    ) -> str:
        """Build authorization URL for OAuth2 flow.
        
        Args:
            redirect_uri: URI to redirect after authorization
            state: State parameter for CSRF protection
            code_challenge: PKCE code challenge
            code_challenge_method: PKCE method (S256 or plain)
            
        Returns:
            Authorization URL
        """
        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(self.config.scopes),
            "state": state,
        }
        
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = code_challenge_method or "S256"
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.config.authorization_endpoint}?{query}"
    
    async def validate_token(self, token: str) -> AuthResult:
        """Validate an OAuth2 token.
        
        Args:
            token: OAuth2 access token
            
        Returns:
            AuthResult containing validation outcome
        """
        credentials = Credentials(
            provider=AuthProvider.OAUTH2,
            token=token,
        )
        return await self.authenticate(credentials)
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
