"""Azure Active Directory authentication backend.

This module provides Azure AD (Entra ID) authentication for
enterprise SSO integration.
"""

from typing import Any
from uuid import UUID

import httpx

from src.auth.base import AuthenticationBackend, AuthProvider, AuthResult, Credentials, User
from src.auth.jwt import JWTHandler


class AzureADConfig:
    """Configuration for Azure AD authentication.
    
    Attributes:
        tenant_id: Azure AD tenant ID
        client_id: Application (client) ID
        client_secret: Application client secret
        authority: Azure AD authority URL
        scopes: Requested scopes
        allowed_groups: Optional list of allowed group IDs
        allowed_domains: Optional list of allowed email domains
    """

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str | None = None,
        scopes: list[str] | None = None,
        allowed_groups: list[str] | None = None,
        allowed_domains: list[str] | None = None,
    ):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"
        self.scopes = scopes or ["openid", "profile", "email", "User.Read"]
        self.allowed_groups = allowed_groups or []
        self.allowed_domains = allowed_domains or []

    @property
    def authorization_endpoint(self) -> str:
        """Get authorization endpoint URL."""
        return f"{self.authority}/oauth2/v2.0/authorize"

    @property
    def token_endpoint(self) -> str:
        """Get token endpoint URL."""
        return f"{self.authority}/oauth2/v2.0/token"

    @property
    def userinfo_endpoint(self) -> str:
        """Get userinfo endpoint URL."""
        return "https://graph.microsoft.com/v1.0/me"

    @property
    def jwks_uri(self) -> str:
        """Get JWKS endpoint URL for token validation."""
        return f"{self.authority}/discovery/v2.0/keys"


class AzureADAuth(AuthenticationBackend):
    """Azure Active Directory authentication backend.
    
    This backend authenticates users via Azure AD (Entra ID) for
    enterprise single sign-on.
    
    Example:
        config = AzureADConfig(
            tenant_id="...",
            client_id="...",
            client_secret="...",
        )
        auth = AzureADAuth(config)
        credentials = Credentials(
            provider=AuthProvider.AZURE_AD,
            token="eyJ..."
        )
        result = await auth.authenticate(credentials)
    """

    def __init__(
        self,
        config: AzureADConfig,
        jwt_handler: JWTHandler | None = None,
    ):
        """Initialize Azure AD authentication.
        
        Args:
            config: Azure AD configuration
            jwt_handler: Optional JWT handler for session tokens
        """
        self.config = config
        self.jwt_handler = jwt_handler
        self._http_client: httpx.AsyncClient | None = None

    @property
    def provider_type(self) -> AuthProvider:
        """Return the authentication provider type."""
        return AuthProvider.AZURE_AD

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def authenticate(self, credentials: Credentials) -> AuthResult:
        """Authenticate using Azure AD token.
        
        Args:
            credentials: Credentials containing Azure AD token
            
        Returns:
            AuthResult with user if authenticated
        """
        if not credentials.token:
            return AuthResult.failure_result(
                "Azure AD token is required",
                "MISSING_TOKEN"
            )

        token = credentials.token

        # Validate token by fetching user info from Microsoft Graph
        try:
            userinfo = await self._fetch_userinfo(token)
            member_groups = await self._fetch_member_groups(token)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return AuthResult.failure_result(
                    "Invalid or expired Azure AD token",
                    "INVALID_TOKEN"
                )
            return AuthResult.failure_result(
                f"Azure AD error: {e}",
                "PROVIDER_ERROR"
            )
        except Exception as e:
            return AuthResult.failure_result(
                f"Failed to validate token: {e}",
                "VALIDATION_ERROR"
            )

        # Check domain restrictions
        email = userinfo.get("mail") or userinfo.get("userPrincipalName", "")
        if self.config.allowed_domains:
            domain = email.split("@")[-1] if "@" in email else ""
            if domain not in self.config.allowed_domains:
                return AuthResult.failure_result(
                    f"Domain '{domain}' is not allowed",
                    "DOMAIN_NOT_ALLOWED"
                )

        # Check group membership restrictions
        if self.config.allowed_groups:
            user_group_ids = {g["id"] for g in member_groups.get("value", [])}
            allowed_set = set(self.config.allowed_groups)
            if not user_group_ids.intersection(allowed_set):
                return AuthResult.failure_result(
                    "User is not in an authorized group",
                    "GROUP_NOT_AUTHORIZED"
                )

        # Create user from Azure AD info
        user = self._create_user_from_graph(userinfo, member_groups)

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
                    "auth_provider": "azure_ad",
                    "tenant_id": self.config.tenant_id,
                    "oid": userinfo.get("id"),
                },
            )

        return AuthResult.success_result(
            user=user,
            metadata={
                "auth_method": "azure_ad",
                "tenant_id": self.config.tenant_id,
                "oid": userinfo.get("id"),
                "session_token": session_token,
            }
        )

    async def _fetch_userinfo(self, token: str) -> dict[str, Any]:
        """Fetch user info from Microsoft Graph.
        
        Args:
            token: Azure AD access token
            
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

    async def _fetch_member_groups(self, token: str) -> dict[str, Any]:
        """Fetch user's group memberships from Microsoft Graph.
        
        Args:
            token: Azure AD access token
            
        Returns:
            Groups dictionary
        """
        headers = {"Authorization": f"Bearer {token}"}
        response = await self.http_client.get(
            f"{self.config.userinfo_endpoint}/memberOf",
            headers=headers,
        )
        if response.status_code == 200:
            return response.json()
        return {"value": []}

    def _create_user_from_graph(
        self,
        userinfo: dict[str, Any],
        member_groups: dict[str, Any],
    ) -> User:
        """Create User object from Microsoft Graph data.
        
        Args:
            userinfo: User info from Microsoft Graph
            member_groups: User's group memberships
            
        Returns:
            User object
        """
        # Extract user info
        oid = userinfo.get("id", "")
        email = userinfo.get("mail") or userinfo.get("userPrincipalName", "")
        username = userinfo.get("userPrincipalName", "").split("@")[0]
        display_name = userinfo.get("displayName", "")
        given_name = userinfo.get("givenName", "")
        surname = userinfo.get("surname", "")
        job_title = userinfo.get("jobTitle", "")
        department = userinfo.get("department", "")

        # Map Azure AD groups to roles
        groups = member_groups.get("value", [])
        group_names = [g.get("displayName", "").lower() for g in groups]

        # Simple role mapping based on group names
        roles = ["viewer"]  # Default role

        # Check for admin groups
        admin_keywords = ["admin", "administrator", "owner"]
        if any(keyword in name for name in group_names for keyword in admin_keywords):
            roles.append("admin")

        # Check for operator groups
        operator_keywords = ["operator", "devops", "sre"]
        if any(keyword in name for name in group_names for keyword in operator_keywords):
            roles.append("operator")

        # Check for developer groups
        dev_keywords = ["developer", "engineer", "dev"]
        if any(keyword in name for name in group_names for keyword in dev_keywords):
            roles.append("developer")

        # Determine primary role (highest privilege)
        role_priority = ["admin", "operator", "developer", "viewer"]
        primary_role = "viewer"
        for role in role_priority:
            if role in roles:
                primary_role = role
                break

        # Generate deterministic UUID from Azure AD object ID
        # Remove dashes and pad/truncate to 32 chars
        oid_clean = oid.replace("-", "").ljust(32, "0")[:32]
        try:
            user_id = UUID(oid_clean)
        except ValueError:
            # Fallback: hash the OID
            import hashlib
            oid_hash = hashlib.sha256(oid.encode()).hexdigest()[:32]
            user_id = UUID(oid_hash)

        return User(
            id=user_id,
            email=email,
            username=username,
            role=primary_role,
            roles=list(set(roles)),  # Remove duplicates
            auth_provider=AuthProvider.AZURE_AD,
            permissions=[],  # Will be determined by RBAC
            metadata={
                "display_name": display_name,
                "given_name": given_name,
                "surname": surname,
                "job_title": job_title,
                "department": department,
                "azure_ad_oid": oid,
                "tenant_id": self.config.tenant_id,
                "groups": [g.get("displayName") for g in groups],
            },
        )

    async def exchange_code(
        self,
        code: str,
        redirect_uri: str,
        code_verifier: str | None = None,
    ) -> dict[str, Any]:
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
            "code": code,
            "redirect_uri": redirect_uri,
            "scope": " ".join(self.config.scopes),
        }

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

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
        code_challenge: str | None = None,
        code_challenge_method: str | None = None,
    ) -> str:
        """Build authorization URL for Azure AD flow.
        
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
            "response_mode": "query",
        }

        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = code_challenge_method or "S256"

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.config.authorization_endpoint}?{query}"

    async def validate_token(self, token: str) -> AuthResult:
        """Validate an Azure AD token.
        
        Args:
            token: Azure AD access token
            
        Returns:
            AuthResult containing validation outcome
        """
        credentials = Credentials(
            provider=AuthProvider.AZURE_AD,
            token=token,
        )
        return await self.authenticate(credentials)

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
