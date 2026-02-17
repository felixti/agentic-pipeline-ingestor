"""Azure Blob Storage source plugin.

This plugin provides connectivity to Azure Blob Storage for
file ingestion with support for:
- SAS tokens
- Managed Identity
- Connection strings
- Event Grid integration
- Container-level and blob-level operations
"""

import logging
import mimetypes
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from src.plugins.base import (
    Connection,
    HealthStatus,
    PluginMetadata,
    PluginType,
    RetrievedFile,
    SourceFile,
    SourcePlugin,
    ValidationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class AzureBlobConfig:
    """Configuration for Azure Blob Storage connection.
    
    Attributes:
        account_name: Storage account name
        container: Container name
        connection_string: Full connection string (alternative to individual fields)
        account_key: Storage account key
        sas_token: Shared Access Signature token
        tenant_id: Azure AD tenant ID for managed identity
        client_id: Client ID for managed identity
        prefix: Optional prefix/path filter
        endpoint_suffix: Endpoint suffix (for sovereign clouds)
    """
    account_name: str
    container: str
    connection_string: str | None = None
    account_key: str | None = None
    sas_token: str | None = None
    tenant_id: str | None = None
    client_id: str | None = None
    prefix: str = ""
    endpoint_suffix: str = "core.windows.net"


class AzureBlobSourcePlugin(SourcePlugin):
    """Source plugin for Azure Blob Storage.
    
    Supports:
    - Connection string authentication
    - Account key authentication
    - SAS token authentication
    - Managed Identity authentication
    - Event Grid event handling
    """

    def __init__(self) -> None:
        """Initialize the Azure Blob source plugin."""
        self.logger = logger
        self._blob_service = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="azure_blob",
            name="Azure Blob Storage Source",
            version="1.0.0",
            type=PluginType.SOURCE,
            description="Ingest files from Azure Blob Storage",
            author="Agentic Pipeline Team",
            supported_formats=["*"],  # Supports all file types
            requires_auth=True,
            config_schema={
                "type": "object",
                "required": ["account_name", "container"],
                "properties": {
                    "account_name": {"type": "string", "description": "Storage account name"},
                    "container": {"type": "string", "description": "Container name"},
                    "connection_string": {"type": "string"},
                    "account_key": {"type": "string"},
                    "sas_token": {"type": "string"},
                    "tenant_id": {"type": "string"},
                    "client_id": {"type": "string"},
                    "prefix": {"type": "string", "default": ""},
                    "endpoint_suffix": {"type": "string", "default": "core.windows.net"},
                },
            },
        )

    def _get_blob_service(self, config: AzureBlobConfig):
        """Get or create BlobServiceClient.
        
        Args:
            config: Azure Blob configuration
            
        Returns:
            BlobServiceClient
        """
        if self._blob_service is None:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.storage.blob import BlobServiceClient

                # Build connection
                if config.connection_string:
                    # Use connection string
                    self._blob_service = BlobServiceClient.from_connection_string(
                        config.connection_string
                    )
                elif config.sas_token:
                    # Use SAS token
                    account_url = f"https://{config.account_name}.blob.{config.endpoint_suffix}"
                    self._blob_service = BlobServiceClient(
                        account_url=account_url,
                        credential=config.sas_token,
                    )
                elif config.account_key:
                    # Use account key
                    account_url = f"https://{config.account_name}.blob.{config.endpoint_suffix}"
                    from azure.storage.blob import StorageSharedKeyCredential
                    credential = StorageSharedKeyCredential(
                        config.account_name,
                        config.account_key,
                    )
                    self._blob_service = BlobServiceClient(
                        account_url=account_url,
                        credential=credential,
                    )
                else:
                    # Use managed identity / DefaultAzureCredential
                    account_url = f"https://{config.account_name}.blob.{config.endpoint_suffix}"
                    credential = DefaultAzureCredential(
                        managed_identity_client_id=config.client_id,
                    )
                    self._blob_service = BlobServiceClient(
                        account_url=account_url,
                        credential=credential,
                    )

            except ImportError:
                raise RuntimeError(
                    "azure-storage-blob and azure-identity are required. "
                    "Install with: pip install azure-storage-blob azure-identity"
                )

        return self._blob_service

    def _get_container_client(self, config: AzureBlobConfig):
        """Get container client.
        
        Args:
            config: Azure Blob configuration
            
        Returns:
            ContainerClient
        """
        blob_service = self._get_blob_service(config)
        return blob_service.get_container_client(config.container)

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Establish connection to Azure Blob Storage.
        
        Args:
            config: Connection configuration
            
        Returns:
            Connection handle
        """
        blob_config = AzureBlobConfig(**config)

        # Test connection by getting container properties
        try:
            container_client = self._get_container_client(blob_config)
            container_client.get_container_properties()
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Azure Blob container {blob_config.container}: {e}"
            )

        return Connection(
            id=uuid4(),
            plugin_id=self.metadata.id,
            config=config,
            metadata={
                "account_name": blob_config.account_name,
                "container": blob_config.container,
                "prefix": blob_config.prefix,
                "endpoint_suffix": blob_config.endpoint_suffix,
            },
        )

    async def list_files(
        self,
        conn: Connection,
        path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[SourceFile]:
        """List files in Azure Blob container.
        
        Args:
            conn: Connection handle
            path: Prefix path to list
            recursive: Whether to list recursively
            pattern: Optional glob pattern to filter files
            
        Returns:
            List of SourceFile objects
        """
        config = AzureBlobConfig(**conn.config)
        container_client = self._get_container_client(config)

        # Build prefix
        prefix = path or config.prefix
        if config.prefix and not prefix.startswith(config.prefix):
            prefix = f"{config.prefix}/{prefix}".strip("/")

        files: list[SourceFile] = []

        try:
            if recursive:
                # List all blobs with prefix
                blob_list = container_client.list_blobs(name_starts_with=prefix)
            else:
                # List only direct children
                blob_list = container_client.walk_blobs(name_starts_with=prefix)

            for blob in blob_list:
                # Skip directories
                if blob.name.endswith("/"):
                    continue

                # Apply pattern filter
                if pattern:
                    import fnmatch
                    if not fnmatch.fnmatch(blob.name, pattern):
                        continue

                # Determine MIME type
                mime_type, _ = mimetypes.guess_type(blob.name)

                files.append(SourceFile(
                    path=blob.name,
                    name=blob.name.split("/")[-1],
                    size=blob.size,
                    mime_type=mime_type or "application/octet-stream",
                    modified_at=blob.last_modified,
                    metadata={
                        "etag": blob.etag.strip('"') if blob.etag else "",
                        "content_type": blob.content_settings.content_type if blob.content_settings else None,
                    },
                ))

        except Exception as e:
            self.logger.error(f"Failed to list Azure Blob files: {e}")
            raise

        return files

    async def get_file(
        self,
        conn: Connection,
        path: str,
        download_to: str | None = None,
    ) -> RetrievedFile:
        """Retrieve a file from Azure Blob Storage.
        
        Args:
            conn: Connection handle
            path: Blob name (path)
            download_to: Optional local path to save the file
            
        Returns:
            RetrievedFile with content and metadata
        """
        config = AzureBlobConfig(**conn.config)
        container_client = self._get_container_client(config)
        blob_client = container_client.get_blob_client(path)

        try:
            # Get blob properties
            properties = blob_client.get_blob_properties()

            # Build source file info
            mime_type, _ = mimetypes.guess_type(path)
            source_file = SourceFile(
                path=path,
                name=path.split("/")[-1],
                size=properties.size,
                mime_type=mime_type or properties.content_settings.content_type or "application/octet-stream",
                modified_at=properties.last_modified,
                metadata={
                    "etag": properties.etag.strip('"') if properties.etag else "",
                    "version_id": properties.version_id,
                    "tags": properties.tags,
                },
            )

            # Download content
            if download_to:
                # Download to file
                with open(download_to, "wb") as file:
                    download_stream = blob_client.download_blob()
                    file.write(download_stream.readall())

                # Calculate hash
                import hashlib
                sha256 = hashlib.sha256()
                with open(download_to, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)

                return RetrievedFile(
                    source_file=source_file,
                    content=b"",  # Content is on disk
                    content_hash=sha256.hexdigest(),
                    local_path=download_to,
                )
            else:
                # Download to memory
                download_stream = blob_client.download_blob()
                content = download_stream.readall()

                # Calculate hash
                import hashlib
                content_hash = hashlib.sha256(content).hexdigest()

                return RetrievedFile(
                    source_file=source_file,
                    content=content,
                    content_hash=content_hash,
                )

        except Exception as e:
            if "BlobNotFound" in str(e):
                raise FileNotFoundError(f"Azure Blob not found: {path}")
            self.logger.error(f"Failed to get Azure Blob {path}: {e}")
            raise

    async def stream_file(
        self,
        conn: Connection,
        path: str,
        chunk_size: int = 8192,
    ):
        """Stream a file from Azure Blob Storage.
        
        Args:
            conn: Connection handle
            path: Blob name
            chunk_size: Size of chunks to stream
            
        Yields:
            File content chunks
        """
        config = AzureBlobConfig(**conn.config)
        container_client = self._get_container_client(config)
        blob_client = container_client.get_blob_client(path)

        try:
            download_stream = blob_client.download_blob()

            for chunk in download_stream.chunks():
                yield chunk

        except Exception as e:
            self.logger.error(f"Failed to stream Azure Blob {path}: {e}")
            raise

    async def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate Azure Blob configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check required fields
        if not config.get("account_name"):
            errors.append("account_name is required")

        if not config.get("container"):
            errors.append("container is required")

        # Validate credentials
        has_connection_string = bool(config.get("connection_string"))
        has_account_key = bool(config.get("account_key"))
        has_sas = bool(config.get("sas_token"))

        if not (has_connection_string or has_account_key or has_sas):
            warnings.append(
                "No explicit credentials provided. Will attempt to use "
                "DefaultAzureCredential (managed identity or Azure CLI)."
            )

        # Check for container name validity
        container = config.get("container", "")
        if container:
            # Container names must be lowercase, alphanumeric, and 3-63 characters
            if not container.islower() or not container.replace("-", "").isalnum():
                warnings.append(
                    "Container name should be lowercase alphanumeric with hyphens only"
                )
            if len(container) < 3 or len(container) > 63:
                errors.append("Container name must be 3-63 characters")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Check Azure Blob Storage health.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus
        """
        if config is None:
            return HealthStatus.UNKNOWN

        try:
            blob_config = AzureBlobConfig(**config)
            container_client = self._get_container_client(blob_config)

            # Try to get container properties
            container_client.get_container_properties()

            # Try to list one blob to verify read permissions
            blobs = list(container_client.list_blobs(max_results=1))

            return HealthStatus.HEALTHY

        except Exception as e:
            self.logger.warning(f"Azure Blob health check failed: {e}")
            return HealthStatus.UNHEALTHY

    async def handle_event_grid_notification(
        self,
        event: dict[str, Any],
        config: dict[str, Any],
    ) -> list[SourceFile]:
        """Handle Event Grid notification.
        
        Processes Event Grid events for blob storage changes.
        
        Args:
            event: Event Grid event
            config: Azure Blob configuration
            
        Returns:
            List of affected SourceFiles
        """
        files: list[SourceFile] = []

        try:
            event_type = event.get("eventType", "")

            # Only process blob created events
            if "BlobCreated" in event_type:
                data = event.get("data", {})
                url = data.get("url", "")

                # Parse URL to get blob path
                # URL format: https://account.blob.core.windows.net/container/path
                if "/" in url:
                    parts = url.split("/")
                    if len(parts) >= 5:
                        # Reconstruct blob path from URL
                        blob_path = "/".join(parts[5:])  # Skip protocol, account, 'blob', endpoint, container

                        # URL decode
                        from urllib.parse import unquote
                        blob_path = unquote(blob_path)

                        mime_type, _ = mimetypes.guess_type(blob_path)

                        files.append(SourceFile(
                            path=blob_path,
                            name=blob_path.split("/")[-1],
                            mime_type=mime_type or "application/octet-stream",
                            modified_at=datetime.utcnow(),
                            metadata={
                                "event_type": event_type,
                                "url": url,
                            },
                        ))

        except Exception as e:
            self.logger.error(f"Failed to process Event Grid event: {e}")

        return files
