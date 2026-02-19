"""Amazon S3 source plugin.

This plugin provides connectivity to Amazon S3 buckets for
file ingestion with support for:
- IAM roles and access keys
- Event-driven ingestion via S3 notifications
- Prefix-based filtering
- Cross-region access
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
class S3Config:
    """Configuration for S3 connection.
    
    Attributes:
        bucket: S3 bucket name
        region: AWS region
        access_key_id: AWS access key ID (optional, use IAM role if not provided)
        secret_access_key: AWS secret access key (optional)
        session_token: AWS session token for temporary credentials
        prefix: Optional prefix/path filter
        endpoint_url: Custom endpoint URL (for S3-compatible services)
        use_ssl: Whether to use SSL
        verify_ssl: Whether to verify SSL certificates
    """
    bucket: str
    region: str = "us-east-1"
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    prefix: str = ""
    endpoint_url: str | None = None
    use_ssl: bool = True
    verify_ssl: bool = True


class S3SourcePlugin(SourcePlugin):
    """Source plugin for Amazon S3.
    
    Supports:
    - Listing files with prefix filtering
    - Downloading files
    - IAM role-based authentication
    - Access key authentication
    - Event notification handling
    """

    def __init__(self) -> None:
        """Initialize the S3 source plugin."""
        self.logger = logger
        self._s3_client = None
        self._s3_resource = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="s3",
            name="Amazon S3 Source",
            version="1.0.0",
            type=PluginType.SOURCE,
            description="Ingest files from Amazon S3 buckets",
            author="Agentic Pipeline Team",
            supported_formats=["*"],  # Supports all file types
            requires_auth=True,
            config_schema={
                "type": "object",
                "required": ["bucket"],
                "properties": {
                    "bucket": {"type": "string", "description": "S3 bucket name"},
                    "region": {"type": "string", "default": "us-east-1"},
                    "access_key_id": {"type": "string"},
                    "secret_access_key": {"type": "string"},
                    "session_token": {"type": "string"},
                    "prefix": {"type": "string", "default": ""},
                    "endpoint_url": {"type": "string"},
                    "use_ssl": {"type": "boolean", "default": True},
                    "verify_ssl": {"type": "boolean", "default": True},
                },
            },
        )

    def _get_s3_client(self, config: S3Config) -> Any:
        """Get or create S3 client.
        
        Args:
            config: S3 configuration
            
        Returns:
            Boto3 S3 client
        """
        if self._s3_client is None:
            try:
                import boto3
                from botocore.config import Config

                # Build client arguments
                client_kwargs: dict[str, Any] = {
                    "region_name": config.region,
                    "use_ssl": config.use_ssl,
                    "verify": config.verify_ssl,
                }

                # Add endpoint URL if provided
                if config.endpoint_url:
                    client_kwargs["endpoint_url"] = config.endpoint_url

                # Add credentials if provided
                if config.access_key_id and config.secret_access_key:
                    client_kwargs["aws_access_key_id"] = config.access_key_id
                    client_kwargs["aws_secret_access_key"] = config.secret_access_key
                    if config.session_token:
                        client_kwargs["aws_session_token"] = config.session_token

                # Configure retries
                boto_config = Config(
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    max_pool_connections=25,
                )
                client_kwargs["config"] = boto_config

                self._s3_client = boto3.client("s3", **client_kwargs)
                self._s3_resource = boto3.resource("s3", **client_kwargs)

            except ImportError:
                raise RuntimeError(
                    "boto3 is required for S3 support. "
                    "Install with: pip install boto3"
                )

        return self._s3_client

    async def connect(self, config: dict[str, Any]) -> Connection:
        """Establish connection to S3.
        
        Args:
            config: Connection configuration
            
        Returns:
            Connection handle
        """
        s3_config = S3Config(**config)

        # Test connection by listing objects
        client = self._get_s3_client(s3_config)
        try:
            client.head_bucket(Bucket=s3_config.bucket)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to S3 bucket {s3_config.bucket}: {e}")

        return Connection(
            id=uuid4(),
            plugin_id=self.metadata.id,
            config=config,
            metadata={
                "bucket": s3_config.bucket,
                "region": s3_config.region,
                "prefix": s3_config.prefix,
            },
        )

    async def list_files(
        self,
        conn: Connection,
        path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[SourceFile]:
        """List files in S3 bucket.
        
        Args:
            conn: Connection handle
            path: Prefix path to list
            recursive: Whether to list recursively
            pattern: Optional glob pattern to filter files
            
        Returns:
            List of SourceFile objects
        """
        config = S3Config(**conn.config)
        client = self._get_s3_client(config)

        # Build prefix
        prefix = path or config.prefix
        if config.prefix and not prefix.startswith(config.prefix):
            prefix = f"{config.prefix}/{prefix}".strip("/")

        files: list[SourceFile] = []

        try:
            paginator = client.get_paginator("list_objects_v2")

            # If not recursive, use delimiter to get only direct children
            delimiter = "/" if not recursive else None

            page_iterator = paginator.paginate(
                Bucket=config.bucket,
                Prefix=prefix,
                Delimiter=delimiter,
            )

            for page in page_iterator:
                # Process files (Contents)
                for obj in page.get("Contents", []):
                    key = obj["Key"]

                    # Skip "directory" markers
                    if key.endswith("/"):
                        continue

                    # Apply pattern filter if specified
                    if pattern:
                        import fnmatch
                        if not fnmatch.fnmatch(key, pattern):
                            continue

                    # Determine MIME type
                    mime_type, _ = mimetypes.guess_type(key)

                    files.append(SourceFile(
                        path=key,
                        name=key.split("/")[-1],
                        size=obj["Size"],
                        mime_type=mime_type or "application/octet-stream",
                        modified_at=obj["LastModified"],
                        metadata={
                            "etag": obj["ETag"].strip('"'),
                            "storage_class": obj.get("StorageClass", "STANDARD"),
                        },
                    ))

                # If recursive, we don't need to process CommonPrefixes
                # as the pagination will get all objects

        except Exception as e:
            self.logger.error(f"Failed to list S3 files: {e}")
            raise

        return files

    async def get_file(
        self,
        conn: Connection,
        path: str,
        download_to: str | None = None,
    ) -> RetrievedFile:
        """Retrieve a file from S3.
        
        Args:
            conn: Connection handle
            path: S3 key (path) to the file
            download_to: Optional local path to save the file
            
        Returns:
            RetrievedFile with content and metadata
        """
        config = S3Config(**conn.config)
        client = self._get_s3_client(config)

        try:
            # Get object metadata first
            head_response = client.head_object(
                Bucket=config.bucket,
                Key=path,
            )

            # Build source file info
            mime_type, _ = mimetypes.guess_type(path)
            source_file = SourceFile(
                path=path,
                name=path.split("/")[-1],
                size=head_response.get("ContentLength"),
                mime_type=mime_type or head_response.get("ContentType", "application/octet-stream"),
                modified_at=head_response.get("LastModified"),
                metadata={
                    "etag": head_response["ETag"].strip('"'),
                    "version_id": head_response.get("VersionId"),
                },
            )

            # Download content
            if download_to:
                # Download to file
                client.download_file(config.bucket, path, download_to)

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
                response = client.get_object(
                    Bucket=config.bucket,
                    Key=path,
                )
                content = response["Body"].read()

                # Calculate hash
                import hashlib
                content_hash = hashlib.sha256(content).hexdigest()

                return RetrievedFile(
                    source_file=source_file,
                    content=content,
                    content_hash=content_hash,
                )

        except client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"S3 object not found: {path}")
        except Exception as e:
            self.logger.error(f"Failed to get S3 file {path}: {e}")
            raise

    async def stream_file(
        self,
        conn: Connection,
        path: str,
        chunk_size: int = 8192,
    ) -> Any:
        """Stream a file from S3.
        
        Args:
            conn: Connection handle
            path: S3 key to the file
            chunk_size: Size of chunks to stream
            
        Yields:
            File content chunks
        """
        config = S3Config(**conn.config)
        client = self._get_s3_client(config)

        try:
            response = client.get_object(
                Bucket=config.bucket,
                Key=path,
            )

            body = response["Body"]
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        except Exception as e:
            self.logger.error(f"Failed to stream S3 file {path}: {e}")
            raise

    async def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate S3 configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check required fields
        if not config.get("bucket"):
            errors.append("bucket is required")

        # Validate credentials or IAM role
        has_access_key = config.get("access_key_id") and config.get("secret_access_key")
        # Note: We can't easily check for IAM role here, so we just warn
        if not has_access_key:
            warnings.append(
                "No access keys provided. Ensure IAM role is configured."
            )

        # Validate region
        valid_regions = [
            "us-east-1", "us-east-2", "us-west-1", "us-west-2",
            "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1",
            "ap-northeast-1", "ap-northeast-2", "ap-southeast-1", "ap-southeast-2",
            "sa-east-1", "ca-central-1",
        ]
        region = config.get("region", "us-east-1")
        if region not in valid_regions:
            warnings.append(f"Unusual region: {region}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Check S3 health.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus
        """
        if config is None:
            return HealthStatus.UNKNOWN

        try:
            s3_config = S3Config(**config)
            client = self._get_s3_client(s3_config)

            # Try to head the bucket
            client.head_bucket(Bucket=s3_config.bucket)

            # Try to list a single object to verify permissions
            response = client.list_objects_v2(
                Bucket=s3_config.bucket,
                MaxKeys=1,
            )

            return HealthStatus.HEALTHY

        except Exception as e:
            self.logger.warning(f"S3 health check failed: {e}")
            return HealthStatus.UNHEALTHY

    async def handle_s3_notification(
        self,
        notification: dict[str, Any],
        config: dict[str, Any],
    ) -> list[SourceFile]:
        """Handle S3 event notification.
        
        Processes S3 event notifications (from SNS, SQS, or EventBridge)
        and returns affected files.
        
        Args:
            notification: S3 event notification
            config: S3 configuration
            
        Returns:
            List of affected SourceFiles
        """
        files: list[SourceFile] = []

        try:
            # Parse S3 notification
            records = notification.get("Records", [])

            for record in records:
                event_name = record.get("eventName", "")
                s3_info = record.get("s3", {})
                bucket_name = s3_info.get("bucket", {}).get("name")
                object_info = s3_info.get("object", {})
                key = object_info.get("key")

                # URL decode the key
                from urllib.parse import unquote_plus
                key = unquote_plus(key)

                # Only process object created events
                if "ObjectCreated" in event_name and key:
                    mime_type, _ = mimetypes.guess_type(key)

                    files.append(SourceFile(
                        path=key,
                        name=key.split("/")[-1],
                        size=object_info.get("size"),
                        mime_type=mime_type or "application/octet-stream",
                        modified_at=datetime.utcnow(),  # Use current time as approximation
                        metadata={
                            "etag": object_info.get("eTag", "").strip('"'),
                            "event": event_name,
                            "bucket": bucket_name,
                        },
                    ))

        except Exception as e:
            self.logger.error(f"Failed to process S3 notification: {e}")

        return files

    async def authorize(
        self,
        user_id: str,
        resource: str,
        action: str,
        config: dict[str, Any],
    ) -> bool:
        """Check if user is authorized to access an S3 resource.
        
        For S3, authorization is primarily handled through IAM policies
        associated with the credentials. This method provides an additional
        layer for application-level authorization.
        
        Args:
            user_id: User identifier
            resource: S3 key/path
            action: Action to perform (read, write, delete)
            config: S3 configuration
            
        Returns:
            True if user is authorized
        """
        # S3 authorization primarily relies on IAM policies
        # The IAM credentials determine what actions are allowed

        # Application-level authorization can be added here
        # For example, checking if user has access to this specific bucket

        bucket = config.get("bucket", "")

        # Check if there's a bucket whitelist
        allowed_buckets = config.get("allowed_buckets", [])
        if allowed_buckets and bucket not in allowed_buckets:
            self.logger.warning(
                f"User {user_id} attempted to access non-allowed bucket: {bucket}"
            )
            return False

        # Check path-based restrictions
        allowed_prefixes = config.get("allowed_prefixes", [])
        if allowed_prefixes:
            resource_allowed = any(
                resource.startswith(prefix) for prefix in allowed_prefixes
            )
            if not resource_allowed:
                self.logger.warning(
                    f"User {user_id} attempted to access restricted path: {resource}"
                )
                return False

        # For read operations, we assume IAM policies handle the actual access control
        # The S3 API will return 403 if IAM policies don't allow access
        if action == "read":
            return True

        # For write/delete operations, require explicit configuration
        if action in ("write", "delete"):
            allow_write = config.get("allow_write_operations", False)
            if not allow_write:
                return False

        return True
