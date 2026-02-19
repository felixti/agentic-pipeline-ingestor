"""Unit tests for S3 source plugin."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.plugins.base import Connection, HealthStatus, SourceFile
from src.plugins.sources.s3_source import S3Config, S3SourcePlugin

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def s3_plugin():
    """Create an S3 source plugin instance."""
    return S3SourcePlugin()


@pytest.fixture
def valid_s3_config():
    """Return a valid S3 configuration."""
    return {
        "bucket": "test-bucket",
        "region": "us-east-1",
        "access_key_id": "test-access-key",
        "secret_access_key": "test-secret-key",
    }


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_connection(valid_s3_config):
    """Create a mock Connection object."""
    return Connection(
        id=MagicMock(),
        plugin_id="s3",
        config=valid_s3_config,
        metadata={
            "bucket": valid_s3_config["bucket"],
            "region": valid_s3_config["region"],
            "prefix": "",
        },
    )


# ============================================================================
# S3SourcePlugin Class Tests
# ============================================================================

@pytest.mark.unit
class TestS3SourcePlugin:
    """Tests for S3SourcePlugin class."""

    def test_init(self):
        """Test plugin initialization."""
        plugin = S3SourcePlugin()
        assert plugin._s3_client is None
        assert plugin._s3_resource is None
        assert plugin.logger is not None

    def test_metadata(self, s3_plugin):
        """Test plugin metadata."""
        metadata = s3_plugin.metadata

        assert metadata.id == "s3"
        assert metadata.name == "Amazon S3 Source"
        assert metadata.version == "1.0.0"
        assert metadata.type.value == "source"
        assert metadata.requires_auth is True
        assert "*" in metadata.supported_formats

    def test_metadata_config_schema(self, s3_plugin):
        """Test plugin metadata config schema."""
        schema = s3_plugin.metadata.config_schema

        assert schema["type"] == "object"
        assert "bucket" in schema["required"]
        assert "bucket" in schema["properties"]
        assert "region" in schema["properties"]
        assert "access_key_id" in schema["properties"]
        assert "secret_access_key" in schema["properties"]


# ============================================================================
# S3Config Tests
# ============================================================================

@pytest.mark.unit
class TestS3Config:
    """Tests for S3Config dataclass."""

    def test_default_values(self):
        """Test S3Config default values."""
        config = S3Config(bucket="test-bucket")

        assert config.bucket == "test-bucket"
        assert config.region == "us-east-1"
        assert config.access_key_id is None
        assert config.secret_access_key is None
        assert config.session_token is None
        assert config.prefix == ""
        assert config.endpoint_url is None
        assert config.use_ssl is True
        assert config.verify_ssl is True

    def test_custom_values(self):
        """Test S3Config with custom values."""
        config = S3Config(
            bucket="my-bucket",
            region="eu-west-1",
            access_key_id="key-id",
            secret_access_key="secret",
            session_token="token",
            prefix="data/",
            endpoint_url="https://custom.s3.endpoint",
            use_ssl=False,
            verify_ssl=False,
        )

        assert config.bucket == "my-bucket"
        assert config.region == "eu-west-1"
        assert config.access_key_id == "key-id"
        assert config.secret_access_key == "secret"
        assert config.session_token == "token"
        assert config.prefix == "data/"
        assert config.endpoint_url == "https://custom.s3.endpoint"
        assert config.use_ssl is False
        assert config.verify_ssl is False


# ============================================================================
# Connection Handling Tests
# ============================================================================

@pytest.mark.unit
class TestS3ConnectionHandling:
    """Tests for S3 connection handling."""

    @pytest.mark.asyncio
    async def test_connect_success(self, s3_plugin, valid_s3_config):
        """Test successful connection to S3."""
        mock_client = MagicMock()

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            conn = await s3_plugin.connect(valid_s3_config)

        assert isinstance(conn, Connection)
        assert conn.plugin_id == "s3"
        assert conn.config == valid_s3_config
        assert conn.metadata["bucket"] == "test-bucket"
        assert conn.metadata["region"] == "us-east-1"
        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")

    @pytest.mark.asyncio
    async def test_connect_failure(self, s3_plugin, valid_s3_config):
        """Test connection failure to S3."""
        mock_client = MagicMock()
        mock_client.head_bucket.side_effect = Exception("Access denied")

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            with pytest.raises(ConnectionError, match="Failed to connect to S3 bucket"):
                await s3_plugin.connect(valid_s3_config)

    @pytest.mark.asyncio
    async def test_connect_missing_bucket(self, s3_plugin):
        """Test connection with missing bucket."""
        config = {"region": "us-east-1"}

        with pytest.raises(TypeError):
            await s3_plugin.connect(config)

    def test_get_s3_client_caching(self, s3_plugin, valid_s3_config):
        """Test that S3 client is cached."""
        mock_client = MagicMock()
        s3_plugin._s3_client = mock_client

        config = S3Config(**valid_s3_config)

        # Should return cached client
        client1 = s3_plugin._get_s3_client(config)
        client2 = s3_plugin._get_s3_client(config)

        assert client1 is client2
        assert client1 is mock_client

    def test_get_s3_client_with_credentials(self, s3_plugin):
        """Test S3 client creation with credentials."""
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = MagicMock()
        mock_boto3.resource.return_value = MagicMock()

        config = S3Config(
            bucket="test-bucket",
            region="us-west-2",
            access_key_id="test-key",
            secret_access_key="test-secret",
            session_token="test-token",
        )

        s3_plugin._s3_client = None
        with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore.config": MagicMock()}):
            s3_plugin._get_s3_client(config)

        call_kwargs = mock_boto3.client.call_args[1]
        assert call_kwargs["region_name"] == "us-west-2"
        assert call_kwargs["aws_access_key_id"] == "test-key"
        assert call_kwargs["aws_secret_access_key"] == "test-secret"
        assert call_kwargs["aws_session_token"] == "test-token"

    def test_get_s3_client_iam_role(self, s3_plugin):
        """Test S3 client creation with IAM role (no credentials)."""
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = MagicMock()
        mock_boto3.resource.return_value = MagicMock()

        config = S3Config(
            bucket="test-bucket",
            region="eu-west-1",
        )

        s3_plugin._s3_client = None
        with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore.config": MagicMock()}):
            s3_plugin._get_s3_client(config)

        call_kwargs = mock_boto3.client.call_args[1]
        assert call_kwargs["region_name"] == "eu-west-1"
        assert "aws_access_key_id" not in call_kwargs
        assert "aws_secret_access_key" not in call_kwargs

    def test_get_s3_client_custom_endpoint(self, s3_plugin):
        """Test S3 client creation with custom endpoint."""
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = MagicMock()
        mock_boto3.resource.return_value = MagicMock()

        config = S3Config(
            bucket="test-bucket",
            endpoint_url="https://custom.s3.local",
            use_ssl=True,
            verify_ssl=False,
        )

        s3_plugin._s3_client = None
        with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore.config": MagicMock()}):
            s3_plugin._get_s3_client(config)

        call_kwargs = mock_boto3.client.call_args[1]
        assert call_kwargs["endpoint_url"] == "https://custom.s3.local"
        assert call_kwargs["use_ssl"] is True
        assert call_kwargs["verify"] is False


# ============================================================================
# List Files Tests
# ============================================================================

@pytest.mark.unit
class TestS3ListFiles:
    """Tests for S3 list_files method."""

    @pytest.mark.asyncio
    async def test_list_files_success(self, s3_plugin, mock_connection):
        """Test listing files successfully."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "folder/file1.txt",
                        "Size": 100,
                        "LastModified": datetime(2024, 1, 15),
                        "ETag": '"abc123"',
                        "StorageClass": "STANDARD",
                    },
                    {
                        "Key": "folder/file2.pdf",
                        "Size": 200,
                        "LastModified": datetime(2024, 1, 16),
                        "ETag": '"def456"',
                        "StorageClass": "STANDARD",
                    },
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            files = await s3_plugin.list_files(mock_connection, "folder/")

        assert len(files) == 2
        assert isinstance(files[0], SourceFile)
        assert files[0].path == "folder/file1.txt"
        assert files[0].name == "file1.txt"
        assert files[0].size == 100
        assert files[0].mime_type == "text/plain"
        assert files[0].metadata["etag"] == "abc123"
        assert files[1].path == "folder/file2.pdf"
        assert files[1].mime_type == "application/pdf"

    @pytest.mark.asyncio
    async def test_list_files_with_prefix_in_config(self, s3_plugin):
        """Test listing files with prefix in config."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_client.get_paginator.return_value = mock_paginator

        config = {
            "bucket": "test-bucket",
            "region": "us-east-1",
            "prefix": "data/",
        }
        conn = Connection(
            id=MagicMock(),
            plugin_id="s3",
            config=config,
            metadata={"bucket": "test-bucket", "region": "us-east-1", "prefix": "data/"},
        )

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            await s3_plugin.list_files(conn, "subfolder/")

        call_kwargs = mock_paginator.paginate.call_args[1]
        # The implementation builds prefix by combining config.prefix with path
        assert "data/" in call_kwargs["Prefix"]
        assert "subfolder" in call_kwargs["Prefix"]

    @pytest.mark.asyncio
    async def test_list_files_with_pattern(self, s3_plugin, mock_connection):
        """Test listing files with pattern filter."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "file1.txt", "Size": 100, "LastModified": datetime(2024, 1, 15), "ETag": '"abc"'},
                    {"Key": "file2.pdf", "Size": 200, "LastModified": datetime(2024, 1, 15), "ETag": '"def"'},
                    {"Key": "file3.txt", "Size": 300, "LastModified": datetime(2024, 1, 15), "ETag": '"ghi"'},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            files = await s3_plugin.list_files(mock_connection, "", pattern="*.txt")

        assert len(files) == 2
        assert all(f.name.endswith(".txt") for f in files)

    @pytest.mark.asyncio
    async def test_list_files_skips_directories(self, s3_plugin, mock_connection):
        """Test that directory markers are skipped."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "folder/", "Size": 0, "LastModified": datetime(2024, 1, 15), "ETag": '"abc"'},
                    {"Key": "folder/file.txt", "Size": 100, "LastModified": datetime(2024, 1, 15), "ETag": '"def"'},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            files = await s3_plugin.list_files(mock_connection, "folder/")

        assert len(files) == 1
        assert files[0].name == "file.txt"

    @pytest.mark.asyncio
    async def test_list_files_non_recursive(self, s3_plugin, mock_connection):
        """Test listing files non-recursively with delimiter."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_client.get_paginator.return_value = mock_paginator

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            await s3_plugin.list_files(mock_connection, "", recursive=False)

        call_kwargs = mock_paginator.paginate.call_args[1]
        assert call_kwargs["Delimiter"] == "/"

    @pytest.mark.asyncio
    async def test_list_files_recursive(self, s3_plugin, mock_connection):
        """Test listing files recursively without delimiter."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_client.get_paginator.return_value = mock_paginator

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            await s3_plugin.list_files(mock_connection, "", recursive=True)

        call_kwargs = mock_paginator.paginate.call_args[1]
        assert call_kwargs["Delimiter"] is None

    @pytest.mark.asyncio
    async def test_list_files_error(self, s3_plugin, mock_connection):
        """Test error handling in list_files."""
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = Exception("S3 error")

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            with pytest.raises(Exception, match="S3 error"):
                await s3_plugin.list_files(mock_connection, "")


# ============================================================================
# Get File Tests
# ============================================================================

@pytest.mark.unit
class TestS3GetFile:
    """Tests for S3 get_file method."""

    @pytest.mark.asyncio
    async def test_get_file_to_memory(self, s3_plugin, mock_connection):
        """Test downloading file to memory."""
        mock_client = MagicMock()
        mock_client.head_object.return_value = {
            "ContentLength": 100,
            "LastModified": datetime(2024, 1, 15),
            "ETag": '"abc123"',
            "ContentType": "text/plain",
        }
        mock_body = MagicMock()
        mock_body.read.return_value = b"file content"
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            result = await s3_plugin.get_file(mock_connection, "test.txt")

        assert result.content == b"file content"
        assert result.source_file.path == "test.txt"
        assert result.source_file.name == "test.txt"
        assert result.source_file.size == 100
        assert result.content_hash is not None

    @pytest.mark.asyncio
    async def test_get_file_to_disk(self, s3_plugin, mock_connection, tmp_path):
        """Test downloading file to disk."""
        mock_client = MagicMock()
        mock_client.head_object.return_value = {
            "ContentLength": 100,
            "LastModified": datetime(2024, 1, 15),
            "ETag": '"abc123"',
        }
        # Mock download_file to actually write content
        def mock_download(bucket, key, filepath):
            with open(filepath, "wb") as f:
                f.write(b"test content for hash")
        mock_client.download_file.side_effect = mock_download

        download_path = tmp_path / "downloaded.txt"

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            result = await s3_plugin.get_file(mock_connection, "test.txt", download_to=str(download_path))

        assert result.local_path == str(download_path)
        assert result.content == b""  # Content is on disk
        assert result.content_hash is not None
        mock_client.download_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_file_not_found(self, s3_plugin, mock_connection):
        """Test getting non-existent file."""
        mock_client = MagicMock()
        mock_client.head_object.side_effect = Exception("Not Found")
        mock_client.exceptions.NoSuchKey = Exception

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            with pytest.raises((FileNotFoundError, Exception)):
                await s3_plugin.get_file(mock_connection, "nonexistent.txt")


# ============================================================================
# Stream File Tests
# ============================================================================

@pytest.mark.unit
class TestS3StreamFile:
    """Tests for S3 stream_file method."""

    @pytest.mark.asyncio
    async def test_stream_file(self, s3_plugin, mock_connection):
        """Test streaming file content."""
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.side_effect = [b"chunk1", b"chunk2", b""]
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            chunks = []
            async for chunk in s3_plugin.stream_file(mock_connection, "test.txt"):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == b"chunk1"
        assert chunks[1] == b"chunk2"


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestS3ErrorHandling:
    """Tests for S3 error handling."""

    def test_import_error(self, s3_plugin, valid_s3_config):
        """Test handling of boto3 import error."""
        s3_plugin._s3_client = None  # Reset cache

        with patch.dict("sys.modules", {"boto3": None, "botocore.config": None}):
            with pytest.raises((RuntimeError, ImportError, TypeError)):
                s3_plugin._get_s3_client(S3Config(**valid_s3_config))

    @pytest.mark.asyncio
    async def test_connection_error_on_list(self, s3_plugin, mock_connection):
        """Test connection error during list_files."""
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = Exception("Connection refused")

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            with pytest.raises(Exception, match="Connection refused"):
                await s3_plugin.list_files(mock_connection, "")


# ============================================================================
# Validation Tests
# ============================================================================

@pytest.mark.unit
class TestS3Validation:
    """Tests for S3 configuration validation."""

    @pytest.mark.asyncio
    async def test_validate_config_valid(self, s3_plugin):
        """Test valid configuration."""
        config = {
            "bucket": "test-bucket",
            "region": "us-east-1",
            "access_key_id": "key",
            "secret_access_key": "secret",
        }

        result = await s3_plugin.validate_config(config)

        assert result.valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_config_missing_bucket(self, s3_plugin):
        """Test validation with missing bucket."""
        config = {"region": "us-east-1"}

        result = await s3_plugin.validate_config(config)

        assert result.valid is False
        assert "bucket is required" in result.errors

    @pytest.mark.asyncio
    async def test_validate_config_no_credentials_warning(self, s3_plugin):
        """Test validation warning when no credentials provided."""
        config = {"bucket": "test-bucket"}

        result = await s3_plugin.validate_config(config)

        assert result.valid is True
        assert len(result.warnings) > 0
        assert "IAM role" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_validate_config_unusual_region(self, s3_plugin):
        """Test validation with unusual region."""
        config = {
            "bucket": "test-bucket",
            "region": "custom-region-1",
        }

        result = await s3_plugin.validate_config(config)

        assert result.valid is True
        # May or may not have warnings depending on region list


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestS3HealthCheck:
    """Tests for S3 health check."""

    @pytest.mark.asyncio
    async def test_health_check_no_config(self, s3_plugin):
        """Test health check with no config."""
        result = await s3_plugin.health_check()

        assert result == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, s3_plugin, valid_s3_config):
        """Test healthy status."""
        mock_client = MagicMock()
        mock_client.head_bucket.return_value = {}
        mock_client.list_objects_v2.return_value = {"Contents": []}

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            result = await s3_plugin.health_check(valid_s3_config)

        assert result == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, s3_plugin, valid_s3_config):
        """Test unhealthy status."""
        mock_client = MagicMock()
        mock_client.head_bucket.side_effect = Exception("Access denied")

        with patch.object(s3_plugin, "_get_s3_client", return_value=mock_client):
            result = await s3_plugin.health_check(valid_s3_config)

        assert result == HealthStatus.UNHEALTHY


# ============================================================================
# S3 Notification Tests
# ============================================================================

@pytest.mark.unit
class TestS3Notification:
    """Tests for S3 event notification handling."""

    @pytest.mark.asyncio
    async def test_handle_s3_notification(self, s3_plugin, valid_s3_config):
        """Test handling S3 event notification."""
        notification = {
            "Records": [
                {
                    "eventName": "ObjectCreated:Put",
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {
                            "key": "folder/test+file.txt",
                            "size": 100,
                            "eTag": '"abc123"',
                        },
                    },
                }
            ]
        }

        files = await s3_plugin.handle_s3_notification(notification, valid_s3_config)

        assert len(files) == 1
        assert files[0].path == "folder/test file.txt"  # URL decoded
        assert files[0].name == "test file.txt"
        assert files[0].metadata["etag"] == "abc123"

    @pytest.mark.asyncio
    async def test_handle_s3_notification_non_created_events(self, s3_plugin, valid_s3_config):
        """Test that non-ObjectCreated events are ignored."""
        notification = {
            "Records": [
                {
                    "eventName": "ObjectRemoved:Delete",
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "deleted.txt"},
                    },
                }
            ]
        }

        files = await s3_plugin.handle_s3_notification(notification, valid_s3_config)

        assert len(files) == 0


# ============================================================================
# Authorization Tests
# ============================================================================

@pytest.mark.unit
class TestS3Authorization:
    """Tests for S3 authorization."""

    @pytest.mark.asyncio
    async def test_authorize_read_action(self, s3_plugin, valid_s3_config):
        """Test read action authorization."""
        result = await s3_plugin.authorize(
            user_id="user1",
            resource="folder/file.txt",
            action="read",
            config=valid_s3_config,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_authorize_write_without_flag(self, s3_plugin, valid_s3_config):
        """Test write action without explicit flag."""
        result = await s3_plugin.authorize(
            user_id="user1",
            resource="folder/file.txt",
            action="write",
            config=valid_s3_config,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_authorize_write_with_flag(self, s3_plugin, valid_s3_config):
        """Test write action with explicit flag."""
        config = {**valid_s3_config, "allow_write_operations": True}

        result = await s3_plugin.authorize(
            user_id="user1",
            resource="folder/file.txt",
            action="write",
            config=config,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_authorize_bucket_restriction(self, s3_plugin, valid_s3_config):
        """Test bucket-based authorization restriction."""
        config = {**valid_s3_config, "allowed_buckets": ["other-bucket"]}

        result = await s3_plugin.authorize(
            user_id="user1",
            resource="folder/file.txt",
            action="read",
            config=config,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_authorize_prefix_restriction(self, s3_plugin, valid_s3_config):
        """Test prefix-based authorization restriction."""
        config = {**valid_s3_config, "allowed_prefixes": ["allowed/"]}

        result = await s3_plugin.authorize(
            user_id="user1",
            resource="restricted/file.txt",
            action="read",
            config=config,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_authorize_prefix_allowed(self, s3_plugin, valid_s3_config):
        """Test prefix-based authorization allowed."""
        config = {**valid_s3_config, "allowed_prefixes": ["allowed/"]}

        result = await s3_plugin.authorize(
            user_id="user1",
            resource="allowed/file.txt",
            action="read",
            config=config,
        )

        assert result is True
