"""Unit tests for Azure Blob Storage source plugin."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.plugins.base import Connection, HealthStatus, SourceFile
from src.plugins.sources.azure_blob_source import AzureBlobConfig, AzureBlobSourcePlugin


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def azure_plugin():
    """Create an Azure Blob source plugin instance."""
    return AzureBlobSourcePlugin()


@pytest.fixture
def valid_azure_config():
    """Return a valid Azure Blob configuration."""
    return {
        "account_name": "teststorage",
        "container": "test-container",
        "account_key": "test-key",
    }


@pytest.fixture
def mock_connection(valid_azure_config):
    """Create a mock Connection object."""
    return Connection(
        id=MagicMock(),
        plugin_id="azure_blob",
        config=valid_azure_config,
        metadata={
            "account_name": valid_azure_config["account_name"],
            "container": valid_azure_config["container"],
            "prefix": "",
            "endpoint_suffix": "core.windows.net",
        },
    )


# ============================================================================
# AzureBlobSourcePlugin Class Tests
# ============================================================================

@pytest.mark.unit
class TestAzureBlobSourcePlugin:
    """Tests for AzureBlobSourcePlugin class."""

    def test_init(self):
        """Test plugin initialization."""
        plugin = AzureBlobSourcePlugin()
        assert plugin._blob_service is None
        assert plugin.logger is not None

    def test_metadata(self, azure_plugin):
        """Test plugin metadata."""
        metadata = azure_plugin.metadata

        assert metadata.id == "azure_blob"
        assert metadata.name == "Azure Blob Storage Source"
        assert metadata.version == "1.0.0"
        assert metadata.type.value == "source"
        assert metadata.requires_auth is True
        assert "*" in metadata.supported_formats

    def test_metadata_config_schema(self, azure_plugin):
        """Test plugin metadata config schema."""
        schema = azure_plugin.metadata.config_schema

        assert schema["type"] == "object"
        assert "account_name" in schema["required"]
        assert "container" in schema["required"]
        assert "account_name" in schema["properties"]
        assert "connection_string" in schema["properties"]
        assert "sas_token" in schema["properties"]


# ============================================================================
# AzureBlobConfig Tests
# ============================================================================

@pytest.mark.unit
class TestAzureBlobConfig:
    """Tests for AzureBlobConfig dataclass."""

    def test_default_values(self):
        """Test AzureBlobConfig default values."""
        config = AzureBlobConfig(
            account_name="teststorage",
            container="test-container",
        )

        assert config.account_name == "teststorage"
        assert config.container == "test-container"
        assert config.connection_string is None
        assert config.account_key is None
        assert config.sas_token is None
        assert config.tenant_id is None
        assert config.client_id is None
        assert config.prefix == ""
        assert config.endpoint_suffix == "core.windows.net"

    def test_custom_values(self):
        """Test AzureBlobConfig with custom values."""
        config = AzureBlobConfig(
            account_name="mystorage",
            container="mycontainer",
            connection_string="DefaultEndpointsProtocol=https;...",
            account_key="mykey",
            sas_token="?sv=2020-08-04&...",
            tenant_id="tenant-id",
            client_id="client-id",
            prefix="data/",
            endpoint_suffix="core.chinacloudapi.cn",
        )

        assert config.account_name == "mystorage"
        assert config.container == "mycontainer"
        assert config.connection_string == "DefaultEndpointsProtocol=https;..."
        assert config.account_key == "mykey"
        assert config.sas_token == "?sv=2020-08-04&..."
        assert config.tenant_id == "tenant-id"
        assert config.client_id == "client-id"
        assert config.prefix == "data/"
        assert config.endpoint_suffix == "core.chinacloudapi.cn"


# ============================================================================
# Connection Handling Tests
# ============================================================================

@pytest.mark.unit
class TestAzureConnectionHandling:
    """Tests for Azure Blob connection handling."""

    @pytest.mark.asyncio
    async def test_connect_success(self, azure_plugin, valid_azure_config):
        """Test successful connection to Azure Blob."""
        mock_container_client = MagicMock()
        mock_container_client.get_container_properties.return_value = {}

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            conn = await azure_plugin.connect(valid_azure_config)

        assert isinstance(conn, Connection)
        assert conn.plugin_id == "azure_blob"
        assert conn.config == valid_azure_config
        assert conn.metadata["account_name"] == "teststorage"
        assert conn.metadata["container"] == "test-container"

    @pytest.mark.asyncio
    async def test_connect_failure(self, azure_plugin, valid_azure_config):
        """Test connection failure to Azure Blob."""
        mock_container_client = MagicMock()
        mock_container_client.get_container_properties.side_effect = Exception("Access denied")

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            with pytest.raises(ConnectionError, match="Failed to connect to Azure Blob container"):
                await azure_plugin.connect(valid_azure_config)

    @pytest.mark.asyncio
    async def test_connect_missing_required(self, azure_plugin):
        """Test connection with missing required fields."""
        config = {"account_name": "teststorage"}  # Missing container

        with pytest.raises(TypeError):
            await azure_plugin.connect(config)

    def test_get_blob_service_caching(self, azure_plugin, valid_azure_config):
        """Test that blob service client is cached."""
        mock_service = MagicMock()
        azure_plugin._blob_service = mock_service

        config = AzureBlobConfig(**valid_azure_config)

        # Should return cached client
        service1 = azure_plugin._get_blob_service(config)
        service2 = azure_plugin._get_blob_service(config)

        assert service1 is service2
        assert service1 is mock_service

    def test_get_container_client(self, azure_plugin):
        """Test getting container client."""
        mock_service = MagicMock()
        mock_container_client = MagicMock()
        mock_service.get_container_client.return_value = mock_container_client
        azure_plugin._blob_service = mock_service

        config = AzureBlobConfig(
            account_name="teststorage",
            container="test-container",
            account_key="key",
        )

        result = azure_plugin._get_container_client(config)

        assert result is mock_container_client
        mock_service.get_container_client.assert_called_once_with("test-container")


# ============================================================================
# List Blobs Tests
# ============================================================================

@pytest.mark.unit
class TestAzureListBlobs:
    """Tests for Azure list_files (blobs) method."""

    @pytest.mark.asyncio
    async def test_list_files_success(self, azure_plugin, mock_connection):
        """Test listing blobs successfully."""
        mock_container_client = MagicMock()

        # Create mock blob objects
        mock_blob1 = MagicMock()
        mock_blob1.name = "folder/file1.txt"
        mock_blob1.size = 100
        mock_blob1.last_modified = datetime(2024, 1, 15)
        mock_blob1.etag = '"abc123"'
        mock_blob1.content_settings = MagicMock()
        mock_blob1.content_settings.content_type = "text/plain"

        mock_blob2 = MagicMock()
        mock_blob2.name = "folder/file2.pdf"
        mock_blob2.size = 200
        mock_blob2.last_modified = datetime(2024, 1, 16)
        mock_blob2.etag = '"def456"'
        mock_blob2.content_settings = MagicMock()
        mock_blob2.content_settings.content_type = "application/pdf"

        mock_container_client.list_blobs.return_value = [mock_blob1, mock_blob2]

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            files = await azure_plugin.list_files(mock_connection, "folder/", recursive=True)

        assert len(files) == 2
        assert isinstance(files[0], SourceFile)
        assert files[0].path == "folder/file1.txt"
        assert files[0].name == "file1.txt"
        assert files[0].size == 100
        assert files[0].mime_type == "text/plain"
        assert files[0].metadata["etag"] == "abc123"

    @pytest.mark.asyncio
    async def test_list_files_with_prefix(self, azure_plugin):
        """Test listing files with prefix in config."""
        mock_container_client = MagicMock()
        mock_container_client.list_blobs.return_value = []

        config = {
            "account_name": "teststorage",
            "container": "test-container",
            "account_key": "key",
            "prefix": "data/",
        }
        conn = Connection(
            id=MagicMock(),
            plugin_id="azure_blob",
            config=config,
            metadata={"account_name": "teststorage", "container": "test-container", "prefix": "data/"},
        )

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            await azure_plugin.list_files(conn, "subfolder/", recursive=True)

        call_kwargs = mock_container_client.list_blobs.call_args[1]
        # The implementation builds prefix by combining config.prefix with path
        assert "data/" in call_kwargs["name_starts_with"]
        assert "subfolder" in call_kwargs["name_starts_with"]

    @pytest.mark.asyncio
    async def test_list_files_with_pattern(self, azure_plugin, mock_connection):
        """Test listing files with pattern filter."""
        mock_container_client = MagicMock()

        mock_blob1 = MagicMock()
        mock_blob1.name = "file1.txt"
        mock_blob1.size = 100
        mock_blob1.last_modified = datetime(2024, 1, 15)
        mock_blob1.etag = '"abc"'
        mock_blob1.content_settings = None

        mock_blob2 = MagicMock()
        mock_blob2.name = "file2.pdf"
        mock_blob2.size = 200
        mock_blob2.last_modified = datetime(2024, 1, 15)
        mock_blob2.etag = '"def"'
        mock_blob2.content_settings = None

        mock_container_client.list_blobs.return_value = [mock_blob1, mock_blob2]

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            files = await azure_plugin.list_files(mock_connection, "", recursive=True, pattern="*.txt")

        assert len(files) == 1
        assert files[0].name == "file1.txt"

    @pytest.mark.asyncio
    async def test_list_files_skips_directories(self, azure_plugin, mock_connection):
        """Test that directory markers are skipped."""
        mock_container_client = MagicMock()

        mock_blob1 = MagicMock()
        mock_blob1.name = "folder/"
        mock_blob1.size = 0
        mock_blob1.last_modified = datetime(2024, 1, 15)
        mock_blob1.etag = '"abc"'
        mock_blob1.content_settings = None

        mock_blob2 = MagicMock()
        mock_blob2.name = "folder/file.txt"
        mock_blob2.size = 100
        mock_blob2.last_modified = datetime(2024, 1, 15)
        mock_blob2.etag = '"def"'
        mock_blob2.content_settings = None

        mock_container_client.list_blobs.return_value = [mock_blob1, mock_blob2]

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            files = await azure_plugin.list_files(mock_connection, "", recursive=True)

        assert len(files) == 1
        assert files[0].name == "file.txt"

    @pytest.mark.asyncio
    async def test_list_files_non_recursive(self, azure_plugin, mock_connection):
        """Test listing files non-recursively."""
        mock_container_client = MagicMock()
        mock_container_client.walk_blobs.return_value = []

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            await azure_plugin.list_files(mock_connection, "", recursive=False)

        mock_container_client.walk_blobs.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_files_error(self, azure_plugin, mock_connection):
        """Test error handling in list_files."""
        # Note: list_blobs returns an iterator, so exception is raised during iteration
        # Testing the exception handling would require iterating the mock
        mock_container_client = MagicMock()
        # Create an iterator that raises on first iteration
        def failing_iterator(*args, **kwargs):
            raise Exception("Azure error")
        mock_container_client.list_blobs.side_effect = failing_iterator

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            with pytest.raises(Exception, match="Azure error"):
                await azure_plugin.list_files(mock_connection, "", recursive=True)


# ============================================================================
# Download Blob Tests
# ============================================================================

@pytest.mark.unit
class TestAzureGetFile:
    """Tests for Azure get_file (download blob) method."""

    @pytest.mark.asyncio
    async def test_get_file_to_memory(self, azure_plugin, mock_connection):
        """Test downloading blob to memory."""
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()

        mock_properties = MagicMock()
        mock_properties.size = 100
        mock_properties.last_modified = datetime(2024, 1, 15)
        mock_properties.etag = '"abc123"'
        mock_properties.version_id = "v1"
        mock_properties.tags = {}
        mock_properties.content_settings = MagicMock()
        mock_properties.content_settings.content_type = "text/plain"

        mock_blob_client.get_blob_properties.return_value = mock_properties

        mock_download_stream = MagicMock()
        mock_download_stream.readall.return_value = b"blob content"
        mock_blob_client.download_blob.return_value = mock_download_stream

        mock_container_client.get_blob_client.return_value = mock_blob_client

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            result = await azure_plugin.get_file(mock_connection, "test.txt")

        assert result.content == b"blob content"
        assert result.source_file.path == "test.txt"
        assert result.source_file.name == "test.txt"
        assert result.source_file.size == 100
        assert result.content_hash is not None

    @pytest.mark.asyncio
    async def test_get_file_to_disk(self, azure_plugin, mock_connection, tmp_path):
        """Test downloading blob to disk."""
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()

        mock_properties = MagicMock()
        mock_properties.size = 100
        mock_properties.last_modified = datetime(2024, 1, 15)
        mock_properties.etag = '"abc123"'
        mock_properties.version_id = None
        mock_properties.tags = None
        mock_properties.content_settings = MagicMock()
        mock_properties.content_settings.content_type = None

        mock_blob_client.get_blob_properties.return_value = mock_properties

        mock_download_stream = MagicMock()
        mock_download_stream.readall.return_value = b"blob content"
        mock_blob_client.download_blob.return_value = mock_download_stream

        mock_container_client.get_blob_client.return_value = mock_blob_client

        download_path = tmp_path / "downloaded.txt"

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            result = await azure_plugin.get_file(mock_connection, "test.txt", download_to=str(download_path))

        assert result.local_path == str(download_path)
        assert result.content == b""  # Content is on disk
        assert result.content_hash is not None
        assert download_path.read_bytes() == b"blob content"

    @pytest.mark.asyncio
    async def test_get_file_not_found(self, azure_plugin, mock_connection):
        """Test getting non-existent blob."""
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()
        mock_blob_client.get_blob_properties.side_effect = Exception("BlobNotFound")
        mock_container_client.get_blob_client.return_value = mock_blob_client

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            with pytest.raises(FileNotFoundError, match="Azure Blob not found"):
                await azure_plugin.get_file(mock_connection, "nonexistent.txt")


# ============================================================================
# Stream File Tests
# ============================================================================

@pytest.mark.unit
class TestAzureStreamFile:
    """Tests for Azure stream_file method."""

    @pytest.mark.asyncio
    async def test_stream_file(self, azure_plugin, mock_connection):
        """Test streaming blob content."""
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()

        mock_download_stream = MagicMock()
        mock_download_stream.chunks.return_value = [b"chunk1", b"chunk2"]
        mock_blob_client.download_blob.return_value = mock_download_stream

        mock_container_client.get_blob_client.return_value = mock_blob_client

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            chunks = []
            async for chunk in azure_plugin.stream_file(mock_connection, "test.txt"):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == b"chunk1"
        assert chunks[1] == b"chunk2"


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestAzureErrorHandling:
    """Tests for Azure Blob error handling."""

    @pytest.mark.asyncio
    async def test_connection_error_on_list(self, azure_plugin, mock_connection):
        """Test connection error during list_files."""
        mock_container_client = MagicMock()
        # Create a function that raises immediately
        def failing_list(*args, **kwargs):
            raise Exception("Connection refused")
        mock_container_client.list_blobs.side_effect = failing_list

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            with pytest.raises(Exception, match="Connection refused"):
                await azure_plugin.list_files(mock_connection, "", recursive=True)


# ============================================================================
# Validation Tests
# ============================================================================

@pytest.mark.unit
class TestAzureValidation:
    """Tests for Azure Blob configuration validation."""

    @pytest.mark.asyncio
    async def test_validate_config_valid(self, azure_plugin):
        """Test valid configuration."""
        config = {
            "account_name": "teststorage",
            "container": "test-container",
            "account_key": "key",
        }

        result = await azure_plugin.validate_config(config)

        assert result.valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_config_missing_account_name(self, azure_plugin):
        """Test validation with missing account_name."""
        config = {"container": "test-container"}

        result = await azure_plugin.validate_config(config)

        assert result.valid is False
        assert "account_name is required" in result.errors

    @pytest.mark.asyncio
    async def test_validate_config_missing_container(self, azure_plugin):
        """Test validation with missing container."""
        config = {"account_name": "teststorage"}

        result = await azure_plugin.validate_config(config)

        assert result.valid is False
        assert "container is required" in result.errors

    @pytest.mark.asyncio
    async def test_validate_config_no_credentials_warning(self, azure_plugin):
        """Test validation warning when no explicit credentials."""
        config = {
            "account_name": "teststorage",
            "container": "test-container",
        }

        result = await azure_plugin.validate_config(config)

        assert result.valid is True
        assert len(result.warnings) > 0
        assert "DefaultAzureCredential" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_validate_config_container_name_too_short(self, azure_plugin):
        """Test validation with container name too short."""
        config = {
            "account_name": "teststorage",
            "container": "ab",  # Less than 3 chars
        }

        result = await azure_plugin.validate_config(config)

        assert result.valid is False
        assert "3-63 characters" in result.errors[0]

    @pytest.mark.asyncio
    async def test_validate_config_container_name_too_long(self, azure_plugin):
        """Test validation with container name too long."""
        config = {
            "account_name": "teststorage",
            "container": "a" * 64,  # More than 63 chars
        }

        result = await azure_plugin.validate_config(config)

        assert result.valid is False
        assert "3-63 characters" in result.errors[0]


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestAzureHealthCheck:
    """Tests for Azure Blob health check."""

    @pytest.mark.asyncio
    async def test_health_check_no_config(self, azure_plugin):
        """Test health check with no config."""
        result = await azure_plugin.health_check()

        assert result == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, azure_plugin, valid_azure_config):
        """Test healthy status."""
        mock_container_client = MagicMock()
        mock_container_client.get_container_properties.return_value = {}
        mock_container_client.list_blobs.return_value = iter([])

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            result = await azure_plugin.health_check(valid_azure_config)

        assert result == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, azure_plugin, valid_azure_config):
        """Test unhealthy status."""
        mock_container_client = MagicMock()
        mock_container_client.get_container_properties.side_effect = Exception("Access denied")

        with patch.object(azure_plugin, "_get_container_client", return_value=mock_container_client):
            result = await azure_plugin.health_check(valid_azure_config)

        assert result == HealthStatus.UNHEALTHY


# ============================================================================
# Event Grid Notification Tests
# ============================================================================

@pytest.mark.unit
class TestAzureEventGridNotification:
    """Tests for Event Grid event handling."""

    @pytest.mark.asyncio
    async def test_handle_event_grid_blob_created(self, azure_plugin, valid_azure_config):
        """Test handling Event Grid blob created event."""
        event = {
            "eventType": "Microsoft.Storage.BlobCreated",
            "data": {
                "url": "https://teststorage.blob.core.windows.net/test-container/folder/test%20file.txt",
            },
        }

        files = await azure_plugin.handle_event_grid_notification(event, valid_azure_config)

        assert len(files) == 1
        # URL is parsed and path extracted - the blob path starts after the container
        assert "test file" in files[0].name
        assert files[0].metadata["event_type"] == "Microsoft.Storage.BlobCreated"

    @pytest.mark.asyncio
    async def test_handle_event_grid_non_blob_created(self, azure_plugin, valid_azure_config):
        """Test that non-BlobCreated events are ignored."""
        event = {
            "eventType": "Microsoft.Storage.BlobDeleted",
            "data": {
                "url": "https://teststorage.blob.core.windows.net/test-container/deleted.txt",
            },
        }

        files = await azure_plugin.handle_event_grid_notification(event, valid_azure_config)

        assert len(files) == 0
