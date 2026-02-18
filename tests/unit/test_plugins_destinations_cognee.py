"""Unit tests for Cognee destination plugin."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import httpx
import pytest

from src.plugins.base import Connection, HealthStatus, TransformedData, ValidationResult, WriteResult
from src.plugins.destinations.cognee import CogneeDestination, CogneeMockDestination


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def cognee_destination():
    """Create an initialized Cognee destination."""
    dest = CogneeDestination()
    await dest.initialize({
        "api_url": "https://api.cognee.example.com",
        "api_key": "test-api-key",
        "timeout": 30,
    })
    return dest


@pytest.fixture
def sample_transformed_data():
    """Create sample transformed data for testing."""
    return TransformedData(
        job_id=UUID("12345678-1234-1234-1234-123456789abc"),
        chunks=[
            {"content": "Chunk 1 content", "metadata": {"index": 0}},
            {"content": "Chunk 2 content", "metadata": {"index": 1}},
        ],
        metadata={"title": "Test Document", "author": "Test Author"},
        lineage={"source": "test", "parser": "test_parser"},
        original_format="pdf",
        output_format="json",
    )


@pytest.fixture
def sample_connection():
    """Create a sample connection."""
    return Connection(
        id=UUID(int=hash("test-dataset") % (2**32)),
        plugin_id="cognee",
        config={"dataset_id": "test-dataset", "graph_name": "default"},
    )


# ============================================================================
# CogneeDestination Class Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeDestination:
    """Tests for CogneeDestination class."""

    def test_init(self):
        """Test destination initialization."""
        dest = CogneeDestination()
        assert dest._api_url is None
        assert dest._api_key is None
        assert dest._config == {}
        assert dest._client is None

    def test_metadata(self):
        """Test destination metadata."""
        dest = CogneeDestination()
        metadata = dest.metadata

        assert metadata.id == "cognee"
        assert metadata.name == "Cognee Knowledge Graph"
        assert metadata.version == "1.0.0"
        assert "json" in metadata.supported_formats
        assert "markdown" in metadata.supported_formats
        assert "text" in metadata.supported_formats
        assert metadata.requires_auth is True
        assert "api_url" in metadata.config_schema["properties"]
        assert "api_key" in metadata.config_schema["properties"]

    @pytest.mark.asyncio
    async def test_initialize_with_config(self):
        """Test initialization with config parameters."""
        dest = CogneeDestination()
        await dest.initialize({
            "api_url": "https://api.cognee.example.com",
            "api_key": "test-key",
            "timeout": 45,
        })

        assert dest._api_url == "https://api.cognee.example.com"
        assert dest._api_key == "test-key"
        assert dest._config["timeout"] == 45
        assert dest._client is not None

    @pytest.mark.asyncio
    async def test_initialize_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(os.environ, {
            "COGNEE_API_URL": "https://env.cognee.example.com",
            "COGNEE_API_KEY": "env-api-key",
        }):
            dest = CogneeDestination()
            await dest.initialize({})

            assert dest._api_url == "https://env.cognee.example.com"
            assert dest._api_key == "env-api-key"

    @pytest.mark.asyncio
    async def test_initialize_without_api_url(self):
        """Test initialization without API URL shows warning."""
        dest = CogneeDestination()
        with patch("src.plugins.destinations.cognee.logger") as mock_logger:
            await dest.initialize({})
            mock_logger.warning.assert_called_once()
            assert "not configured" in mock_logger.warning.call_args[0][0].lower()


# ============================================================================
# Connection Handling Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeConnectionHandling:
    """Tests for Cognee connection handling."""

    @pytest.mark.asyncio
    async def test_connect_creates_connection(self, cognee_destination):
        """Test connect creates a connection."""
        with patch.object(cognee_destination, "_ensure_dataset", new_callable=AsyncMock):
            conn = await cognee_destination.connect({"dataset_id": "my-dataset"})

            assert isinstance(conn, Connection)
            assert conn.plugin_id == "cognee"
            assert conn.config["dataset_id"] == "my-dataset"
            assert conn.config["graph_name"] == "default"

    @pytest.mark.asyncio
    async def test_connect_with_graph_name(self, cognee_destination):
        """Test connect with custom graph name."""
        with patch.object(cognee_destination, "_ensure_dataset", new_callable=AsyncMock):
            conn = await cognee_destination.connect({
                "dataset_id": "my-dataset",
                "graph_name": "custom-graph",
            })

            assert conn.config["graph_name"] == "custom-graph"

    @pytest.mark.asyncio
    async def test_connect_ensures_dataset(self, cognee_destination):
        """Test connect ensures dataset exists."""
        with patch.object(cognee_destination, "_ensure_dataset", new_callable=AsyncMock) as mock_ensure:
            await cognee_destination.connect({"dataset_id": "test-dataset"})
            mock_ensure.assert_called_once_with("test-dataset")

    @pytest.mark.asyncio
    async def test_ensure_dataset_exists(self, cognee_destination):
        """Test _ensure_dataset when dataset exists."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        cognee_destination._client = AsyncMock()
        cognee_destination._client.get = AsyncMock(return_value=mock_response)

        await cognee_destination._ensure_dataset("existing-dataset")

        cognee_destination._client.get.assert_called_once_with("/v1/datasets/existing-dataset")
        cognee_destination._client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_dataset_creates_new(self, cognee_destination):
        """Test _ensure_dataset creates dataset when not exists."""
        mock_get_response = MagicMock()
        mock_get_response.status_code = 404

        mock_post_response = MagicMock()
        mock_post_response.status_code = 201

        cognee_destination._client = AsyncMock()
        cognee_destination._client.get = AsyncMock(return_value=mock_get_response)
        cognee_destination._client.post = AsyncMock(return_value=mock_post_response)

        with patch("src.plugins.destinations.cognee.logger") as mock_logger:
            await cognee_destination._ensure_dataset("new-dataset")

            cognee_destination._client.post.assert_called_once()
            call_args = cognee_destination._client.post.call_args
            assert call_args[0][0] == "/v1/datasets"
            assert call_args[1]["json"]["id"] == "new-dataset"
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_dataset_handles_error(self, cognee_destination):
        """Test _ensure_dataset handles errors gracefully."""
        cognee_destination._client = AsyncMock()
        cognee_destination._client.get = AsyncMock(side_effect=Exception("Network error"))

        with patch("src.plugins.destinations.cognee.logger") as mock_logger:
            await cognee_destination._ensure_dataset("test-dataset")
            mock_logger.warning.assert_called_once()


# ============================================================================
# Write Data Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeWriteData:
    """Tests for Cognee write operations."""

    @pytest.mark.asyncio
    async def test_write_success(self, cognee_destination, sample_connection, sample_transformed_data):
        """Test successful write operation."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "doc-123"}

        cognee_destination._client = AsyncMock()
        cognee_destination._client.post = AsyncMock(return_value=mock_response)

        result = await cognee_destination.write(sample_connection, sample_transformed_data)

        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.destination_id == "cognee"
        assert "/datasets/test-dataset/documents/doc-123" in result.destination_uri
        assert result.records_written == 2
        assert result.metadata["document_id"] == "doc-123"
        assert result.metadata["chunks_count"] == 2

    @pytest.mark.asyncio
    async def test_write_not_initialized(self, sample_connection, sample_transformed_data):
        """Test write when client not initialized."""
        dest = CogneeDestination()
        # Don't initialize

        result = await dest.write(sample_connection, sample_transformed_data)

        assert isinstance(result, WriteResult)
        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_write_builds_correct_payload(self, cognee_destination, sample_connection, sample_transformed_data):
        """Test that write builds correct API payload."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "doc-123"}

        cognee_destination._client = AsyncMock()
        cognee_destination._client.post = AsyncMock(return_value=mock_response)

        await cognee_destination.write(sample_connection, sample_transformed_data)

        call_args = cognee_destination._client.post.call_args
        assert call_args[0][0] == "/v1/datasets/test-dataset/documents"

        payload = call_args[1]["json"]
        assert payload["job_id"] == str(sample_transformed_data.job_id)
        assert payload["original_format"] == "pdf"
        assert payload["output_format"] == "json"
        assert len(payload["chunks"]) == 2
        assert payload["chunks"][0]["index"] == 0
        assert payload["chunks"][0]["content"] == "Chunk 1 content"

    @pytest.mark.asyncio
    async def test_write_with_embeddings(self, cognee_destination, sample_connection):
        """Test write with embeddings."""
        data_with_embeddings = TransformedData(
            job_id=UUID("12345678-1234-1234-1234-123456789abc"),
            chunks=[{"content": "Chunk 1"}],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "doc-456"}

        cognee_destination._client = AsyncMock()
        cognee_destination._client.post = AsyncMock(return_value=mock_response)

        await cognee_destination.write(sample_connection, data_with_embeddings)

        payload = cognee_destination._client.post.call_args[1]["json"]
        assert payload["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @pytest.mark.asyncio
    async def test_build_payload_structure(self, cognee_destination):
        """Test _build_payload creates correct structure."""
        data = TransformedData(
            job_id=UUID("12345678-1234-1234-1234-123456789abc"),
            chunks=[{"content": "Test chunk", "metadata": {"key": "value"}}],
            metadata={"title": "Test"},
            lineage={"source": "test"},
            original_format="pdf",
            output_format="json",
        )

        payload = cognee_destination._build_payload(data, {"dataset_id": "test"})

        assert payload["job_id"] == "12345678-1234-1234-1234-123456789abc"
        assert payload["original_format"] == "pdf"
        assert payload["output_format"] == "json"
        assert payload["metadata"] == {"title": "Test"}
        assert payload["lineage"] == {"source": "test"}
        assert len(payload["chunks"]) == 1
        assert payload["chunks"][0]["index"] == 0
        assert payload["chunks"][0]["content"] == "Test chunk"


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeErrorHandling:
    """Tests for Cognee error handling."""

    @pytest.mark.asyncio
    async def test_write_http_error(self, cognee_destination, sample_connection, sample_transformed_data):
        """Test write handles HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        error = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        cognee_destination._client = AsyncMock()
        cognee_destination._client.post = AsyncMock(side_effect=error)

        with patch("src.plugins.destinations.cognee.logger") as mock_logger:
            result = await cognee_destination.write(sample_connection, sample_transformed_data)

            assert isinstance(result, WriteResult)
            assert result.success is False
            assert "500" in result.error
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_network_error(self, cognee_destination, sample_connection, sample_transformed_data):
        """Test write handles network errors."""
        cognee_destination._client = AsyncMock()
        cognee_destination._client.post = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("src.plugins.destinations.cognee.logger") as mock_logger:
            result = await cognee_destination.write(sample_connection, sample_transformed_data)

            assert isinstance(result, WriteResult)
            assert result.success is False
            assert "Connection refused" in result.error
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_timeout_error(self, cognee_destination, sample_connection, sample_transformed_data):
        """Test write handles timeout errors."""
        cognee_destination._client = AsyncMock()
        cognee_destination._client.post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))

        with patch("src.plugins.destinations.cognee.logger") as mock_logger:
            result = await cognee_destination.write(sample_connection, sample_transformed_data)

            assert isinstance(result, WriteResult)
            assert result.success is False
            assert "timed out" in result.error.lower() or "Request timed out" in result.error


# ============================================================================
# Config Validation Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeConfigValidation:
    """Tests for Cognee configuration validation."""

    @pytest.mark.asyncio
    async def test_validate_config_valid(self):
        """Test validation with valid config."""
        dest = CogneeDestination()
        result = await dest.validate_config({
            "api_url": "https://api.cognee.example.com",
            "api_key": "test-key",
        })

        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_config_missing_api_url(self):
        """Test validation with missing API URL."""
        dest = CogneeDestination()
        result = await dest.validate_config({})

        assert result.valid is False
        assert len(result.errors) == 1
        assert "api_url" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_config_invalid_url(self):
        """Test validation with invalid URL."""
        dest = CogneeDestination()
        result = await dest.validate_config({"api_url": "not-a-url"})

        assert result.valid is False
        assert len(result.errors) == 1
        assert "http" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_config_missing_api_key(self):
        """Test validation with missing API key (warning only)."""
        dest = CogneeDestination()
        result = await dest.validate_config({
            "api_url": "https://api.cognee.example.com",
        })

        assert result.valid is True
        assert len(result.warnings) == 1
        assert "api_key" in result.warnings[0].lower()

    @pytest.mark.asyncio
    async def test_validate_config_with_env_vars(self):
        """Test validation with environment variables."""
        with patch.dict(os.environ, {"COGNEE_API_URL": "https://env.example.com"}):
            dest = CogneeDestination()
            result = await dest.validate_config({})

            assert result.valid is True


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeHealthCheck:
    """Tests for Cognee health check."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, cognee_destination):
        """Test health check returns healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        cognee_destination._client = AsyncMock()
        cognee_destination._client.get = AsyncMock(return_value=mock_response)

        health = await cognee_destination.health_check()

        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, cognee_destination):
        """Test health check returns degraded for non-500 errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429  # Too many requests

        cognee_destination._client = AsyncMock()
        cognee_destination._client.get = AsyncMock(return_value=mock_response)

        health = await cognee_destination.health_check()

        assert health == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_500(self, cognee_destination):
        """Test health check returns unhealthy for 500 errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        cognee_destination._client = AsyncMock()
        cognee_destination._client.get = AsyncMock(return_value=mock_response)

        health = await cognee_destination.health_check()

        assert health == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, cognee_destination):
        """Test health check handles timeout."""
        cognee_destination._client = AsyncMock()
        cognee_destination._client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        health = await cognee_destination.health_check()

        assert health == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        dest = CogneeDestination()
        health = await dest.health_check()

        assert health == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_no_api_url(self, cognee_destination):
        """Test health check when no API URL set."""
        cognee_destination._api_url = None

        health = await cognee_destination.health_check()

        assert health == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_exception(self, cognee_destination):
        """Test health check handles exceptions."""
        cognee_destination._client = AsyncMock()
        cognee_destination._client.get = AsyncMock(side_effect=Exception("Unexpected error"))

        with patch("src.plugins.destinations.cognee.logger") as mock_logger:
            health = await cognee_destination.health_check()

            assert health == HealthStatus.UNHEALTHY
            mock_logger.warning.assert_called_once()


# ============================================================================
# Shutdown Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeShutdown:
    """Tests for Cognee shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self, cognee_destination):
        """Test shutdown closes HTTP client."""
        mock_client = AsyncMock()
        cognee_destination._client = mock_client

        with patch("src.plugins.destinations.cognee.logger") as mock_logger:
            await cognee_destination.shutdown()

            mock_client.aclose.assert_called_once()
            assert cognee_destination._client is None
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_no_client(self):
        """Test shutdown when no client exists."""
        dest = CogneeDestination()
        dest._client = None

        # Should not raise
        await dest.shutdown()


# ============================================================================
# CogneeMockDestination Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeMockDestination:
    """Tests for CogneeMockDestination."""

    def test_init(self):
        """Test mock destination initialization."""
        dest = CogneeMockDestination()
        assert dest._storage == {}
        assert dest._config == {}

    def test_metadata(self):
        """Test mock destination metadata."""
        dest = CogneeMockDestination()
        metadata = dest.metadata

        assert metadata.id == "cognee_mock"
        assert metadata.name == "Cognee Mock (Testing)"
        assert metadata.requires_auth is False

    @pytest.mark.asyncio
    async def test_mock_initialize(self):
        """Test mock initialization."""
        dest = CogneeMockDestination()
        await dest.initialize({"test": "config"})

        assert dest._config == {"test": "config"}

    @pytest.mark.asyncio
    async def test_mock_connect(self):
        """Test mock connect."""
        dest = CogneeMockDestination()
        await dest.initialize({})

        conn = await dest.connect({"dataset_id": "test-dataset"})

        assert isinstance(conn, Connection)
        assert conn.plugin_id == "cognee_mock"
        assert conn.config["dataset_id"] == "test-dataset"
        assert "test-dataset" in dest._storage

    @pytest.mark.asyncio
    async def test_mock_write(self):
        """Test mock write."""
        dest = CogneeMockDestination()
        await dest.initialize({})

        conn = await dest.connect({"dataset_id": "test-dataset"})
        data = TransformedData(
            job_id=UUID("12345678-1234-1234-1234-123456789abc"),
            chunks=[{"content": "Chunk 1"}, {"content": "Chunk 2"}],
        )

        result = await dest.write(conn, data)

        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.destination_id == "cognee_mock"
        assert result.records_written == 2

        # Check stored data
        stored = dest.get_stored_data("test-dataset")
        assert len(stored) == 1
        assert stored[0]["job_id"] == str(data.job_id)

    @pytest.mark.asyncio
    async def test_mock_health_check(self):
        """Test mock health check always returns healthy."""
        dest = CogneeMockDestination()

        health = await dest.health_check()

        assert health == HealthStatus.HEALTHY

    def test_mock_get_stored_data_empty(self):
        """Test getting stored data when empty."""
        dest = CogneeMockDestination()

        stored = dest.get_stored_data("nonexistent")

        assert stored == []

    def test_mock_clear_storage(self):
        """Test clearing storage."""
        dest = CogneeMockDestination()
        dest._storage = {"dataset1": [{"data": "test"}]}

        dest.clear_storage()

        assert dest._storage == {}
