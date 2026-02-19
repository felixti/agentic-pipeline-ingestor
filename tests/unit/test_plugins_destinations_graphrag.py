"""Unit tests for GraphRAG destination plugin."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import httpx
import pytest

from src.plugins.base import (
    Connection,
    HealthStatus,
    TransformedData,
    ValidationResult,
    WriteResult,
)
from src.plugins.destinations.graphrag import GraphRAGDestination, GraphRAGMockDestination

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def graphrag_destination():
    """Create an initialized GraphRAG destination."""
    dest = GraphRAGDestination()
    await dest.initialize({
        "api_url": "https://api.graphrag.example.com",
        "api_key": "test-api-key",
        "timeout": 120,
        "auto_extract_entities": True,
        "community_detection": True,
    })
    return dest


@pytest.fixture
def sample_transformed_data():
    """Create sample transformed data for testing."""
    return TransformedData(
        job_id=UUID("12345678-1234-1234-1234-123456789abc"),
        chunks=[
            {"content": "Entity A is related to Entity B", "metadata": {"index": 0}},
            {"content": "Entity B connects to Entity C", "metadata": {"index": 1}},
            {"content": "Entity C has properties", "metadata": {"index": 2}},
        ],
        metadata={"title": "Test Graph Document", "author": "Test Author"},
        lineage={"source": "test", "parser": "test_parser"},
        original_format="pdf",
        output_format="json",
    )


@pytest.fixture
def sample_connection():
    """Create a sample connection."""
    return Connection(
        id=UUID(int=hash("test-graph") % (2**32)),
        plugin_id="graphrag",
        config={
            "graph_id": "test-graph",
            "index_name": "default",
            "auto_extract_entities": True,
            "community_detection": True,
        },
    )


# ============================================================================
# GraphRAGDestination Class Tests
# ============================================================================

@pytest.mark.unit
class TestGraphRAGDestination:
    """Tests for GraphRAGDestination class."""

    def test_init(self):
        """Test destination initialization."""
        dest = GraphRAGDestination()
        assert dest._api_url is None
        assert dest._api_key is None
        assert dest._config == {}
        assert dest._client is None

    def test_metadata(self):
        """Test destination metadata."""
        dest = GraphRAGDestination()
        metadata = dest.metadata

        assert metadata.id == "graphrag"
        assert metadata.name == "Microsoft GraphRAG"
        assert metadata.version == "1.0.0"
        assert "json" in metadata.supported_formats
        assert "text" in metadata.supported_formats
        assert "markdown" in metadata.supported_formats
        assert metadata.requires_auth is True
        assert "api_url" in metadata.config_schema["properties"]
        assert "auto_extract_entities" in metadata.config_schema["properties"]
        assert "community_detection" in metadata.config_schema["properties"]

    @pytest.mark.asyncio
    async def test_initialize_with_config(self):
        """Test initialization with config parameters."""
        dest = GraphRAGDestination()
        await dest.initialize({
            "api_url": "https://api.graphrag.example.com",
            "api_key": "test-key",
            "timeout": 60,
            "auto_extract_entities": False,
            "community_detection": False,
        })

        assert dest._api_url == "https://api.graphrag.example.com"
        assert dest._api_key == "test-key"
        assert dest._config["timeout"] == 60
        assert dest._config["auto_extract_entities"] is False
        assert dest._config["community_detection"] is False
        assert dest._client is not None

    @pytest.mark.asyncio
    async def test_initialize_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(os.environ, {
            "GRAPHRAG_API_URL": "https://env.graphrag.example.com",
            "GRAPHRAG_API_KEY": "env-api-key",
        }):
            dest = GraphRAGDestination()
            await dest.initialize({})

            assert dest._api_url == "https://env.graphrag.example.com"
            assert dest._api_key == "env-api-key"

    @pytest.mark.asyncio
    async def test_initialize_without_api_url(self):
        """Test initialization without API URL shows warning."""
        dest = GraphRAGDestination()
        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            await dest.initialize({})
            mock_logger.warning.assert_called_once()
            assert "not configured" in mock_logger.warning.call_args[0][0].lower()


# ============================================================================
# Connection Handling Tests
# ============================================================================

@pytest.mark.unit
class TestGraphRAGConnectionHandling:
    """Tests for GraphRAG connection handling."""

    @pytest.mark.asyncio
    async def test_connect_creates_connection(self, graphrag_destination):
        """Test connect creates a connection."""
        with patch.object(graphrag_destination, "_ensure_graph", new_callable=AsyncMock):
            conn = await graphrag_destination.connect({"graph_id": "my-graph"})

            assert isinstance(conn, Connection)
            assert conn.plugin_id == "graphrag"
            assert conn.config["graph_id"] == "my-graph"
            assert conn.config["index_name"] == "default"
            assert conn.config["auto_extract_entities"] is True
            assert conn.config["community_detection"] is True

    @pytest.mark.asyncio
    async def test_connect_with_options(self, graphrag_destination):
        """Test connect with custom options."""
        with patch.object(graphrag_destination, "_ensure_graph", new_callable=AsyncMock):
            conn = await graphrag_destination.connect({
                "graph_id": "my-graph",
                "index_name": "custom-index",
                "auto_extract_entities": False,
                "community_detection": False,
            })

            assert conn.config["index_name"] == "custom-index"
            assert conn.config["auto_extract_entities"] is False
            assert conn.config["community_detection"] is False

    @pytest.mark.asyncio
    async def test_connect_uses_default_config(self, graphrag_destination):
        """Test connect uses defaults from initialize config."""
        graphrag_destination._config = {
            "auto_extract_entities": True,
            "community_detection": False,
        }

        with patch.object(graphrag_destination, "_ensure_graph", new_callable=AsyncMock):
            conn = await graphrag_destination.connect({"graph_id": "my-graph"})

            # Should use default from _config since not overridden
            assert conn.config["auto_extract_entities"] is True

    @pytest.mark.asyncio
    async def test_ensure_graph_exists(self, graphrag_destination):
        """Test _ensure_graph when graph exists."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(return_value=mock_response)

        await graphrag_destination._ensure_graph("existing-graph")

        graphrag_destination._client.get.assert_called_once_with("/v1/graphs/existing-graph")
        graphrag_destination._client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_graph_creates_new(self, graphrag_destination):
        """Test _ensure_graph creates graph when not exists."""
        mock_get_response = MagicMock()
        mock_get_response.status_code = 404

        mock_post_response = MagicMock()
        mock_post_response.status_code = 201

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(return_value=mock_get_response)
        graphrag_destination._client.post = AsyncMock(return_value=mock_post_response)

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            await graphrag_destination._ensure_graph("new-graph")

            graphrag_destination._client.post.assert_called_once()
            call_args = graphrag_destination._client.post.call_args
            assert call_args[0][0] == "/v1/graphs"
            assert call_args[1]["json"]["id"] == "new-graph"
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_graph_handles_error(self, graphrag_destination):
        """Test _ensure_graph handles errors gracefully."""
        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(side_effect=Exception("Network error"))

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            await graphrag_destination._ensure_graph("test-graph")
            mock_logger.warning.assert_called_once()


# ============================================================================
# Write Operations Tests
# ============================================================================

@pytest.mark.unit
class TestGraphRAGWriteOperations:
    """Tests for GraphRAG write operations."""

    @pytest.mark.asyncio
    async def test_write_success(self, graphrag_destination, sample_connection, sample_transformed_data):
        """Test successful write operation."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "document_id": "doc-123",
            "entities_count": 5,
            "relationships_count": 10,
        }

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.post = AsyncMock(return_value=mock_response)

        result = await graphrag_destination.write(sample_connection, sample_transformed_data)

        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.destination_id == "graphrag"
        assert "/graphs/test-graph/documents/doc-123" in result.destination_uri
        assert result.records_written == 3
        assert result.metadata["document_id"] == "doc-123"
        assert result.metadata["entities_extracted"] == 5
        assert result.metadata["relationships_created"] == 10

    @pytest.mark.asyncio
    async def test_write_not_initialized(self, sample_connection, sample_transformed_data):
        """Test write when client not initialized."""
        dest = GraphRAGDestination()
        # Don't initialize

        result = await dest.write(sample_connection, sample_transformed_data)

        assert isinstance(result, WriteResult)
        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_write_triggers_entity_extraction(self, graphrag_destination, sample_connection, sample_transformed_data):
        """Test write triggers entity extraction when enabled."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"document_id": "doc-123"}

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.post = AsyncMock(return_value=mock_response)

        with patch.object(graphrag_destination, "_trigger_entity_extraction", new_callable=AsyncMock) as mock_extract:
            with patch.object(graphrag_destination, "_trigger_community_detection", new_callable=AsyncMock):
                await graphrag_destination.write(sample_connection, sample_transformed_data)

                mock_extract.assert_called_once_with("test-graph", "doc-123")

    @pytest.mark.asyncio
    async def test_write_triggers_community_detection(self, graphrag_destination, sample_connection, sample_transformed_data):
        """Test write triggers community detection when enabled."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"document_id": "doc-123"}

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.post = AsyncMock(return_value=mock_response)

        with patch.object(graphrag_destination, "_trigger_entity_extraction", new_callable=AsyncMock):
            with patch.object(graphrag_destination, "_trigger_community_detection", new_callable=AsyncMock) as mock_community:
                await graphrag_destination.write(sample_connection, sample_transformed_data)

                mock_community.assert_called_once_with("test-graph")

    @pytest.mark.asyncio
    async def test_write_skips_triggers_when_disabled(self, graphrag_destination, sample_transformed_data):
        """Test write skips triggers when disabled in config."""
        conn = Connection(
            id=UUID(int=hash("test-graph") % (2**32)),
            plugin_id="graphrag",
            config={
                "graph_id": "test-graph",
                "auto_extract_entities": False,
                "community_detection": False,
            },
        )

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"document_id": "doc-123"}

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.post = AsyncMock(return_value=mock_response)

        with patch.object(graphrag_destination, "_trigger_entity_extraction", new_callable=AsyncMock) as mock_extract:
            with patch.object(graphrag_destination, "_trigger_community_detection", new_callable=AsyncMock) as mock_community:
                await graphrag_destination.write(conn, sample_transformed_data)

                mock_extract.assert_not_called()
                mock_community.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_entity_extraction(self, graphrag_destination):
        """Test entity extraction trigger."""
        graphrag_destination._client = AsyncMock()

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            await graphrag_destination._trigger_entity_extraction("test-graph", "doc-123")

            graphrag_destination._client.post.assert_called_once_with(
                "/v1/graphs/test-graph/extract-entities",
                json={"document_id": "doc-123"},
            )
            mock_logger.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_entity_extraction_no_client(self, graphrag_destination):
        """Test entity extraction with no client."""
        graphrag_destination._client = None

        # Should not raise
        await graphrag_destination._trigger_entity_extraction("test-graph", "doc-123")

    @pytest.mark.asyncio
    async def test_trigger_entity_extraction_no_document_id(self, graphrag_destination):
        """Test entity extraction with no document ID."""
        graphrag_destination._client = AsyncMock()

        # Should not call API when document_id is None
        await graphrag_destination._trigger_entity_extraction("test-graph", None)

        graphrag_destination._client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_community_detection(self, graphrag_destination):
        """Test community detection trigger."""
        graphrag_destination._client = AsyncMock()

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            await graphrag_destination._trigger_community_detection("test-graph")

            graphrag_destination._client.post.assert_called_once_with(
                "/v1/graphs/test-graph/detect-communities",
                json={},
            )
            mock_logger.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_payload_structure(self, graphrag_destination):
        """Test _build_payload creates correct structure."""
        data = TransformedData(
            job_id=UUID("12345678-1234-1234-1234-123456789abc"),
            chunks=[{"content": "Test chunk", "metadata": {"key": "value"}}],
            metadata={"title": "Test"},
            lineage={"source": "test"},
            original_format="pdf",
        )

        payload = graphrag_destination._build_payload(data, {"auto_extract_entities": True})

        assert payload["job_id"] == "12345678-1234-1234-1234-123456789abc"
        assert payload["original_format"] == "pdf"
        assert payload["metadata"] == {"title": "Test"}
        assert payload["lineage"] == {"source": "test"}
        assert len(payload["text_chunks"]) == 1
        assert payload["text_chunks"][0]["index"] == 0
        assert payload["text_chunks"][0]["content"] == "Test chunk"
        assert payload["options"]["extract_entities"] is True
        assert payload["options"]["extract_relationships"] is True
        assert payload["options"]["build_communities"] is True


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestGraphRAGErrorHandling:
    """Tests for GraphRAG error handling."""

    @pytest.mark.asyncio
    async def test_write_http_error(self, graphrag_destination, sample_connection, sample_transformed_data):
        """Test write handles HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        error = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.post = AsyncMock(side_effect=error)

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            result = await graphrag_destination.write(sample_connection, sample_transformed_data)

            assert isinstance(result, WriteResult)
            assert result.success is False
            assert "500" in result.error
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_network_error(self, graphrag_destination, sample_connection, sample_transformed_data):
        """Test write handles network errors."""
        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.post = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            result = await graphrag_destination.write(sample_connection, sample_transformed_data)

            assert isinstance(result, WriteResult)
            assert result.success is False
            assert "Connection refused" in result.error
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_entity_extraction_error(self, graphrag_destination):
        """Test entity extraction handles errors gracefully."""
        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.post = AsyncMock(side_effect=Exception("API Error"))

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            await graphrag_destination._trigger_entity_extraction("test-graph", "doc-123")
            mock_logger.warning.assert_called_once()


# ============================================================================
# Search Operations Tests
# ============================================================================

@pytest.mark.unit
class TestGraphRAGSearchOperations:
    """Tests for GraphRAG search operations."""

    @pytest.mark.asyncio
    async def test_search_success(self, graphrag_destination):
        """Test successful search operation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": "ent-1", "name": "Entity A", "score": 0.95},
                {"id": "ent-2", "name": "Entity B", "score": 0.87},
            ],
        }

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.post = AsyncMock(return_value=mock_response)

        result = await graphrag_destination.search("test-graph", "test query", "hybrid", 10)

        assert result["results"][0]["name"] == "Entity A"
        graphrag_destination._client.post.assert_called_once()
        call_args = graphrag_destination._client.post.call_args
        assert call_args[0][0] == "/v1/graphs/test-graph/search"
        assert call_args[1]["json"]["query"] == "test query"
        assert call_args[1]["json"]["search_type"] == "hybrid"
        assert call_args[1]["json"]["top_k"] == 10

    @pytest.mark.asyncio
    async def test_search_not_initialized(self):
        """Test search when not initialized."""
        dest = GraphRAGDestination()

        with pytest.raises(RuntimeError, match="not initialized"):
            await dest.search("test-graph", "query")

    @pytest.mark.asyncio
    async def test_search_error(self, graphrag_destination):
        """Test search handles errors."""
        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.post = AsyncMock(side_effect=Exception("Search failed"))

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            with pytest.raises(Exception, match="Search failed"):
                await graphrag_destination.search("test-graph", "query")

    @pytest.mark.asyncio
    async def test_get_entities_success(self, graphrag_destination):
        """Test successful get_entities operation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "entities": [
                {"id": "ent-1", "name": "Entity A", "type": "person"},
                {"id": "ent-2", "name": "Entity B", "type": "organization"},
            ],
        }

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(return_value=mock_response)

        entities = await graphrag_destination.get_entities("test-graph", "person", 50)

        assert len(entities) == 2
        assert entities[0]["name"] == "Entity A"
        graphrag_destination._client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entities_error(self, graphrag_destination):
        """Test get_entities handles errors."""
        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(side_effect=Exception("API Error"))

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            entities = await graphrag_destination.get_entities("test-graph")

            assert entities == []

    @pytest.mark.asyncio
    async def test_get_relationships_success(self, graphrag_destination):
        """Test successful get_relationships operation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "relationships": [
                {"source": "ent-1", "target": "ent-2", "type": "related_to"},
            ],
        }

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(return_value=mock_response)

        relationships = await graphrag_destination.get_relationships("test-graph", "ent-1", 25)

        assert len(relationships) == 1
        assert relationships[0]["type"] == "related_to"

    @pytest.mark.asyncio
    async def test_get_relationships_error(self, graphrag_destination):
        """Test get_relationships handles errors."""
        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(side_effect=Exception("API Error"))

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            relationships = await graphrag_destination.get_relationships("test-graph")

            assert relationships == []


# ============================================================================
# Config Validation Tests
# ============================================================================

@pytest.mark.unit
class TestGraphRAGConfigValidation:
    """Tests for GraphRAG configuration validation."""

    @pytest.mark.asyncio
    async def test_validate_config_valid(self):
        """Test validation with valid config."""
        dest = GraphRAGDestination()
        result = await dest.validate_config({
            "api_url": "https://api.graphrag.example.com",
            "api_key": "test-key",
        })

        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_config_missing_api_url(self):
        """Test validation with missing API URL."""
        dest = GraphRAGDestination()
        result = await dest.validate_config({})

        assert result.valid is False
        assert len(result.errors) == 1
        assert "api_url" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_config_invalid_url(self):
        """Test validation with invalid URL."""
        dest = GraphRAGDestination()
        result = await dest.validate_config({"api_url": "not-a-url"})

        assert result.valid is False
        assert len(result.errors) == 1
        assert "http" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_config_missing_api_key(self):
        """Test validation with missing API key (warning only)."""
        dest = GraphRAGDestination()
        result = await dest.validate_config({
            "api_url": "https://api.graphrag.example.com",
        })

        assert result.valid is True
        assert len(result.warnings) == 1
        assert "api_key" in result.warnings[0].lower()


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestGraphRAGHealthCheck:
    """Tests for GraphRAG health check."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, graphrag_destination):
        """Test health check returns healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(return_value=mock_response)

        health = await graphrag_destination.health_check()

        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, graphrag_destination):
        """Test health check returns degraded."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(return_value=mock_response)

        health = await graphrag_destination.health_check()

        assert health == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, graphrag_destination):
        """Test health check returns unhealthy."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(return_value=mock_response)

        health = await graphrag_destination.health_check()

        assert health == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, graphrag_destination):
        """Test health check handles timeout."""
        graphrag_destination._client = AsyncMock()
        graphrag_destination._client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        health = await graphrag_destination.health_check()

        assert health == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        dest = GraphRAGDestination()
        health = await dest.health_check()

        assert health == HealthStatus.UNHEALTHY


# ============================================================================
# Shutdown Tests
# ============================================================================

@pytest.mark.unit
class TestGraphRAGShutdown:
    """Tests for GraphRAG shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self, graphrag_destination):
        """Test shutdown closes HTTP client."""
        mock_client = AsyncMock()
        graphrag_destination._client = mock_client

        with patch("src.plugins.destinations.graphrag.logger") as mock_logger:
            await graphrag_destination.shutdown()

            mock_client.aclose.assert_called_once()
            assert graphrag_destination._client is None
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_no_client(self):
        """Test shutdown when no client exists."""
        dest = GraphRAGDestination()
        dest._client = None

        # Should not raise
        await dest.shutdown()


# ============================================================================
# GraphRAGMockDestination Tests
# ============================================================================

@pytest.mark.unit
class TestGraphRAGMockDestination:
    """Tests for GraphRAGMockDestination."""

    def test_init(self):
        """Test mock destination initialization."""
        dest = GraphRAGMockDestination()
        assert dest._storage == {}
        assert dest._entities == {}
        assert dest._relationships == {}

    @pytest.mark.asyncio
    async def test_mock_write(self):
        """Test mock write."""
        dest = GraphRAGMockDestination()
        await dest.initialize({})

        conn = await dest.connect({"graph_id": "test-graph"})
        data = TransformedData(
            job_id=UUID("12345678-1234-1234-1234-123456789abc"),
            chunks=[{"content": "Chunk 1"}, {"content": "Chunk 2"}],
        )

        result = await dest.write(conn, data)

        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.destination_id == "graphrag_mock"
        assert result.records_written == 2
        assert "entities_count" in result.metadata

        # Check stored data
        stored = dest.get_stored_data("test-graph")
        assert len(stored) == 1

    @pytest.mark.asyncio
    async def test_mock_get_entities(self):
        """Test mock get_entities."""
        dest = GraphRAGMockDestination()
        await dest.initialize({})
        await dest.connect({"graph_id": "test-graph"})

        # Add some mock entities
        dest._entities["test-graph"] = [
            {"id": "ent-1", "name": "Entity 1", "type": "person"},
            {"id": "ent-2", "name": "Entity 2", "type": "organization"},
        ]

        entities = await dest.get_entities("test-graph", "person")

        assert len(entities) == 1
        assert entities[0]["type"] == "person"

    @pytest.mark.asyncio
    async def test_mock_get_relationships(self):
        """Test mock get_relationships."""
        dest = GraphRAGMockDestination()
        await dest.initialize({})

        relationships = await dest.get_relationships("test-graph")

        assert relationships == []

    def test_mock_clear_storage(self):
        """Test clearing storage."""
        dest = GraphRAGMockDestination()
        dest._storage = {"graph1": [{"data": "test"}]}
        dest._entities = {"graph1": [{"id": "ent-1"}]}
        dest._relationships = {"graph1": [{"id": "rel-1"}]}

        dest.clear_storage()

        assert dest._storage == {}
        assert dest._entities == {}
        assert dest._relationships == {}
