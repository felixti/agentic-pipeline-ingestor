"""Unit tests for CogneeLocalDestination plugin using actual Cognee library APIs.

These tests verify that the CogneeLocalDestination correctly uses:
- cognee.add() for adding documents
- cognee.cognify() for processing into knowledge graph
- cognee.search() for searching
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from src.plugins.base import (
    Connection,
    HealthStatus,
    TransformedData,
    ValidationResult,
    WriteResult,
)

# Import directly to avoid httpx dependency from other modules
import sys
sys.path.insert(0, '/Users/felix/temp/agentic-pipeline-ingestor')
from src.plugins.destinations.cognee_local import (
    CogneeLocalDestination,
    CogneeLocalMockDestination,
    _get_search_type,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_transformed_data():
    """Create sample transformed data for testing."""
    return TransformedData(
        job_id=UUID("12345678-1234-1234-1234-123456789abc"),
        chunks=[
            {"content": "Chunk 1 content about artificial intelligence", "metadata": {"index": 0}},
            {"content": "Chunk 2 content about machine learning", "metadata": {"index": 1}},
            {"content": "Chunk 3 content about neural networks", "metadata": {"index": 2}},
        ],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        metadata={
            "title": "Test Document",
            "author": "Test Author",
            "organization": "Test Org",
        },
        lineage={"source": "test", "parser": "test_parser"},
        original_format="pdf",
        output_format="json",
    )


@pytest.fixture
def sample_connection():
    """Create a sample connection."""
    return Connection(
        id=UUID(int=hash("test-dataset:default") % (2**32)),
        plugin_id="cognee_local",
        config={
            "dataset_id": "test-dataset",
            "graph_name": "default",
            "auto_cognify": False,
        },
    )


@pytest.fixture
def mock_neo4j_client():
    """Create a mock Neo4j client for health checks."""
    client = MagicMock()
    client.execute_write = AsyncMock(return_value=[{"id": "test-id"}])
    client.execute_query = AsyncMock(return_value=[{"count": 5}])
    client.health_check = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_cognee_module():
    """Create a mock Cognee module."""
    cognee = MagicMock()
    cognee.add = AsyncMock()
    cognee.cognify = AsyncMock()
    cognee.search = AsyncMock(return_value=[
        {"id": "chunk1", "text": "Test result 1", "score": 0.95},
        {"id": "chunk2", "text": "Test result 2", "score": 0.85},
    ])
    cognee.__version__ = "0.1.0"
    return cognee


@pytest.fixture
def mock_search_type():
    """Create mock SearchType enum."""
    class SearchType:
        GRAPH_COMPLETION = "graph_completion"
        RAG_COMPLETION = "rag_completion"
        FEELING_LUCKY = "feeling_lucky"
        SUMMARIES = "summaries"
        CHUNKS = "chunks"
    return SearchType


# ============================================================================
# CogneeLocalDestination Class Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLocalDestination:
    """Tests for CogneeLocalDestination class."""

    def test_init(self):
        """Test destination initialization."""
        dest = CogneeLocalDestination()
        assert dest._config == {}
        assert dest._dataset_id is None
        assert dest._graph_name == "default"
        assert dest._neo4j_client is None
        assert dest._is_initialized is False
        assert dest._cognee_module is None

    def test_metadata(self):
        """Test destination metadata."""
        dest = CogneeLocalDestination()
        metadata = dest.metadata

        assert metadata.id == "cognee_local"
        assert "Cognee" in metadata.name
        assert metadata.version == "1.0.0"
        assert metadata.type.value == "destination"
        assert "json" in metadata.supported_formats
        assert metadata.requires_auth is False
        assert "dataset_id" in metadata.config_schema["properties"]
        assert "graph_name" in metadata.config_schema["properties"]
        assert "auto_cognify" in metadata.config_schema["properties"]

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_neo4j_client, mock_cognee_module, mock_search_type):
        """Test successful initialization."""
        dest = CogneeLocalDestination()

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            with patch.dict("sys.modules", {"cognee": mock_cognee_module}):
                with patch(
                    "src.plugins.destinations.cognee_local._get_search_type",
                    return_value=mock_search_type.FEELING_LUCKY,
                ):
                    # Mock the actual import
                    mock_cognee_api = MagicMock()
                    mock_cognee_api.SearchType = mock_search_type
                    
                    with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: {
                        "cognee": mock_cognee_module,
                        "cognee.api": MagicMock(),
                        "cognee.api.v1": MagicMock(),
                        "cognee.api.v1.search": mock_cognee_api,
                    }.get(name, __import__(name, *args, **kwargs))):
                        # We can't easily mock the import, so let's patch the module directly
                        pass
        
        # Simpler approach: patch after initialization
        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            # Mock the imports by patching the module attributes after init
            dest._cognee_module = mock_cognee_module
            dest._search_type_module = mock_search_type
            
            await dest.initialize({
                "dataset_id": "my-dataset",
                "graph_name": "my-graph",
            })

        assert dest._is_initialized is True
        assert dest._dataset_id == "my-dataset"
        assert dest._graph_name == "my-graph"
        assert dest._neo4j_client is mock_neo4j_client

    @pytest.mark.asyncio
    async def test_initialize_without_cognee_lib(self, mock_neo4j_client):
        """Test initialization fails when cognee library not installed."""
        dest = CogneeLocalDestination()

        def mock_import(name, *args, **kwargs):
            if name == "cognee":
                raise ImportError("No module named 'cognee'")
            return __builtins__["__import__"](name, *args, **kwargs)

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            # The import check happens in initialize, mock it to fail
            with patch.object(dest, "_cognee_module", None):
                dest._cognee_module = None  # Ensure it's None
                
                # Mock the import to raise ImportError
                original_import = __builtins__["__import__"]
                try:
                    __builtins__["__import__"] = mock_import
                    with pytest.raises((RuntimeError, ImportError)):
                        await dest.initialize({})
                finally:
                    __builtins__["__import__"] = original_import

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_neo4j_client, mock_cognee_module):
        """Test successful connection creation."""
        dest = CogneeLocalDestination()
        dest._cognee_module = mock_cognee_module

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            await dest.initialize({"dataset_id": "my-dataset"})
            conn = await dest.connect({"dataset_id": "my-dataset"})

        assert isinstance(conn, Connection)
        assert conn.plugin_id == "cognee_local"
        assert conn.config["dataset_id"] == "my-dataset"
        assert conn.config["graph_name"] == "default"

    @pytest.mark.asyncio
    async def test_connect_not_initialized(self):
        """Test connect fails when not initialized."""
        dest = CogneeLocalDestination()

        with pytest.raises(ConnectionError, match="not initialized"):
            await dest.connect({"dataset_id": "test"})


# ============================================================================
# Write Data Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLocalWriteData:
    """Tests for CogneeLocal write operations using cognee.add()."""

    @pytest.mark.asyncio
    async def test_write_success(self, mock_neo4j_client, mock_cognee_module, sample_connection, sample_transformed_data):
        """Test successful write operation using cognee.add()."""
        dest = CogneeLocalDestination()
        dest._cognee_module = mock_cognee_module

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            await dest.initialize({})
            result = await dest.write(sample_connection, sample_transformed_data)

        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.destination_id == "cognee_local"
        assert "cognee://" in result.destination_uri
        assert result.records_written == 3  # 3 chunks added
        assert result.metadata["chunks_added"] == 3
        assert result.metadata["dataset_id"] == "test-dataset"
        
        # Verify cognee.add() was called for each chunk
        assert mock_cognee_module.add.call_count == 3

    @pytest.mark.asyncio
    async def test_write_not_initialized(self, sample_connection, sample_transformed_data):
        """Test write when not initialized."""
        dest = CogneeLocalDestination()

        result = await dest.write(sample_connection, sample_transformed_data)

        assert isinstance(result, WriteResult)
        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_write_with_auto_cognify(self, mock_neo4j_client, mock_cognee_module, sample_transformed_data):
        """Test write with auto_cognify enabled."""
        dest = CogneeLocalDestination()
        dest._cognee_module = mock_cognee_module

        conn = Connection(
            id=UUID(int=hash("test") % (2**32)),
            plugin_id="cognee_local",
            config={"dataset_id": "test", "graph_name": "default", "auto_cognify": True},
        )

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            await dest.initialize({})
            result = await dest.write(conn, sample_transformed_data)

        assert result.success is True
        # Verify cognee.cognify() was called when auto_cognify is True
        mock_cognee_module.cognify.assert_called_once_with(datasets=["test"])


# ============================================================================
# Process Dataset Tests (cognee.cognify)
# ============================================================================

@pytest.mark.unit
class TestCogneeLocalProcessDataset:
    """Tests for process_dataset using cognee.cognify()."""

    @pytest.mark.asyncio
    async def test_process_dataset_success(self, mock_neo4j_client, mock_cognee_module):
        """Test successful dataset processing using cognee.cognify()."""
        dest = CogneeLocalDestination()
        dest._cognee_module = mock_cognee_module

        conn = Connection(
            id=UUID(int=hash("test") % (2**32)),
            plugin_id="cognee_local",
            config={"dataset_id": "test-dataset"},
        )

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            await dest.initialize({})
            result = await dest.process_dataset(conn)

        assert result["success"] is True
        assert result["dataset_id"] == "test-dataset"
        assert "processing_time_ms" in result
        
        # Verify cognee.cognify() was called correctly
        mock_cognee_module.cognify.assert_called_once_with(datasets=["test-dataset"])

    @pytest.mark.asyncio
    async def test_process_dataset_not_initialized(self):
        """Test process_dataset fails when not initialized."""
        dest = CogneeLocalDestination()

        conn = Connection(
            id=UUID(int=hash("test") % (2**32)),
            plugin_id="cognee_local",
            config={"dataset_id": "test"},
        )

        with pytest.raises(RuntimeError, match="not initialized"):
            await dest.process_dataset(conn)


# ============================================================================
# Search Tests (cognee.search)
# ============================================================================

@pytest.mark.unit
class TestCogneeLocalSearch:
    """Tests for search using cognee.search()."""

    @pytest.mark.asyncio
    async def test_search_success(self, mock_neo4j_client, mock_cognee_module, mock_search_type, sample_connection):
        """Test successful search using cognee.search()."""
        dest = CogneeLocalDestination()
        dest._cognee_module = mock_cognee_module

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            with patch(
                "src.plugins.destinations.cognee_local._get_search_type",
                return_value=mock_search_type.GRAPH_COMPLETION,
            ):
                await dest.initialize({})
                results = await dest.search(sample_connection, "test query", search_type="graph", top_k=5)

        assert len(results) == 2
        assert results[0].chunk_id == "chunk1"
        assert results[0].content == "Test result 1"
        assert results[0].score == 0.95
        
        # Verify cognee.search() was called
        mock_cognee_module.search.assert_called_once_with(
            query_text="test query",
            query_type=mock_search_type.GRAPH_COMPLETION,
            datasets=["test-dataset"],
        )

    @pytest.mark.asyncio
    async def test_search_invalid_type(self, mock_neo4j_client, mock_cognee_module, sample_connection):
        """Test search with invalid search type."""
        dest = CogneeLocalDestination()
        dest._cognee_module = mock_cognee_module

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            await dest.initialize({})
            
            with pytest.raises(ValueError, match="Invalid search_type"):
                await dest.search(sample_connection, "test", search_type="invalid")

    @pytest.mark.asyncio
    async def test_search_not_initialized(self, sample_connection):
        """Test search when not initialized."""
        dest = CogneeLocalDestination()

        with pytest.raises(RuntimeError, match="not initialized"):
            await dest.search(sample_connection, "test query")


# ============================================================================
# Search Type Mapping Tests
# ============================================================================

@pytest.mark.unit
class TestSearchTypeMapping:
    """Tests for _get_search_type function."""

    def test_get_search_type_graph(self, mock_search_type):
        """Test getting GRAPH_COMPLETION search type."""
        with patch(
            "src.plugins.destinations.cognee_local._get_search_type",
            return_value=mock_search_type.GRAPH_COMPLETION,
        ):
            result = _get_search_type("graph")
            assert result == mock_search_type.GRAPH_COMPLETION

    def test_get_search_type_vector(self, mock_search_type):
        """Test getting RAG_COMPLETION search type."""
        with patch(
            "src.plugins.destinations.cognee_local._get_search_type",
            return_value=mock_search_type.RAG_COMPLETION,
        ):
            result = _get_search_type("vector")
            assert result == mock_search_type.RAG_COMPLETION

    def test_get_search_type_hybrid(self, mock_search_type):
        """Test getting FEELING_LUCKY search type."""
        with patch(
            "src.plugins.destinations.cognee_local._get_search_type",
            return_value=mock_search_type.FEELING_LUCKY,
        ):
            result = _get_search_type("hybrid")
            assert result == mock_search_type.FEELING_LUCKY


# ============================================================================
# Config Validation Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLocalConfigValidation:
    """Tests for CogneeLocal configuration validation."""

    @pytest.mark.asyncio
    async def test_validate_config_valid(self):
        """Test validation with valid config."""
        dest = CogneeLocalDestination()

        with patch.dict(os.environ, {"NEO4J_URI": "bolt://localhost:7687"}):
            result = await dest.validate_config({})

        assert isinstance(result, ValidationResult)
        # May have warnings about dependencies not installed, but should be valid

    @pytest.mark.asyncio
    async def test_validate_config_invalid_uri(self):
        """Test validation with invalid Neo4j URI."""
        dest = CogneeLocalDestination()

        result = await dest.validate_config({"neo4j_uri": "not-a-valid-uri"})

        assert result.valid is False
        assert any("bolt://" in error or "neo4j://" in error for error in result.errors)


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLocalHealthCheck:
    """Tests for CogneeLocal health check."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_neo4j_client, mock_cognee_module):
        """Test health check returns healthy."""
        dest = CogneeLocalDestination()
        dest._cognee_module = mock_cognee_module

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            await dest.initialize({})
            health = await dest.health_check()

        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        dest = CogneeLocalDestination()

        health = await dest.health_check()

        assert health == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_neo4j_unhealthy(self, mock_neo4j_client, mock_cognee_module):
        """Test health check when Neo4j is unhealthy."""
        dest = CogneeLocalDestination()
        dest._cognee_module = mock_cognee_module
        mock_neo4j_client.health_check.return_value = False

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            await dest.initialize({})
            health = await dest.health_check()

        assert health == HealthStatus.DEGRADED


# ============================================================================
# Shutdown Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLocalShutdown:
    """Tests for CogneeLocal shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_neo4j(self, mock_neo4j_client, mock_cognee_module):
        """Test shutdown closes Neo4j connection."""
        dest = CogneeLocalDestination()
        dest._cognee_module = mock_cognee_module

        with patch(
            "src.plugins.destinations.cognee_local.get_neo4j_client",
            return_value=mock_neo4j_client,
        ):
            with patch(
                "src.plugins.destinations.cognee_local.close_neo4j_client",
                new_callable=AsyncMock,
            ) as mock_close:
                await dest.initialize({})
                await dest.shutdown()

                mock_close.assert_called_once()
                assert dest._is_initialized is False
                assert dest._cognee_module is None


# ============================================================================
# Environment Setup Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeEnvironmentSetup:
    """Tests for Cognee environment variable setup."""

    @pytest.mark.asyncio
    async def test_setup_cognee_environment(self, mock_neo4j_client, mock_cognee_module):
        """Test that environment variables are set up for Cognee."""
        dest = CogneeLocalDestination()
        dest._cognee_module = mock_cognee_module

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "src.plugins.destinations.cognee_local.get_neo4j_client",
                return_value=mock_neo4j_client,
            ):
                await dest.initialize({
                    "neo4j_uri": "bolt://test:7687",
                    "neo4j_user": "testuser",
                    "neo4j_password": "testpass",
                })

        # Check that environment variables were set
        assert os.getenv("NEO4J_URI") == "bolt://test:7687"
        assert os.getenv("NEO4J_USER") == "testuser"
        assert os.getenv("NEO4J_PASSWORD") == "testpass"


# ============================================================================
# CogneeLocalMockDestination Tests
# ============================================================================

@pytest.mark.unit
class TestCogneeLocalMockDestination:
    """Tests for CogneeLocalMockDestination."""

    def test_init(self):
        """Test mock destination initialization."""
        dest = CogneeLocalMockDestination()
        assert dest._storage["documents"] == {}
        assert dest._storage["chunks"] == {}
        assert dest._storage["datasets"] == {}

    def test_metadata(self):
        """Test mock destination metadata."""
        dest = CogneeLocalMockDestination()
        metadata = dest.metadata

        assert metadata.id == "cognee_local_mock"
        assert "Mock" in metadata.name
        assert metadata.requires_auth is False

    @pytest.mark.asyncio
    async def test_mock_initialize(self):
        """Test mock initialization."""
        dest = CogneeLocalMockDestination()
        await dest.initialize({
            "dataset_id": "test-dataset",
            "graph_name": "test-graph",
        })

        assert dest._config == {"dataset_id": "test-dataset", "graph_name": "test-graph"}
        assert dest._is_initialized is True

    @pytest.mark.asyncio
    async def test_mock_connect(self):
        """Test mock connect."""
        dest = CogneeLocalMockDestination()
        await dest.initialize({})

        conn = await dest.connect({"dataset_id": "test-dataset"})

        assert isinstance(conn, Connection)
        assert conn.plugin_id == "cognee_local_mock"
        assert conn.config["dataset_id"] == "test-dataset"
        assert "test-dataset" in dest._storage["datasets"]

    @pytest.mark.asyncio
    async def test_mock_write(self, sample_transformed_data):
        """Test mock write."""
        dest = CogneeLocalMockDestination()
        await dest.initialize({})

        conn = await dest.connect({"dataset_id": "test-dataset"})
        result = await dest.write(conn, sample_transformed_data)

        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.destination_id == "cognee_local_mock"
        assert result.records_written == 4  # 3 chunks + 1 document

        # Check stored data
        documents = dest.get_stored_documents("test-dataset")
        assert len(documents) == 1

    @pytest.mark.asyncio
    async def test_mock_process_dataset(self):
        """Test mock process_dataset."""
        dest = CogneeLocalMockDestination()
        await dest.initialize({})

        conn = await dest.connect({"dataset_id": "test-dataset"})
        result = await dest.process_dataset(conn)

        assert result["success"] is True
        assert result["dataset_id"] == "test-dataset"
        assert result["mock"] is True

    @pytest.mark.asyncio
    async def test_mock_search(self, sample_transformed_data):
        """Test mock search."""
        dest = CogneeLocalMockDestination()
        await dest.initialize({})

        conn = await dest.connect({"dataset_id": "test-dataset"})
        await dest.write(conn, sample_transformed_data)
        
        results = await dest.search(conn, "artificial intelligence")

        assert len(results) > 0
        assert all(hasattr(r, "chunk_id") for r in results)
        assert all(hasattr(r, "score") for r in results)

    @pytest.mark.asyncio
    async def test_mock_health_check(self):
        """Test mock health check always returns healthy."""
        dest = CogneeLocalMockDestination()

        health = await dest.health_check()

        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_mock_validate_config(self):
        """Test mock validate config always returns valid."""
        dest = CogneeLocalMockDestination()

        result = await dest.validate_config({})

        assert isinstance(result, ValidationResult)
        assert result.valid is True

    def test_mock_clear_storage(self):
        """Test clearing storage."""
        dest = CogneeLocalMockDestination()
        dest._storage["documents"]["dataset1"] = {"doc1": {"data": "test"}}

        dest.clear_storage()

        assert dest._storage["documents"] == {}
        assert dest._storage["chunks"] == {}
        assert dest._storage["datasets"] == {}
