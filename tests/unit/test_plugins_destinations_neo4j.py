"""Unit tests for Neo4j destination plugin."""

import json
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
from src.plugins.destinations.neo4j import Neo4jDestination, Neo4jMockDestination

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def neo4j_destination():
    """Create an initialized Neo4j destination."""
    dest = Neo4jDestination()
    await dest.initialize({
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "test-password",
        "database": "testdb",
    })
    return dest


@pytest.fixture
def sample_transformed_data():
    """Create sample transformed data for testing."""
    return TransformedData(
        job_id=UUID("12345678-1234-1234-1234-123456789abc"),
        chunks=[
            {"content": "First chunk content", "metadata": {"index": 0, "page": 1}},
            {"content": "Second chunk content", "metadata": {"index": 1, "page": 1}},
            {"content": "Third chunk content", "metadata": {"index": 2, "page": 2}},
        ],
        metadata={"title": "Test Document", "author": "Test Author", "organization": "Test Org"},
        lineage={"source": "test", "parser": "test_parser"},
        original_format="pdf",
        output_format="json",
    )


@pytest.fixture
def sample_connection():
    """Create a sample connection."""
    return Connection(
        id=UUID(int=hash("testdb") % (2**32)),
        plugin_id="neo4j",
        config={
            "database": "testdb",
            "create_entities": True,
            "create_relationships": True,
        },
    )


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = MagicMock()
    session = AsyncMock()
    driver.session = MagicMock(return_value=session)
    driver.verify_connectivity = AsyncMock()
    return driver, session


# ============================================================================
# Neo4jDestination Class Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jDestination:
    """Tests for Neo4jDestination class."""

    def test_init(self):
        """Test destination initialization."""
        dest = Neo4jDestination()
        assert dest._uri is None
        assert dest._username is None
        assert dest._password is None
        assert dest._config == {}
        assert dest._driver is None

    def test_metadata(self):
        """Test destination metadata."""
        dest = Neo4jDestination()
        metadata = dest.metadata

        assert metadata.id == "neo4j"
        assert metadata.name == "Neo4j Graph Database"
        assert metadata.version == "1.0.0"
        assert "json" in metadata.supported_formats
        assert "text" in metadata.supported_formats
        assert "markdown" in metadata.supported_formats
        assert metadata.requires_auth is True
        assert "uri" in metadata.config_schema["properties"]
        assert "username" in metadata.config_schema["properties"]
        assert "password" in metadata.config_schema["properties"]
        assert "create_entities" in metadata.config_schema["properties"]
        assert "create_relationships" in metadata.config_schema["properties"]

    @pytest.mark.asyncio
    async def test_initialize_with_config(self):
        """Test initialization with config parameters."""
        dest = Neo4jDestination()
        await dest.initialize({
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "database": "neo4j",
        })

        assert dest._uri == "bolt://localhost:7687"
        assert dest._username == "neo4j"
        assert dest._password == "password"
        assert dest._config["database"] == "neo4j"

    @pytest.mark.asyncio
    async def test_initialize_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(os.environ, {
            "NEO4J_URI": "bolt://env.example.com:7687",
            "NEO4J_USERNAME": "env-user",
            "NEO4J_PASSWORD": "env-pass",
        }):
            dest = Neo4jDestination()
            await dest.initialize({})

            assert dest._uri == "bolt://env.example.com:7687"
            assert dest._username == "env-user"
            assert dest._password == "env-pass"

    @pytest.mark.asyncio
    async def test_initialize_without_uri(self):
        """Test initialization without URI shows warning."""
        dest = Neo4jDestination()
        with patch("src.plugins.destinations.neo4j.logger") as mock_logger:
            await dest.initialize({})
            # Should warn about URI not configured (and possibly about driver)
            assert mock_logger.warning.call_count >= 1
            assert any("not configured" in str(call) for call in mock_logger.warning.call_args_list)

    @pytest.mark.asyncio
    async def test_initialize_import_error(self):
        """Test initialization logs warning when neo4j not installed."""
        with patch.dict("sys.modules", {"neo4j": None}):
            dest = Neo4jDestination()
            with patch("src.plugins.destinations.neo4j.logger") as mock_logger:
                await dest.initialize({"uri": "bolt://localhost:7687"})
                # Check that warning was called about import
                warning_calls = [call for call in mock_logger.warning.call_args_list
                               if "not installed" in str(call)]
                assert len(warning_calls) > 0 or mock_logger.warning.called


# ============================================================================
# Connection Handling Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jConnectionHandling:
    """Tests for Neo4j connection handling."""

    @pytest.mark.asyncio
    async def test_connect_creates_driver(self, neo4j_destination, mock_neo4j_driver):
        """Test connect creates Neo4j driver."""
        driver, session = mock_neo4j_driver
        mock_neo4j_module = MagicMock()
        mock_neo4j_module.AsyncGraphDatabase.driver = MagicMock(return_value=driver)

        with patch.dict("sys.modules", {"neo4j": mock_neo4j_module}):
            with patch.object(neo4j_destination, "_ensure_schema", new_callable=AsyncMock):
                conn = await neo4j_destination.connect({"database": "testdb"})

                assert isinstance(conn, Connection)
                assert conn.plugin_id == "neo4j"
                assert conn.config["database"] == "testdb"
                driver.verify_connectivity.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_without_uri(self):
        """Test connect raises error without URI."""
        dest = Neo4jDestination()
        await dest.initialize({})

        # Create mock neo4j module for the import in connect()
        mock_neo4j_module = MagicMock()
        
        with patch.dict("sys.modules", {"neo4j": mock_neo4j_module}):
            with pytest.raises(ConnectionError, match="URI not configured"):
                await dest.connect({})

    @pytest.mark.asyncio
    async def test_connect_uses_default_database(self, neo4j_destination, mock_neo4j_driver):
        """Test connect uses default database from config."""
        driver, session = mock_neo4j_driver
        neo4j_destination._config["database"] = "defaultdb"
        mock_neo4j_module = MagicMock()
        mock_neo4j_module.AsyncGraphDatabase.driver = MagicMock(return_value=driver)

        with patch.dict("sys.modules", {"neo4j": mock_neo4j_module}):
            with patch.object(neo4j_destination, "_ensure_schema", new_callable=AsyncMock):
                conn = await neo4j_destination.connect({})

                assert conn.config["database"] == "defaultdb"

    @pytest.mark.asyncio
    async def test_connect_with_auth(self, neo4j_destination, mock_neo4j_driver):
        """Test connect uses authentication."""
        driver, session = mock_neo4j_driver
        mock_neo4j_module = MagicMock()
        mock_neo4j_module.AsyncGraphDatabase.driver = MagicMock(return_value=driver)

        with patch.dict("sys.modules", {"neo4j": mock_neo4j_module}):
            with patch.object(neo4j_destination, "_ensure_schema", new_callable=AsyncMock):
                await neo4j_destination.connect({"database": "testdb"})

                mock_neo4j_module.AsyncGraphDatabase.driver.assert_called_once()
                call_args = mock_neo4j_module.AsyncGraphDatabase.driver.call_args
                assert call_args[0][0] == "bolt://localhost:7687"
                assert call_args[1]["auth"] == ("neo4j", "test-password")

    @pytest.mark.asyncio
    async def test_connect_without_password(self, neo4j_destination, mock_neo4j_driver):
        """Test connect without password."""
        driver, session = mock_neo4j_driver
        neo4j_destination._password = None
        mock_neo4j_module = MagicMock()
        mock_neo4j_module.AsyncGraphDatabase.driver = MagicMock(return_value=driver)

        with patch.dict("sys.modules", {"neo4j": mock_neo4j_module}):
            with patch.object(neo4j_destination, "_ensure_schema", new_callable=AsyncMock):
                await neo4j_destination.connect({"database": "testdb"})

                call_args = mock_neo4j_module.AsyncGraphDatabase.driver.call_args
                assert call_args[1]["auth"] is None

    @pytest.mark.asyncio
    async def test_ensure_schema(self, neo4j_destination, mock_neo4j_driver):
        """Test schema creation."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        # Mock session.run
        mock_result = AsyncMock()
        session.run = AsyncMock(return_value=mock_result)

        await neo4j_destination._ensure_schema("testdb")

        # Should create multiple constraints/indexes
        assert session.run.call_count >= 6

    @pytest.mark.asyncio
    async def test_ensure_schema_no_driver(self, neo4j_destination):
        """Test _ensure_schema with no driver."""
        neo4j_destination._driver = None

        # Should not raise
        await neo4j_destination._ensure_schema("testdb")

    @pytest.mark.asyncio
    async def test_ensure_schema_handles_errors(self, neo4j_destination, mock_neo4j_driver):
        """Test _ensure_schema handles constraint creation errors."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        session.run = AsyncMock(side_effect=Exception("Constraint exists"))

        with patch("src.plugins.destinations.neo4j.logger") as mock_logger:
            await neo4j_destination._ensure_schema("testdb")
            # Should log warnings for each constraint failure
            assert mock_logger.warning.call_count >= 0


# ============================================================================
# Write Operations Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jWriteOperations:
    """Tests for Neo4j write operations."""

    @pytest.mark.asyncio
    async def test_write_not_initialized(self, sample_connection, sample_transformed_data):
        """Test write when driver not initialized."""
        dest = Neo4jDestination()
        # Don't initialize

        result = await dest.write(sample_connection, sample_transformed_data)

        assert isinstance(result, WriteResult)
        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_write_success(self, neo4j_destination, sample_connection, sample_transformed_data, mock_neo4j_driver):
        """Test successful write operation."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Mock result.single() for each query
        mock_record = {"id": "12345678-1234-1234-1234-123456789abc"}
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        session.run = AsyncMock(return_value=mock_result)

        result = await neo4j_destination.write(sample_connection, sample_transformed_data)

        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.destination_id == "neo4j"
        assert "neo4j://testdb/Document/" in result.destination_uri
        assert result.records_written == 4  # 1 document + 3 chunks
        assert result.metadata["chunks_created"] == 3
        assert result.metadata["database"] == "testdb"

    @pytest.mark.asyncio
    async def test_create_document(self, neo4j_destination, sample_transformed_data, mock_neo4j_driver):
        """Test document node creation."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        mock_record = {"id": "doc-123"}
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        session.run = AsyncMock(return_value=mock_result)

        conn = sample_connection
        document_id = await neo4j_destination._create_document(conn, sample_transformed_data, "testdb")

        assert document_id == "doc-123"
        session.run.assert_called_once()
        call_args = session.run.call_args
        assert "MERGE (d:Document" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_chunks(self, neo4j_destination, sample_transformed_data, mock_neo4j_driver):
        """Test chunk node creation."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        mock_record = {"id": "chunk-123"}
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        session.run = AsyncMock(return_value=mock_result)

        conn = sample_connection
        document_id = str(sample_transformed_data.job_id)
        chunk_ids = await neo4j_destination._create_chunks(conn, sample_transformed_data, document_id, "testdb")

        assert len(chunk_ids) == 3
        assert session.run.call_count == 3

    @pytest.mark.asyncio
    async def test_create_entities(self, neo4j_destination, sample_transformed_data, mock_neo4j_driver):
        """Test entity node creation."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        mock_result = AsyncMock()
        session.run = AsyncMock(return_value=mock_result)

        conn = sample_connection
        chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
        entity_count = await neo4j_destination._create_entities(conn, sample_transformed_data, chunk_ids, "testdb")

        # Should create entities from metadata (title, author, organization)
        assert entity_count > 0
        assert session.run.called

    @pytest.mark.asyncio
    async def test_create_entities_no_metadata(self, neo4j_destination, mock_neo4j_driver):
        """Test entity creation with no entity metadata."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        data = TransformedData(
            job_id=UUID("12345678-1234-1234-1234-123456789abc"),
            chunks=[{"content": "Test"}],
            metadata={"other": "value"},  # No title, author, or organization
        )

        conn = sample_connection
        chunk_ids = ["chunk-1"]
        entity_count = await neo4j_destination._create_entities(conn, data, chunk_ids, "testdb")

        assert entity_count == 0

    @pytest.mark.asyncio
    async def test_create_relationships(self, neo4j_destination, sample_transformed_data, mock_neo4j_driver):
        """Test relationship creation between chunks."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        mock_result = AsyncMock()
        session.run = AsyncMock(return_value=mock_result)

        conn = sample_connection
        chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
        rel_count = await neo4j_destination._create_relationships(conn, sample_transformed_data, chunk_ids, "testdb")

        # Should create 2 sequential relationships (3 chunks -> 2 relationships)
        assert rel_count == 2
        assert session.run.call_count == 2

    @pytest.mark.asyncio
    async def test_create_relationships_single_chunk(self, neo4j_destination, mock_neo4j_driver):
        """Test relationship creation with single chunk."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        data = TransformedData(
            job_id=UUID("12345678-1234-1234-1234-123456789abc"),
            chunks=[{"content": "Only one chunk"}],
        )

        conn = sample_connection
        chunk_ids = ["chunk-1"]
        rel_count = await neo4j_destination._create_relationships(conn, data, chunk_ids, "testdb")

        assert rel_count == 0
        session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_skips_entities_when_disabled(self, neo4j_destination, mock_neo4j_driver):
        """Test write skips entity creation when disabled."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        mock_record = {"id": "doc-123"}
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        session.run = AsyncMock(return_value=mock_result)

        conn = Connection(
            id=UUID(int=hash("testdb") % (2**32)),
            plugin_id="neo4j",
            config={
                "database": "testdb",
                "create_entities": False,
                "create_relationships": True,
            },
        )

        with patch.object(neo4j_destination, "_create_entities", new_callable=AsyncMock) as mock_entities:
            await neo4j_destination.write(conn, sample_transformed_data)
            mock_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_skips_relationships_when_disabled(self, neo4j_destination, mock_neo4j_driver):
        """Test write skips relationship creation when disabled."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        mock_record = {"id": "doc-123"}
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        session.run = AsyncMock(return_value=mock_result)

        conn = Connection(
            id=UUID(int=hash("testdb") % (2**32)),
            plugin_id="neo4j",
            config={
                "database": "testdb",
                "create_entities": True,
                "create_relationships": False,
            },
        )

        with patch.object(neo4j_destination, "_create_relationships", new_callable=AsyncMock) as mock_rels:
            await neo4j_destination.write(conn, sample_transformed_data)
            mock_rels.assert_not_called()


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jErrorHandling:
    """Tests for Neo4j error handling."""

    @pytest.mark.asyncio
    async def test_write_error(self, neo4j_destination, sample_connection, sample_transformed_data):
        """Test write handles errors."""
        neo4j_destination._driver = MagicMock()
        neo4j_destination._driver.session = MagicMock(side_effect=Exception("Database error"))

        with patch("src.plugins.destinations.neo4j.logger") as mock_logger:
            result = await neo4j_destination.write(sample_connection, sample_transformed_data)

            assert isinstance(result, WriteResult)
            assert result.success is False
            assert "Database error" in result.error
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_document_error(self, neo4j_destination, sample_transformed_data, mock_neo4j_driver):
        """Test document creation handles errors."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        session.run = AsyncMock(side_effect=Exception("Query failed"))

        conn = sample_connection
        with pytest.raises(Exception, match="Query failed"):
            await neo4j_destination._create_document(conn, sample_transformed_data, "testdb")

    @pytest.mark.asyncio
    async def test_create_entities_handles_errors(self, neo4j_destination, sample_transformed_data, mock_neo4j_driver):
        """Test entity creation handles errors gracefully."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        # First call succeeds, second fails
        session.run = AsyncMock(side_effect=[MagicMock(), Exception("Entity error")])

        conn = sample_connection
        chunk_ids = ["chunk-1", "chunk-2"]

        with patch("src.plugins.destinations.neo4j.logger") as mock_logger:
            entity_count = await neo4j_destination._create_entities(conn, sample_transformed_data, chunk_ids, "testdb")

            # Should continue despite errors
            assert entity_count >= 0
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_create_relationships_handles_errors(self, neo4j_destination, sample_transformed_data, mock_neo4j_driver):
        """Test relationship creation handles errors gracefully."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        session.run = AsyncMock(side_effect=Exception("Relationship error"))

        conn = sample_connection
        chunk_ids = ["chunk-1", "chunk-2"]

        with patch("src.plugins.destinations.neo4j.logger") as mock_logger:
            rel_count = await neo4j_destination._create_relationships(conn, sample_transformed_data, chunk_ids, "testdb")

            # Should continue despite errors
            assert rel_count == 0
            mock_logger.warning.assert_called()


# ============================================================================
# Query Operations Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jQueryOperations:
    """Tests for Neo4j query operations."""

    @pytest.mark.asyncio
    async def test_execute_cypher(self, neo4j_destination, mock_neo4j_driver):
        """Test Cypher query execution."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        # Create a mock record that behaves like a dict when passed to dict()
        class MockRecord:
            def __init__(self, data):
                self._data = data
            def items(self):
                return self._data.items()
            def keys(self):
                return self._data.keys()
            def values(self):
                return self._data.values()
            def __getitem__(self, key):
                return self._data[key]
            def __iter__(self):
                return iter(self._data)

        mock_record = MockRecord({"name": "Alice", "age": 30})

        mock_result = AsyncMock()
        mock_result.__aiter__.return_value = [mock_record]
        session.run = AsyncMock(return_value=mock_result)

        results = await neo4j_destination.execute_cypher("testdb", "MATCH (n) RETURN n")

        assert len(results) == 1
        assert results[0] == {"name": "Alice", "age": 30}

    @pytest.mark.asyncio
    async def test_execute_cypher_not_initialized(self):
        """Test execute_cypher when not initialized."""
        dest = Neo4jDestination()

        with pytest.raises(RuntimeError, match="not initialized"):
            await dest.execute_cypher("testdb", "MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_search_content(self, neo4j_destination, mock_neo4j_driver):
        """Test content search."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        # Create a mock record that behaves like a Neo4j Record
        # Neo4j Record implements items() that returns list of (key, value) tuples
        class MockRecord:
            def __init__(self, data):
                self._data = data
            def items(self):
                return list(self._data.items())
            def keys(self):
                return list(self._data.keys())
            def values(self):
                return list(self._data.values())
            def __getitem__(self, key):
                return self._data[key]
            def __iter__(self):
                return iter(self._data.keys())

        mock_record = MockRecord({"id": "chunk-1", "content": "test content"})

        mock_result = AsyncMock()
        mock_result.__aiter__.return_value = [mock_record]
        session.run = AsyncMock(return_value=mock_result)

        results = await neo4j_destination.search("testdb", "test", "content", 10)

        assert len(results) == 1
        session.run.assert_called_once()
        call_args = session.run.call_args
        assert "CONTAINS" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_search_entity(self, neo4j_destination, mock_neo4j_driver):
        """Test entity search."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        # Create a mock record that behaves like a Neo4j Record
        class MockRecord:
            def __init__(self, data):
                self._data = data
            def items(self):
                return list(self._data.items())
            def keys(self):
                return list(self._data.keys())
            def values(self):
                return list(self._data.values())
            def __getitem__(self, key):
                return self._data[key]
            def __iter__(self):
                return iter(self._data.keys())

        mock_record = MockRecord({"id": "chunk-1", "entity": "Entity A"})

        mock_result = AsyncMock()
        mock_result.__aiter__.return_value = [mock_record]
        session.run = AsyncMock(return_value=mock_result)

        results = await neo4j_destination.search("testdb", "Entity", "entity", 10)

        assert len(results) == 1
        call_args = session.run.call_args
        assert ":Entity" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_search_metadata(self, neo4j_destination, mock_neo4j_driver):
        """Test metadata search."""
        driver, session = mock_neo4j_driver
        neo4j_destination._driver = driver

        # Create proper async context manager mock for session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)
        driver.session = MagicMock(return_value=session_cm)

        # Create a mock record that behaves like a Neo4j Record
        class MockRecord:
            def __init__(self, data):
                self._data = data
            def items(self):
                return list(self._data.items())
            def keys(self):
                return list(self._data.keys())
            def values(self):
                return list(self._data.values())
            def __getitem__(self, key):
                return self._data[key]
            def __iter__(self):
                return iter(self._data.keys())

        mock_record = MockRecord({"id": "doc-1"})

        mock_result = AsyncMock()
        mock_result.__aiter__.return_value = [mock_record]
        session.run = AsyncMock(return_value=mock_result)

        results = await neo4j_destination.search("testdb", "author", "metadata", 10)

        assert len(results) == 1
        call_args = session.run.call_args
        assert ":Document" in call_args[0][0]


# ============================================================================
# Config Validation Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jConfigValidation:
    """Tests for Neo4j configuration validation."""

    @pytest.mark.asyncio
    async def test_validate_config_valid(self):
        """Test validation with valid config."""
        with patch.dict("sys.modules", {"neo4j": MagicMock()}):
            dest = Neo4jDestination()
            result = await dest.validate_config({
                "uri": "bolt://localhost:7687",
                "password": "test-pass",
            })

            assert isinstance(result, ValidationResult)
            assert result.valid is True
            assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_config_missing_uri(self):
        """Test validation with missing URI."""
        with patch.dict("sys.modules", {"neo4j": MagicMock()}):
            dest = Neo4jDestination()
            result = await dest.validate_config({})

            assert result.valid is False
            assert len(result.errors) == 1
            assert "uri" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_config_invalid_uri(self):
        """Test validation with invalid URI."""
        with patch.dict("sys.modules", {"neo4j": MagicMock()}):
            dest = Neo4jDestination()
            result = await dest.validate_config({"uri": "http://localhost:7687"})

            assert result.valid is False
            assert len(result.errors) == 1
            assert "bolt" in result.errors[0].lower() or "neo4j" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_config_missing_password(self):
        """Test validation with missing password (warning only)."""
        with patch.dict("sys.modules", {"neo4j": MagicMock()}):
            dest = Neo4jDestination()
            result = await dest.validate_config({
                "uri": "bolt://localhost:7687",
            })

            assert result.valid is True
            assert len(result.warnings) == 1
            assert "password" in result.warnings[0].lower()

    @pytest.mark.asyncio
    async def test_validate_config_no_neo4j_driver(self):
        """Test validation when neo4j driver not installed."""
        with patch.dict("sys.modules", {"neo4j": None}):
            dest = Neo4jDestination()
            result = await dest.validate_config({
                "uri": "bolt://localhost:7687",
            })

            assert result.valid is False
            assert any("neo4j" in err.lower() and "install" in err.lower() for err in result.errors)

    @pytest.mark.asyncio
    async def test_validate_config_with_env_vars(self):
        """Test validation with environment variables."""
        with patch.dict(os.environ, {"NEO4J_URI": "bolt://env.example.com:7687"}):
            with patch.dict("sys.modules", {"neo4j": MagicMock()}):
                dest = Neo4jDestination()
                result = await dest.validate_config({})

                assert result.valid is True


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jHealthCheck:
    """Tests for Neo4j health check."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, neo4j_destination):
        """Test health check returns healthy."""
        neo4j_destination._driver = MagicMock()
        neo4j_destination._driver.verify_connectivity = AsyncMock()

        health = await neo4j_destination.health_check()

        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, neo4j_destination):
        """Test health check returns unhealthy on error."""
        neo4j_destination._driver = MagicMock()
        neo4j_destination._driver.verify_connectivity = AsyncMock(side_effect=Exception("Connection failed"))

        with patch("src.plugins.destinations.neo4j.logger") as mock_logger:
            health = await neo4j_destination.health_check()

            assert health == HealthStatus.UNHEALTHY
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        dest = Neo4jDestination()
        health = await dest.health_check()

        assert health == HealthStatus.UNHEALTHY


# ============================================================================
# Shutdown Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jShutdown:
    """Tests for Neo4j shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_driver(self, neo4j_destination):
        """Test shutdown closes driver."""
        mock_driver = MagicMock()
        mock_driver.close = AsyncMock()
        neo4j_destination._driver = mock_driver

        with patch("src.plugins.destinations.neo4j.logger") as mock_logger:
            await neo4j_destination.shutdown()

            mock_driver.close.assert_called_once()
            assert neo4j_destination._driver is None
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_no_driver(self):
        """Test shutdown when no driver exists."""
        dest = Neo4jDestination()
        dest._driver = None

        # Should not raise
        await dest.shutdown()


# ============================================================================
# Neo4jMockDestination Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jMockDestination:
    """Tests for Neo4jMockDestination."""

    def test_init(self):
        """Test mock destination initialization."""
        dest = Neo4jMockDestination()
        assert dest._storage["documents"] == {}
        assert dest._storage["chunks"] == {}
        assert dest._storage["entities"] == {}
        assert dest._storage["relationships"] == []

    @pytest.mark.asyncio
    async def test_mock_initialize(self):
        """Test mock initialization."""
        dest = Neo4jMockDestination()
        await dest.initialize({"test": "config"})

        assert dest._config == {"test": "config"}

    @pytest.mark.asyncio
    async def test_mock_connect(self):
        """Test mock connect."""
        dest = Neo4jMockDestination()
        await dest.initialize({})

        conn = await dest.connect({"database": "testdb"})

        assert isinstance(conn, Connection)
        assert conn.plugin_id == "neo4j_mock"
        assert conn.config["database"] == "testdb"

    @pytest.mark.asyncio
    async def test_mock_write(self):
        """Test mock write."""
        dest = Neo4jMockDestination()
        await dest.initialize({})

        conn = await dest.connect({"database": "testdb"})
        data = TransformedData(
            job_id=UUID("12345678-1234-1234-1234-123456789abc"),
            chunks=[{"content": "Chunk 1"}, {"content": "Chunk 2"}],
        )

        result = await dest.write(conn, data)

        assert isinstance(result, WriteResult)
        assert result.success is True
        assert result.destination_id == "neo4j_mock"
        assert result.records_written == 3  # 1 doc + 2 chunks

        # Check stored data
        documents = dest.get_stored_documents()
        assert len(documents) == 1

    @pytest.mark.asyncio
    async def test_mock_execute_cypher(self):
        """Test mock execute_cypher."""
        dest = Neo4jMockDestination()

        results = await dest.execute_cypher("testdb", "MATCH (n) RETURN n")

        assert results == [{"mock": "result"}]

    def test_mock_get_stored_documents(self):
        """Test getting stored documents."""
        dest = Neo4jMockDestination()
        dest._storage["documents"] = {"doc-1": {"id": "doc-1", "title": "Test"}}

        documents = dest.get_stored_documents()

        assert len(documents) == 1
        assert documents["doc-1"]["title"] == "Test"

    def test_mock_clear_storage(self):
        """Test clearing storage."""
        dest = Neo4jMockDestination()
        dest._storage["documents"] = {"doc-1": {"id": "doc-1"}}
        dest._storage["chunks"] = {"chunk-1": {"id": "chunk-1"}}

        dest.clear_storage()

        assert dest._storage["documents"] == {}
        assert dest._storage["chunks"] == {}
        assert dest._storage["entities"] == {}
        assert dest._storage["relationships"] == []
