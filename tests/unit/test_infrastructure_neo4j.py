"""Unit tests for Neo4j client infrastructure.

Tests for Neo4jClient class and utility functions:
- get_neo4j_client()
- close_neo4j_client()
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j.exceptions import AuthError, ServiceUnavailable

from src.infrastructure.neo4j.client import (
    Neo4jClient,
    close_neo4j_client,
    get_neo4j_client,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j async driver."""
    driver = MagicMock()
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    
    # Mock session
    session = MagicMock()
    session.run = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    
    # Mock session context manager
    driver.session = MagicMock(return_value=session)
    
    return driver


@pytest.fixture
def reset_global_client():
    """Reset global client before and after test."""
    # Reset before test
    import src.infrastructure.neo4j.client as neo4j_module
    original_client = neo4j_module._global_client
    neo4j_module._global_client = None
    
    yield
    
    # Reset after test
    neo4j_module._global_client = original_client


# ============================================================================
# Neo4jClient Class Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jClient:
    """Tests for Neo4jClient class."""

    def test_init_default_values(self):
        """Test client initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            client = Neo4jClient()
            
            assert client.uri == "bolt://neo4j:7687"
            assert client.user == "neo4j"
            assert client._password == "cognee-graph-db"
            assert client._driver is None
            assert client.is_connected is False

    def test_init_with_env_vars(self):
        """Test client initialization with environment variables."""
        env_vars = {
            "NEO4J_URI": "bolt://custom:7687",
            "NEO4J_USER": "custom_user",
            "NEO4J_PASSWORD": "custom_pass",
        }
        with patch.dict(os.environ, env_vars):
            client = Neo4jClient()
            
            assert client.uri == "bolt://custom:7687"
            assert client.user == "custom_user"
            assert client._password == "custom_pass"

    def test_init_with_explicit_params(self):
        """Test client initialization with explicit parameters."""
        with patch.dict(os.environ, {}, clear=True):
            client = Neo4jClient(
                uri="bolt://explicit:7687",
                user="explicit_user",
                password="explicit_pass",
            )
            
            assert client.uri == "bolt://explicit:7687"
            assert client.user == "explicit_user"
            assert client._password == "explicit_pass"

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_neo4j_driver):
        """Test successful connection to Neo4j."""
        client = Neo4jClient()
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
        
        assert client.is_connected is True
        assert client._driver is mock_neo4j_driver
        mock_neo4j_driver.verify_connectivity.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_auth_error(self, mock_neo4j_driver):
        """Test connection fails with authentication error."""
        client = Neo4jClient()
        mock_neo4j_driver.verify_connectivity.side_effect = AuthError("Invalid credentials")
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            with pytest.raises(ConnectionError, match="authentication failed"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_connect_service_unavailable(self, mock_neo4j_driver):
        """Test connection fails when service is unavailable."""
        client = Neo4jClient()
        mock_neo4j_driver.verify_connectivity.side_effect = ServiceUnavailable("Service down")
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            with pytest.raises(ConnectionError, match="service unavailable"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_connect_generic_error(self, mock_neo4j_driver):
        """Test connection fails with generic error."""
        client = Neo4jClient()
        mock_neo4j_driver.verify_connectivity.side_effect = Exception("Unknown error")
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_close_connection(self, mock_neo4j_driver):
        """Test closing connection."""
        client = Neo4jClient()
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            assert client.is_connected is True
            
            await client.close()
            
            assert client.is_connected is False
            assert client._driver is None
            mock_neo4j_driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_connection(self):
        """Test closing connection when not connected."""
        client = Neo4jClient()
        
        # Should not raise any error
        await client.close()
        
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_execute_query_success(self, mock_neo4j_driver):
        """Test executing Cypher query."""
        client = Neo4jClient()
        
        # Mock result
        mock_result = MagicMock()
        mock_result.data = AsyncMock(return_value=[
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ])
        
        # Mock session run
        session = MagicMock()
        session.run = AsyncMock(return_value=mock_result)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_driver.session = MagicMock(return_value=session)
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            records = await client.execute_query(
                "MATCH (n:Person) RETURN n.name as name, n.age as age",
                {"limit": 10},
            )
        
        assert len(records) == 2
        assert records[0]["name"] == "Alice"
        assert records[1]["age"] == 25
        session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_not_connected(self):
        """Test query execution fails when not connected."""
        client = Neo4jClient()
        
        with pytest.raises(RuntimeError, match="not connected"):
            await client.execute_query("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_execute_query_error(self, mock_neo4j_driver):
        """Test query execution handles errors."""
        client = Neo4jClient()
        
        session = MagicMock()
        session.run = AsyncMock(side_effect=Exception("Query failed"))
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_driver.session = MagicMock(return_value=session)
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            
            with pytest.raises(Exception, match="Query failed"):
                await client.execute_query("INVALID QUERY")

    @pytest.mark.asyncio
    async def test_execute_write_success(self, mock_neo4j_driver):
        """Test executing write query within transaction."""
        client = Neo4jClient()
        
        # Mock transaction
        mock_tx = MagicMock()
        mock_tx.run = AsyncMock()
        mock_tx.commit = AsyncMock()
        
        mock_result = MagicMock()
        mock_result.data = AsyncMock(return_value=[{"id": "123"}])
        mock_tx.run.return_value = mock_result
        
        # Mock session with transaction
        session = MagicMock()
        session.begin_transaction = MagicMock(return_value=mock_tx)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_driver.session = MagicMock(return_value=session)
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            records = await client.execute_write(
                "CREATE (n:Person {name: $name}) RETURN id(n) as id",
                {"name": "Alice"},
            )
        
        assert len(records) == 1
        assert records[0]["id"] == "123"

    @pytest.mark.asyncio
    async def test_execute_write_not_connected(self):
        """Test write execution fails when not connected."""
        client = Neo4jClient()
        
        with pytest.raises(RuntimeError, match="not connected"):
            await client.execute_write("CREATE (n) RETURN n")

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_neo4j_driver):
        """Test health check returns True when healthy."""
        client = Neo4jClient()
        
        # Mock health query result
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value={"health": 1})
        
        session = MagicMock()
        session.run = AsyncMock(return_value=mock_result)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_driver.session = MagicMock(return_value=session)
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            is_healthy = await client.health_check()
        
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self):
        """Test health check returns False when not connected."""
        client = Neo4jClient()
        
        is_healthy = await client.health_check()
        
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_query_fails(self, mock_neo4j_driver):
        """Test health check returns False when query fails."""
        client = Neo4jClient()
        
        session = MagicMock()
        session.run = AsyncMock(side_effect=Exception("Query failed"))
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_driver.session = MagicMock(return_value=session)
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            is_healthy = await client.health_check()
        
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_wrong_result(self, mock_neo4j_driver):
        """Test health check returns False when result is unexpected."""
        client = Neo4jClient()
        
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value={"health": 0})
        
        session = MagicMock()
        session.run = AsyncMock(return_value=mock_result)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_driver.session = MagicMock(return_value=session)
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            is_healthy = await client.health_check()
        
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_get_database_info_success(self, mock_neo4j_driver):
        """Test getting database information."""
        client = Neo4jClient()
        
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value={
            "name": "Neo4j Kernel",
            "version": "5.15.0",
            "edition": "community",
        })
        
        session = MagicMock()
        session.run = AsyncMock(return_value=mock_result)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_driver.session = MagicMock(return_value=session)
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            info = await client.get_database_info()
        
        assert info["name"] == "Neo4j Kernel"
        assert info["version"] == "5.15.0"
        assert info["edition"] == "community"
        assert "uri" in info

    @pytest.mark.asyncio
    async def test_get_database_info_not_connected(self):
        """Test get_database_info fails when not connected."""
        client = Neo4jClient()
        
        with pytest.raises(RuntimeError, match="not connected"):
            await client.get_database_info()

    @pytest.mark.asyncio
    async def test_get_database_info_no_result(self, mock_neo4j_driver):
        """Test get_database_info handles no result."""
        client = Neo4jClient()
        
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value=None)
        
        session = MagicMock()
        session.run = AsyncMock(return_value=mock_result)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_driver.session = MagicMock(return_value=session)
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            info = await client.get_database_info()
        
        assert "error" in info


# ============================================================================
# Utility Function Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jUtilityFunctions:
    """Tests for get_neo4j_client and close_neo4j_client functions."""

    @pytest.mark.asyncio
    async def test_get_neo4j_client_creates_new(self, mock_neo4j_driver, reset_global_client):
        """Test get_neo4j_client creates new client."""
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            client = await get_neo4j_client()
        
        assert isinstance(client, Neo4jClient)
        assert client.is_connected is True

    @pytest.mark.asyncio
    async def test_get_neo4j_client_reuses_existing(self, mock_neo4j_driver, reset_global_client):
        """Test get_neo4j_client reuses existing connected client."""
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            client1 = await get_neo4j_client()
            client2 = await get_neo4j_client()
        
        # Should be the same instance
        assert client1 is client2

    @pytest.mark.asyncio
    async def test_get_neo4j_client_force_new(self, mock_neo4j_driver, reset_global_client):
        """Test get_neo4j_client with force_new creates new client."""
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            client1 = await get_neo4j_client()
            client2 = await get_neo4j_client(force_new=True)
        
        # Should be different instances
        assert client1 is not client2

    @pytest.mark.asyncio
    async def test_get_neo4j_client_with_custom_params(self, mock_neo4j_driver, reset_global_client):
        """Test get_neo4j_client with custom connection parameters."""
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            client = await get_neo4j_client(
                uri="bolt://custom:7687",
                user="custom_user",
                password="custom_pass",
            )
        
        assert client.uri == "bolt://custom:7687"
        assert client.user == "custom_user"

    @pytest.mark.asyncio
    async def test_close_neo4j_client_closes_global(self, mock_neo4j_driver, reset_global_client):
        """Test close_neo4j_client closes global client."""
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            client = await get_neo4j_client()
            assert client.is_connected is True
            
            await close_neo4j_client()
            
            # Global client should be reset
            import src.infrastructure.neo4j.client as neo4j_module
            assert neo4j_module._global_client is None

    @pytest.mark.asyncio
    async def test_close_neo4j_client_no_global(self, reset_global_client):
        """Test close_neo4j_client when no global client exists."""
        # Should not raise any error
        await close_neo4j_client()
        
        import src.infrastructure.neo4j.client as neo4j_module
        assert neo4j_module._global_client is None


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.unit
class TestNeo4jClientEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_execute_query_empty_params(self, mock_neo4j_driver):
        """Test query execution with empty params."""
        client = Neo4jClient()
        
        mock_result = MagicMock()
        mock_result.data = AsyncMock(return_value=[])
        
        session = MagicMock()
        session.run = AsyncMock(return_value=mock_result)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_driver.session = MagicMock(return_value=session)
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            # Should work with None params
            records = await client.execute_query("MATCH (n) RETURN n", None)
        
        assert records == []

    @pytest.mark.asyncio
    async def test_execute_query_with_database_param(self, mock_neo4j_driver):
        """Test query execution with specific database."""
        client = Neo4jClient()
        
        mock_result = MagicMock()
        mock_result.data = AsyncMock(return_value=[])
        
        session = MagicMock()
        session.run = AsyncMock(return_value=mock_result)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_driver.session = MagicMock(return_value=session)
        
        with patch(
            "src.infrastructure.neo4j.client.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await client.connect()
            await client.execute_query(
                "MATCH (n) RETURN n",
                database="custom_db",
            )
        
        # Verify session was created with custom database
        mock_neo4j_driver.session.assert_called_with(database="custom_db")

    def test_password_not_exposed(self):
        """Test that password is not exposed via properties."""
        client = Neo4jClient(password="secret_password")
        
        # Password should not be accessible via public properties
        assert not hasattr(client, "password")
        assert client._password == "secret_password"
