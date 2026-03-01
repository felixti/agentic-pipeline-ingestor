"""Neo4j async client for graph database operations.

This module provides a high-level async client for Neo4j operations,
using the official neo4j Python driver's async API.
"""

import os
from typing import Any

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from src.observability.logging import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """Async Neo4j client for graph database operations.
    
    This client provides a simple interface for executing Cypher queries
    against a Neo4j database using the async driver.
    
    Example:
        >>> client = Neo4jClient(
        ...     uri="bolt://neo4j:7687",
        ...     user="neo4j",
        ...     password="cognee-graph-db"
        ... )
        >>> await client.connect()
        >>> result = await client.execute_query(
        ...     "MATCH (n) RETURN count(n) as count"
        ... )
        >>> await client.close()
    
    Attributes:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        _driver: Internal Neo4j async driver instance
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        """Initialize the Neo4j client.
        
        Args:
            uri: Neo4j connection URI (default: NEO4J_URI env var or bolt://neo4j:7687)
            user: Neo4j username (default: NEO4J_USER env var or neo4j)
            password: Neo4j password (default: NEO4J_PASSWORD env var or cognee-graph-db)
        """
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "cognee-graph-db")
        self._driver: AsyncGraphDatabase | None = None

        logger.debug(
            "neo4j_client_initialized",
            uri=self._uri,
            user=self._user,
        )

    @property
    def uri(self) -> str:
        """Return the Neo4j connection URI."""
        return self._uri

    @property
    def user(self) -> str:
        """Return the Neo4j username."""
        return self._user

    @property
    def is_connected(self) -> bool:
        """Check if the client has an active driver connection."""
        return self._driver is not None

    async def connect(self) -> None:
        """Create async driver connection to Neo4j.
        
        Raises:
            ConnectionError: If connection to Neo4j fails
            AuthError: If authentication fails
        
        Example:
            >>> client = Neo4jClient()
            >>> await client.connect()
        """
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            
            # Verify connectivity
            await self._driver.verify_connectivity()
            
            logger.info(
                "neo4j_connected",
                uri=self._uri,
                user=self._user,
            )
        
        except AuthError as e:
            logger.error(
                "neo4j_auth_failed",
                uri=self._uri,
                user=self._user,
                error=str(e),
            )
            raise ConnectionError(f"Neo4j authentication failed: {e}") from e
        
        except ServiceUnavailable as e:
            logger.error(
                "neo4j_service_unavailable",
                uri=self._uri,
                error=str(e),
            )
            raise ConnectionError(f"Neo4j service unavailable: {e}") from e
        
        except Exception as e:
            logger.error(
                "neo4j_connection_failed",
                uri=self._uri,
                error=str(e),
            )
            raise ConnectionError(f"Failed to connect to Neo4j: {e}") from e

    async def close(self) -> None:
        """Close the Neo4j driver connection.
        
        This method safely closes the driver connection. It is safe to call
        even if the client was never connected.
        
        Example:
            >>> await client.close()
        """
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j_disconnected")

    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query.
        
        Args:
            query: Cypher query string
            params: Query parameters (optional)
            database: Database name (default: neo4j)
        
        Returns:
            List of records as dictionaries
        
        Raises:
            RuntimeError: If the client is not connected
            Exception: If query execution fails
        
        Example:
            >>> result = await client.execute_query(
            ...     "MATCH (n:Person {name: $name}) RETURN n",
            ...     {"name": "Alice"}
            ... )
        """
        if not self._driver:
            raise RuntimeError("Neo4j client not connected. Call connect() first.")

        params = params or {}
        
        try:
            async with self._driver.session(database=database) as session:
                result = await session.run(query, params)
                records = await result.data()
                
                logger.debug(
                    "neo4j_query_executed",
                    query=query[:100],  # Log first 100 chars for debugging
                    param_keys=list(params.keys()),
                    record_count=len(records),
                    database=database,
                )
                
                return records
        
        except Exception as e:
            logger.error(
                "neo4j_query_failed",
                query=query[:100],
                error=str(e),
            )
            raise

    async def execute_write(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> list[dict[str, Any]]:
        """Execute a write Cypher query within a transaction.
        
        Args:
            query: Cypher query string
            params: Query parameters (optional)
            database: Database name (default: neo4j)
        
        Returns:
            List of records as dictionaries
        
        Raises:
            RuntimeError: If the client is not connected
            Exception: If query execution fails
        
        Example:
            >>> result = await client.execute_write(
            ...     "CREATE (n:Person {name: $name}) RETURN n",
            ...     {"name": "Alice"}
            ... )
        """
        if not self._driver:
            raise RuntimeError("Neo4j client not connected. Call connect() first.")

        params = params or {}
        
        try:
            async with self._driver.session(database=database) as session:
                async with session.begin_transaction() as tx:
                    result = await tx.run(query, params)
                    records = await result.data()
                    await tx.commit()
                    
                    logger.debug(
                        "neo4j_write_executed",
                        query=query[:100],
                        param_keys=list(params.keys()),
                        record_count=len(records),
                        database=database,
                    )
                    
                    return records
        
        except Exception as e:
            logger.error(
                "neo4j_write_failed",
                query=query[:100],
                error=str(e),
            )
            raise

    async def health_check(self) -> bool:
        """Verify Neo4j connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        
        Example:
            >>> is_healthy = await client.health_check()
            >>> if not is_healthy:
            ...     await client.connect()
        """
        if not self._driver:
            logger.warning("neo4j_health_check_no_driver")
            return False

        try:
            await self._driver.verify_connectivity()
            
            # Try a simple query to ensure full connectivity
            async with self._driver.session() as session:
                result = await session.run("RETURN 1 as health")
                record = await result.single()
                if record and record.get("health") == 1:
                    logger.debug("neo4j_health_check_passed")
                    return True
                return False
        
        except Exception as e:
            logger.warning(
                "neo4j_health_check_failed",
                error=str(e),
            )
            return False

    async def get_database_info(self) -> dict[str, Any]:
        """Get Neo4j database information.
        
        Returns:
            Dictionary with database information
        
        Raises:
            RuntimeError: If the client is not connected
        
        Example:
            >>> info = await client.get_database_info()
            >>> print(info["version"])
        """
        if not self._driver:
            raise RuntimeError("Neo4j client not connected. Call connect() first.")

        query = """
        CALL dbms.components() 
        YIELD name, versions, edition
        RETURN name, versions[0] as version, edition
        """
        
        try:
            async with self._driver.session() as session:
                result = await session.run(query)
                record = await result.single()
                
                if record:
                    return {
                        "name": record.get("name"),
                        "version": record.get("version"),
                        "edition": record.get("edition"),
                        "uri": self._uri,
                    }
                return {"error": "No database info returned"}
        
        except Exception as e:
            logger.error(
                "neo4j_database_info_failed",
                error=str(e),
            )
            raise


# Global client instance for singleton pattern
_global_client: Neo4jClient | None = None


async def get_neo4j_client(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    force_new: bool = False,
) -> Neo4jClient:
    """Get or create a Neo4j client instance.
    
    This function provides a singleton pattern for the Neo4j client.
    If a global client exists and is connected, it returns that instance.
    Otherwise, it creates a new client and connects it.
    
    Args:
        uri: Neo4j connection URI (optional, uses env var or default)
        user: Neo4j username (optional, uses env var or default)
        password: Neo4j password (optional, uses env var or default)
        force_new: If True, always create a new client instance
    
    Returns:
        Connected Neo4jClient instance
    
    Example:
        >>> client = await get_neo4j_client()
        >>> result = await client.execute_query("MATCH (n) RETURN count(n)")
    """
    global _global_client
    
    if not force_new and _global_client is not None and _global_client.is_connected:
        logger.debug("neo4j_client_reusing_existing")
        return _global_client
    
    client = Neo4jClient(uri=uri, user=user, password=password)
    await client.connect()
    
    if not force_new:
        _global_client = client
    
    return client


async def close_neo4j_client() -> None:
    """Close the global Neo4j client instance.
    
    This function should be called during application shutdown to ensure
    proper cleanup of Neo4j connections.
    
    Example:
        >>> await close_neo4j_client()
    """
    global _global_client
    
    if _global_client is not None:
        await _global_client.close()
        _global_client = None
        logger.info("neo4j_global_client_closed")
