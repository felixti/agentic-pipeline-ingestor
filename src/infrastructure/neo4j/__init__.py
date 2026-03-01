"""Neo4j infrastructure module.

This module provides Neo4j graph database connectivity for the application,
including async client support for Cypher query execution.

Example:
    >>> from src.infrastructure.neo4j import get_neo4j_client, close_neo4j_client
    >>> client = await get_neo4j_client()
    >>> result = await client.execute_query("MATCH (n) RETURN count(n) as count")
    >>> await close_neo4j_client()
"""

from src.infrastructure.neo4j.client import (
    Neo4jClient,
    close_neo4j_client,
    get_neo4j_client,
)

__all__ = [
    "Neo4jClient",
    "get_neo4j_client",
    "close_neo4j_client",
]
