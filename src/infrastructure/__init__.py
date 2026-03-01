"""Infrastructure module for external service integrations.

This module provides clients and utilities for connecting to external
services including databases, message queues, and graph databases.

Modules:
    neo4j: Neo4j graph database client
"""

from src.infrastructure.neo4j import (
    Neo4jClient,
    close_neo4j_client,
    get_neo4j_client,
)

__all__ = [
    "Neo4jClient",
    "get_neo4j_client",
    "close_neo4j_client",
]
