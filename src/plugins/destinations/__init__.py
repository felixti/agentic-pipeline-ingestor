"""Destination plugins package.

This package contains destination plugins for output routing.
"""

from src.plugins.destinations.cognee import CogneeDestination, CogneeMockDestination
from src.plugins.destinations.graphrag import GraphRAGDestination, GraphRAGMockDestination
from src.plugins.destinations.neo4j import Neo4jDestination, Neo4jMockDestination
from src.plugins.destinations.pinecone import PineconeDestination, PineconeMockDestination
from src.plugins.destinations.weaviate import WeaviateDestination, WeaviateMockDestination

__all__ = [
    "CogneeDestination",
    "CogneeMockDestination",
    "GraphRAGDestination",
    "GraphRAGMockDestination",
    "Neo4jDestination",
    "Neo4jMockDestination",
    "PineconeDestination",
    "PineconeMockDestination",
    "WeaviateDestination",
    "WeaviateMockDestination",
]
