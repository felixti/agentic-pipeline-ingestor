"""GraphRAG integration module.

This module provides knowledge graph building and community detection
capabilities for the Agentic Data Pipeline Ingestor.
"""

from src.core.graphrag.knowledge_graph import (
    Community,
    Entity,
    KnowledgeGraph,
    KnowledgeGraphBuilder,
    Relationship,
)
from src.core.graphrag.community_detection import (
    CommunityDetectionResult,
    CommunityDetector,
    GraphClusteringAlgorithm,
)

__all__ = [
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
    "Entity",
    "Relationship",
    "Community",
    "CommunityDetector",
    "CommunityDetectionResult",
    "GraphClusteringAlgorithm",
]
