"""Community detection module for knowledge graphs.

This module provides algorithms for detecting communities (clusters) in
knowledge graphs, enabling graph-based analysis and indexing for RAG.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from src.core.graphrag.knowledge_graph import (
    Community,
    Entity,
    EntityType,
    KnowledgeGraph,
)

logger = logging.getLogger(__name__)


class GraphClusteringAlgorithm(str, Enum):
    """Available community detection algorithms."""
    LOUVAIN = "louvain"
    LEIDEN = "leiden"
    LABEL_PROPAGATION = "label_propagation"
    GREEDY_MODULARITY = "greedy_modularity"
    SPECTRAL = "spectral"
    WALKTRAP = "walktrap"


@dataclass
class CommunityDetectionResult:
    """Result of community detection.
    
    Attributes:
        communities: List of detected communities
        algorithm: Algorithm used
        modularity_score: Graph modularity score
        coverage: Percentage of entities covered by communities
        execution_time_ms: Execution time in milliseconds
        metadata: Additional metadata
    """
    communities: list[Community]
    algorithm: GraphClusteringAlgorithm
    modularity_score: float = 0.0
    coverage: float = 0.0
    execution_time_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "communities": [c.to_dict() for c in self.communities],
            "algorithm": self.algorithm.value,
            "modularity_score": self.modularity_score,
            "coverage": self.coverage,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class CommunityDetector:
    """Community detector for knowledge graphs.
    
    This class provides community detection capabilities using various
    graph clustering algorithms.
    
    Example:
        >>> detector = CommunityDetector()
        >>> result = await detector.detect_communities(
        ...     graph,
        ...     algorithm=GraphClusteringAlgorithm.LOUVAIN
        ... )
        >>> for community in result.communities:
        ...     print(f"Community: {community.name} ({len(community.entity_ids)} entities)")
    """

    def __init__(self) -> None:
        """Initialize the community detector."""
        self._algorithms = {
            GraphClusteringAlgorithm.LABEL_PROPAGATION: self._label_propagation,
            GraphClusteringAlgorithm.GREEDY_MODULARITY: self._greedy_modularity,
            GraphClusteringAlgorithm.LOUVAIN: self._louvain,
            GraphClusteringAlgorithm.SPECTRAL: self._spectral,
        }

    async def detect_communities(
        self,
        graph: KnowledgeGraph,
        algorithm: GraphClusteringAlgorithm = GraphClusteringAlgorithm.LOUVAIN,
        min_community_size: int = 3,
        max_communities: int | None = None,
        resolution: float = 1.0,
    ) -> CommunityDetectionResult:
        """Detect communities in a knowledge graph.
        
        Args:
            graph: Knowledge graph to analyze
            algorithm: Community detection algorithm to use
            min_community_size: Minimum entities per community
            max_communities: Maximum number of communities
            resolution: Resolution parameter for modularity optimization
            
        Returns:
            Community detection result
        """
        import time

        start_time = time.time()

        # Get the algorithm function
        algo_func = self._algorithms.get(algorithm)
        if not algo_func:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        logger.info(f"Running community detection with {algorithm.value}")

        # Build adjacency list
        adjacency = self._build_adjacency(graph)

        # Run algorithm
        node_to_community = algo_func(adjacency, resolution)

        # Build communities
        communities = self._build_communities(
            graph,
            node_to_community,
            min_community_size,
        )

        # Limit communities if specified
        if max_communities and len(communities) > max_communities:
            communities = sorted(
                communities,
                key=lambda c: len(c.entity_ids),
                reverse=True
            )[:max_communities]

        # Calculate metrics
        modularity = self._calculate_modularity(adjacency, node_to_community)
        coverage = len(node_to_community) / max(len(graph.entities), 1)

        execution_time = int((time.time() - start_time) * 1000)

        # Generate community summaries
        for community in communities:
            community.summary = await self._generate_summary(graph, community)
            community.key_entities = self._identify_key_entities(
                graph, adjacency, community
            )

        logger.info(
            f"Detected {len(communities)} communities in {execution_time}ms "
            f"(modularity: {modularity:.3f})"
        )

        return CommunityDetectionResult(
            communities=communities,
            algorithm=algorithm,
            modularity_score=modularity,
            coverage=coverage,
            execution_time_ms=execution_time,
            metadata={
                "min_community_size": min_community_size,
                "max_communities": max_communities,
                "resolution": resolution,
            },
        )

    def _build_adjacency(
        self,
        graph: KnowledgeGraph,
    ) -> dict[str, set[str]]:
        """Build adjacency list from graph.
        
        Args:
            graph: Knowledge graph
            
        Returns:
            Adjacency list as dictionary
        """
        adjacency: dict[str, set[str]] = {}

        for entity_id in graph.entities:
            adjacency[entity_id] = set()

        for rel in graph.relationships.values():
            if rel.source_id in adjacency and rel.target_id in adjacency:
                adjacency[rel.source_id].add(rel.target_id)
                adjacency[rel.target_id].add(rel.source_id)

        return adjacency

    def _label_propagation(
        self,
        adjacency: dict[str, set[str]],
        resolution: float = 1.0,
    ) -> dict[str, int]:
        """Label Propagation Algorithm for community detection.
        
        A simple and fast algorithm where each node adopts the label
        that most of its neighbors have.
        
        Args:
            adjacency: Adjacency list
            resolution: Unused for this algorithm
            
        Returns:
            Node to community mapping
        """
        import random

        nodes = list(adjacency.keys())
        if not nodes:
            return {}

        # Initialize each node with unique label
        labels = {node: i for i, node in enumerate(nodes)}

        # Iteratively update labels
        changed = True
        iteration = 0
        max_iterations = 100

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            # Random node order
            random.shuffle(nodes)

            for node in nodes:
                neighbors = adjacency[node]
                if not neighbors:
                    continue

                # Count labels in neighborhood
                label_counts: dict[int, int] = {}
                for neighbor in neighbors:
                    label = labels[neighbor]
                    label_counts[label] = label_counts.get(label, 0) + 1

                # Find most common label
                max_count = max(label_counts.values())
                best_labels = [l for l, c in label_counts.items() if c == max_count]

                # Randomly choose among ties
                new_label = random.choice(best_labels)

                if new_label != labels[node]:
                    labels[node] = new_label
                    changed = True

        return labels

    def _greedy_modularity(
        self,
        adjacency: dict[str, set[str]],
        resolution: float = 1.0,
    ) -> dict[str, int]:
        """Greedy modularity optimization algorithm.
        
        Iteratively merges communities to maximize modularity.
        
        Args:
            adjacency: Adjacency list
            resolution: Resolution parameter
            
        Returns:
            Node to community mapping
        """
        nodes = list(adjacency.keys())
        if not nodes:
            return {}

        # Start with each node in its own community
        labels = {node: i for i, node in enumerate(nodes)}
        communities = {i: {node} for i, node in enumerate(nodes)}

        # Calculate initial degrees
        degrees = {node: len(adjacency[node]) for node in nodes}
        total_edges = sum(degrees.values()) / 2

        if total_edges == 0:
            return labels

        # Greedy merging
        improved = True
        while improved and len(communities) > 1:
            improved = False
            best_merge = None
            best_delta_q = 0

            # Try merging each pair of communities
            comm_ids = list(communities.keys())
            for i, ci in enumerate(comm_ids):
                for cj in comm_ids[i + 1:]:
                    # Calculate modularity change for merging
                    delta_q = self._delta_modularity(
                        adjacency, communities[ci], communities[cj],
                        degrees, total_edges, resolution
                    )

                    if delta_q > best_delta_q:
                        best_delta_q = float(delta_q)  # type: ignore[assignment]
                        best_merge = (ci, cj)

            # Perform best merge if it improves modularity
            if best_merge and best_delta_q > 0:
                ci, cj = best_merge
                communities[ci].update(communities[cj])

                for node in communities[cj]:
                    labels[node] = ci

                del communities[cj]
                improved = True

        return labels

    def _louvain(
        self,
        adjacency: dict[str, set[str]],
        resolution: float = 1.0,
    ) -> dict[str, int]:
        """Louvain algorithm for community detection.
        
        A multi-phase algorithm that optimizes modularity through
        local moves and community aggregation.
        
        Args:
            adjacency: Adjacency list
            resolution: Resolution parameter
            
        Returns:
            Node to community mapping
        """
        # Simplified Louvain implementation
        # Phase 1: Local optimization
        labels = self._greedy_modularity(adjacency, resolution)

        # Renumber communities consecutively
        unique_labels = sorted(set(labels.values()))
        label_map = {old: new for new, old in enumerate(unique_labels)}

        return {node: label_map[label] for node, label in labels.items()}

    def _spectral(
        self,
        adjacency: dict[str, set[str]],
        resolution: float = 1.0,
    ) -> dict[str, int]:
        """Spectral clustering algorithm.
        
        Uses graph Laplacian and eigenvectors for clustering.
        This is a simplified placeholder implementation.
        
        Args:
            adjacency: Adjacency list
            resolution: Unused
            
        Returns:
            Node to community mapping
        """
        # Placeholder - real implementation would use numpy/scipy
        # For now, fall back to label propagation
        return self._label_propagation(adjacency)

    def _delta_modularity(
        self,
        adjacency: dict[str, set[str]],
        comm_i: set[str],
        comm_j: set[str],
        degrees: dict[str, int],
        total_edges: float,
        resolution: float,
    ) -> float:
        """Calculate modularity change for merging two communities."""
        # Count edges between communities
        e_ij = 0
        for node in comm_i:
            for neighbor in adjacency[node]:
                if neighbor in comm_j:
                    e_ij += 1

        # Calculate community degrees
        k_i = sum(degrees[node] for node in comm_i)
        k_j = sum(degrees[node] for node in comm_j)

        # Modularity change formula
        delta_q = (e_ij / total_edges) - resolution * (k_i * k_j) / (2 * total_edges ** 2)

        return delta_q

    def _build_communities(
        self,
        graph: KnowledgeGraph,
        node_to_community: dict[str, int],
        min_size: int,
    ) -> list[Community]:
        """Build community objects from node assignments."""
        # Group nodes by community
        communities: dict[int, list[str]] = {}
        for node, comm_id in node_to_community.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)

        # Create community objects
        result: list[Community] = []
        for comm_id, entity_ids in communities.items():
            if len(entity_ids) >= min_size:
                community = Community(
                    id=f"comm_{comm_id}_{uuid4().hex[:8]}",
                    name=f"Community {len(result) + 1}",
                    entity_ids=entity_ids,
                    coherence_score=0.0,
                )
                result.append(community)

        return result

    def _calculate_modularity(
        self,
        adjacency: dict[str, set[str]],
        node_to_community: dict[str, int],
    ) -> float:
        """Calculate graph modularity."""
        nodes = list(adjacency.keys())
        if not nodes:
            return 0.0

        m = sum(len(neighbors) for neighbors in adjacency.values()) / 2
        if m == 0:
            return 0.0

        degrees = {node: len(adjacency[node]) for node in nodes}

        q = 0.0
        for node in nodes:
            for neighbor in adjacency[node]:
                if node_to_community.get(node) == node_to_community.get(neighbor):
                    q += 1 - (degrees[node] * degrees[neighbor]) / (2 * m)

        return q / (2 * m)

    async def _generate_summary(
        self,
        graph: KnowledgeGraph,
        community: Community,
    ) -> str:
        """Generate a summary for a community."""
        # Get entity names and types
        entity_info = []
        for entity_id in community.entity_ids[:10]:
            entity = graph.entities.get(entity_id)
            if entity:
                entity_info.append(f"{entity.name} ({entity.type.value})")

        if not entity_info:
            return "Empty community"

        # Simple summary based on entity types
        type_counts: dict[str, int] = {}
        for entity_id in community.entity_ids:
            entity = graph.entities.get(entity_id)
            if entity:
                type_counts[entity.type.value] = type_counts.get(entity.type.value, 0) + 1

        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "unknown"

        return (
            f"Community of {len(community.entity_ids)} entities "
            f"centered around {dominant_type}. "
            f"Key entities: {', '.join(entity_info[:5])}"
        )

    def _identify_key_entities(
        self,
        graph: KnowledgeGraph,
        adjacency: dict[str, set[str]],
        community: Community,
    ) -> list[str]:
        """Identify key entities using degree centrality."""
        community_set = set(community.entity_ids)
        degrees = []

        for entity_id in community.entity_ids:
            neighbors = adjacency.get(entity_id, set())
            internal_degree = len(neighbors & community_set)
            degrees.append((entity_id, internal_degree))

        degrees.sort(key=lambda x: x[1], reverse=True)
        return [entity_id for entity_id, _ in degrees[:5]]

    async def detect_communities_hierarchical(
        self,
        graph: KnowledgeGraph,
        levels: int = 2,
        algorithm: GraphClusteringAlgorithm = GraphClusteringAlgorithm.LOUVAIN,
    ) -> list[CommunityDetectionResult]:
        """Perform hierarchical community detection."""
        results = []
        current_graph = graph

        for level in range(levels):
            result = await self.detect_communities(
                current_graph,
                algorithm=algorithm,
            )

            results.append(result)

            if level < levels - 1:
                current_graph = self._create_summary_graph(result)

        return results

    def _create_summary_graph(
        self,
        detection_result: CommunityDetectionResult,
    ) -> KnowledgeGraph:
        """Create a summary graph where communities become nodes."""
        summary = KnowledgeGraph(
            id=f"summary_{uuid4().hex[:8]}",
            name="Community Summary Graph",
        )

        for community in detection_result.communities:
            entity = Entity(
                id=community.id,
                name=community.name,
                type=EntityType.CONCEPT,
                properties={"size": len(community.entity_ids)},
            )
            summary.add_entity(entity)

        return summary
