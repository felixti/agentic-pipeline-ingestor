"""Knowledge graph building module for GraphRAG.

This module provides knowledge graph construction capabilities including
entity extraction, relationship mapping, and graph storage.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    DOCUMENT = "document"
    CHUNK = "chunk"
    UNKNOWN = "unknown"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    WORKS_AT = "works_at"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    FOUNDED = "founded"
    KNOWS = "knows"
    MENTIONS = "mentions"
    RELATED_TO = "related_to"
    CONTAINS = "contains"
    DERIVED_FROM = "derived_from"
    APPEARS_IN = "appears_in"
    CUSTOM = "custom"


@dataclass
class Entity:
    """Represents an entity in the knowledge graph.
    
    Attributes:
        id: Unique entity identifier
        name: Entity name/text
        type: Entity type
        properties: Additional entity properties
        source_chunks: IDs of chunks where entity was found
        confidence: Extraction confidence (0.0 - 1.0)
        created_at: Creation timestamp
    """
    id: str
    name: str
    type: EntityType
    properties: dict[str, Any] = field(default_factory=dict)
    source_chunks: list[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "properties": self.properties,
            "source_chunks": self.source_chunks,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """Create entity from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            type=EntityType(data.get("type", "unknown")),
            properties=data.get("properties", {}),
            source_chunks=data.get("source_chunks", []),
            confidence=data.get("confidence", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
        )


@dataclass
class Relationship:
    """Represents a relationship between entities.
    
    Attributes:
        id: Unique relationship identifier
        source_id: Source entity ID
        target_id: Target entity ID
        type: Relationship type
        properties: Additional relationship properties
        confidence: Extraction confidence (0.0 - 1.0)
        created_at: Creation timestamp
    """
    id: str
    source_id: str
    target_id: str
    type: RelationshipType
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Relationship):
            return self.id == other.id
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relationship":
        """Create relationship from dictionary."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=RelationshipType(data.get("type", "custom")),
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
        )


@dataclass
class Community:
    """Represents a community (cluster) of entities in the graph.
    
    Attributes:
        id: Unique community identifier
        name: Community name/description
        entity_ids: IDs of entities in this community
        summary: Community summary/description
        key_entities: Key entities representing the community
        coherence_score: Community coherence score (0.0 - 1.0)
        created_at: Creation timestamp
    """
    id: str
    name: str
    entity_ids: list[str] = field(default_factory=list)
    summary: str | None = None
    key_entities: list[str] = field(default_factory=list)
    coherence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert community to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_ids": self.entity_ids,
            "summary": self.summary,
            "key_entities": self.key_entities,
            "coherence_score": self.coherence_score,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class KnowledgeGraph:
    """Represents a knowledge graph with entities and relationships.
    
    Attributes:
        id: Graph identifier
        name: Graph name
        entities: Dictionary of entities by ID
        relationships: Dictionary of relationships by ID
        communities: Dictionary of communities by ID
        metadata: Additional graph metadata
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    id: str
    name: str
    entities: dict[str, Entity] = field(default_factory=dict)
    relationships: dict[str, Relationship] = field(default_factory=dict)
    communities: dict[str, Community] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        self.updated_at = datetime.utcnow()

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph."""
        self.relationships[relationship.id] = relationship
        self.updated_at = datetime.utcnow()

    def add_community(self, community: Community) -> None:
        """Add a community to the graph."""
        self.communities[community.id] = community
        self.updated_at = datetime.utcnow()

    def get_entity_neighbors(
        self,
        entity_id: str,
        relationship_type: RelationshipType | None = None,
    ) -> list[tuple[Entity, Relationship]]:
        """Get neighboring entities connected to the given entity.
        
        Args:
            entity_id: Entity ID
            relationship_type: Optional relationship type filter
            
        Returns:
            List of (entity, relationship) tuples
        """
        neighbors = []

        for rel in self.relationships.values():
            if rel.source_id == entity_id or rel.target_id == entity_id:
                if relationship_type and rel.type != relationship_type:
                    continue

                neighbor_id = rel.target_id if rel.source_id == entity_id else rel.source_id
                neighbor = self.entities.get(neighbor_id)
                if neighbor:
                    neighbors.append((neighbor, rel))

        return neighbors

    def get_subgraph(
        self,
        entity_ids: list[str],
        include_relationships: bool = True,
    ) -> "KnowledgeGraph":
        """Extract a subgraph containing specified entities.
        
        Args:
            entity_ids: IDs of entities to include
            include_relationships: Whether to include relationships between entities
            
        Returns:
            Subgraph containing specified entities
        """
        subgraph = KnowledgeGraph(
            id=f"{self.id}_subgraph_{uuid4().hex[:8]}",
            name=f"{self.name} (subgraph)",
        )

        entity_id_set = set(entity_ids)

        # Add entities
        for entity_id in entity_ids:
            if entity_id in self.entities:
                subgraph.add_entity(self.entities[entity_id])

        # Add relationships if both endpoints are in subgraph
        if include_relationships:
            for rel in self.relationships.values():
                if rel.source_id in entity_id_set and rel.target_id in entity_id_set:
                    subgraph.add_relationship(rel)

        return subgraph

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "community_count": len(self.communities),
            "entity_types": {
                entity_type.value: sum(1 for e in self.entities.values() if e.type == entity_type)
                for entity_type in EntityType
            },
            "relationship_types": {
                rel_type.value: sum(1 for r in self.relationships.values() if r.type == rel_type)
                for rel_type in RelationshipType
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relationships": {k: v.to_dict() for k, v in self.relationships.items()},
            "communities": {k: v.to_dict() for k, v in self.communities.items()},
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class KnowledgeGraphBuilder:
    """Builder for constructing knowledge graphs from documents.
    
    This class provides methods for building knowledge graphs from
    processed documents, including entity extraction and relationship
    detection.
    
    Example:
        >>> builder = KnowledgeGraphBuilder()
        >>> graph = await builder.build_from_document(
        ...     document_id="doc123",
        ...     chunks=[{"content": "Apple Inc. was founded by Steve Jobs."}]
        ... )
    """

    def __init__(self, entity_extractor=None):
        """Initialize the graph builder.
        
        Args:
            entity_extractor: Optional entity extractor to use
        """
        self._entity_extractor = entity_extractor

    async def build_from_document(
        self,
        document_id: str,
        chunks: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeGraph:
        """Build a knowledge graph from a document.
        
        Args:
            document_id: Document identifier
            chunks: List of document chunks
            metadata: Optional document metadata
            
        Returns:
            Constructed knowledge graph
        """
        graph = KnowledgeGraph(
            id=f"kg_{document_id}",
            name=f"Knowledge Graph for {document_id}",
            metadata={"source_document": document_id, **(metadata or {})},
        )

        # Create document entity
        doc_entity = Entity(
            id=f"doc_{document_id}",
            name=metadata.get("title", document_id) if metadata else document_id,
            type=EntityType.DOCUMENT,
            properties=metadata or {},
        )
        graph.add_entity(doc_entity)

        # Process each chunk
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{idx}"

            # Create chunk entity
            chunk_entity = Entity(
                id=chunk_id,
                name=f"Chunk {idx}",
                type=EntityType.CHUNK,
                properties={
                    "index": idx,
                    "content_preview": chunk.get("content", "")[:200],
                },
                source_chunks=[chunk_id],
            )
            graph.add_entity(chunk_entity)

            # Link chunk to document
            rel = Relationship(
                id=f"rel_{chunk_id}_part_of_doc",
                source_id=chunk_id,
                target_id=doc_entity.id,
                type=RelationshipType.PART_OF,
            )
            graph.add_relationship(rel)

            # Extract entities from chunk if extractor available
            if self._entity_extractor:
                entities = await self._extract_entities_from_chunk(
                    chunk, chunk_id
                )
                for entity in entities:
                    # Check if entity already exists (by name and type)
                    existing = self._find_existing_entity(graph, entity)
                    if existing:
                        # Merge source chunks
                        if chunk_id not in existing.source_chunks:
                            existing.source_chunks.append(chunk_id)
                        entity = existing
                    else:
                        graph.add_entity(entity)

                    # Link entity to chunk
                    rel = Relationship(
                        id=f"rel_{entity.id}_appears_in_{chunk_id}",
                        source_id=entity.id,
                        target_id=chunk_id,
                        type=RelationshipType.APPEARS_IN,
                        properties={"confidence": entity.confidence},
                    )
                    graph.add_relationship(rel)

        # Extract relationships between entities
        await self._extract_relationships(graph)

        return graph

    async def _extract_entities_from_chunk(
        self,
        chunk: dict[str, Any],
        chunk_id: str,
    ) -> list[Entity]:
        """Extract entities from a chunk.
        
        Args:
            chunk: Document chunk
            chunk_id: Chunk identifier
            
        Returns:
            List of extracted entities
        """
        content = chunk.get("content", "")

        if not self._entity_extractor or not content:
            return []

        try:
            result = await self._entity_extractor.extract_entities(content)

            entities = []
            for extracted in result.entities:
                entity_id = self._generate_entity_id(extracted.text, extracted.type.value)

                entity = Entity(
                    id=entity_id,
                    name=extracted.text,
                    type=EntityType(extracted.type.value.lower()) if extracted.type.value.lower() in [e.value for e in EntityType] else EntityType.UNKNOWN,
                    properties={"extracted_type": extracted.type.value},
                    source_chunks=[chunk_id],
                    confidence=extracted.confidence,
                )
                entities.append(entity)

            return entities

        except Exception as e:
            logger.warning(f"Entity extraction failed for chunk {chunk_id}: {e}")
            return []

    async def _extract_relationships(self, graph: KnowledgeGraph) -> None:
        """Extract relationships between entities in the graph.
        
        This is a simplified implementation. In a real scenario, this would
        use more sophisticated NLP techniques or LLM-based extraction.
        
        Args:
            graph: Knowledge graph to enhance with relationships
        """
        # For now, create simple co-occurrence relationships
        # Entities that appear in the same chunk are related

        chunk_entities: dict[str, list[str]] = {}

        for entity in graph.entities.values():
            for chunk_id in entity.source_chunks:
                if chunk_id not in chunk_entities:
                    chunk_entities[chunk_id] = []
                chunk_entities[chunk_id].append(entity.id)

        # Create relationships for co-occurring entities
        for chunk_id, entity_ids in chunk_entities.items():
            if len(entity_ids) > 1:
                for i, source_id in enumerate(entity_ids):
                    for target_id in entity_ids[i + 1:]:
                        if source_id != target_id:
                            rel_id = f"rel_{source_id}_{target_id}_cooccurs"
                            if rel_id not in graph.relationships:
                                rel = Relationship(
                                    id=rel_id,
                                    source_id=source_id,
                                    target_id=target_id,
                                    type=RelationshipType.RELATED_TO,
                                    properties={"basis": "co-occurrence", "chunk_id": chunk_id},
                                )
                                graph.add_relationship(rel)

    def _find_existing_entity(
        self,
        graph: KnowledgeGraph,
        new_entity: Entity,
    ) -> Entity | None:
        """Find an existing entity with the same name and type.
        
        Args:
            graph: Knowledge graph
            new_entity: Entity to find
            
        Returns:
            Existing entity or None
        """
        for entity in graph.entities.values():
            if (entity.name.lower() == new_entity.name.lower() and
                entity.type == new_entity.type):
                return entity
        return None

    @staticmethod
    def _generate_entity_id(name: str, entity_type: str) -> str:
        """Generate a unique entity ID from name and type.
        
        Args:
            name: Entity name
            entity_type: Entity type
            
        Returns:
            Unique entity ID
        """
        content = f"{name.lower()}:{entity_type.lower()}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"ent_{hash_val}"

    async def merge_graphs(
        self,
        graphs: list[KnowledgeGraph],
        name: str | None = None,
    ) -> KnowledgeGraph:
        """Merge multiple knowledge graphs into one.
        
        Args:
            graphs: List of graphs to merge
            name: Optional name for merged graph
            
        Returns:
            Merged knowledge graph
        """
        merged = KnowledgeGraph(
            id=f"merged_{uuid4().hex[:8]}",
            name=name or "Merged Knowledge Graph",
        )

        # Track entity name -> ID mappings for deduplication
        entity_name_map: dict[tuple[str, EntityType], str] = {}

        for graph in graphs:
            # Merge entities
            for entity in graph.entities.values():
                key = (entity.name.lower(), entity.type)

                if key in entity_name_map:
                    # Entity exists, merge properties
                    existing_id = entity_name_map[key]
                    existing = merged.entities[existing_id]
                    existing.source_chunks.extend(entity.source_chunks)
                    existing.properties.update(entity.properties)
                    existing.confidence = max(existing.confidence, entity.confidence)
                else:
                    # New entity
                    merged.add_entity(entity)
                    entity_name_map[key] = entity.id

            # Merge relationships
            for rel in graph.relationships.values():
                # Update relationship endpoints if entities were merged
                source_key = None
                target_key = None

                for entity in graph.entities.values():
                    if entity.id == rel.source_id:
                        source_key = (entity.name.lower(), entity.type)
                    if entity.id == rel.target_id:
                        target_key = (entity.name.lower(), entity.type)

                if source_key and target_key:
                    new_source = entity_name_map.get(source_key, rel.source_id)
                    new_target = entity_name_map.get(target_key, rel.target_id)

                    if new_source != rel.source_id or new_target != rel.target_id:
                        rel.source_id = new_source
                        rel.target_id = new_target
                        rel.id = f"rel_{new_source}_{new_target}_{rel.type.value}"

                if rel.id not in merged.relationships:
                    merged.add_relationship(rel)

        return merged
