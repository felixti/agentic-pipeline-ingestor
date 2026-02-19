"""Entity extraction module using LLM for NER.

This module provides entity extraction capabilities using LLM models
to identify named entities in text content.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.llm.provider import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Standard entity types."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    PERCENT = "percent"
    PRODUCT = "product"
    EVENT = "event"
    WORK_OF_ART = "work_of_art"
    LAW = "law"
    LANGUAGE = "language"
    GPE = "gpe"  # Geopolitical Entity
    FAC = "fac"  # Facility
    NORP = "norp"  # Nationalities or religious or political groups
    CARDINAL = "cardinal"
    ORDINAL = "ordinal"
    QUANTITY = "quantity"
    CUSTOM = "custom"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"


@dataclass
class Entity:
    """Represents an extracted entity.
    
    Attributes:
        text: The entity text as it appears in the document
        type: The entity type
        start_pos: Start position in the text (optional)
        end_pos: End position in the text (optional)
        confidence: Confidence score (0.0 - 1.0)
        metadata: Additional entity metadata
    """
    text: str
    type: EntityType
    start_pos: int | None = None
    end_pos: int | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "text": self.text,
            "type": self.type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """Create entity from dictionary."""
        return cls(
            text=data["text"],
            type=EntityType(data.get("type", "custom")),
            start_pos=data.get("start_pos"),
            end_pos=data.get("end_pos"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: list[Entity] = field(default_factory=list)
    processing_time_ms: int = 0
    model_used: str = ""

    def get_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get entities of a specific type."""
        return [e for e in self.entities if e.type == entity_type]

    def get_unique(self) -> list[Entity]:
        """Get unique entities (by text and type)."""
        seen: set[tuple[str, EntityType]] = set()
        unique = []
        for entity in self.entities:
            key = (entity.text.lower(), entity.type)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique


class EntityExtractor:
    """Entity extractor using LLM for Named Entity Recognition.
    
    This class uses LLM models to extract entities from text,
    supporting multiple entity types with confidence scoring.
    
    Example:
        >>> extractor = EntityExtractor(llm_provider)
        >>> result = await extractor.extract_entities(
        ...     "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        ... )
        >>> for entity in result.entities:
        ...     print(f"{entity.text} ({entity.type.value})")
    """

    # Default entity types to extract
    DEFAULT_ENTITY_TYPES = [
        EntityType.PERSON,
        EntityType.ORGANIZATION,
        EntityType.LOCATION,
        EntityType.DATE,
        EntityType.GPE,
    ]

    def __init__(
        self,
        llm_provider: LLMProvider,
        entity_types: list[EntityType] | None = None,
        min_confidence: float = 0.7,
    ):
        """Initialize the entity extractor.
        
        Args:
            llm_provider: LLM provider for entity extraction
            entity_types: List of entity types to extract
            min_confidence: Minimum confidence threshold
        """
        self._llm = llm_provider
        self._entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self._min_confidence = min_confidence

    async def extract_entities(
        self,
        text: str,
        entity_types: list[EntityType] | None = None,
    ) -> ExtractionResult:
        """Extract entities from text.
        
        Args:
            text: Input text to analyze
            entity_types: Optional override for entity types to extract
            
        Returns:
            ExtractionResult with extracted entities
        """
        import time

        start_time = time.time()

        types_to_extract = entity_types or self._entity_types

        try:
            # Use LLM for entity extraction
            entities = await self._extract_with_llm(text, types_to_extract)

            # Filter by confidence
            entities = [e for e in entities if e.confidence >= self._min_confidence]

            processing_time = int((time.time() - start_time) * 1000)

            return ExtractionResult(
                entities=entities,
                processing_time_ms=processing_time,
                model_used=getattr(self._llm, "get_model_name", lambda: "unknown")(),
            )

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            return ExtractionResult(
                entities=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
                model_used="error",
            )

    async def extract_from_chunks(
        self,
        chunks: list[dict[str, Any]],
        entity_types: list[EntityType] | None = None,
    ) -> ExtractionResult:
        """Extract entities from multiple text chunks.
        
        Args:
            chunks: List of text chunks with 'content' field
            entity_types: Optional entity types to extract
            
        Returns:
            Combined extraction result
        """
        all_entities = []
        total_time = 0

        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                continue

            result = await self.extract_entities(content, entity_types)
            all_entities.extend(result.entities)
            total_time += result.processing_time_ms

        # Deduplicate entities
        unique_entities = self._deduplicate_entities(all_entities)

        return ExtractionResult(
            entities=unique_entities,
            processing_time_ms=total_time,
            model_used=getattr(self._llm, "get_model_name", lambda: "unknown")(),
        )

    async def _extract_with_llm(
        self,
        text: str,
        entity_types: list[EntityType],
    ) -> list[Entity]:
        """Extract entities using LLM.
        
        Args:
            text: Input text
            entity_types: Entity types to extract
            
        Returns:
            List of extracted entities
        """
        # Build the prompt
        type_list = ", ".join([t.value for t in entity_types])

        prompt = f"""Extract named entities from the following text.

Text:
{text[:4000]}  # Limit text length

Extract entities of these types: {type_list}

Return a JSON array of entities with this format:
[
  {{"text": "entity text", "type": "entity_type", "confidence": 0.95}}
]

Rules:
- Only extract entities that clearly match the specified types
- Provide confidence scores between 0.0 and 1.0
- Return an empty array if no entities are found
- Do not include explanations, only the JSON array
"""

        try:
            response = await self._llm.chat_completion(
                messages=[
                    ChatMessage.system("You are a precise named entity recognition system."),
                    ChatMessage.user(prompt),
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            # Parse the response
            content = response.content.strip()

            # Extract JSON from response
            entities_data = self._parse_json_response(content)

            # Convert to Entity objects
            entities = []
            for item in entities_data:
                try:
                    entity = Entity(
                        text=item["text"],
                        type=EntityType(item["type"].lower()),
                        confidence=item.get("confidence", 0.8),
                    )
                    entities.append(entity)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Invalid entity data: {item}, error: {e}")
                    continue

            return entities

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

    def _parse_json_response(self, content: str) -> list[dict[str, Any]]:
        """Parse JSON from LLM response.
        
        Args:
            content: LLM response content
            
        Returns:
            Parsed JSON data
        """
        # Try to find JSON array in response
        content = content.strip()

        # Remove markdown code blocks if present
        content = content.removeprefix("```json")
        content = content.removeprefix("```")
        content = content.removesuffix("```")

        content = content.strip()

        try:
            result: list[dict[str, Any]] = json.loads(content)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON array
            import re
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    return result
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse JSON from: {content[:200]}")
            return []

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """Deduplicate entities by text and type.
        
        Args:
            entities: List of entities
            
        Returns:
            Deduplicated list
        """
        seen: dict[tuple[str, EntityType], Entity] = {}

        for entity in entities:
            key = (entity.text.lower().strip(), entity.type)

            if key in seen:
                # Keep the one with higher confidence
                if entity.confidence > seen[key].confidence:
                    seen[key] = entity
            else:
                seen[key] = entity

        return list(seen.values())

    async def extract_relationships(
        self,
        text: str,
        entities: list[Entity] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract relationships between entities.
        
        Args:
            text: Input text
            entities: Optional pre-extracted entities
            
        Returns:
            List of relationships
        """
        if not entities:
            result = await self.extract_entities(text)
            entities = result.entities

        if len(entities) < 2:
            return []

        entity_text = "\n".join([
            f"- {e.text} ({e.type.value})"
            for e in entities[:20]  # Limit to 20 entities
        ])

        prompt = f"""Extract relationships between the following entities found in the text.

Entities:
{entity_text}

Text:
{text[:3000]}

Return a JSON array of relationships with this format:
[
  {{
    "subject": "entity text",
    "predicate": "relationship type",
    "object": "entity text",
    "confidence": 0.85
  }}
]

Common relationship types: works_at, founded, located_in, part_of, owns, knows, employs, acquired
"""

        try:
            response = await self._llm.chat_completion(
                messages=[
                    ChatMessage.system("You are a relationship extraction system."),
                    ChatMessage.user(prompt),
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            content = response.content.strip()
            relationships = self._parse_json_response(content)

            return relationships

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []


class RegexEntityExtractor:
    """Fallback entity extractor using regex patterns.
    
    This class provides basic entity extraction using regex patterns
    when LLM is unavailable or for simple cases.
    """

    # Regex patterns for entity types
    PATTERNS = {
        EntityType.EMAIL: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        EntityType.URL: r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*",
        EntityType.PHONE: r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        EntityType.DATE: r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        EntityType.MONEY: r"\$[\d,]+(?:\.\d{2})?|\b\d+\s*(?:USD|EUR|GBP)\b",
        EntityType.PERCENT: r"\b\d+(?:\.\d+)?%",
    }

    def __init__(self, min_confidence: float = 0.8):
        """Initialize the regex extractor.
        
        Args:
            min_confidence: Minimum confidence threshold
        """
        self._min_confidence = min_confidence
        import re
        self._compiled = {
            entity_type: re.compile(pattern, re.IGNORECASE)
            for entity_type, pattern in self.PATTERNS.items()
        }

    def extract_entities(self, text: str) -> ExtractionResult:
        """Extract entities using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            ExtractionResult with extracted entities
        """
        import time

        start_time = time.time()
        entities = []

        for entity_type, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                entity = Entity(
                    text=match.group(),
                    type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9,  # Regex matches are fairly confident
                )
                entities.append(entity)

        processing_time = int((time.time() - start_time) * 1000)

        return ExtractionResult(
            entities=entities,
            processing_time_ms=processing_time,
            model_used="regex",
        )


class HybridEntityExtractor:
    """Hybrid entity extractor combining LLM and regex approaches.
    
    This class uses LLM for complex entity extraction and regex
    for pattern-based entities like emails, URLs, etc.
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        min_confidence: float = 0.7,
    ):
        """Initialize the hybrid extractor.
        
        Args:
            llm_provider: Optional LLM provider
            min_confidence: Minimum confidence threshold
        """
        self._llm_extractor = None
        if llm_provider:
            self._llm_extractor = EntityExtractor(llm_provider, min_confidence=min_confidence)

        self._regex_extractor = RegexEntityExtractor(min_confidence=min_confidence)

    async def extract_entities(
        self,
        text: str,
        entity_types: list[EntityType] | None = None,
    ) -> ExtractionResult:
        """Extract entities using both methods.
        
        Args:
            text: Input text
            entity_types: Entity types to extract
            
        Returns:
            Combined extraction result
        """
        all_entities = []
        total_time = 0
        models_used = []

        # Get regex entities (for pattern-based types)
        regex_result = self._regex_extractor.extract_entities(text)
        all_entities.extend(regex_result.entities)
        total_time += regex_result.processing_time_ms
        models_used.append("regex")

        # Get LLM entities if available
        if self._llm_extractor:
            llm_result = await self._llm_extractor.extract_entities(text, entity_types)
            all_entities.extend(llm_result.entities)
            total_time += llm_result.processing_time_ms
            models_used.append(llm_result.model_used)

        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in all_entities:
            key = (entity.text.lower(), entity.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return ExtractionResult(
            entities=unique_entities,
            processing_time_ms=total_time,
            model_used="+".join(models_used),
        )
