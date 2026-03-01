"""LLM provider for Cognee using existing litellm integration.

This module provides the CogneeLLMProvider class which uses the existing
litellm-based LLM provider for entity extraction, relationship extraction,
and search result summarization in the Cognee GraphRAG pipeline.
"""

import json
from dataclasses import dataclass
from typing import Any

from src.llm.provider import ChatMessage, LLMProvider
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Entity extraction prompt template
ENTITY_EXTRACTION_PROMPT = """You are an expert in named entity recognition and extraction.

Extract named entities from the following text. For each entity, identify:
- name: The entity name (proper noun or specific term)
- type: The entity type (PERSON, ORGANIZATION, LOCATION, DATE, TECHNOLOGY, PRODUCT, CONCEPT, etc.)
- description: A brief description of the entity

Return ONLY a valid JSON object in this exact format:
{{
    "entities": [
        {{"name": "Entity Name", "type": "ENTITY_TYPE", "description": "Brief description"}},
        ...
    ]
}}

Important:
- Extract only real, specific entities mentioned in the text
- Do not create generic or inferred entities
- Ensure the response is valid JSON
- If no entities are found, return {"entities": []}

Text to analyze:
{text}
"""

# Relationship extraction prompt template
RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert in relationship extraction and knowledge graph construction.

Given the following entities and text, extract relationships between the entities.
For each relationship, identify:
- source: The source entity name (must match one of the provided entities)
- target: The target entity name (must match one of the provided entities)
- type: The relationship type (WORKS_AT, LOCATED_IN, PART_OF, USES, CREATED, MENTIONS, RELATED_TO, etc.)
- confidence: Confidence score (0.0 to 1.0)

Return ONLY a valid JSON object in this exact format:
{{
    "relationships": [
        {{"source": "Entity A", "target": "Entity B", "type": "RELATIONSHIP_TYPE", "confidence": 0.95}},
        ...
    ]
}}

Important:
- Only create relationships that are explicitly stated or strongly implied in the text
- Both source and target must be from the provided entity list
- Ensure the response is valid JSON
- If no relationships are found, return {"relationships": []}

Entities:
{entities}

Text to analyze:
{text}
"""

# Summarization prompt template
SUMMARIZATION_PROMPT = """You are an expert at synthesizing information and creating concise summaries.

Given a user query and a set of relevant context passages, provide a clear and accurate summary
that directly answers the query based on the provided context.

Instructions:
- Focus on information relevant to the query
- Synthesize information from multiple sources if needed
- Be concise but comprehensive
- If the context doesn't contain relevant information, state that clearly
- Do not make up or infer information not present in the context

Query: {query}

Context passages:
{context}

Provide a summary that answers the query based on the context above.
"""


@dataclass
class ExtractedEntity:
    """Represents an extracted entity.
    
    Attributes:
        name: Entity name
        type: Entity type (PERSON, ORGANIZATION, etc.)
        description: Brief description of the entity
    """
    name: str
    type: str
    description: str = ""


@dataclass
class ExtractedRelationship:
    """Represents an extracted relationship between entities.
    
    Attributes:
        source: Source entity name
        target: Target entity name
        type: Relationship type
        confidence: Confidence score (0.0 to 1.0)
    """
    source: str
    target: str
    type: str
    confidence: float = 0.8


class CogneeLLMProvider:
    """LLM provider for Cognee using existing litellm integration.
    
    This class provides LLM capabilities for the Cognee GraphRAG pipeline:
    - Entity extraction from text
    - Relationship extraction between entities
    - Search result summarization
    
    It wraps the existing LLMProvider to provide Cognee-specific functionality
    with proper error handling and logging.
    
    Example:
        >>> provider = CogneeLLMProvider()
        >>> entities = await provider.extract_entities(
        ...     "Apple Inc. was founded by Steve Jobs in California."
        ... )
        >>> print(entities[0].name)  # "Apple Inc."
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        """Initialize the Cognee LLM provider.
        
        Args:
            llm_provider: Optional existing LLMProvider instance. If not provided,
                         a new one will be created.
        """
        self._llm = llm_provider or LLMProvider()
        logger.info("cognee_llm_provider_initialized")

    async def extract_entities(self, text: str) -> list[ExtractedEntity]:
        """Extract entities from text using LLM.
        
        Uses the configured LLM to identify and extract named entities
        from the provided text, including their types and descriptions.
        
        Args:
            text: The text to analyze for entities
            
        Returns:
            List of ExtractedEntity objects
            
        Example:
            >>> entities = await provider.extract_entities(
            ...     "Microsoft was founded by Bill Gates."
            ... )
            >>> len(entities)
            2
        """
        if not text or not text.strip():
            logger.debug("cognee_extract_entities_empty_text")
            return []

        try:
            prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:8000])  # Limit text length
            
            response = await self._llm.simple_completion(
                prompt=prompt,
                system_prompt="You are a precise entity extraction system. Return only valid JSON.",
                model="enrichment",  # Use enrichment model for entity extraction
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=2000,
            )

            # Parse JSON response
            try:
                data = json.loads(response.strip())
                entities_data = data.get("entities", [])
            except json.JSONDecodeError as e:
                logger.warning(
                    "cognee_entity_extraction_json_parse_failed",
                    error=str(e),
                    response_preview=response[:200],
                )
                # Try to extract JSON from markdown code blocks
                response = self._extract_json_from_markdown(response)
                data = json.loads(response)
                entities_data = data.get("entities", [])

            # Convert to ExtractedEntity objects
            entities = []
            for entity_data in entities_data:
                try:
                    entity = ExtractedEntity(
                        name=str(entity_data.get("name", "")).strip(),
                        type=str(entity_data.get("type", "UNKNOWN")).strip().upper(),
                        description=str(entity_data.get("description", "")).strip(),
                    )
                    if entity.name:  # Only add if name is not empty
                        entities.append(entity)
                except Exception as e:
                    logger.warning(
                        "cognee_entity_parse_warning",
                        entity_data=entity_data,
                        error=str(e),
                    )

            logger.info(
                "cognee_entities_extracted",
                entity_count=len(entities),
                text_length=len(text),
            )
            return entities

        except Exception as e:
            logger.error(
                "cognee_entity_extraction_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

    async def extract_relationships(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelationship]:
        """Extract relationships between entities using LLM.
        
        Analyzes the text to identify relationships between the provided
        entities, including the relationship type and confidence score.
        
        Args:
            text: The text to analyze for relationships
            entities: List of entities to find relationships between
            
        Returns:
            List of ExtractedRelationship objects
            
        Example:
            >>> entities = [ExtractedEntity("Microsoft", "ORGANIZATION")]
            >>> relationships = await provider.extract_relationships(text, entities)
        """
        if not text or not entities or len(entities) < 2:
            logger.debug("cognee_extract_relationships_insufficient_data")
            return []

        try:
            # Format entities for the prompt
            entity_list = [
                f"- {e.name} ({e.type})" for e in entities[:50]  # Limit entity count
            ]
            entities_text = "\n".join(entity_list)

            prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
                entities=entities_text,
                text=text[:8000],  # Limit text length
            )

            response = await self._llm.simple_completion(
                prompt=prompt,
                system_prompt="You are a precise relationship extraction system. Return only valid JSON.",
                model="enrichment",
                temperature=0.1,
                max_tokens=2000,
            )

            # Parse JSON response
            try:
                data = json.loads(response.strip())
                relationships_data = data.get("relationships", [])
            except json.JSONDecodeError as e:
                logger.warning(
                    "cognee_relationship_extraction_json_parse_failed",
                    error=str(e),
                    response_preview=response[:200],
                )
                response = self._extract_json_from_markdown(response)
                data = json.loads(response)
                relationships_data = data.get("relationships", [])

            # Convert to ExtractedRelationship objects
            relationships = []
            entity_names = {e.name for e in entities}

            for rel_data in relationships_data:
                try:
                    source = str(rel_data.get("source", "")).strip()
                    target = str(rel_data.get("target", "")).strip()
                    
                    # Validate that source and target are in our entity list
                    if source not in entity_names or target not in entity_names:
                        continue

                    relationship = ExtractedRelationship(
                        source=source,
                        target=target,
                        type=str(rel_data.get("type", "RELATED_TO")).strip().upper(),
                        confidence=float(rel_data.get("confidence", 0.8)),
                    )
                    relationships.append(relationship)
                except Exception as e:
                    logger.warning(
                        "cognee_relationship_parse_warning",
                        rel_data=rel_data,
                        error=str(e),
                    )

            logger.info(
                "cognee_relationships_extracted",
                relationship_count=len(relationships),
                entity_count=len(entities),
            )
            return relationships

        except Exception as e:
            logger.error(
                "cognee_relationship_extraction_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

    async def summarize_for_search(
        self,
        query: str,
        context: list[str],
        max_length: int = 500,
    ) -> str:
        """Summarize search results for a query.
        
        Creates a coherent summary that answers the user's query based on
        the provided context passages.
        
        Args:
            query: The user's search query
            context: List of context passages from search results
            max_length: Maximum length of the summary in characters
            
        Returns:
            Summarized answer to the query
            
        Example:
            >>> summary = await provider.summarize_for_search(
            ...     "What is GraphRAG?",
            ...     ["GraphRAG is a technique...", "It uses knowledge graphs..."]
            ... )
        """
        if not context:
            logger.debug("cognee_summarize_empty_context")
            return "No relevant information found."

        try:
            # Format context passages
            formatted_context = "\n\n---\n\n".join(
                f"[{i+1}] {passage[:1000]}"  # Limit each passage
                for i, passage in enumerate(context[:10])  # Limit number of passages
            )

            prompt = SUMMARIZATION_PROMPT.format(
                query=query,
                context=formatted_context,
            )

            summary = await self._llm.simple_completion(
                prompt=prompt,
                system_prompt="You are a helpful assistant that synthesizes information accurately.",
                model="agentic-decisions",  # Use agentic model for summarization
                temperature=0.3,
                max_tokens=1000,
            )

            # Truncate if necessary
            if len(summary) > max_length:
                summary = summary[:max_length].rsplit(" ", 1)[0] + "..."

            logger.info(
                "cognee_search_summary_generated",
                query_length=len(query),
                context_count=len(context),
                summary_length=len(summary),
            )
            return summary.strip()

        except Exception as e:
            logger.error(
                "cognee_summarization_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return a fallback concatenation of contexts
            return " ".join(context[:3])[:max_length]

    async def extract_query_entities(self, query: str) -> list[str]:
        """Extract key entities from a search query.
        
        Identifies the key terms and entities in a user query that
        should be used for graph traversal and entity matching.
        
        Args:
            query: The user's search query
            
        Returns:
            List of entity names/keywords extracted from the query
        """
        if not query or not query.strip():
            return []

        try:
            prompt = f"""Extract the key entities, names, and important terms from this search query.
Return as a simple comma-separated list.

Query: {query}

Key entities:"""

            response = await self._llm.simple_completion(
                prompt=prompt,
                system_prompt="Extract only the most relevant entities. Be concise.",
                model="enrichment",
                temperature=0.1,
                max_tokens=200,
            )

            # Parse comma-separated list
            entities = [
                e.strip()
                for e in response.replace("\n", ",").split(",")
                if e.strip()
            ]

            logger.debug(
                "cognee_query_entities_extracted",
                query=query,
                entities=entities,
            )
            return entities

        except Exception as e:
            logger.warning(
                "cognee_query_entity_extraction_failed",
                error=str(e),
            )
            # Fallback: return query words
            return [w for w in query.split() if len(w) > 3][:5]

    def _extract_json_from_markdown(self, text: str) -> str:
        """Extract JSON from markdown code blocks.
        
        Args:
            text: Text that may contain markdown code blocks
            
        Returns:
            Extracted JSON string or original text
        """
        # Look for JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        return text

    async def health_check(self) -> dict[str, Any]:
        """Check the health of the LLM provider.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Try a simple completion
            result = await self._llm.simple_completion(
                prompt="Say 'healthy' and nothing else.",
                max_tokens=10,
            )
            is_healthy = "healthy" in result.lower()
            
            return {
                "healthy": is_healthy,
                "status": "healthy" if is_healthy else "degraded",
                "message": "LLM provider is responding" if is_healthy else "Unexpected response",
                "response": result[:50] if result else None,
            }
        except Exception as e:
            return {
                "healthy": False,
                "status": "unhealthy",
                "message": f"LLM provider health check failed: {e}",
            }
