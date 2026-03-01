"""LLM provider for HippoRAG using existing litellm integration.

This module provides the HippoRAGLLMProvider class which uses the existing
litellm-based LLM provider for:
- OpenIE triple extraction from documents
- Query entity extraction
- RAG question answering
- Text embeddings

The provider wraps the existing LLMProvider to provide HippoRAG-specific
functionality with proper error handling and logging.

Example:
    >>> provider = HippoRAGLLMProvider(
    ...     llm_model="azure/gpt-4.1",
    ...     embedding_model="azure/text-embedding-3-small",
    ... )
    >>> triples = await provider.extract_triples(
    ...     "Steve Jobs founded Apple in California."
    ... )
    >>> print(triples)  # [("Steve Jobs", "founded", "Apple"), ...]
    >>> answer = await provider.answer_question(
    ...     question="What did Steve Jobs found?",
    ...     context=["Steve Jobs founded Apple in California."]
    ... )

Environment Variables:
    HIPPO_LLM_MODEL: LLM model for OpenIE and QA (default: azure/gpt-4.1)
    HIPPO_EMBEDDING_MODEL: Embedding model (default: azure/text-embedding-3-small)
    AZURE_OPENAI_API_BASE: Azure OpenAI endpoint
    AZURE_OPENAI_API_KEY: Azure OpenAI API key
    OPENROUTER_API_KEY: OpenRouter API key (fallback)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.llm.provider import LLMProvider
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Default models
DEFAULT_LLM_MODEL = os.getenv("HIPPO_LLM_MODEL", "azure/gpt-4.1")
DEFAULT_EMBEDDING_MODEL = os.getenv("HIPPO_EMBEDDING_MODEL", "azure/text-embedding-3-small")

# OpenIE prompt for triple extraction
OPENIE_PROMPT = """Extract subject-predicate-object triples from the text.

Return a JSON object in this exact format:
{{
    "triples": [
        ["subject", "predicate", "object"],
        ["subject", "predicate", "object"],
        ...
    ]
}}

Guidelines:
- Extract factual relationships between entities
- Keep subjects and objects as simple noun phrases
- Use present tense for predicates when possible
- Include location, date, and organization relationships
- Only extract triples explicitly stated or strongly implied in the text
- If no clear triples are found, return an empty list

Text: {text}

JSON Response:"""

# Query entity extraction prompt
QUERY_ENTITY_PROMPT = """Extract the key named entities from this query.

Return a JSON object in this exact format:
{{
    "entities": ["Entity1", "Entity2", ...]
}}

Focus on:
- Proper nouns (names of people, organizations, places)
- Specific concepts or topics
- Important terms that would match a knowledge graph

Query: {query}

JSON Response:"""

# RAG QA prompt
RAG_QA_PROMPT = """Answer the question based on the provided context.

Instructions:
1. Use only the information in the provided context
2. If the context doesn't contain enough information, say so clearly
3. Provide a concise but complete answer
4. Cite specific details from the context when possible

Context:
{context}

Question: {question}

Answer:"""

# Fallback prompt if context is too long
RAG_QA_SHORT_PROMPT = """Based on the following passages, answer the question.

Passages:
{context}

Question: {question}

Provide a brief, accurate answer using only the information above:"""


@dataclass
class ExtractedTriple:
    """Represents an extracted OpenIE triple.
    
    Attributes:
        subject: Subject entity
        predicate: Relationship predicate
        object: Object entity
        confidence: Extraction confidence (0.0 to 1.0)
    """
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


class HippoRAGLLMProvider:
    """LLM provider for HippoRAG operations.
    
    This class provides LLM capabilities for the HippoRAG pipeline:
    - OpenIE triple extraction from text
    - Query entity extraction for PPR
    - RAG question answering
    - Text embeddings
    
    It wraps the existing LLMProvider to provide HippoRAG-specific
    functionality with proper error handling and logging.
    
    Attributes:
        llm_model: Model used for text generation tasks
        embedding_model: Model used for embeddings
        _llm: Internal LLM provider instance
    
    Example:
        >>> provider = HippoRAGLLMProvider()
        >>> triples = await provider.extract_triples(
        ...     "Microsoft was founded by Bill Gates."
        ... )
        >>> print(len(triples))
        2
    """

    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the HippoRAG LLM provider.
        
        Args:
            llm_model: LLM model for generation tasks (OpenIE, QA)
            embedding_model: Model for text embeddings
            llm_provider: Optional existing LLMProvider instance
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._llm = llm_provider or LLMProvider()
        
        logger.info(
            "hipporag_llm_provider_initialized",
            llm_model=llm_model,
            embedding_model=embedding_model,
        )

    async def extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        """Extract OpenIE triples from text.
        
        Uses the configured LLM to extract subject-predicate-object
        triples from the provided text.
        
        Args:
            text: The text to analyze for triples
            
        Returns:
            List of (subject, predicate, object) tuples
            
        Example:
            >>> triples = await provider.extract_triples(
            ...     "Apple Inc. was founded by Steve Jobs in California."
            ... )
            >>> triples
            [('Apple Inc.', 'founded by', 'Steve Jobs'), 
             ('Apple Inc.', 'located in', 'California')]
        """
        if not text or not text.strip():
            logger.debug("hipporag_extract_triples_empty_text")
            return []

        try:
            prompt = OPENIE_PROMPT.format(text=text[:4000])  # Limit text length
            
            response = await self._llm.simple_completion(
                prompt=prompt,
                system_prompt="You are a precise information extraction system. Return only valid JSON.",
                model="enrichment",  # Use enrichment model for extraction
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=2000,
            )

            # Parse JSON response
            triples = self._parse_triples_response(response)
            
            logger.info(
                "hipporag_triples_extracted",
                triple_count=len(triples),
                text_length=len(text),
            )
            return triples

        except Exception as e:
            logger.error(
                "hipporag_triple_extraction_failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
            )
            return []

    def _parse_triples_response(self, response: str) -> list[tuple[str, str, str]]:
        """Parse triples from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        try:
            # Try to parse as JSON
            data = json.loads(response.strip())
            triples_data = data.get("triples", [])
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            try:
                cleaned = self._extract_json_from_markdown(response)
                data = json.loads(cleaned)
                triples_data = data.get("triples", [])
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "hipporag_triple_json_parse_failed",
                    error=str(e),
                    response_preview=response[:200],
                )
                return []
        
        # Convert to tuples
        triples = []
        for triple in triples_data:
            try:
                if isinstance(triple, list) and len(triple) >= 3:
                    subject = str(triple[0]).strip()
                    predicate = str(triple[1]).strip()
                    obj = str(triple[2]).strip()
                    
                    # Validate non-empty
                    if subject and predicate and obj:
                        triples.append((subject, predicate, obj))
                elif isinstance(triple, dict):
                    # Handle dict format
                    subject = str(triple.get("subject", triple.get("s", ""))).strip()
                    predicate = str(triple.get("predicate", triple.get("p", ""))).strip()
                    obj = str(triple.get("object", triple.get("o", ""))).strip()
                    
                    if subject and predicate and obj:
                        triples.append((subject, predicate, obj))
            except Exception as e:
                logger.warning(
                    "hipporag_triple_parse_warning",
                    triple=triple,
                    error=str(e),
                )
        
        return triples

    async def extract_query_entities(self, query: str) -> list[str]:
        """Extract key entities from a search query.
        
        Identifies the key terms and entities in a user query that
        should be used for graph traversal and entity matching.
        
        Args:
            query: The user's search query
            
        Returns:
            List of entity names/keywords extracted from the query
            
        Example:
            >>> entities = await provider.extract_query_entities(
            ...     "What company did Steve Jobs found?"
            ... )
            >>> entities
            ['Steve Jobs']
        """
        if not query or not query.strip():
            return []

        try:
            prompt = QUERY_ENTITY_PROMPT.format(query=query[:500])  # Limit query length
            
            response = await self._llm.simple_completion(
                prompt=prompt,
                system_prompt="Extract only the most relevant named entities. Be concise.",
                model="enrichment",
                temperature=0.1,
                max_tokens=200,
            )

            # Parse JSON response
            try:
                data = json.loads(response.strip())
                entities = data.get("entities", [])
            except json.JSONDecodeError:
                # Try to extract JSON from markdown
                cleaned = self._extract_json_from_markdown(response)
                data = json.loads(cleaned)
                entities = data.get("entities", [])
            
            # Ensure list of strings
            result = [str(e).strip() for e in entities if str(e).strip()]
            
            logger.debug(
                "hipporag_query_entities_extracted",
                query=query[:100],
                entities=result,
            )
            return result

        except Exception as e:
            logger.warning(
                "hipporag_query_entity_extraction_failed",
                query=query[:100],
                error=str(e),
            )
            # Fallback: return query words longer than 3 chars
            return [w for w in query.split() if len(w) > 3][:5]

    async def answer_question(
        self,
        question: str,
        context: list[str],
    ) -> str:
        """Generate answer from question and context.
        
        Uses the configured LLM to generate a concise answer
        based on the provided context passages.
        
        Args:
            question: The question to answer
            context: List of context passages
            
        Returns:
            Generated answer text
            
        Example:
            >>> answer = await provider.answer_question(
            ...     question="What company did Steve Jobs found?",
            ...     context=["Steve Jobs founded Apple Inc. in 1976."]
            ... )
            >>> print(answer)
            Steve Jobs founded Apple Inc.
        """
        if not context:
            logger.debug("hipporag_answer_question_empty_context")
            return "No relevant context provided to answer this question."

        try:
            # Format context passages
            formatted_context = "\n\n".join(
                f"[{i+1}] {passage[:1500]}"  # Limit each passage
                for i, passage in enumerate(context[:10])  # Limit number of passages
            )

            # Choose prompt based on context length
            if len(formatted_context) > 3000:
                prompt = RAG_QA_SHORT_PROMPT.format(
                    context=formatted_context[:3000],
                    question=question[:500],
                )
            else:
                prompt = RAG_QA_PROMPT.format(
                    context=formatted_context,
                    question=question[:500],
                )

            answer = await self._llm.simple_completion(
                prompt=prompt,
                system_prompt="You are a helpful assistant that answers questions accurately based on provided context.",
                model="agentic-decisions",  # Use agentic model for QA
                temperature=0.3,  # Slightly higher for natural answers
                max_tokens=1000,
            )

            logger.info(
                "hipporag_answer_generated",
                question_length=len(question),
                context_count=len(context),
                answer_length=len(answer),
            )
            
            return answer.strip()

        except Exception as e:
            logger.error(
                "hipporag_answer_generation_failed",
                error=str(e),
                error_type=type(e).__name__,
                question=question[:100],
            )
            # Return a fallback concatenation of contexts
            return " ".join(context[:3])[:500]

    async def embed_text(self, text: str) -> np.ndarray | None:
        """Generate embedding for text.
        
        Uses the configured embedding model to generate
        a vector representation of the text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array embedding vector, or None if embedding fails
        """
        if not text or not text.strip():
            return None

        try:
            # Use litellm for embeddings
            import litellm
            
            response = await litellm.aembedding(
                model=self.embedding_model,
                input=text[:8000],  # Limit text length
            )
            
            # Extract embedding vector
            if response and response.data:
                embedding = response.data[0].get("embedding", [])
                if embedding:
                    return np.array(embedding, dtype=np.float32)
            
            logger.warning(
                "hipporag_embedding_empty_response",
                text_length=len(text),
            )
            return None

        except Exception as e:
            logger.warning(
                "hipporag_embedding_failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
            )
            return None

    async def health_check(self) -> dict[str, Any]:
        """Check the health of the LLM provider.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Try a simple completion
            result = await self._llm.simple_completion(
                prompt='Say "healthy" and nothing else.',
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
