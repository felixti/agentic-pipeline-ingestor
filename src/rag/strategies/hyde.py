"""HyDE (Hypothetical Document Embeddings) strategy for RAG optimization.

This module implements the HyDE approach which generates hypothetical documents
that answer the user's query, then uses these documents for vector search instead
of the raw query text. This improves retrieval quality for vague or out-of-domain
queries by finding conceptually related documents beyond exact keyword matches.

Example:
    User asks: "What is vibe coding?"
    LLM generates: "Vibe coding is a programming approach emphasizing intuition..."
    This hypothetical answer is embedded and used for semantic search.
"""

import hashlib
import json
import logging
import time
from typing import Any

from src.config import get_settings
from src.llm.provider import ChatMessage, LLMProvider
from src.rag.models import QueryRewriteResult
from src.rag.strategies.query_rewriting import (
    NullQueryRewritingCache,
    QueryRewriter,
    QueryRewritingCache,
    QueryRewritingError,
)

logger = logging.getLogger(__name__)

DEFAULT_HYDE_SYSTEM_PROMPT = """Generate a focused 2-4 sentence hypothetical document that answers the user's question. This will be used for semantic search, so include key concepts and terminology that would appear in relevant documents."""


class HyDERewritingError(QueryRewritingError):
    """Exception raised when HyDE rewriting fails."""

    pass


class HyDERewriter(QueryRewriter):
    """HyDE Query Rewriter that generates hypothetical documents for semantic search.

    This class extends QueryRewriter to implement the HyDE (Hypothetical Document
    Embeddings) strategy. Instead of using the raw query for vector search, it
    generates a hypothetical document that answers the query, then uses that
    document's embedding for semantic search.

    This approach improves retrieval for:
    - Complex questions requiring multi-step reasoning
    - Vague or ambiguous queries
    - Out-of-domain queries where keyword matching fails

    Attributes:
        llm_provider: LLM provider for generating hypothetical documents
        cache: Cache for storing generated hypothetical documents
        system_prompt: System prompt for hypothetical document generation
        model: Model group to use for generation
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens for hypothetical document
        max_hypothetical_length: Maximum character length for the document
        cache_ttl: Cache time-to-live in seconds
        enable_for_query_types: Query types to enable HyDE for
        fallback_to_standard: Whether to fallback on failure
        max_processing_time_ms: Maximum allowed processing time

    Example:
        >>> rewriter = HyDERewriter()
        >>> result = await rewriter.rewrite("What is vibe coding?")
        >>> print(result.embedding_source_text)
        "Vibe coding is a programming approach that emphasizes..."
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        cache: QueryRewritingCache | NullQueryRewritingCache | None = None,
        system_prompt: str | None = None,
        model: str = "agentic-decisions",
        temperature: float = 0.7,
        max_tokens: int = 300,
        max_hypothetical_length: int = 512,
        cache_ttl: int = 7200,
        enable_for_query_types: list[str] | None = None,
        fallback_to_standard: bool = True,
        max_processing_time_ms: int = 500,
    ):
        """Initialize the HyDE rewriter.

        Args:
            llm_provider: LLM provider instance. If None, creates default.
            cache: Cache instance for hypothetical documents. If None, uses NullCache.
            system_prompt: Custom system prompt for hypothetical document generation.
            model: Model group to use for generation.
            temperature: Sampling temperature (higher for more creative output).
            max_tokens: Maximum tokens for hypothetical document.
            max_hypothetical_length: Maximum character length for the document.
            cache_ttl: Cache time-to-live in seconds (default: 2 hours).
            enable_for_query_types: Query types to enable HyDE for.
            fallback_to_standard: Whether to fallback to standard rewriting on failure.
            max_processing_time_ms: Maximum allowed processing time in milliseconds.
        """
        # Initialize parent with different defaults for HyDE
        super().__init__(
            llm_provider=llm_provider,
            cache=cache or NullQueryRewritingCache(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_ttl=cache_ttl,
        )

        self.system_prompt = system_prompt or DEFAULT_HYDE_SYSTEM_PROMPT
        self.max_hypothetical_length = max_hypothetical_length
        self.enable_for_query_types = enable_for_query_types or [
            "complex_questions",
            "vague_queries",
            "out_of_domain",
        ]
        self.fallback_to_standard = fallback_to_standard
        self.max_processing_time_ms = max_processing_time_ms

        logger.info(
            f"Initialized HyDERewriter with model: {model}, "
            f"max_length: {max_hypothetical_length}, "
            f"cache_ttl: {cache_ttl}s"
        )

    @classmethod
    def from_settings(
        cls, settings: Any | None = None
    ) -> "HyDERewriter":
        """Create a HyDERewriter instance from application settings.

        Args:
            settings: Application settings. If None, loads from get_settings().

        Returns:
            Configured HyDERewriter instance.
        """
        if settings is None:
            settings = get_settings()

        hyde_settings = settings.hyde

        # Create cache if enabled
        cache: QueryRewritingCache | NullQueryRewritingCache = NullQueryRewritingCache()
        if hyde_settings.cache_enabled:
            try:
                import redis.asyncio as redis

                redis_client = redis.from_url(str(settings.redis.url))
                cache = QueryRewritingCache(redis_client=redis_client)
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                cache = NullQueryRewritingCache()

        return cls(
            llm_provider=None,  # Will create default
            cache=cache,
            system_prompt=hyde_settings.system_prompt,
            model=hyde_settings.model,
            temperature=hyde_settings.temperature,
            max_tokens=hyde_settings.max_tokens,
            max_hypothetical_length=hyde_settings.max_hypothetical_length,
            cache_ttl=hyde_settings.cache_ttl,
            enable_for_query_types=hyde_settings.enable_for_query_types,
            fallback_to_standard=hyde_settings.fallback_to_standard,
            max_processing_time_ms=hyde_settings.max_processing_time_ms,
        )

    def _make_cache_key(self, query: str) -> str:
        """Create a cache key for a query.

        Args:
            query: User query string.

        Returns:
            Cache key string.
        """
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:32]
        return f"rag:hyde:{query_hash}"

    def _should_use_hyde(self, query: str, context: dict[str, Any] | None = None) -> bool:
        """Determine if HyDE should be used for this query.

        Checks if the query type is enabled for HyDE processing.

        Args:
            query: User query string.
            context: Optional conversation context.

        Returns:
            True if HyDE should be used, False otherwise.
        """
        # If no specific query types are enabled, use HyDE for all
        if not self.enable_for_query_types:
            return True

        # Check for @knowledgebase trigger - always enable for knowledge base queries
        if "@knowledgebase" in query.lower():
            return True

        # For now, enable for all queries if the list is not empty
        # Future enhancement: classify query types and filter accordingly
        return True

    def _truncate_hypothetical(self, text: str) -> str:
        """Truncate hypothetical document to max length.

        Args:
            text: Generated hypothetical document.

        Returns:
            Truncated text if needed.
        """
        if len(text) > self.max_hypothetical_length:
            # Try to truncate at a sentence boundary
            truncated = text[: self.max_hypothetical_length]
            last_period = truncated.rfind(".")
            if last_period > self.max_hypothetical_length * 0.5:
                return truncated[: last_period + 1]
            return truncated
        return text

    def _create_llm_prompt(self, query: str) -> str:
        """Create the LLM prompt for the final response.

        Args:
            query: Original user query.

        Returns:
            Formatted prompt for LLM generation.
        """
        return (
            f"Based on the provided context, answer the following question: {query}"
        )

    async def generate_hypothetical_document(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate a hypothetical document that answers the query.

        Uses the LLM to generate a focused 2-4 sentence document that answers
        the user's question. This hypothetical document is then used for
        semantic search embedding.

        Args:
            query: Original user query.
            context: Optional conversation context.

        Returns:
            Generated hypothetical document as a string.

        Raises:
            HyDERewritingError: If generation fails.
        """
        start_time = time.time()

        try:
            # Build the prompt
            prompt = f"Question: {query}\n\nGenerate a hypothetical document that answers this question."

            # Call LLM for hypothetical document generation
            messages = [
                ChatMessage.system(self.system_prompt),
                ChatMessage.user(prompt),
            ]

            response = await self.llm_provider.chat_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            hypothetical = response.content.strip()

            # Truncate if necessary
            hypothetical = self._truncate_hypothetical(hypothetical)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "Generated hypothetical document",
                extra={
                    "query_length": len(query),
                    "hypothetical_length": len(hypothetical),
                    "elapsed_ms": elapsed_ms,
                },
            )

            return hypothetical

        except Exception as e:
            logger.error(f"Failed to generate hypothetical document: {e}")
            raise HyDERewritingError(
                f"Failed to generate hypothetical document: {e}"
            ) from e

    async def rewrite(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> QueryRewriteResult:
        """Rewrite user query using HyDE strategy.

        Generates a hypothetical document that answers the query, then returns
        it as the embedding_source_text for vector search.

        Args:
            query: Original user query.
            context: Optional conversation context with keys like:
                - previous_queries: List of previous user queries
                - previous_responses: List of previous assistant responses
                - session_id: Conversation session identifier

        Returns:
            QueryRewriteResult with hypothetical document as embedding_source_text.

        Raises:
            HyDERewritingError: If rewriting fails and fallback is disabled.
        """
        start_time = time.time()

        # Validate input
        if not query or not query.strip():
            logger.warning("Empty query received, returning default result")
            return QueryRewriteResult(
                search_rag=False,
                embedding_source_text="",
                llm_query="",
            )

        # Check if HyDE should be used for this query
        if not self._should_use_hyde(query, context):
            logger.debug("HyDE not enabled for this query type, using standard rewrite")
            return await super().rewrite(query, context)

        try:
            # Check cache first (use HyDE-specific cache key)
            cache_key = self._make_cache_key(query)
            if hasattr(self.cache, "redis") and self.cache.redis:
                cached_data = await self.cache.redis.get(cache_key)
                if cached_data:
                    try:
                        record_dict = json.loads(cached_data)
                        hypothetical = record_dict.get("hypothetical", "")
                        logger.debug(f"Cache hit for HyDE query: {query[:50]}...")
                        return QueryRewriteResult(
                            search_rag="@knowledgebase" in query.lower(),
                            embedding_source_text=hypothetical,
                            llm_query=self._create_llm_prompt(query),
                        )
                    except Exception as e:
                        logger.debug(f"Failed to parse cached HyDE result: {e}")

            # Generate hypothetical document
            hypothetical = await self.generate_hypothetical_document(query, context)

            # Cache the result
            if hasattr(self.cache, "redis") and self.cache.redis:
                try:
                    cache_data = json.dumps({"hypothetical": hypothetical})
                    await self.cache.redis.setex(
                        cache_key, self.cache_ttl, cache_data
                    )
                except Exception as e:
                    logger.debug(f"Failed to cache HyDE result: {e}")

            elapsed_ms = (time.time() - start_time) * 1000

            # Check processing time constraint
            if elapsed_ms > self.max_processing_time_ms:
                logger.warning(
                    f"HyDE processing time ({elapsed_ms:.2f}ms) exceeded limit "
                    f"({self.max_processing_time_ms}ms)"
                )

            logger.info(
                "HyDE rewrite completed",
                extra={
                    "query_length": len(query),
                    "elapsed_ms": elapsed_ms,
                    "search_rag": "@knowledgebase" in query.lower(),
                },
            )

            return QueryRewriteResult(
                search_rag="@knowledgebase" in query.lower(),
                embedding_source_text=hypothetical,
                llm_query=self._create_llm_prompt(query),
            )

        except Exception as e:
            logger.error(f"HyDE rewrite failed: {e}")

            if self.fallback_to_standard:
                logger.info("Falling back to standard query rewriting")
                return await super().rewrite(query, context)

            raise HyDERewritingError(f"HyDE rewrite failed: {e}") from e

    async def rewrite_batch(
        self,
        queries: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[QueryRewriteResult]:
        """Rewrite multiple queries using HyDE strategy.

        Args:
            queries: List of user queries.
            context: Optional conversation context.

        Returns:
            List of QueryRewriteResult objects.
        """
        results = []
        for query in queries:
            try:
                result = await self.rewrite(query, context)
                results.append(result)
            except HyDERewritingError as e:
                logger.warning(f"Failed to rewrite query '{query[:50]}...': {e}")
                # Return fallback result on failure
                results.append(
                    QueryRewriteResult(
                        search_rag="@knowledgebase" in query.lower(),
                        embedding_source_text=query,
                        llm_query=query,
                    )
                )
        return results
