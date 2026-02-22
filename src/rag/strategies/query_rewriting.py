"""Query Rewriting Service for RAG optimization.

This module provides LLM-based query rewriting to separate user intent
into optimized prompts for vector search and LLM generation.
"""

import hashlib
import json
import logging
import time
from typing import Any

import redis.asyncio as redis

from src.llm.provider import ChatMessage, LLMProvider
from src.rag.models import ConversationContext, QueryRewriteResult

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a query rewriting assistant for a RAG (Retrieval-Augmented Generation) system.

Your task is to analyze user queries and rewrite them into two separate components:

1. **Search Component**: Extract core topic keywords for embedding/vector search.
   - Remove instruction words like "search", "find", "summarize", "list", "explain"
   - Keep only the essential topic terms
   - Make it concise and search-friendly

2. **Generation Component**: Create a clear instruction for the LLM.
   - Include the original intent (summarize, compare, explain, etc.)
   - Reference that answers should be based on "the provided context"
   - Include instructions to cite sources when appropriate

You must respond with a JSON object containing exactly these fields:
- "search_rag": boolean - True if query contains "@knowledgebase"
- "embedding_source_text": string - Core topic keywords only, ignoring instruction words
- "llm_query": string - Clear instruction for the LLM with appropriate context reference

Example:
Input: "@knowledgebase search on vibe coding, then summarize, list pros and cons"
Output:
{
  "search_rag": true,
  "embedding_source_text": "vibe coding programming approach",
  "llm_query": "Based on the provided context, explain what vibe coding is, including its pros and cons, and cite sources."
}"""


class QueryRewritingError(Exception):
    """Base exception for query rewriting errors."""
    pass


class QueryRewritingValidationError(QueryRewritingError):
    """Exception raised when the LLM response fails validation."""
    pass


class QueryRewritingCache:
    """Redis cache for query rewriting results.
    
    Provides caching layer to improve performance for repeated queries.
    """
    
    CACHE_PREFIX = "rag:query_rewrite:"
    DEFAULT_TTL = 3600  # 1 hour in seconds
    
    def __init__(self, redis_client: redis.Redis | None = None):
        """Initialize cache.
        
        Args:
            redis_client: Redis async client. If None, cache operations will fail gracefully.
        """
        self.redis = redis_client
    
    def _make_key(self, query: str, context_hash: str = "") -> str:
        """Create cache key from query and context.
        
        Args:
            query: User query string
            context_hash: Hash of conversation context
            
        Returns:
            Cache key string
        """
        # Create deterministic hash of query + context
        key_data = f"{query}:{context_hash}"
        query_hash = hashlib.sha256(key_data.encode()).hexdigest()[:32]
        return f"{self.CACHE_PREFIX}{query_hash}"
    
    def _hash_context(self, context: dict[str, Any] | None) -> str:
        """Create hash of conversation context.
        
        Args:
            context: Optional conversation context
            
        Returns:
            Hash string of context
        """
        if not context:
            return ""
        context_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]
    
    async def get(
        self,
        query: str,
        context: dict[str, Any] | None = None
    ) -> QueryRewriteResult | None:
        """Get cached rewrite result.
        
        Args:
            query: User query string
            context: Optional conversation context
            
        Returns:
            Cached result if found, None otherwise
        """
        if not self.redis:
            return None
        
        try:
            context_hash = self._hash_context(context)
            key = self._make_key(query, context_hash)
            data = await self.redis.get(key)
            
            if data is None:
                return None
            
            record_dict = json.loads(data)
            return QueryRewriteResult(**record_dict)
            
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
            return None
    
    async def set(
        self,
        query: str,
        result: QueryRewriteResult,
        context: dict[str, Any] | None = None,
        ttl: int | None = None
    ) -> bool:
        """Cache rewrite result.
        
        Args:
            query: User query string
            result: Rewrite result to cache
            context: Optional conversation context
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            context_hash = self._hash_context(context)
            key = self._make_key(query, context_hash)
            data = result.model_dump_json()
            
            await self.redis.setex(
                key,
                ttl or self.DEFAULT_TTL,
                data
            )
            return True
            
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
            return False
    
    async def delete(
        self,
        query: str,
        context: dict[str, Any] | None = None
    ) -> bool:
        """Delete cached rewrite result.
        
        Args:
            query: User query string
            context: Optional conversation context
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            context_hash = self._hash_context(context)
            key = self._make_key(query, context_hash)
            result = await self.redis.delete(key)
            return bool(result > 0)
        except Exception as e:
            logger.debug(f"Cache delete failed: {e}")
            return False
    
    async def clear_all(self) -> int:
        """Clear all query rewrite cache entries.
        
        Warning: Use with caution in production!
        
        Returns:
            Number of keys deleted
        """
        if not self.redis:
            return 0
        
        try:
            pattern = f"{self.CACHE_PREFIX}*"
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return int(await self.redis.delete(*keys))
            return 0
        except Exception as e:
            logger.debug(f"Cache clear failed: {e}")
            return 0


class NullQueryRewritingCache:
    """Null object pattern for when caching is disabled."""
    
    async def get(
        self,
        query: str,
        context: dict[str, Any] | None = None
    ) -> None:
        """Always returns None."""
        return None
    
    async def set(
        self,
        query: str,
        result: QueryRewriteResult,
        context: dict[str, Any] | None = None,
        ttl: int | None = None
    ) -> bool:
        """Always returns False."""
        return False
    
    async def delete(
        self,
        query: str,
        context: dict[str, Any] | None = None
    ) -> bool:
        """Always returns False."""
        return False
    
    async def clear_all(self) -> int:
        """Always returns 0."""
        return 0


class QueryRewriter:
    """Query Rewriting Service for RAG optimization.
    
    This class uses an LLM to separate user intent into:
    1. Search-optimized keywords (for embedding/vector search)
    2. Generation-optimized prompts (for LLM response)
    
    Example:
        >>> rewriter = QueryRewriter()
        >>> result = await rewriter.rewrite(
        ...     "@knowledgebase search on vibe coding, then summarize"
        ... )
        >>> print(result.search_rag)  # True
        >>> print(result.embedding_source_text)  # "vibe coding programming"
        >>> print(result.llm_query)  # "Based on the provided context..."
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        cache: QueryRewritingCache | NullQueryRewritingCache | None = None,
        system_prompt: str | None = None,
        model: str = "agentic-decisions",
        temperature: float = 0.1,
        max_tokens: int = 500,
        cache_ttl: int = 3600,
        max_query_length: int = 4000,
    ):
        """Initialize the query rewriter.
        
        Args:
            llm_provider: LLM provider instance. If None, creates default.
            cache: Cache instance for query results. If None, uses NullCache.
            system_prompt: Custom system prompt for the LLM.
            model: Model group to use for rewriting.
            temperature: Sampling temperature (lower for more deterministic output).
            max_tokens: Maximum tokens for rewrite response.
            cache_ttl: Cache time-to-live in seconds.
            max_query_length: Maximum allowed query length.
        """
        self.llm_provider = llm_provider or LLMProvider()
        self.cache = cache or NullQueryRewritingCache()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl
        self.max_query_length = max_query_length
        
        logger.info(f"Initialized QueryRewriter with model: {model}")
    
    async def rewrite(
        self,
        query: str,
        context: dict[str, Any] | None = None
    ) -> QueryRewriteResult:
        """Rewrite user query for optimal retrieval and generation.
        
        This method:
        1. Checks cache for existing result
        2. If not cached, calls LLM to rewrite the query
        3. Validates and parses the JSON response
        4. Caches the result for future use
        
        Args:
            query: Original user query
            context: Optional conversation context with keys like:
                - previous_queries: List of previous user queries
                - previous_responses: List of previous assistant responses
                - session_id: Conversation session identifier
                
        Returns:
            QueryRewriteResult with search and generation components
            
        Raises:
            QueryRewritingError: If rewriting fails
            QueryRewritingValidationError: If LLM response is invalid
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
        
        # Check query length
        if len(query) > self.max_query_length:
            logger.warning(f"Query too long ({len(query)} chars), truncating")
            query = query[:self.max_query_length]
        
        try:
            # Check cache first
            cached_result = await self.cache.get(query, context)
            if cached_result is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
            
            # Build the prompt with optional context
            prompt = self._build_prompt(query, context)
            
            # Call LLM for rewriting
            messages = [
                ChatMessage.system(self.system_prompt),
                ChatMessage.user(prompt),
            ]
            
            response = await self.llm_provider.chat_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            
            # Parse and validate the result
            result = self._parse_response(response.content, query)
            
            # Cache the result
            await self.cache.set(query, result, context, self.cache_ttl)
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "Query rewrite completed",
                extra={
                    "query_length": len(query),
                    "elapsed_ms": elapsed_ms,
                    "search_rag": result.search_rag,
                }
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise QueryRewritingValidationError(f"Invalid JSON response: {e}") from e
        except Exception as e:
            if isinstance(e, QueryRewritingError):
                raise
            logger.error(f"Query rewriting failed: {e}")
            raise QueryRewritingError(f"Failed to rewrite query: {e}") from e
    
    def _build_prompt(
        self,
        query: str,
        context: dict[str, Any] | None = None
    ) -> str:
        """Build the prompt for the LLM.
        
        Args:
            query: User query
            context: Optional conversation context
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [f'Query: "{query}"']
        
        # Add conversation context if available
        if context:
            previous_queries = context.get("previous_queries", [])
            previous_responses = context.get("previous_responses", [])
            
            if previous_queries:
                prompt_parts.append("\nPrevious queries in this conversation:")
                for i, prev_query in enumerate(previous_queries[-3:], 1):  # Last 3 only
                    prompt_parts.append(f"  {i}. {prev_query}")
            
            if previous_responses:
                prompt_parts.append("\nThis is a follow-up question.")
        
        prompt_parts.append(
            "\nRewrite this query into the required JSON format."
        )
        
        return "\n".join(prompt_parts)
    
    def _parse_response(self, content: str, original_query: str) -> QueryRewriteResult:
        """Parse and validate the LLM response.
        
        Args:
            content: LLM response content
            original_query: Original user query for fallback
            
        Returns:
            Parsed QueryRewriteResult
            
        Raises:
            QueryRewritingValidationError: If response is invalid
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise QueryRewritingValidationError(f"Invalid JSON: {e}") from e
        
        # Validate required fields
        required_fields = ["search_rag", "embedding_source_text", "llm_query"]
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            raise QueryRewritingValidationError(
                f"Missing required fields: {missing_fields}"
            )
        
        # Validate field types
        if not isinstance(data["search_rag"], bool):
            raise QueryRewritingValidationError(
                "Field 'search_rag' must be a boolean"
            )
        
        if not isinstance(data["embedding_source_text"], str):
            raise QueryRewritingValidationError(
                "Field 'embedding_source_text' must be a string"
            )
        
        if not isinstance(data["llm_query"], str):
            raise QueryRewritingValidationError(
                "Field 'llm_query' must be a string"
            )
        
        # If @knowledgebase is in original query but not detected, force it
        search_rag = data["search_rag"]
        if "@knowledgebase" in original_query.lower() and not search_rag:
            search_rag = True
            logger.debug("Forcing search_rag=True due to @knowledgebase in query")
        
        return QueryRewriteResult(
            search_rag=search_rag,
            embedding_source_text=data["embedding_source_text"].strip(),
            llm_query=data["llm_query"].strip(),
        )
    
    async def rewrite_batch(
        self,
        queries: list[str],
        context: dict[str, Any] | None = None
    ) -> list[QueryRewriteResult]:
        """Rewrite multiple queries in batch.
        
        Args:
            queries: List of user queries
            context: Optional conversation context
            
        Returns:
            List of QueryRewriteResult objects
        """
        results = []
        for query in queries:
            try:
                result = await self.rewrite(query, context)
                results.append(result)
            except QueryRewritingError as e:
                logger.warning(f"Failed to rewrite query '{query[:50]}...': {e}")
                # Return default result on failure
                results.append(QueryRewriteResult(
                    search_rag="@knowledgebase" in query.lower(),
                    embedding_source_text=query,
                    llm_query=query,
                ))
        return results
    
    async def health_check(self) -> dict[str, Any]:
        """Check the health of the query rewriter.
        
        Returns:
            Dictionary with health status information
        """
        result: dict[str, Any] = {
            "healthy": False,
            "components": {},
        }
        
        # Check LLM provider
        try:
            llm_health = await self.llm_provider.health_check(self.model)
            result["components"]["llm_provider"] = llm_health
            result["healthy"] = llm_health.get("healthy", False)
        except Exception as e:
            result["components"]["llm_provider"] = {
                "healthy": False,
                "error": str(e),
            }
        
        # Check cache (non-critical)
        if isinstance(self.cache, QueryRewritingCache) and self.cache.redis:
            result["components"]["cache"] = {"healthy": True, "enabled": True}
        else:
            result["components"]["cache"] = {"healthy": True, "enabled": False}
        
        return result
