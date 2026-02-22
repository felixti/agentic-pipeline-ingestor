"""Query Classification Service for RAG strategy selection.

This module provides LLM-based query classification to determine the optimal
RAG retrieval strategy based on query type. It supports:
- LLM-based classification with confidence scoring
- Pattern-based fallback for fast classification
- Caching of classification results
- Strategy selection matrix per query type
"""

import hashlib
import json
import logging
import re
import time
from typing import Any

import redis.asyncio as redis

from src.llm.provider import ChatMessage, LLMProvider
from src.rag.models import QueryClassification, QueryType

logger = logging.getLogger(__name__)

# Default system prompt for query classification
DEFAULT_CLASSIFICATION_PROMPT = """You are a query classification assistant for a RAG (Retrieval-Augmented Generation) system.

Your task is to analyze user queries and classify them into one of these types:

**FACTUAL**: Simple fact lookup questions
- Examples: "What is X?", "Who is Y?", "When did Z happen?", "Where is W located?"
- Characteristics: Asks for specific, discrete facts; usually has a single correct answer

**ANALYTICAL**: Requires explanation, synthesis, or reasoning
- Examples: "Explain X", "How does Y work?", "Why is Z important?", "Analyze the causes of W"
- Characteristics: Needs deeper understanding, explanation of mechanisms, or reasoning

**COMPARATIVE**: Compares multiple items or asks for pros/cons
- Examples: "Compare X and Y", "What are the differences between A and B?", "Pros and cons of Z"
- Characteristics: Explicitly compares entities or asks for trade-offs

**VAGUE**: Unclear, broad, or needs clarification
- Examples: "Tell me about...", "Information on...", "Stuff related to X", "Anything about Y"
- Characteristics: Lacks specificity, too broad to answer directly

**MULTI_HOP**: Requires multiple retrieval or reasoning steps
- Examples: "What did the author of X say about Y?", "Companies founded by people who worked at Z"
- Characteristics: Requires connecting information across multiple documents or steps

Respond with a JSON object containing exactly these fields:
- "query_type": One of "factual", "analytical", "comparative", "vague", "multi_hop"
- "confidence": Number between 0.0 and 1.0 indicating classification confidence
- "reasoning": Brief explanation (1-2 sentences) of why this classification was chosen

Example response:
{
  "query_type": "factual",
  "confidence": 0.95,
  "reasoning": "Query asks for a specific definition with clear subject"
}"""

# Strategy selection matrix based on query type
# Maps query types to recommended RAG strategies
STRATEGY_MATRIX: dict[QueryType, dict[str, bool]] = {
    QueryType.FACTUAL: {
        "hyde": False,
        "reranking": True,
        "hybrid": True,
        "query_rewrite": True,
    },
    QueryType.ANALYTICAL: {
        "hyde": True,
        "reranking": True,
        "hybrid": True,
        "query_rewrite": True,
    },
    QueryType.COMPARATIVE: {
        "hyde": False,
        "reranking": True,
        "hybrid": True,
        "query_rewrite": True,
    },
    QueryType.VAGUE: {
        "hyde": True,
        "reranking": True,
        "hybrid": True,
        "query_rewrite": True,
    },
    QueryType.MULTI_HOP: {
        "hyde": True,
        "reranking": True,
        "hybrid": True,
        "query_rewrite": True,
    },
}

# Pattern-based classification rules (used as fallback or for fast path)
# Each tuple: (pattern, query_type, confidence)
CLASSIFICATION_PATTERNS: list[tuple[re.Pattern[str], QueryType, float]] = [
    # Factual patterns
    (re.compile(r"^(what is|what's|who is|who's|when did|where is|where's|how many|how much|what year|what date)\s", re.IGNORECASE), QueryType.FACTUAL, 0.90),
    (re.compile(r"^(define|explain what|tell me what|what does .+ mean)\s", re.IGNORECASE), QueryType.FACTUAL, 0.85),
    (re.compile(r"^(is |are |was |were |does |do |can |could |would |will |has |have |had )", re.IGNORECASE), QueryType.FACTUAL, 0.80),
    
    # Comparative patterns
    (re.compile(r"(compare|versus|vs\.|vs |difference between|similarities between|pros and cons|advantages?|disadvantages?)", re.IGNORECASE), QueryType.COMPARATIVE, 0.90),
    (re.compile(r"^(which is better|which is worse|which is more|which is less|which has|which does)\s", re.IGNORECASE), QueryType.COMPARATIVE, 0.85),
    (re.compile(r"\b(better than|worse than|different from|similar to)\b", re.IGNORECASE), QueryType.COMPARATIVE, 0.80),
    
    # Multi-hop patterns
    (re.compile(r"(author of|writer of|creator of).+(said|say|think|thought|opinion|believe|wrote|wrote about|on|about)", re.IGNORECASE), QueryType.MULTI_HOP, 0.90),
    (re.compile(r"(founded by|created by|written by|developed by).+(who|what|when|where)", re.IGNORECASE), QueryType.MULTI_HOP, 0.85),
    (re.compile(r"\b(company|companies).+(founded|started|established).+(by|where|when)", re.IGNORECASE), QueryType.MULTI_HOP, 0.80),
    (re.compile(r"\b(people who|person who|author who|writer who).+(worked|created|founded|started).+(at|for|in|about|on)\b", re.IGNORECASE), QueryType.MULTI_HOP, 0.85),
    
    # Analytical patterns
    (re.compile(r"^(explain|analyze|how does|why does|what causes|what led to|reasons? for|impact of|effect of)\s", re.IGNORECASE), QueryType.ANALYTICAL, 0.85),
    (re.compile(r"\b(purpose|reason|cause|effect|impact|significance|importance|implications?)\b", re.IGNORECASE), QueryType.ANALYTICAL, 0.75),
    
    # Vague patterns
    (re.compile(r"^(tell me about|information on|stuff about|anything about|something about|details? about)\s", re.IGNORECASE), QueryType.VAGUE, 0.90),
    (re.compile(r"^(what can you tell me|what do you know|i want to know)\s", re.IGNORECASE), QueryType.VAGUE, 0.85),
    (re.compile(r"^(?:related|connected|associated)\s+(?:to|with)", re.IGNORECASE), QueryType.VAGUE, 0.75),
]


class ClassificationError(Exception):
    """Base exception for classification errors."""
    pass


class ClassificationValidationError(ClassificationError):
    """Exception raised when the LLM response fails validation."""
    pass


class ClassificationCache:
    """Redis cache for query classification results.
    
    Provides caching layer to improve performance for repeated queries.
    """
    
    CACHE_PREFIX = "rag:classification:"
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
    ) -> QueryClassification | None:
        """Get cached classification result.
        
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
            return QueryClassification(**record_dict)
            
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
            return None
    
    async def set(
        self,
        query: str,
        result: QueryClassification,
        context: dict[str, Any] | None = None,
        ttl: int | None = None
    ) -> bool:
        """Cache classification result.
        
        Args:
            query: User query string
            result: Classification result to cache
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
        """Delete cached classification result.
        
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
        """Clear all classification cache entries.
        
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


class NullClassificationCache:
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
        result: QueryClassification,
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


class QueryClassifier:
    """Query Classification Service for RAG strategy selection.
    
    This class uses an LLM to classify queries into types (factual, analytical,
    comparative, vague, multi_hop) to determine the optimal RAG retrieval strategy.
    It also provides pattern-based fallback classification for fast processing.
    
    Attributes:
        llm_provider: LLM provider for classification
        cache: Cache for storing classification results
        system_prompt: System prompt for the LLM classifier
        model: Model group to use for classification
        temperature: Sampling temperature for classification
        max_tokens: Maximum tokens for classification response
        min_confidence_threshold: Minimum confidence for accepting classification
        use_pattern_fallback: Whether to use pattern-based fallback
        max_query_length: Maximum allowed query length
    
    Example:
        >>> classifier = QueryClassifier()
        >>> result = await classifier.classify("What is machine learning?")
        >>> print(result.query_type)  # QueryType.FACTUAL
        >>> print(result.confidence)  # 0.95
        >>> print(result.suggested_strategies)  # ["query_rewrite", "reranking", "hybrid_search"]
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        cache: ClassificationCache | NullClassificationCache | None = None,
        system_prompt: str | None = None,
        model: str = "agentic-decisions",
        temperature: float = 0.1,
        max_tokens: int = 300,
        cache_ttl: int = 3600,
        min_confidence_threshold: float = 0.7,
        use_pattern_fallback: bool = True,
        max_query_length: int = 4000,
    ):
        """Initialize the query classifier.
        
        Args:
            llm_provider: LLM provider instance. If None, creates default.
            cache: Cache instance for classification results. If None, uses NullCache.
            system_prompt: Custom system prompt for the LLM.
            model: Model group to use for classification.
            temperature: Sampling temperature (lower for more deterministic output).
            max_tokens: Maximum tokens for classification response.
            cache_ttl: Cache time-to-live in seconds.
            min_confidence_threshold: Minimum confidence for accepting classification.
            use_pattern_fallback: Whether to use pattern-based fallback classification.
            max_query_length: Maximum allowed query length.
        """
        self.llm_provider = llm_provider or LLMProvider()
        self.cache = cache or NullClassificationCache()
        self.system_prompt = system_prompt or DEFAULT_CLASSIFICATION_PROMPT
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl
        self.min_confidence_threshold = min_confidence_threshold
        self.use_pattern_fallback = use_pattern_fallback
        self.max_query_length = max_query_length
        
        logger.info(f"Initialized QueryClassifier with model: {model}")
    
    def _classify_by_pattern(self, query: str) -> QueryClassification | None:
        """Classify query using regex patterns (fast path).
        
        This method provides fast classification using predefined regex patterns
        without requiring an LLM call.
        
        Args:
            query: User query string
            
        Returns:
            QueryClassification if pattern match found, None otherwise
        """
        if not self.use_pattern_fallback:
            return None
        
        query_lower = query.lower().strip()
        
        # Check patterns in order of confidence
        for pattern, query_type, confidence in CLASSIFICATION_PATTERNS:
            if pattern.search(query_lower):
                # Get suggested strategies for this query type
                strategies = self._get_suggested_strategies(query_type)
                
                return QueryClassification(
                    query_type=query_type,
                    confidence=confidence,
                    reasoning=f"Pattern match: {pattern.pattern[:50]}...",
                    suggested_strategies=strategies,
                )
        
        return None
    
    def _get_suggested_strategies(self, query_type: QueryType) -> list[str]:
        """Get suggested RAG strategies for a query type.
        
        Args:
            query_type: The classified query type
            
        Returns:
            List of suggested strategy names
        """
        strategies = STRATEGY_MATRIX.get(query_type, {})
        return [name for name, enabled in strategies.items() if enabled]
    
    def _build_prompt(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Build the classification prompt for the LLM.
        
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
            
            if previous_queries:
                prompt_parts.append("\nPrevious queries in this conversation:")
                for i, prev_query in enumerate(previous_queries[-3:], 1):  # Last 3 only
                    prompt_parts.append(f"  {i}. {prev_query}")
        
        prompt_parts.append("\nClassify this query and respond with JSON.")
        
        return "\n".join(prompt_parts)
    
    def _parse_response(self, content: str) -> QueryClassification:
        """Parse and validate the LLM classification response.
        
        Args:
            content: LLM response content (JSON string)
            
        Returns:
            Parsed QueryClassification
            
        Raises:
            ClassificationValidationError: If response is invalid
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ClassificationValidationError(f"Invalid JSON: {e}") from e
        
        # Validate required fields
        required_fields = ["query_type", "confidence", "reasoning"]
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            raise ClassificationValidationError(
                f"Missing required fields: {missing_fields}"
            )
        
        # Validate query_type
        query_type_str = data["query_type"].lower()
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            valid_types = [qt.value for qt in QueryType]
            raise ClassificationValidationError(
                f"Invalid query_type: {query_type_str}. Must be one of: {valid_types}"
            )
        
        # Validate confidence
        confidence = data["confidence"]
        if not isinstance(confidence, (int, float)):
            raise ClassificationValidationError("Field 'confidence' must be a number")
        
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ClassificationValidationError(
                f"Field 'confidence' must be between 0.0 and 1.0, got {confidence}"
            )
        
        # Validate reasoning
        reasoning = data["reasoning"]
        if not isinstance(reasoning, str):
            raise ClassificationValidationError("Field 'reasoning' must be a string")
        
        # Get suggested strategies
        suggested_strategies = self._get_suggested_strategies(query_type)
        
        return QueryClassification(
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning.strip(),
            suggested_strategies=suggested_strategies,
        )
    
    async def classify(
        self,
        query: str,
        context: dict[str, Any] | None = None
    ) -> QueryClassification:
        """Classify query to determine optimal RAG strategy.
        
        This method:
        1. Checks cache for existing classification
        2. Attempts pattern-based classification (fast path)
        3. If no pattern match or low confidence, calls LLM for classification
        4. Caches the result for future use
        
        Args:
            query: Original user query
            context: Optional conversation context with keys like:
                - previous_queries: List of previous user queries
                - session_id: Conversation session identifier
                
        Returns:
            QueryClassification with query type, confidence, reasoning, and strategies
            
        Raises:
            ClassificationError: If classification fails
        """
        start_time = time.time()
        
        # Validate input
        if not query or not query.strip():
            logger.warning("Empty query received, returning default classification")
            return QueryClassification(
                query_type=QueryType.VAGUE,
                confidence=0.5,
                reasoning="Empty query, defaulting to vague",
                suggested_strategies=self._get_suggested_strategies(QueryType.VAGUE),
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
            
            # Try pattern-based classification first (fast path)
            pattern_result = self._classify_by_pattern(query)
            if pattern_result is not None and pattern_result.confidence >= self.min_confidence_threshold:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    "Pattern-based classification completed",
                    extra={
                        "query_length": len(query),
                        "elapsed_ms": elapsed_ms,
                        "query_type": pattern_result.query_type.value,
                        "confidence": pattern_result.confidence,
                        "pattern_matched": True,
                    }
                )
                
                # Cache the pattern result
                await self.cache.set(query, pattern_result, context, self.cache_ttl)
                
                return pattern_result
            
            # Build the prompt with optional context
            prompt = self._build_prompt(query, context)
            
            # Call LLM for classification
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
            result = self._parse_response(response.content)
            
            # Cache the result
            await self.cache.set(query, result, context, self.cache_ttl)
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "LLM classification completed",
                extra={
                    "query_length": len(query),
                    "elapsed_ms": elapsed_ms,
                    "query_type": result.query_type.value,
                    "confidence": result.confidence,
                    "pattern_matched": False,
                }
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ClassificationValidationError(f"Invalid JSON response: {e}") from e
        except Exception as e:
            if isinstance(e, ClassificationError):
                raise
            logger.error(f"Classification failed: {e}")
            raise ClassificationError(f"Failed to classify query: {e}") from e
    
    async def classify_batch(
        self,
        queries: list[str],
        context: dict[str, Any] | None = None
    ) -> list[QueryClassification]:
        """Classify multiple queries in batch.
        
        Args:
            queries: List of user queries
            context: Optional conversation context
            
        Returns:
            List of QueryClassification objects
        """
        results = []
        for query in queries:
            try:
                result = await self.classify(query, context)
                results.append(result)
            except ClassificationError as e:
                logger.warning(f"Failed to classify query '{query[:50]}...': {e}")
                # Return default classification on failure
                results.append(QueryClassification(
                    query_type=QueryType.VAGUE,
                    confidence=0.5,
                    reasoning=f"Classification failed: {e}",
                    suggested_strategies=self._get_suggested_strategies(QueryType.VAGUE),
                ))
        return results
    
    async def get_strategy_config(self, query: str, context: dict[str, Any] | None = None) -> dict[str, bool]:
        """Get RAG strategy configuration for a query.
        
        This is a convenience method that classifies the query and returns
        the recommended strategy configuration.
        
        Args:
            query: User query string
            context: Optional conversation context
            
        Returns:
            Dictionary mapping strategy names to enabled status
        """
        classification = await self.classify(query, context)
        return STRATEGY_MATRIX.get(classification.query_type, STRATEGY_MATRIX[QueryType.VAGUE])
    
    async def health_check(self) -> dict[str, Any]:
        """Check the health of the query classifier.
        
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
        if isinstance(self.cache, ClassificationCache) and self.cache.redis:
            result["components"]["cache"] = {"healthy": True, "enabled": True}
        else:
            result["components"]["cache"] = {"healthy": True, "enabled": False}
        
        return result


def get_strategy_matrix() -> dict[QueryType, dict[str, bool]]:
    """Get the strategy selection matrix.
    
    Returns:
        Dictionary mapping query types to strategy configurations
    """
    import copy
    return copy.deepcopy(STRATEGY_MATRIX)


def get_classification_patterns() -> list[tuple[str, str, float]]:
    """Get the pattern-based classification rules.
    
    Returns:
        List of (pattern, query_type, confidence) tuples
    """
    return [
        (pattern.pattern, query_type.value, confidence)
        for pattern, query_type, confidence in CLASSIFICATION_PATTERNS
    ]
