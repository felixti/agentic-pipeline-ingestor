"""Hybrid search strategy with Reciprocal Rank Fusion (RRF) and metadata filtering.

This module provides enhanced hybrid search capabilities that combine vector similarity
search (semantic) with full-text search (lexical) using Reciprocal Rank Fusion (RRF)
with configurable weights. It supports metadata filtering, weight presets, and query
expansion for improved retrieval quality.

Example:
    >>> from src.rag.strategies.hybrid_search import EnhancedHybridSearch
    >>> search = EnhancedHybridSearch(vector_service, text_service)
    >>> results = await search.search(
    ...     query="machine learning",
    ...     embedding=[0.1, 0.2, ...],
    ...     filters={"source_type": "pdf", "author": "John Doe"},
    ...     weight_preset="balanced",
    ...     limit=10
    ... )
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.config import settings
from src.db.models import DocumentChunkModel
from src.observability.logging import get_logger
from src.services.text_search_service import TextSearchResult, TextSearchService
from src.services.vector_search_service import SearchResult, VectorSearchService

logger = get_logger(__name__)


class HybridSearchError(Exception):
    """Base exception for hybrid search errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}


class InvalidFusionMethodError(HybridSearchError):
    """Raised when an invalid fusion method is specified."""
    pass


class InvalidWeightError(HybridSearchError):
    """Raised when weights are invalid."""
    pass


class InvalidPresetError(HybridSearchError):
    """Raised when an invalid weight preset is specified."""
    pass


class InvalidFilterError(HybridSearchError):
    """Raised when invalid filters are specified."""
    pass


class FusionMethod(str, Enum):
    """Fusion methods for combining vector and text search results.

    Attributes:
        WEIGHTED_SUM: Linear combination of normalized scores
        RECIPROCAL_RANK_FUSION: Rank-based fusion using RRF formula with weights
    """

    WEIGHTED_SUM = "weighted_sum"
    RECIPROCAL_RANK_FUSION = "rrf"


class WeightPreset(str, Enum):
    """Predefined weight configurations for different search focuses.

    Attributes:
        SEMANTIC_FOCUS: Emphasizes vector/semantic search (vector: 0.9, text: 0.1)
        BALANCED: Equal balance between semantic and lexical (vector: 0.7, text: 0.3)
        LEXICAL_FOCUS: Emphasizes text/lexical search (vector: 0.3, text: 0.7)
    """

    SEMANTIC_FOCUS = "semantic_focus"
    BALANCED = "balanced"
    LEXICAL_FOCUS = "lexical_focus"


@dataclass
class HybridSearchResult:
    """Result of a hybrid search operation.

    Attributes:
        chunk: The matching DocumentChunkModel
        hybrid_score: Final combined score from fusion method (0-1)
        vector_score: Cosine similarity score from vector search (0-1), or None
        text_score: BM25 or normalized score from text search (0-1), or None
        vector_rank: Rank in vector search results (1-based), or None
        text_rank: Rank in text search results (1-based), or None
        fusion_method: Name of the fusion method used
        rank: Final rank in combined results (1-based)
        matched_filters: List of filters that matched for this result
    """

    chunk: DocumentChunkModel
    hybrid_score: float
    vector_score: float | None = None
    text_score: float | None = None
    vector_rank: int | None = None
    text_rank: int | None = None
    fusion_method: str = "rrf"
    rank: int = 0
    matched_filters: list[str] = field(default_factory=list)


@dataclass
class MetadataFilter:
    """Metadata filter configuration.

    Attributes:
        field: Filter field name (source_type, document_type, created_date, author, tags)
        value: Filter value (can include operators like >, <, >=, <= for dates)
        operator: Comparison operator (eq, ne, gt, lt, gte, lte, in, contains)
    """

    field: str
    value: Any
    operator: str = "eq"

    # Valid filterable fields
    VALID_FIELDS = {
        "source_type",
        "document_type",
        "created_date",
        "author",
        "tags",
        "job_id",
        "chunk_index",
        "content_hash",
    }

    # Valid operators
    VALID_OPERATORS = {"eq", "ne", "gt", "lt", "gte", "lte", "in", "contains"}

    def __post_init__(self) -> None:
        """Validate filter configuration."""
        if self.field not in self.VALID_FIELDS:
            raise InvalidFilterError(
                f"Invalid filter field: {self.field}. "
                f"Must be one of: {', '.join(sorted(self.VALID_FIELDS))}"
            )
        if self.operator not in self.VALID_OPERATORS:
            raise InvalidFilterError(
                f"Invalid filter operator: {self.operator}. "
                f"Must be one of: {', '.join(sorted(self.VALID_OPERATORS))}"
            )


@dataclass
class QueryExpansionResult:
    """Result of query expansion.

    Attributes:
        expanded_query: Query with expanded terms
        original_terms: Original query terms
        expanded_terms: Additional terms added
        expansion_method: Method used for expansion
    """

    expanded_query: str
    original_terms: list[str]
    expanded_terms: list[str]
    expansion_method: str = "synonym"


class QueryExpander:
    """Query expansion for better lexical matching.

    Expands search queries with synonyms and related terms to improve
    recall in full-text search.
    """

    # Common synonym mappings for query expansion
    SYNONYMS: dict[str, list[str]] = {
        "ml": ["machine learning"],
        "ai": ["artificial intelligence"],
        "nlp": ["natural language processing"],
        "cv": ["computer vision"],
        "dl": ["deep learning"],
        "nn": ["neural network"],
        "llm": ["large language model"],
        "rag": ["retrieval augmented generation"],
        "api": ["application programming interface"],
        "db": ["database"],
        "ui": ["user interface"],
        "ux": ["user experience"],
    }

    def __init__(self, max_expanded_terms: int = 5):
        """Initialize query expander.

        Args:
            max_expanded_terms: Maximum number of expanded terms to add
        """
        self.max_expanded_terms = max_expanded_terms

    def expand(self, query: str) -> QueryExpansionResult:
        """Expand query with synonyms and related terms.

        Args:
            query: Original search query

        Returns:
            QueryExpansionResult with expanded query
        """
        original_terms = query.lower().split()
        expanded_terms: list[str] = []

        # Check for acronym expansions
        for term in original_terms:
            if term in self.SYNONYMS:
                synonyms = self.SYNONYMS[term][: self.max_expanded_terms]
                expanded_terms.extend(synonyms)

        # Add expanded terms to query
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
        else:
            expanded_query = query

        return QueryExpansionResult(
            expanded_query=expanded_query,
            original_terms=original_terms,
            expanded_terms=expanded_terms,
            expansion_method="synonym",
        )

    def expand_with_patterns(self, query: str) -> QueryExpansionResult:
        """Expand query using pattern matching and stemming.

        Args:
            query: Original search query

        Returns:
            QueryExpansionResult with expanded query
        """
        original_terms = query.lower().split()
        expanded_terms: list[str] = []

        # Add common variations
        for term in original_terms:
            # Add plural form if singular
            if not term.endswith("s") and len(term) > 2:
                expanded_terms.append(f"{term}s")
            # Add singular form if plural
            elif term.endswith("s") and len(term) > 3:
                expanded_terms.append(term[:-1])

        # Add expanded terms to query (deduplicated)
        all_terms = list(dict.fromkeys(original_terms + expanded_terms))
        expanded_query = " ".join(all_terms[: len(original_terms) + self.max_expanded_terms])

        return QueryExpansionResult(
            expanded_query=expanded_query,
            original_terms=original_terms,
            expanded_terms=list(set(expanded_terms)),
            expansion_method="pattern",
        )


class EnhancedHybridSearch:
    """Enhanced hybrid search with RRF fusion and metadata filtering.

    Provides methods for hybrid search that combines results from vector
    similarity search (semantic) and text search (lexical) using configurable
    fusion methods: weighted sum or Reciprocal Rank Fusion (RRF) with weights.

    Supports metadata filtering, weight presets, and query expansion.

    Example:
        >>> search = EnhancedHybridSearch(vector_service, text_service)
        >>> results = await search.search(
        ...     query="machine learning",
        ...     embedding=[0.1, 0.2, ...],
        ...     filters={"source_type": "pdf"},
        ...     weight_preset="semantic_focus",
        ...     limit=10
        ... )
    """

    # Valid fallback modes
    FALLBACK_MODES = {"auto", "vector", "text", "strict"}

    def __init__(
        self,
        vector_service: VectorSearchService,
        text_service: TextSearchService,
        config: dict[str, Any] | None = None,
    ):
        """Initialize the enhanced hybrid search.

        Args:
            vector_service: VectorSearchService for semantic similarity search
            text_service: TextSearchService for text-based search
            config: Optional configuration dictionary
        """
        self.vector_service = vector_service
        self.text_service = text_service
        self.config = config or {}
        self.logger = logger

        # Initialize query expander
        self.query_expander = QueryExpander(
            max_expanded_terms=settings.hybrid_search.query_expansion_max_terms
        )

        # Use settings from config if not provided
        self.rrf_k = self.config.get("rrf_k", settings.hybrid_search.rrf_k)
        self.default_vector_weight = self.config.get(
            "default_vector_weight", settings.hybrid_search.default_vector_weight
        )
        self.default_text_weight = self.config.get(
            "default_text_weight", settings.hybrid_search.default_text_weight
        )
        self.default_fusion_method = FusionMethod(
            self.config.get("default_fusion_method", settings.hybrid_search.default_fusion_method)
        )
        self.weight_presets = settings.hybrid_search.weight_presets
        self.query_expansion_enabled = self.config.get(
            "query_expansion_enabled", settings.hybrid_search.query_expansion_enabled
        )

    async def search(
        self,
        query: str,
        embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        vector_weight: float | None = None,
        text_weight: float | None = None,
        weight_preset: str | None = None,
        fusion_method: FusionMethod | None = None,
        use_query_expansion: bool = True,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search with RRF fusion and metadata filtering.

        Args:
            query: Text query for full-text search
            embedding: Vector embedding for similarity search
            filters: Metadata filters (e.g., {"source_type": "pdf", "date": ">2024-01-01"})
            limit: Number of results to return
            vector_weight: Weight for vector scores (overrides preset)
            text_weight: Weight for text scores (overrides preset)
            weight_preset: Weight preset name (semantic_focus, balanced, lexical_focus)
            fusion_method: Fusion method to use (default: rrf)
            use_query_expansion: Whether to expand query for better lexical matching

        Returns:
            Fused and ranked search results

        Raises:
            HybridSearchError: If search fails
            InvalidWeightError: If weights don't sum to 1.0
            InvalidPresetError: If weight preset is invalid
        """
        start_time = time.monotonic()

        try:
            # Determine weights (preset overrides defaults, explicit weights override preset)
            if weight_preset:
                preset_weights = self._get_preset_weights(weight_preset)
                vector_weight = vector_weight or preset_weights["vector"]
                text_weight = text_weight or preset_weights["text"]
            else:
                vector_weight = vector_weight or self.default_vector_weight
                text_weight = text_weight or self.default_text_weight

            # Validate weights
            self._validate_weights(vector_weight, text_weight)

            # Use default fusion method if not specified
            fusion_method = fusion_method or self.default_fusion_method

            self.logger.info(
                "enhanced_hybrid_search_started",
                query=query[:100],
                limit=limit,
                vector_weight=vector_weight,
                text_weight=text_weight,
                weight_preset=weight_preset,
                fusion_method=fusion_method.value,
                filters=filters,
            )

            # Expand query if enabled
            expanded_query = query
            expansion_result = None
            if use_query_expansion and self.query_expansion_enabled:
                expansion_result = self.query_expander.expand(query)
                expanded_query = expansion_result.expanded_query
                self.logger.debug(
                    "query_expanded",
                    original_query=query[:100],
                    expanded_query=expanded_query[:100],
                    expanded_terms=expansion_result.expanded_terms,
                )

            # Parse and validate filters
            metadata_filters = self._parse_filters(filters)

            # Get vector results (fetch more for better fusion)
            vector_results = await self.vector_service.search_by_vector(
                query_embedding=embedding,
                top_k=limit * 2,
                filters=self._convert_filters_for_vector(metadata_filters),
            )

            # Get text results
            text_results = await self.text_service.search_by_text(
                query=expanded_query,
                top_k=limit * 2,
                filters=self._convert_filters_for_text(metadata_filters),
            )

            self.logger.debug(
                "search_results_collected",
                vector_count=len(vector_results),
                text_count=len(text_results),
            )

            # Apply fusion based on method
            if fusion_method == FusionMethod.WEIGHTED_SUM:
                fused_results = self._fuse_weighted_sum(
                    vector_results, text_results, vector_weight, text_weight
                )
            elif fusion_method == FusionMethod.RECIPROCAL_RANK_FUSION:
                fused_results = self._fuse_rrf_weighted(
                    vector_results, text_results, vector_weight, text_weight, k=self.rrf_k
                )
            else:
                raise InvalidFusionMethodError(f"Unsupported fusion method: {fusion_method}")

            # Sort by hybrid score descending and assign final ranks
            fused_results.sort(key=lambda x: x.hybrid_score, reverse=True)
            for rank, result in enumerate(fused_results, start=1):
                result.rank = rank

            # Limit to requested number
            fused_results = fused_results[:limit]

            duration_ms = (time.monotonic() - start_time) * 1000

            self.logger.info(
                "enhanced_hybrid_search_completed",
                query=query[:100],
                result_count=len(fused_results),
                duration_ms=round(duration_ms, 2),
                fusion_method=fusion_method.value,
                weight_preset=weight_preset,
            )

            return fused_results

        except HybridSearchError:
            raise
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "enhanced_hybrid_search_error",
                query=query[:100],
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise HybridSearchError(
                f"Hybrid search failed: {e}",
                context={"query": query[:100]},
            ) from e

    def _fuse_rrf_weighted(
        self,
        vector_results: list[SearchResult],
        text_results: list[TextSearchResult],
        vector_weight: float,
        text_weight: float,
        k: int = 60,
    ) -> list[HybridSearchResult]:
        """Fuse results using weighted Reciprocal Rank Fusion (RRF).

        Formula: score = sum(weight_i / (k + rank_i)) for each result's rank in each search type

        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            vector_weight: Weight for vector search contributions
            text_weight: Weight for text search contributions
            k: RRF constant (default: 60)

        Returns:
            List of HybridSearchResult with weighted RRF scores
        """
        # Build lookup maps by chunk ID with ranks and scores
        vector_map: dict[str, tuple[SearchResult, int]] = {}
        text_map: dict[str, tuple[TextSearchResult, int]] = {}

        for v_rank, v_res in enumerate(vector_results, start=1):
            chunk_id = str(v_res.chunk.id)
            vector_map[chunk_id] = (v_res, v_rank)

        for t_rank, t_res in enumerate(text_results, start=1):
            chunk_id = str(t_res.chunk.id)
            text_map[chunk_id] = (t_res, t_rank)

        # Get all unique chunk IDs
        all_chunk_ids = set(vector_map.keys()) | set(text_map.keys())

        fused_results: list[HybridSearchResult] = []

        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            vector_rank = None
            text_rank = None
            vector_score = None
            text_score = None
            chunk = None

            # Add weighted contribution from vector search if present
            if chunk_id in vector_map:
                v_result, v_rank = vector_map[chunk_id]
                rrf_score += vector_weight / (k + v_rank)
                vector_rank = v_rank
                vector_score = v_result.similarity_score
                chunk = v_result.chunk

            # Add weighted contribution from text search if present
            if chunk_id in text_map:
                t_result, t_rank = text_map[chunk_id]
                rrf_score += text_weight / (k + t_rank)
                text_rank = t_rank
                text_score = t_result.rank_score
                chunk = t_result.chunk

            fused_results.append(
                HybridSearchResult(
                    chunk=chunk,  # type: ignore[arg-type]
                    hybrid_score=round(rrf_score, 4),
                    vector_score=round(vector_score, 4) if vector_score else None,
                    text_score=round(text_score, 4) if text_score else None,
                    vector_rank=vector_rank,
                    text_rank=text_rank,
                    fusion_method=FusionMethod.RECIPROCAL_RANK_FUSION.value,
                )
            )

        self.logger.debug(
            "weighted_rrf_fusion_completed",
            input_vector_count=len(vector_results),
            input_text_count=len(text_results),
            output_count=len(fused_results),
            rrf_k=k,
            vector_weight=vector_weight,
            text_weight=text_weight,
        )

        return fused_results

    def _fuse_weighted_sum(
        self,
        vector_results: list[SearchResult],
        text_results: list[TextSearchResult],
        vector_weight: float,
        text_weight: float,
    ) -> list[HybridSearchResult]:
        """Fuse results using weighted sum of normalized scores.

        Formula: hybrid_score = (vector_weight * vector_score) + (text_weight * text_score)

        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            vector_weight: Weight for vector scores
            text_weight: Weight for text scores

        Returns:
            List of HybridSearchResult with fused scores
        """
        # Build lookup maps by chunk ID
        vector_map: dict[str, SearchResult] = {}
        text_map: dict[str, TextSearchResult] = {}

        for rank, v_res in enumerate(vector_results, start=1):
            chunk_id = str(v_res.chunk.id)
            vector_map[chunk_id] = v_res

        for rank, t_res in enumerate(text_results, start=1):
            chunk_id = str(t_res.chunk.id)
            text_map[chunk_id] = t_res

        # Get all unique chunk IDs
        all_chunk_ids = set(vector_map.keys()) | set(text_map.keys())

        # Normalize text scores to 0-1 range if needed
        text_scores_normalized = self._normalize_text_scores(text_results)

        fused_results: list[HybridSearchResult] = []

        for chunk_id in all_chunk_ids:
            vector_result = vector_map.get(chunk_id)
            text_result = text_map.get(chunk_id)

            # Get scores (use 0 if not present in one search)
            vector_score = vector_result.similarity_score if vector_result else 0.0
            text_score = text_scores_normalized.get(chunk_id, 0.0)

            # Get ranks
            vector_rank = vector_result.rank if vector_result else None
            text_rank = text_result.rank if text_result else None

            # Calculate weighted sum
            hybrid_score = (vector_weight * vector_score) + (text_weight * text_score)

            # Get the chunk (prefer vector result if available, otherwise text)
            chunk = vector_result.chunk if vector_result else text_result.chunk if text_result else None

            fused_results.append(
                HybridSearchResult(
                    chunk=chunk,  # type: ignore[arg-type]
                    hybrid_score=round(hybrid_score, 4),
                    vector_score=round(vector_score, 4) if vector_result else None,
                    text_score=round(text_score, 4) if text_result else None,
                    vector_rank=vector_rank,
                    text_rank=text_rank,
                    fusion_method=FusionMethod.WEIGHTED_SUM.value,
                )
            )

        self.logger.debug(
            "weighted_sum_fusion_completed",
            input_vector_count=len(vector_results),
            input_text_count=len(text_results),
            output_count=len(fused_results),
        )

        return fused_results

    def _normalize_text_scores(
        self, results: list[TextSearchResult]
    ) -> dict[str, float]:
        """Normalize text search scores to 0-1 range.

        Text search scores (BM25) can vary widely, so we normalize them
        to a 0-1 range for fair comparison with vector scores.

        Args:
            results: Text search results with raw scores

        Returns:
            Dictionary mapping chunk ID to normalized score (0-1)
        """
        if not results:
            return {}

        # Get max score for normalization
        max_score = max(r.rank_score for r in results)

        if max_score == 0:
            # All scores are 0, return as-is
            return {str(r.chunk.id): 0.0 for r in results}

        # Normalize each score
        normalized: dict[str, float] = {}
        for result in results:
            chunk_id = str(result.chunk.id)
            # Normalize to 0-1 range
            normalized_score = result.rank_score / max_score
            normalized[chunk_id] = min(1.0, max(0.0, normalized_score))

        return normalized

    def _get_preset_weights(self, preset: str) -> dict[str, float]:
        """Get weight configuration for a preset.

        Args:
            preset: Preset name (semantic_focus, balanced, lexical_focus)

        Returns:
            Dictionary with "vector" and "text" weights

        Raises:
            InvalidPresetError: If preset is not recognized
        """
        if preset not in self.weight_presets:
            valid_presets = ", ".join(self.weight_presets.keys())
            raise InvalidPresetError(
                f"Invalid weight preset: {preset}. Must be one of: {valid_presets}"
            )

        weights = self.weight_presets[preset]
        return {"vector": weights["vector"], "text": weights["text"]}

    def _parse_filters(self, filters: dict[str, Any] | None) -> list[MetadataFilter]:
        """Parse filter dictionary into MetadataFilter objects.

        Supports various filter formats:
        - Simple equality: {"source_type": "pdf"}
        - With operators: {"created_date": {">=": "2024-01-01"}}
        - Nested metadata: {"metadata": {"author": "John"}}

        Args:
            filters: Raw filter dictionary

        Returns:
            List of MetadataFilter objects
        """
        if not filters:
            return []

        parsed_filters: list[MetadataFilter] = []

        for field, value in filters.items():
            # Handle nested metadata filters
            if field == "metadata" and isinstance(value, dict):
                for meta_field, meta_value in value.items():
                    parsed_filters.append(
                        MetadataFilter(field=meta_field, value=meta_value, operator="eq")
                    )
                continue

            # Handle operators in value (e.g., {">=": "2024-01-01"})
            if isinstance(value, dict):
                for op, op_value in value.items():
                    # Map operator symbols to names
                    op_map: dict[str, str] = {
                        "=": "eq",
                        "!=": "ne",
                        ">": "gt",
                        "<": "lt",
                        ">=": "gte",
                        "<=": "lte",
                    }
                    operator_str: str = op_map.get(op, op) or "eq"
                    parsed_filters.append(
                        MetadataFilter(field=field, value=op_value, operator=operator_str)
                    )
            else:
                # Simple equality filter
                parsed_filters.append(
                    MetadataFilter(field=field, value=value, operator="eq")
                )

        return parsed_filters

    def _convert_filters_for_vector(
        self, filters: list[MetadataFilter]
    ) -> dict[str, Any] | None:
        """Convert metadata filters to vector service filter format.

        Args:
            filters: List of MetadataFilter objects

        Returns:
            Filter dictionary for vector service
        """
        if not filters:
            return None

        result: dict[str, Any] = {"metadata": {}}

        for f in filters:
            # Handle special fields
            if f.field == "job_id":
                result["job_id"] = f.value
            elif f.field == "chunk_index":
                result["chunk_index"] = f.value
            elif f.field == "content_hash":
                result["content_hash"] = f.value
            else:
                # Add to metadata filters
                result["metadata"][f.field] = f.value

        # Remove empty metadata dict
        if not result["metadata"]:
            del result["metadata"]

        return result or None

    def _convert_filters_for_text(
        self, filters: list[MetadataFilter]
    ) -> dict[str, Any] | None:
        """Convert metadata filters to text service filter format.

        Args:
            filters: List of MetadataFilter objects

        Returns:
            Filter dictionary for text service
        """
        # Same format as vector service for now
        return self._convert_filters_for_vector(filters)

    def _validate_weights(self, vector_weight: float, text_weight: float) -> None:
        """Validate that weights sum to approximately 1.0.

        Args:
            vector_weight: Weight for vector scores
            text_weight: Weight for text scores

        Raises:
            InvalidWeightError: If weights are invalid
        """
        if vector_weight < 0 or text_weight < 0:
            raise InvalidWeightError("Weights must be non-negative")

        weight_sum = vector_weight + text_weight
        if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point error
            raise InvalidWeightError(
                f"Weights must sum to 1.0, got {weight_sum} "
                f"(vector={vector_weight}, text={text_weight})"
            )


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    text_results: list[TextSearchResult],
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Standalone function for Reciprocal Rank Fusion.

    This is a convenience function for performing RRF without instantiating
    the EnhancedHybridSearch class.

    Formula: RRF_score(d) = sum(weight_i / (k + rank_i(d)))

    Args:
        vector_results: Results from vector search
        text_results: Results from text search
        vector_weight: Weight for vector scores (default: 0.7)
        text_weight: Weight for text scores (default: 0.3)
        k: RRF constant (default: 60)

    Returns:
        List of (chunk_id, rrf_score) tuples sorted by score descending

    Example:
        >>> fused = reciprocal_rank_fusion(
        ...     vector_results=[SearchResult(...), ...],
        ...     text_results=[TextSearchResult(...), ...],
        ...     vector_weight=0.7,
        ...     text_weight=0.3,
        ...     k=60
        ... )
        >>> for chunk_id, score in fused:
        ...     print(f"{chunk_id}: {score}")
    """
    scores: defaultdict[str, float] = defaultdict(float)

    # Score from vector results
    for v_rank, v_result in enumerate(vector_results):
        chunk_id = str(v_result.chunk.id)
        scores[chunk_id] += vector_weight / (k + v_rank + 1)  # +1 for 1-based rank

    # Score from text results
    for t_rank, t_result in enumerate(text_results):
        chunk_id = str(t_result.chunk.id)
        scores[chunk_id] += text_weight / (k + t_rank + 1)  # +1 for 1-based rank

    # Sort by fused score descending
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results
