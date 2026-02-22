"""Enhanced hybrid search with Reciprocal Rank Fusion (RRF) and metadata filtering.

This module provides an enhanced hybrid search service that combines vector similarity
and full-text search using Reciprocal Rank Fusion (RRF) with configurable weights.
It supports advanced metadata filtering and weight presets for different search strategies.

Example:
    >>> service = HybridSearchService(vector_service, text_service)
    >>> results = await service.search(
    ...     query="machine learning",
    ...     embedding=[0.1, 0.2, ...],
    ...     filters={"source_type": "pdf", "author": "John Doe"},
    ...     limit=10,
    ...     vector_weight=0.7,
    ...     text_weight=0.3
    ... )
"""

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from src.db.models import DocumentChunkModel
from src.observability.logging import get_logger
from src.services.text_search_service import TextSearchResult, TextSearchService
from src.services.vector_search_service import SearchResult, VectorSearchService

logger = get_logger(__name__)


class HybridSearchError(Exception):
    """Base exception for hybrid search errors."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}


class InvalidFusionMethodError(HybridSearchError):
    """Raised when an invalid fusion method is specified."""
    pass


class InvalidWeightError(HybridSearchError):
    """Raised when weights are invalid."""
    pass


class InvalidFilterError(HybridSearchError):
    """Raised when metadata filters are invalid."""
    pass


class FusionMethod(Enum):
    """Fusion methods for combining vector and text search results.

    Attributes:
        WEIGHTED_SUM: Linear combination of normalized scores
        RECIPROCAL_RANK_FUSION: Rank-based fusion using weighted RRF formula
    """

    WEIGHTED_SUM = "weighted_sum"
    RECIPROCAL_RANK_FUSION = "rrf"


# Weight presets for different search strategies
WEIGHT_PRESETS = {
    "semantic_focus": {"vector": 0.9, "text": 0.1},
    "balanced": {"vector": 0.7, "text": 0.3},
    "lexical_focus": {"vector": 0.3, "text": 0.7},
}

# Filterable metadata fields
FILTERABLE_FIELDS = {
    "source_type",
    "document_type",
    "created_date",
    "author",
    "tags",
    "job_id",
    "metadata",
}


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search operations.

    Attributes:
        default_top_k: Default number of results to return (default: 10)
        max_top_k: Maximum allowed top_k to prevent abuse (default: 100)
        default_vector_weight: Default weight for vector search scores (default: 0.7)
        default_text_weight: Default weight for text search scores (default: 0.3)
        default_fusion_method: Default fusion method (default: RECIPROCAL_RANK_FUSION)
        rrf_k: RRF constant k value (default: 60)
        min_similarity: Default minimum similarity threshold (default: 0.5)
        fallback_mode: How to handle empty results (default: "auto")
        apply_weights_to_rrf: Whether to apply weights in RRF formula (default: True)
    """

    default_top_k: int = 10
    max_top_k: int = 100
    default_vector_weight: float = 0.7
    default_text_weight: float = 0.3
    default_fusion_method: FusionMethod = FusionMethod.RECIPROCAL_RANK_FUSION
    rrf_k: int = 60
    min_similarity: float = 0.5
    fallback_mode: str = "auto"
    apply_weights_to_rrf: bool = True


@dataclass
class HybridSearchResult:
    """Result of a hybrid search operation.

    Attributes:
        chunk: The matching DocumentChunkModel
        hybrid_score: Final combined score from fusion method (0-1 for weighted_sum,
            unbounded for RRF - higher is better)
        vector_score: Cosine similarity score from vector search (0-1), or None
        text_score: BM25 or normalized score from text search (0-1), or None
        vector_rank: Rank in vector search results (1-based), or None
        text_rank: Rank in text search results (1-based), or None
        fusion_method: Name of the fusion method used
        rank: Final rank in combined results (1-based)
    """

    chunk: DocumentChunkModel
    hybrid_score: float
    vector_score: float | None = None
    text_score: float | None = None
    vector_rank: int | None = None
    text_rank: int | None = None
    fusion_method: str = "rrf"
    rank: int = 0


@dataclass
class MetadataFilter:
    """Metadata filter for hybrid search.

    Attributes:
        source_type: Filter by source type (e.g., "pdf", "docx")
        document_type: Filter by document type (e.g., "report", "article")
        created_date: Filter by creation date (date or datetime)
        date_range: Filter by date range (tuple of start, end dates)
        author: Filter by author name
        tags: Filter by tags (list of tags, any match)
        job_id: Filter by job UUID
        metadata: Generic metadata dict for additional filters
    """

    source_type: str | None = None
    document_type: str | None = None
    created_date: date | datetime | None = None
    date_range: tuple[date | datetime, date | datetime] | None = None
    author: str | None = None
    tags: list[str] | None = None
    job_id: UUID | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert filter to dictionary format."""
        result: dict[str, Any] = {}
        if self.source_type:
            result["source_type"] = self.source_type
        if self.document_type:
            result["document_type"] = self.document_type
        if self.created_date:
            result["created_date"] = self.created_date
        if self.date_range:
            result["date_range"] = self.date_range
        if self.author:
            result["author"] = self.author
        if self.tags:
            result["tags"] = self.tags
        if self.job_id:
            result["job_id"] = str(self.job_id)
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class HybridSearchService:
    """Enhanced service for hybrid search with RRF and metadata filtering.

    Provides methods for hybrid search that combines results from vector
    similarity search (semantic) and text search (lexical) using:
    - Reciprocal Rank Fusion (RRF) with configurable weights
    - Weighted sum of normalized scores
    - Advanced metadata filtering

    The RRF formula with weights:
        RRF_score(d) = Σ weight_i / (k + rank_i(d))

    Where:
        - k = constant (typically 60)
        - rank_i(d) = rank of document d in result list i
        - weight_i = weight for search type i (vector or text)

    Example:
        >>> service = HybridSearchService(vector_service, text_service)
        >>> # Using weight preset
        >>> weights = WEIGHT_PRESETS["semantic_focus"]
        >>> results = await service.search(
        ...     query="machine learning",
        ...     embedding=query_embedding,
        ...     filters={"source_type": "pdf"},
        ...     limit=10,
        ...     vector_weight=weights["vector"],
        ...     text_weight=weights["text"]
        ... )
    """

    # Valid fallback modes
    FALLBACK_MODES = {"auto", "vector", "text", "strict"}

    def __init__(
        self,
        vector_service: VectorSearchService,
        text_service: TextSearchService,
        config: HybridSearchConfig | None = None,
    ):
        """Initialize the hybrid search service.

        Args:
            vector_service: VectorSearchService for semantic similarity search
            text_service: TextSearchService for text-based search
            config: Optional configuration for search parameters
        """
        self.vector_service = vector_service
        self.text_service = text_service
        self.config = config or HybridSearchConfig()
        self.logger = logger

    async def search(
        self,
        query: str,
        embedding: list[float],
        filters: dict[str, Any] | MetadataFilter | None = None,
        limit: int = 10,
        vector_weight: float | None = None,
        text_weight: float | None = None,
        fusion_method: FusionMethod | None = None,
        min_similarity: float | None = None,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search with fusion ranking.

        Args:
            query: Text query for full-text search
            embedding: Vector embedding for similarity search
            filters: Metadata filters (source_type, date, author, etc.) or MetadataFilter
            limit: Number of results to return (default: 10)
            vector_weight: Weight for vector scores (default: 0.7)
            text_weight: Weight for text scores (default: 0.3)
            fusion_method: Fusion method to use (default: RECIPROCAL_RANK_FUSION)
            min_similarity: Minimum similarity threshold (default: 0.5)

        Returns:
            List of HybridSearchResult ordered by hybrid_score descending

        Raises:
            HybridSearchError: If search or fusion fails
            InvalidWeightError: If weights don't sum to 1.0
            InvalidFilterError: If filters are invalid

        Example:
            >>> results = await service.search(
            ...     query="neural network architecture",
            ...     embedding=query_embedding,
            ...     filters={"source_type": "pdf", "author": "John"},
            ...     limit=10,
            ...     vector_weight=0.7,
            ...     text_weight=0.3,
            ...     fusion_method=FusionMethod.RECIPROCAL_RANK_FUSION
            ... )
        """
        start_time = time.monotonic()

        try:
            # Use defaults if not specified
            vector_weight = vector_weight or self.config.default_vector_weight
            text_weight = text_weight or self.config.default_text_weight
            fusion_method = fusion_method or self.config.default_fusion_method
            min_similarity = min_similarity or self.config.min_similarity

            # Validate inputs
            self._validate_weights(vector_weight, text_weight)
            self._validate_fusion_method(fusion_method)
            limit = min(limit, self.config.max_top_k)

            # Convert filters to dict if needed
            filter_dict = self._prepare_filters(filters)

            self.logger.info(
                "hybrid_search_started",
                query=query[:100],
                limit=limit,
                vector_weight=vector_weight,
                text_weight=text_weight,
                fusion_method=fusion_method.value,
                filters=filter_dict,
            )

            # Fetch more results than needed to ensure good fusion quality
            fetch_limit = min(limit * 2, self.config.max_top_k)

            # Run both searches
            vector_results = await self.vector_service.search_by_vector(
                query_embedding=embedding,
                top_k=fetch_limit,
                min_similarity=min_similarity,
                filters=filter_dict,
            )

            text_results = await self.text_service.search_by_text(
                query=query,
                top_k=fetch_limit,
                filters=filter_dict,
            )

            self.logger.debug(
                "search_results_collected",
                vector_count=len(vector_results),
                text_count=len(text_results),
            )

            # Apply fallback strategy if one search returns empty
            vector_results, text_results = self._apply_fallback_strategy(
                vector_results, text_results, self.config.fallback_mode
            )

            # Check if both are empty after fallback
            if not vector_results and not text_results:
                self.logger.info("hybrid_search_no_results")
                return []

            # Perform fusion
            if fusion_method == FusionMethod.WEIGHTED_SUM:
                fused_results = self._fuse_weighted_sum(
                    vector_results, text_results, vector_weight, text_weight
                )
            elif fusion_method == FusionMethod.RECIPROCAL_RANK_FUSION:
                fused_results = self.reciprocal_rank_fusion(
                    vector_results=vector_results,
                    text_results=text_results,
                    vector_weight=vector_weight,
                    text_weight=text_weight,
                    k=self.config.rrf_k,
                )
            else:
                # Should not reach here due to validation
                raise InvalidFusionMethodError(
                    f"Unsupported fusion method: {fusion_method}"
                )

            # Sort by hybrid score descending and assign final ranks
            fused_results.sort(key=lambda x: x.hybrid_score, reverse=True)
            for rank, result in enumerate(fused_results, start=1):
                result.rank = rank

            # Limit to requested number
            fused_results = fused_results[:limit]

            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.info(
                "hybrid_search_completed",
                query=query[:100],
                result_count=len(fused_results),
                duration_ms=round(duration_ms, 2),
                fusion_method=fusion_method.value,
            )

            return fused_results

        except HybridSearchError:
            raise
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "hybrid_search_error",
                query=query[:100],
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise HybridSearchError(
                f"Hybrid search failed: {e}",
                context={"query": query[:100]},
            ) from e

    def reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        text_results: list[TextSearchResult],
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        k: int = 60,
    ) -> list[HybridSearchResult]:
        """Apply weighted Reciprocal Rank Fusion (RRF) to combine result lists.

        The weighted RRF formula:
            RRF_score(d) = (vector_weight / (k + rank_vector(d))) +
                          (text_weight / (k + rank_text(d)))

        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            vector_weight: Weight for vector search contribution (default: 0.7)
            text_weight: Weight for text search contribution (default: 0.3)
            k: RRF constant (default: 60)

        Returns:
            List of HybridSearchResult with RRF scores

        Example:
            >>> fused = service.reciprocal_rank_fusion(
            ...     vector_results=vec_results,
            ...     text_results=text_results,
            ...     vector_weight=0.9,  # Semantic focus
            ...     text_weight=0.1,
            ...     k=60
            ... )
        """
        # Build lookup maps by chunk ID with ranks
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
                if chunk is None:
                    chunk = t_result.chunk

            fused_results.append(
                HybridSearchResult(
                    chunk=chunk,  # type: ignore[arg-type]
                    hybrid_score=round(rrf_score, 6),
                    vector_score=round(vector_score, 4) if vector_score else None,
                    text_score=round(text_score, 4) if text_score else None,
                    vector_rank=vector_rank,
                    text_rank=text_rank,
                    fusion_method=FusionMethod.RECIPROCAL_RANK_FUSION.value,
                )
            )

        self.logger.debug(
            "rrf_fusion_completed",
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
        text_scores_normalized = self._normalize_scores(text_results)

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

    def _prepare_filters(
        self, filters: dict[str, Any] | MetadataFilter | None
    ) -> dict[str, Any]:
        """Prepare and validate filters for search.

        Converts MetadataFilter to dict format and validates filter fields.

        Args:
            filters: Filters as dict or MetadataFilter

        Returns:
            Dictionary of filters ready for search services

        Raises:
            InvalidFilterError: If filters are invalid
        """
        if filters is None:
            return {}

        # Convert MetadataFilter to dict
        if isinstance(filters, MetadataFilter):
            return filters.to_dict()

        # Validate dict filters
        if not isinstance(filters, dict):
            raise InvalidFilterError(f"Filters must be dict or MetadataFilter, got {type(filters)}")

        # Check for valid filter fields
        for key in filters.keys():
            if key not in FILTERABLE_FIELDS:
                raise InvalidFilterError(
                    f"Invalid filter field: '{key}'. "
                    f"Valid fields: {', '.join(sorted(FILTERABLE_FIELDS))}"
                )

        return filters

    def _normalize_scores(
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

    def _apply_fallback_strategy(
        self,
        vector_results: list[SearchResult],
        text_results: list[TextSearchResult],
        fallback_mode: str,
    ) -> tuple[list[SearchResult], list[TextSearchResult]]:
        """Handle cases where one search type returns no results.

        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            fallback_mode: Fallback strategy to use
                - "auto": Use whichever search returns results
                - "vector": Always prefer vector results
                - "text": Always prefer text results
                - "strict": Keep both as-is (may return empty)

        Returns:
            Tuple of (vector_results, text_results) after applying fallback
        """
        vector_empty = len(vector_results) == 0
        text_empty = len(text_results) == 0

        if not vector_empty and not text_empty:
            # Both have results, no fallback needed
            return vector_results, text_results

        if vector_empty and text_empty:
            # Both empty, nothing to do
            self.logger.debug("fallback_both_empty")
            return vector_results, text_results

        # One is empty, apply fallback strategy
        if fallback_mode == "strict":
            # Strict mode: keep as-is
            self.logger.debug("fallback_strict_mode")
            return vector_results, text_results

        elif fallback_mode == "vector":
            if vector_empty:
                # Vector is empty, but we're in vector preference mode
                # Return empty results
                self.logger.debug("fallback_vector_preference_empty")
                return [], []
            # Vector has results, use them
            return vector_results, []

        elif fallback_mode == "text":
            if text_empty:
                # Text is empty, but we're in text preference mode
                # Return empty results
                self.logger.debug("fallback_text_preference_empty")
                return [], []
            # Text has results, use them
            return [], text_results

        else:  # auto mode
            if vector_empty:
                # Vector empty, use text only
                self.logger.info(
                    "fallback_to_text_only",
                    reason="vector_search_returned_no_results",
                    text_count=len(text_results),
                )
                return [], text_results
            else:
                # Text empty, use vector only
                self.logger.info(
                    "fallback_to_vector_only",
                    reason="text_search_returned_no_results",
                    vector_count=len(vector_results),
                )
                return vector_results, []

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

    def _validate_fusion_method(self, method: FusionMethod) -> None:
        """Validate fusion method.

        Args:
            method: Fusion method to validate

        Raises:
            InvalidFusionMethodError: If method is invalid
        """
        if not isinstance(method, FusionMethod):
            raise InvalidFusionMethodError(
                f"Invalid fusion method: {method}. "
                f"Must be one of: {[m.value for m in FusionMethod]}"
            )

    def get_weight_preset(self, preset_name: str) -> dict[str, float]:
        """Get a weight preset by name.

        Args:
            preset_name: Name of the preset (semantic_focus, balanced, lexical_focus)

        Returns:
            Dictionary with "vector" and "text" weights

        Raises:
            HybridSearchError: If preset doesn't exist

        Example:
            >>> weights = service.get_weight_preset("semantic_focus")
            >>> results = await service.search(..., vector_weight=weights["vector"],
            ...                                text_weight=weights["text"])
        """
        if preset_name not in WEIGHT_PRESETS:
            raise HybridSearchError(
                f"Unknown weight preset: '{preset_name}'. "
                f"Available presets: {', '.join(WEIGHT_PRESETS.keys())}"
            )
        return WEIGHT_PRESETS[preset_name].copy()

    @staticmethod
    def list_weight_presets() -> dict[str, dict[str, float]]:
        """List all available weight presets.

        Returns:
            Dictionary of preset names to weight configurations
        """
        return WEIGHT_PRESETS.copy()


# Convenience function for quick hybrid search
def get_weight_preset(preset_name: str) -> dict[str, float]:
    """Get a weight preset by name.

    Convenience function for accessing presets without instantiating the service.

    Args:
        preset_name: Name of the preset (semantic_focus, balanced, lexical_focus)

    Returns:
        Dictionary with "vector" and "text" weights

    Example:
        >>> weights = get_weight_preset("semantic_focus")
        >>> # Use weights["vector"] = 0.9, weights["text"] = 0.1
    """
    if preset_name not in WEIGHT_PRESETS:
        raise HybridSearchError(
            f"Unknown weight preset: '{preset_name}'. "
            f"Available presets: {', '.join(WEIGHT_PRESETS.keys())}"
        )
    return WEIGHT_PRESETS[preset_name].copy()
