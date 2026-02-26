"""Vector search service for semantic similarity operations.

This module provides a high-level service interface for vector search operations
using pgvector's cosine similarity. It supports filtering by metadata and job_id,
and includes proper error handling and logging.
"""

import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import literal_column, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload

from src.db.models import ChunkEmbeddingModel, DocumentChunkModel
from src.db.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from src.db.repositories.document_chunk_repository import DocumentChunkRepository
from src.observability.logging import get_logger
from src.vector_store_config.vector_store import get_vector_store_config

logger = get_logger(__name__)


class VectorSearchError(Exception):
    """Base exception for vector search errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}


class ChunkNotFoundError(VectorSearchError):
    """Raised when a chunk is not found."""

    pass


class InvalidEmbeddingError(VectorSearchError):
    """Raised when an embedding is invalid."""

    pass


@dataclass
class VectorSearchConfig:
    """Configuration for vector search operations.

    Attributes:
        default_top_k: Default number of results to return
        default_min_similarity: Default minimum similarity threshold (0-1)
        embedding_dimensions: Expected embedding dimensions (default 1536)
        max_top_k: Maximum allowed top_k to prevent abuse
    """

    default_top_k: int = 10
    default_min_similarity: float = 0.7
    embedding_dimensions: int = 1536
    max_top_k: int = 100
    active_model_name: str = "text-embedding-3-small"


@dataclass
class SearchResult:
    """Result of a vector search operation.

    Attributes:
        chunk: The matching DocumentChunkModel
        similarity_score: Cosine similarity score (0-1, where 1 is identical)
        rank: Position in results (1-based)
    """

    chunk: DocumentChunkModel
    similarity_score: float
    rank: int


class VectorSearchService:
    """Service for vector similarity search operations.

    Provides methods for semantic search using cosine similarity with pgvector,
    including support for metadata filtering and finding similar chunks.

    Example:
        >>> service = VectorSearchService(repository)
        >>> results = await service.search_by_vector(
        ...     query_embedding=[0.1, 0.2, ...],
        ...     top_k=10,
        ...     filters={"job_id": "uuid-here"}
        ... )
    """

    def __init__(
        self,
        repository: DocumentChunkRepository,
        config: VectorSearchConfig | None = None,
        chunk_embedding_repository: ChunkEmbeddingRepository | None = None,
    ):
        """Initialize the vector search service.

        Args:
            repository: DocumentChunkRepository for database operations
            config: Optional configuration for search parameters
        """
        self.repository = repository
        if config is None:
            vector_store_config = get_vector_store_config()
            config = VectorSearchConfig(
                default_top_k=vector_store_config.search.default_top_k,
                default_min_similarity=vector_store_config.search.default_min_similarity,
                embedding_dimensions=vector_store_config.embedding.dimensions,
                max_top_k=vector_store_config.search.max_top_k,
                active_model_name=vector_store_config.embedding.model,
            )
        self.config = config
        self.chunk_embedding_repository = (
            chunk_embedding_repository
            if chunk_embedding_repository is not None
            else ChunkEmbeddingRepository(repository.session)
        )
        self.logger = logger

    async def search_by_vector(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        min_similarity: float = 0.7,
        filters: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks using cosine similarity.

        Uses pgvector's cosine distance operator (<=>) to find chunks with
        embeddings similar to the query vector. Results are ordered by
        similarity descending (most similar first).

        Args:
            query_embedding: The query embedding vector (list of floats)
            top_k: Maximum number of results to return (default: 10)
            min_similarity: Minimum similarity threshold (0-1, default: 0.7)
            filters: Optional filters to apply:
                - job_id: Filter by specific job UUID
                - metadata: Dict of metadata key-value pairs to match

        Returns:
            List of SearchResult ordered by similarity descending

        Raises:
            InvalidEmbeddingError: If query embedding has wrong dimensions
            VectorSearchError: If database query fails

        Example:
            >>> results = await service.search_by_vector(
            ...     query_embedding=[0.1, -0.2, ...],  # 1536 dimensions
            ...     top_k=5,
            ...     min_similarity=0.8,
            ...     filters={"job_id": job_uuid}
            ... )
        """
        start_time = time.monotonic()

        try:
            selected_model_name = model_name or self.config.active_model_name
            expected_dimensions = self._resolve_embedding_dimensions(selected_model_name)

            # Validate inputs
            self._validate_embedding(query_embedding, expected_dimensions)
            top_k = min(top_k, self.config.max_top_k)

            self.logger.info(
                "vector_search_started",
                top_k=top_k,
                min_similarity=min_similarity,
                filters=filters,
                model_name=selected_model_name,
                embedding_dimensions=len(query_embedding),
            )

            # Build and execute similarity query
            query = self._build_similarity_query(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=min_similarity,
                filters=filters,
                model_name=selected_model_name,
            )

            result = await self.repository.session.execute(query)
            rows = result.fetchall()

            # Transform to SearchResult objects
            search_results = []
            for rank, row in enumerate(rows, start=1):
                chunk = row[0]  # DocumentChunkModel
                distance = row[1]  # Raw distance from pgvector
                similarity = self._calculate_similarity(distance)

                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        similarity_score=round(similarity, 4),
                        rank=rank,
                    )
                )

            duration_ms = (time.monotonic() - start_time) * 1000

            self.logger.info(
                "vector_search_completed",
                result_count=len(search_results),
                duration_ms=round(duration_ms, 2),
                top_k=top_k,
                min_similarity=min_similarity,
            )

            return search_results

        except SQLAlchemyError as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "vector_search_database_error",
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise VectorSearchError(
                f"Database query failed: {e}",
                context={
                    "query_type": "cosine_similarity",
                    "top_k": top_k,
                    "min_similarity": min_similarity,
                },
            ) from e

    async def find_similar_chunks(
        self,
        chunk_id: UUID,
        top_k: int = 10,
        exclude_self: bool = True,
        model_name: str | None = None,
    ) -> list[SearchResult]:
        """Find chunks similar to a reference chunk.

        Retrieves the embedding of the reference chunk and performs a
        similarity search to find semantically similar chunks.

        Args:
            chunk_id: UUID of the reference chunk
            top_k: Number of similar chunks to return (default: 10)
            exclude_self: If True, exclude the reference chunk from results (default: True)

        Returns:
            List of SearchResult ordered by similarity descending

        Raises:
            ChunkNotFoundError: If reference chunk doesn't exist
            InvalidEmbeddingError: If reference chunk has no embedding
            VectorSearchError: If database query fails

        Example:
            >>> similar = await service.find_similar_chunks(
            ...     chunk_id=uuid,
            ...     top_k=5,
            ...     exclude_self=True
            ... )
        """
        start_time = time.monotonic()

        self.logger.info(
            "find_similar_chunks_started",
            reference_chunk_id=str(chunk_id),
            top_k=top_k,
            exclude_self=exclude_self,
            model_name=model_name or self.config.active_model_name,
        )

        # Get the reference chunk
        reference_chunk = await self.repository.get_by_id(chunk_id)
        if not reference_chunk:
            self.logger.warning(
                "reference_chunk_not_found",
                chunk_id=str(chunk_id),
            )
            raise ChunkNotFoundError(
                f"Reference chunk with ID {chunk_id} not found",
                context={"chunk_id": str(chunk_id)},
            )

        selected_model_name = model_name or self.config.active_model_name
        reference_embedding = await self.chunk_embedding_repository.get_embedding(
            chunk_id=chunk_id,
            model_name=selected_model_name,
        )
        if reference_embedding is None:
            self.logger.warning(
                "reference_chunk_no_embedding",
                chunk_id=str(chunk_id),
                model_name=selected_model_name,
            )
            raise InvalidEmbeddingError(
                f"Reference chunk {chunk_id} does not have an embedding for model "
                f"{selected_model_name}",
                context={
                    "chunk_id": str(chunk_id),
                    "model_name": selected_model_name,
                },
            )

        # Build filters
        filters: dict[str, Any] = {}
        if exclude_self:
            # We'll handle exclusion in the query
            pass

        try:
            # Perform similarity search using the reference chunk's embedding
            query_embedding = reference_embedding

            top_k_adjusted = top_k + (1 if exclude_self else 0)

            results = await self.search_by_vector(
                query_embedding=query_embedding,
                top_k=top_k_adjusted,
                min_similarity=0.0,  # No threshold for similar chunk search
                filters=filters,
                model_name=selected_model_name,
            )

            # Exclude self if requested
            if exclude_self:
                results = [r for r in results if r.chunk.id != chunk_id][:top_k]

            duration_ms = (time.monotonic() - start_time) * 1000

            self.logger.info(
                "find_similar_chunks_completed",
                reference_chunk_id=str(chunk_id),
                result_count=len(results),
                duration_ms=round(duration_ms, 2),
            )

            return results

        except (ChunkNotFoundError, InvalidEmbeddingError):
            raise
        except VectorSearchError:
            raise
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "find_similar_chunks_error",
                chunk_id=str(chunk_id),
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise VectorSearchError(
                f"Failed to find similar chunks: {e}",
                context={"chunk_id": str(chunk_id)},
            ) from e

    def _validate_embedding(
        self,
        embedding: list[float],
        expected_dimensions: int | None = None,
    ) -> None:
        """Validate embedding vector dimensions and values.

        Args:
            embedding: The embedding vector to validate

        Raises:
            InvalidEmbeddingError: If validation fails
        """
        if not embedding:
            raise InvalidEmbeddingError("Query vector cannot be empty")

        resolved_dimensions = expected_dimensions or self.config.embedding_dimensions

        if len(embedding) != resolved_dimensions:
            raise InvalidEmbeddingError(
                f"Query vector dimension {len(embedding)} does not match "
                f"expected {resolved_dimensions}",
                context={
                    "expected_dimensions": resolved_dimensions,
                    "actual_dimensions": len(embedding),
                },
            )

        # Check for NaN or infinite values
        for i, val in enumerate(embedding):
            if not isinstance(val, (int, float)):
                raise InvalidEmbeddingError(
                    f"Invalid vector value at index {i}: {val}",
                    context={"index": i, "value": str(val)},
                )
            import math

            if math.isnan(val) or math.isinf(val):
                raise InvalidEmbeddingError(
                    f"Invalid vector: contains NaN or infinite value at index {i}",
                    context={"index": i, "value": val},
                )

    def _resolve_embedding_dimensions(self, model_name: str) -> int:
        known_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-large": 3072,
        }

        for model_hint, dimensions in known_dimensions.items():
            if model_hint in model_name:
                return dimensions

        if model_name == self.config.active_model_name:
            return self.config.embedding_dimensions

        return self.config.embedding_dimensions

    def _calculate_similarity(self, distance: float) -> float:
        """Convert cosine distance to similarity score.

        pgvector's cosine distance (<=>) returns values from 0 to 2:
        - 0 = identical vectors (most similar)
        - 1 = orthogonal vectors (no similarity)
        - 2 = opposite vectors (least similar)

        We convert this to a 0-1 similarity score where:
        - 1 = identical (distance 0)
        - 0 = orthogonal (distance 1)
        - Negative values clamped to 0 (opposite vectors)

        Args:
            distance: Cosine distance from pgvector (0-2)

        Returns:
            Similarity score (0-1)
        """
        similarity = 1.0 - distance
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))

    def _build_similarity_query(
        self,
        query_embedding: list[float],
        top_k: int,
        min_similarity: float,
        filters: dict[str, Any] | None,
        model_name: str,
    ) -> Any:
        """Build SQLAlchemy query for cosine similarity search.

        Args:
            query_embedding: The query vector
            top_k: Maximum results to return
            min_similarity: Minimum similarity threshold
            filters: Optional filters dictionary

        Returns:
            SQLAlchemy select query
        """
        # Convert embedding to PostgreSQL vector format
        vector_str = f"[{','.join(str(x) for x in query_embedding)}]"

        # Calculate distance threshold from similarity
        # similarity = 1 - distance, so distance = 1 - similarity
        max_distance = 1.0 - min_similarity

        # Build the query using pgvector's <=> operator
        # The <=> operator computes cosine distance
        # Use the embedding column directly from document_chunks table
        # (not from chunk_embeddings table which may be empty)
        distance_expr = literal_column(
            f"document_chunks.embedding::vector <=> '{vector_str}'::vector"
        ).label("distance")

        # Build base query - query directly on document_chunks
        # No join needed since embeddings are stored in document_chunks.embedding
        stmt: Any = (
            select(DocumentChunkModel, distance_expr)
            .where(DocumentChunkModel.embedding.is_not(None))
            .where(distance_expr <= max_distance)  # type: ignore[arg-type,operator]
            .order_by(distance_expr.asc())  # type: ignore[attr-defined]
            .limit(top_k)
        )

        # Apply filters
        if filters:
            stmt = self._apply_filters(stmt, filters)

        # Eager load job relationship
        stmt = stmt.options(joinedload(DocumentChunkModel.job))

        return stmt

    def _apply_filters(self, query: Any, filters: dict[str, Any]) -> Any:
        """Apply metadata filters to query.

        Args:
            query: SQLAlchemy query to filter
            filters: Dictionary of filter conditions:
                - job_id: Filter by job UUID
                - metadata: Dict of metadata key-value pairs

        Returns:
            Filtered SQLAlchemy query
        """
        # Filter by job_id
        if job_id := filters.get("job_id"):
            if isinstance(job_id, str):
                job_id = UUID(job_id)
            query = query.where(DocumentChunkModel.job_id == job_id)

        # Filter by metadata JSONB
        if metadata_filters := filters.get("metadata"):
            if isinstance(metadata_filters, dict):
                query = query.where(DocumentChunkModel.chunk_metadata.op("@>")(metadata_filters))

        # Filter by chunk_index
        chunk_index = filters.get("chunk_index")
        if chunk_index is not None:
            query = query.where(DocumentChunkModel.chunk_index == chunk_index)

        # Filter by content hash
        content_hash = filters.get("content_hash")
        if content_hash is not None:
            query = query.where(DocumentChunkModel.content_hash == content_hash)

        # Filter by chunk quality score
        min_quality = filters.get("min_quality")
        if min_quality is not None:
            quality_score_col = literal_column("document_chunks.quality_score")
            query = query.where(quality_score_col.is_not(None)).where(
                quality_score_col >= float(min_quality)
            )

        return query

    async def search_by_text(
        self,
        query_text: str,
        embedder: Any,
        top_k: int = 10,
        min_similarity: float = 0.7,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search by text query using an embedding model.

        Generates an embedding from the query text and performs vector search.
        Note: This is a placeholder for future implementation that requires
        an embedding service integration.

        Args:
            query_text: Natural language query text
            embedder: Embedding model/encoder with an encode method
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            filters: Optional filters to apply

        Returns:
            List of SearchResult ordered by similarity descending

        Raises:
            InvalidEmbeddingError: If embedding generation fails
            VectorSearchError: If search fails
        """
        self.logger.info(
            "text_search_started",
            query_length=len(query_text),
            top_k=top_k,
        )

        try:
            # Generate embedding from text
            # This is CPU-bound, should ideally run in thread pool
            query_embedding = embedder.encode(query_text)

            if not isinstance(query_embedding, list):
                query_embedding = query_embedding.tolist()

            # Perform vector search
            results = await self.search_by_vector(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=min_similarity,
                filters=filters,
            )

            self.logger.info(
                "text_search_completed",
                result_count=len(results),
            )

            return results

        except Exception as e:
            self.logger.error(
                "text_search_error",
                error=str(e),
            )
            raise VectorSearchError(
                f"Text search failed: {e}",
                context={"query_text": query_text[:100]},
            ) from e
