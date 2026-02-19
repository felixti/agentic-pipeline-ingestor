"""Text search service for full-text and fuzzy search operations.

This module provides a high-level service interface for text-based search using
PostgreSQL's full-text search (tsvector/tsquery) with BM25-like ranking and
pg_trgm for fuzzy trigram matching. Supports highlighting and language configuration.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError

from src.db.models import DocumentChunkModel
from src.db.repositories.document_chunk_repository import DocumentChunkRepository
from src.observability.logging import get_logger

logger = get_logger(__name__)


class TextSearchError(Exception):
    """Base exception for text search errors."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}


class InvalidQueryError(TextSearchError):
    """Raised when a search query is invalid."""
    pass


class LanguageNotSupportedError(TextSearchError):
    """Raised when an unsupported language is specified."""
    pass


@dataclass
class TextSearchConfig:
    """Configuration for text search operations.

    Attributes:
        default_language: Default language for text search (default: 'english')
        default_top_k: Default number of results to return (default: 10)
        max_top_k: Maximum allowed top_k to prevent abuse (default: 100)
        bm25_weights: BM25 weights for ranking (default: {0.1, 0.2, 0.4, 1.0})
        normalization: Normalization option for ts_rank_cd (default: 32)
        default_similarity_threshold: Default trigram similarity threshold (default: 0.3)
        highlight_start_tag: Default highlight start tag (default: '<mark>')
        highlight_end_tag: Default highlight end tag (default: '</mark>')
        highlight_max_words: Default max words per fragment (default: 50)
        highlight_min_words: Default min words per fragment (default: 15)
        highlight_max_fragments: Default max fragments (default: 3)
    """

    default_language: str = "english"
    default_top_k: int = 10
    max_top_k: int = 100
    bm25_weights: tuple[float, float, float, float] = (0.1, 0.2, 0.4, 1.0)
    normalization: int = 32
    default_similarity_threshold: float = 0.3
    highlight_start_tag: str = "<mark>"
    highlight_end_tag: str = "</mark>"
    highlight_max_words: int = 50
    highlight_min_words: int = 15
    highlight_max_fragments: int = 3


@dataclass
class TextSearchResult:
    """Result of a text search operation.

    Attributes:
        chunk: The matching DocumentChunkModel
        rank_score: BM25 or similarity score (higher is better)
        rank: Position in results (1-based)
        highlighted_content: Highlighted content with matched terms marked (if enabled)
        matched_terms: List of terms that matched the query
    """

    chunk: DocumentChunkModel
    rank_score: float
    rank: int
    highlighted_content: str | None = None
    matched_terms: list[str] | None = None


class TextSearchService:
    """Service for text-based search operations.

    Provides methods for full-text search using PostgreSQL's tsvector/tsquery
    with BM25-like ranking, and fuzzy search using pg_trgm trigram similarity.
    Supports highlighting and multiple languages.

    Example:
        >>> service = TextSearchService(repository)
        >>> results = await service.search_by_text(
        ...     query="machine learning",
        ...     top_k=10,
        ...     language="english",
        ...     highlight=True,
        ...     filters={"job_id": job_uuid}
        ... )
    """

    # Supported PostgreSQL text search languages
    SUPPORTED_LANGUAGES = {
        "english",
        "spanish",
        "french",
        "german",
        "portuguese",
        "italian",
        "dutch",
        "russian",
        "chinese",
        "japanese",
        "simple",  # Language-neutral
    }

    def __init__(
        self,
        repository: DocumentChunkRepository,
        config: TextSearchConfig | None = None,
    ):
        """Initialize the text search service.

        Args:
            repository: DocumentChunkRepository for database operations
            config: Optional configuration for search parameters
        """
        self.repository = repository
        self.config = config or TextSearchConfig()
        self.logger = logger

    async def search_by_text(
        self,
        query: str,
        top_k: int = 10,
        language: str = "english",
        use_fuzzy: bool = True,
        highlight: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> list[TextSearchResult]:
        """Search for chunks using full-text search with BM25 ranking.

        Uses PostgreSQL's to_tsvector and to_tsquery for full-text search,
        with ts_rank_cd for BM25-like ranking. Optionally combines with
        pg_trgm fuzzy matching for typo tolerance.

        Args:
            query: Search query text
            top_k: Maximum number of results to return (default: 10)
            language: Text search language configuration (default: 'english')
            use_fuzzy: Whether to include fuzzy trigram matching (default: True)
            highlight: Whether to include highlighted snippets (default: False)
            filters: Optional filters to apply:
                - job_id: Filter by specific job UUID
                - metadata: Dict of metadata key-value pairs to match

        Returns:
            List of TextSearchResult ordered by rank_score descending

        Raises:
            InvalidQueryError: If query is empty or too short
            LanguageNotSupportedError: If language is not supported
            TextSearchError: If database query fails

        Example:
            >>> results = await service.search_by_text(
            ...     query="neural network architecture",
            ...     top_k=10,
            ...     language="english",
            ...     highlight=True,
            ...     filters={"job_id": job_uuid}
            ... )
        """
        start_time = time.monotonic()

        try:
            # Validate inputs
            self._validate_query(query)
            self._validate_language(language)
            top_k = min(top_k, self.config.max_top_k)

            self.logger.info(
                "text_search_started",
                query=query[:100],
                top_k=top_k,
                language=language,
                use_fuzzy=use_fuzzy,
                highlight=highlight,
                filters=filters,
            )

            # Build and execute search query
            search_query = self._build_text_search_query(
                query=query,
                language=language,
                top_k=top_k,
                use_fuzzy=use_fuzzy,
                highlight=highlight,
                filters=filters,
            )

            result = await self.repository.session.execute(search_query)
            rows = result.fetchall()

            # Transform to TextSearchResult objects
            search_results = []
            for rank, row in enumerate(rows, start=1):
                chunk = row[0]  # DocumentChunkModel
                rank_score = float(row[1]) if row[1] is not None else 0.0
                highlighted = row[2] if highlight and len(row) > 2 else None

                # Extract matched terms if highlighting was used
                matched_terms = None
                if highlight and highlighted:
                    matched_terms = self._extract_matched_terms(highlighted)

                search_results.append(
                    TextSearchResult(
                        chunk=chunk,
                        rank_score=round(rank_score, 4),
                        rank=rank,
                        highlighted_content=highlighted,
                        matched_terms=matched_terms,
                    )
                )

            duration_ms = (time.monotonic() - start_time) * 1000

            self.logger.info(
                "text_search_completed",
                result_count=len(search_results),
                duration_ms=round(duration_ms, 2),
                query=query[:100],
                top_k=top_k,
                language=language,
            )

            return search_results

        except (InvalidQueryError, LanguageNotSupportedError):
            raise
        except SQLAlchemyError as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "text_search_database_error",
                error=str(e),
                duration_ms=round(duration_ms, 2),
                query=query[:100],
            )
            raise TextSearchError(
                f"Database query failed: {e}",
                context={
                    "query_type": "full_text_search",
                    "query": query[:100],
                    "language": language,
                },
            ) from e
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "text_search_unexpected_error",
                error=str(e),
                duration_ms=round(duration_ms, 2),
                query=query[:100],
            )
            raise TextSearchError(
                f"Unexpected error during text search: {e}",
                context={"query": query[:100]},
            ) from e

    async def search_fuzzy(
        self,
        query: str,
        similarity_threshold: float = 0.3,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[TextSearchResult]:
        """Search for chunks using trigram similarity (fuzzy matching).

        Uses PostgreSQL's pg_trgm extension for trigram similarity search.
        Good for finding typos, spelling variations, and partial matches.

        Args:
            query: Search query text
            similarity_threshold: Minimum similarity threshold (0.0-1.0, default: 0.3)
            top_k: Maximum number of results to return (default: 10)
            filters: Optional filters to apply:
                - job_id: Filter by specific job UUID
                - metadata: Dict of metadata key-value pairs to match

        Returns:
            List of TextSearchResult ordered by similarity descending

        Raises:
            InvalidQueryError: If query is empty or too short (< 3 chars)
            TextSearchError: If database query fails

        Example:
            >>> results = await service.search_fuzzy(
            ...     query="accomodation",  # typo intentional
            ...     similarity_threshold=0.3,
            ...     top_k=10,
            ...     filters={"job_id": job_uuid}
            ... )
        """
        start_time = time.monotonic()

        try:
            # Validate inputs
            self._validate_query(query, min_length=3)
            similarity_threshold = max(0.0, min(1.0, similarity_threshold))
            top_k = min(top_k, self.config.max_top_k)

            self.logger.info(
                "fuzzy_search_started",
                query=query[:100],
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters,
            )

            # Build and execute fuzzy search query
            fuzzy_query = self._build_fuzzy_search_query(
                query=query,
                similarity_threshold=similarity_threshold,
                top_k=top_k,
                filters=filters,
            )

            result = await self.repository.session.execute(fuzzy_query)
            rows = result.fetchall()

            # Transform to TextSearchResult objects
            search_results = []
            for rank, row in enumerate(rows, start=1):
                chunk = row[0]  # DocumentChunkModel
                similarity = float(row[1]) if row[1] is not None else 0.0

                search_results.append(
                    TextSearchResult(
                        chunk=chunk,
                        rank_score=round(similarity, 4),
                        rank=rank,
                        highlighted_content=None,
                        matched_terms=None,
                    )
                )

            duration_ms = (time.monotonic() - start_time) * 1000

            self.logger.info(
                "fuzzy_search_completed",
                result_count=len(search_results),
                duration_ms=round(duration_ms, 2),
                query=query[:100],
                similarity_threshold=similarity_threshold,
            )

            return search_results

        except InvalidQueryError:
            raise
        except SQLAlchemyError as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "fuzzy_search_database_error",
                error=str(e),
                duration_ms=round(duration_ms, 2),
                query=query[:100],
            )
            raise TextSearchError(
                f"Database query failed: {e}",
                context={
                    "query_type": "fuzzy_search",
                    "query": query[:100],
                    "similarity_threshold": similarity_threshold,
                },
            ) from e
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "fuzzy_search_unexpected_error",
                error=str(e),
                duration_ms=round(duration_ms, 2),
                query=query[:100],
            )
            raise TextSearchError(
                f"Unexpected error during fuzzy search: {e}",
                context={"query": query[:100]},
            ) from e

    def _build_tsquery(self, query: str) -> str:
        """Convert query string to PostgreSQL tsquery format.

        Processes the query to handle:
        - Quoted phrases (converted to phrase search)
        - Boolean operators (AND, OR, NOT)
        - Prefix matching with *

        Args:
            query: Raw search query

        Returns:
            Formatted tsquery string
        """
        # Normalize whitespace
        query = " ".join(query.split())

        # Handle quoted phrases
        phrases = re.findall(r'"([^"]+)"', query)
        for phrase in phrases:
            # Replace quoted phrase with processed version
            processed = " <-> ".join(phrase.split())
            query = query.replace(f'"{phrase}"', processed)

        # Handle NOT operator
        query = re.sub(r"\bNOT\s+(\w+)", r"!\1", query, flags=re.IGNORECASE)

        # Handle AND operator (implicit)
        query = re.sub(r"\bAND\b", "&", query, flags=re.IGNORECASE)

        # Handle OR operator
        query = re.sub(r"\bOR\b", "|", query, flags=re.IGNORECASE)

        # Replace remaining spaces with AND operators for default behavior
        # But preserve already processed operators
        words = query.split()
        if words:
            # Join with & for full-text search AND behavior
            query = " & ".join(words)

        # Clean up multiple operators
        query = re.sub(r"&\s*\|", "|", query)
        query = re.sub(r"\|\s*&", "|", query)
        query = re.sub(r"&+", "&", query)
        query = re.sub(r"\|+", "|", query)

        return query

    def _calculate_bm25_rank(
        self,
        tsvector: Any,
        tsquery: Any,
    ) -> Any:
        """Calculate BM25-like rank using ts_rank_cd.

        Args:
            tsvector: SQLAlchemy expression for tsvector
            tsquery: SQLAlchemy expression for tsquery

        Returns:
            SQLAlchemy expression for rank score
        """
        weights_str = "{" + ",".join(str(w) for w in self.config.bm25_weights) + "}"
        return func.ts_rank_cd(
            text(f"'{weights_str}'"),
            tsvector,
            tsquery,
            self.config.normalization,
        )

    def _highlight_content(
        self,
        content_col: Any,
        tsquery: Any,
        language: str,
    ) -> Any:
        """Generate highlighted content using ts_headline.

        Args:
            content_col: SQLAlchemy column expression for content
            tsquery: SQLAlchemy expression for tsquery
            language: Language configuration

        Returns:
            SQLAlchemy expression for highlighted content
        """
        options = (
            f"StartSel={self.config.highlight_start_tag}, "
            f"StopSel={self.config.highlight_end_tag}, "
            f"MaxWords={self.config.highlight_max_words}, "
            f"MinWords={self.config.highlight_min_words}, "
            f"MaxFragments={self.config.highlight_max_fragments}, "
            f'FragmentDelimiter=" ... "'
        )

        return func.ts_headline(
            language,
            content_col,
            tsquery,
            options,
        )

    def _build_text_search_query(
        self,
        query: str,
        language: str,
        top_k: int,
        use_fuzzy: bool,
        highlight: bool,
        filters: dict[str, Any] | None,
    ) -> Any:
        """Build SQLAlchemy query for full-text search with BM25 ranking.

        Args:
            query: Search query text
            language: Text search language
            top_k: Maximum results to return
            use_fuzzy: Whether to include fuzzy matching
            highlight: Whether to include highlighting
            filters: Optional filters dictionary

        Returns:
            SQLAlchemy select query
        """
        # Build tsquery
        tsquery_str = self._build_tsquery(query)

        # Create tsvector and tsquery expressions
        tsvector = func.to_tsvector(language, DocumentChunkModel.content)
        tsquery = func.to_tsquery(language, tsquery_str)

        # Calculate BM25 rank
        rank = self._calculate_bm25_rank(tsvector, tsquery)

        # Build select columns
        columns = [DocumentChunkModel, rank.label("rank")]

        # Add highlighting if requested
        if highlight:
            highlighted = self._highlight_content(
                DocumentChunkModel.content, tsquery, language
            )
            columns.append(highlighted.label("highlighted"))

        # Build base query with full-text search
        stmt = select(*columns).where(tsvector.op("@@")(tsquery))

        # Add fuzzy matching if requested
        if use_fuzzy:
            # Use OR logic: tsvector match OR trigram similarity
            similarity = func.similarity(DocumentChunkModel.content, query)
            fuzzy_condition = similarity > self.config.default_similarity_threshold
            stmt = stmt.where(fuzzy_condition | tsvector.op("@@")(tsquery))
            # Use the higher of the two scores
            rank = func.greatest(rank, similarity)
            # Update columns with new rank
            columns[1] = rank.label("rank")
            stmt = select(*columns).where(fuzzy_condition | tsvector.op("@@")(tsquery))

        # Apply filters
        if filters:
            stmt = self._apply_filters(stmt, filters)

        # Order by rank descending and limit
        stmt = stmt.order_by(rank.desc()).limit(top_k)

        return stmt

    def _build_fuzzy_search_query(
        self,
        query: str,
        similarity_threshold: float,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> Any:
        """Build SQLAlchemy query for fuzzy trigram search.

        Args:
            query: Search query text
            similarity_threshold: Minimum similarity threshold
            top_k: Maximum results to return
            filters: Optional filters dictionary

        Returns:
            SQLAlchemy select query
        """
        # Calculate similarity
        similarity = func.similarity(DocumentChunkModel.content, query)

        # Build query using trigram similarity operator
        stmt = (
            select(
                DocumentChunkModel,
                similarity.label("similarity"),
            )
            .where(DocumentChunkModel.content.op("%")(query))
            .where(similarity >= similarity_threshold)
            .order_by(similarity.desc())
            .limit(top_k)
        )

        # Apply filters
        if filters:
            stmt = self._apply_filters(stmt, filters)

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
                query = query.where(
                    DocumentChunkModel.chunk_metadata.op("@>")(metadata_filters)
                )

        # Filter by chunk_index
        if chunk_index := filters.get("chunk_index"):
            query = query.where(DocumentChunkModel.chunk_index == chunk_index)

        # Filter by content hash
        if content_hash := filters.get("content_hash"):
            query = query.where(DocumentChunkModel.content_hash == content_hash)

        return query

    def _validate_query(self, query: str, min_length: int = 1) -> None:
        """Validate search query.

        Args:
            query: Search query to validate
            min_length: Minimum query length

        Raises:
            InvalidQueryError: If validation fails
        """
        if not query or not query.strip():
            raise InvalidQueryError("Search query cannot be empty")

        if len(query.strip()) < min_length:
            raise InvalidQueryError(
                f"Search query must be at least {min_length} characters long"
            )

        # Limit query length to prevent abuse
        if len(query) > 1024:
            raise InvalidQueryError("Search query too long (max 1024 characters)")

    def _validate_language(self, language: str) -> None:
        """Validate text search language.

        Args:
            language: Language code to validate

        Raises:
            LanguageNotSupportedError: If language is not supported
        """
        if language.lower() not in self.SUPPORTED_LANGUAGES:
            raise LanguageNotSupportedError(
                f"Language '{language}' is not supported. "
                f"Supported languages: {', '.join(sorted(self.SUPPORTED_LANGUAGES))}"
            )

    def _extract_matched_terms(self, highlighted_content: str) -> list[str]:
        """Extract matched terms from highlighted content.

        Args:
            highlighted_content: Content with highlight tags

        Returns:
            List of matched terms (unique, lowercase)
        """
        # Extract text between highlight tags
        pattern = re.escape(self.config.highlight_start_tag) + "(.+?)" + re.escape(self.config.highlight_end_tag)
        matches = re.findall(pattern, highlighted_content, re.IGNORECASE)

        # Normalize and deduplicate
        terms = list(set(term.lower().strip() for term in matches))

        return sorted(terms) if terms else None
