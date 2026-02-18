"""Unit tests for vector store services.

Tests for VectorSearchService, TextSearchService, HybridSearchService, and EmbeddingService.
Uses mocks for database and external dependencies.
"""

import math
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.db.models import DocumentChunkModel
from src.db.repositories.document_chunk_repository import DocumentChunkRepository
from src.services.embedding_service import (
    EmbeddingCache,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingResult,
    EmbeddingService,
)
from src.services.hybrid_search_service import (
    FusionMethod,
    HybridSearchConfig,
    HybridSearchError,
    HybridSearchResult,
    HybridSearchService,
    InvalidFusionMethodError,
    InvalidWeightError,
)
from src.services.text_search_service import (
    InvalidQueryError,
    LanguageNotSupportedError,
    TextSearchConfig,
    TextSearchError,
    TextSearchResult,
    TextSearchService,
)
from src.services.vector_search_service import (
    ChunkNotFoundError,
    InvalidEmbeddingError,
    SearchResult,
    VectorSearchConfig,
    VectorSearchError,
    VectorSearchService,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    return session


@pytest.fixture
def document_chunk_repository(mock_db_session):
    """Create a DocumentChunkRepository with mocked session."""
    return DocumentChunkRepository(mock_db_session)


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector (1536 dimensions for OpenAI)."""
    # Create a normalized vector for consistent tests
    dims = 1536
    vec = [0.1 * (i % 10) / 10 for i in range(dims)]
    # Normalize
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


@pytest.fixture
def sample_chunk():
    """Create a sample DocumentChunkModel."""
    chunk = MagicMock(spec=DocumentChunkModel)
    chunk.id = uuid4()
    chunk.job_id = uuid4()
    chunk.chunk_index = 0
    chunk.content = "This is a sample chunk content for testing purposes."
    chunk.content_hash = "abc123"
    chunk.chunk_metadata = {"page": 1, "source": "test"}
    chunk.created_at = datetime.utcnow()
    chunk.has_embedding = True
    chunk.embedding = sample_embedding()
    return chunk


@pytest.fixture
def sample_chunk_without_embedding():
    """Create a sample DocumentChunkModel without embedding."""
    chunk = MagicMock(spec=DocumentChunkModel)
    chunk.id = uuid4()
    chunk.job_id = uuid4()
    chunk.chunk_index = 1
    chunk.content = "This chunk has no embedding."
    chunk.content_hash = "def456"
    chunk.chunk_metadata = {"page": 2}
    chunk.created_at = datetime.utcnow()
    chunk.has_embedding = False
    chunk.embedding = None
    return chunk


@pytest.fixture
def vector_search_service(mock_db_session):
    """Create a VectorSearchService with mocked repository."""
    repo = MagicMock(spec=DocumentChunkRepository)
    repo.session = mock_db_session
    config = VectorSearchConfig(
        default_top_k=10,
        default_min_similarity=0.7,
        embedding_dimensions=1536,
        max_top_k=100,
    )
    return VectorSearchService(repo, config)


@pytest.fixture
def text_search_service(mock_db_session):
    """Create a TextSearchService with mocked repository."""
    repo = MagicMock(spec=DocumentChunkRepository)
    repo.session = mock_db_session
    config = TextSearchConfig(
        default_language="english",
        default_top_k=10,
        max_top_k=100,
        default_similarity_threshold=0.3,
    )
    return TextSearchService(repo, config)


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = MagicMock()
    # Create a 1536-dimensional embedding
    dims = 1536
    vec = [0.05 * (i % 20) / 20 for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in vec))
    normalized = [x / norm for x in vec]
    embedder.encode.return_value = normalized
    return embedder


@pytest.fixture
def hybrid_search_service(vector_search_service, text_search_service):
    """Create a HybridSearchService with mocked dependencies."""
    config = HybridSearchConfig(
        default_top_k=10,
        max_top_k=100,
        default_vector_weight=0.7,
        default_text_weight=0.3,
        default_fusion_method=FusionMethod.WEIGHTED_SUM,
        rrf_k=60,
    )
    return HybridSearchService(vector_search_service, text_search_service, config)


# =============================================================================
# VectorSearchService Tests
# =============================================================================

@pytest.mark.unit
class TestVectorSearchService:
    """Tests for VectorSearchService."""

    @pytest.mark.asyncio
    async def test_search_by_vector_success(self, vector_search_service, mock_db_session, sample_chunk):
        """Test successful vector search."""
        # Setup
        query_embedding = [0.1] * 1536
        
        # Mock the database result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (sample_chunk, 0.2),  # chunk, distance
        ]
        mock_db_session.execute.return_value = mock_result
        
        # Execute
        results = await vector_search_service.search_by_vector(
            query_embedding=query_embedding,
            top_k=5,
            min_similarity=0.7,
        )
        
        # Assert
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk == sample_chunk
        assert results[0].similarity_score == 0.8  # 1.0 - 0.2
        assert results[0].rank == 1

    @pytest.mark.asyncio
    async def test_search_by_vector_with_filters(self, vector_search_service, mock_db_session, sample_chunk):
        """Test vector search with job_id filter."""
        query_embedding = [0.1] * 1536
        job_id = uuid4()
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(sample_chunk, 0.15)]
        mock_db_session.execute.return_value = mock_result
        
        results = await vector_search_service.search_by_vector(
            query_embedding=query_embedding,
            top_k=10,
            filters={"job_id": str(job_id)},
        )
        
        assert len(results) == 1
        mock_db_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_vector_empty_embedding_raises_error(self, vector_search_service):
        """Test that empty embedding raises InvalidEmbeddingError."""
        with pytest.raises(InvalidEmbeddingError) as exc_info:
            await vector_search_service.search_by_vector(
                query_embedding=[],
                top_k=10,
            )
        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_by_vector_wrong_dimensions_raises_error(self, vector_search_service):
        """Test that wrong embedding dimensions raises InvalidEmbeddingError."""
        with pytest.raises(InvalidEmbeddingError) as exc_info:
            await vector_search_service.search_by_vector(
                query_embedding=[0.1] * 768,  # Wrong size
                top_k=10,
            )
        assert "dimension" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_by_vector_nan_values_raises_error(self, vector_search_service):
        """Test that NaN values in embedding raises InvalidEmbeddingError."""
        query_embedding = [0.1] * 1536
        query_embedding[0] = float('nan')
        
        with pytest.raises(InvalidEmbeddingError) as exc_info:
            await vector_search_service.search_by_vector(
                query_embedding=query_embedding,
                top_k=10,
            )
        assert "nan" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_by_vector_database_error(self, vector_search_service, mock_db_session):
        """Test that database errors raise VectorSearchError."""
        query_embedding = [0.1] * 1536
        mock_db_session.execute.side_effect = SQLAlchemyError("Connection failed")
        
        with pytest.raises(VectorSearchError) as exc_info:
            await vector_search_service.search_by_vector(
                query_embedding=query_embedding,
                top_k=10,
            )
        assert "database" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_by_vector_top_k_capped(self, vector_search_service, mock_db_session, sample_chunk):
        """Test that top_k is capped at max_top_k."""
        query_embedding = [0.1] * 1536
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(sample_chunk, 0.2)]
        mock_db_session.execute.return_value = mock_result
        
        await vector_search_service.search_by_vector(
            query_embedding=query_embedding,
            top_k=200,  # Exceeds max_top_k
        )
        
        # Verify the query was built with capped limit
        call_args = mock_db_session.execute.call_args[0][0]
        # The limit should be in the query

    @pytest.mark.asyncio
    async def test_find_similar_chunks_success(self, vector_search_service, mock_db_session, sample_chunk):
        """Test finding similar chunks."""
        chunk_id = uuid4()
        
        # Mock repository.get_by_id to return the reference chunk
        vector_search_service.repository.get_by_id = AsyncMock(return_value=sample_chunk)
        
        # Mock the similarity search
        similar_chunk = MagicMock(spec=DocumentChunkModel)
        similar_chunk.id = uuid4()
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (sample_chunk, 0.0),  # Self (distance 0 = identical)
            (similar_chunk, 0.3),
        ]
        mock_db_session.execute.return_value = mock_result
        
        results = await vector_search_service.find_similar_chunks(
            chunk_id=chunk_id,
            top_k=5,
            exclude_self=True,
        )
        
        # Should exclude self and return only the similar chunk
        assert len(results) == 1
        assert results[0].chunk == similar_chunk

    @pytest.mark.asyncio
    async def test_find_similar_chunks_not_found(self, vector_search_service):
        """Test that finding similar chunks raises ChunkNotFoundError when chunk doesn't exist."""
        chunk_id = uuid4()
        vector_search_service.repository.get_by_id = AsyncMock(return_value=None)
        
        with pytest.raises(ChunkNotFoundError) as exc_info:
            await vector_search_service.find_similar_chunks(chunk_id=chunk_id)
        
        assert str(chunk_id) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_find_similar_chunks_no_embedding(self, vector_search_service, sample_chunk_without_embedding):
        """Test that finding similar chunks raises InvalidEmbeddingError when chunk has no embedding."""
        chunk_id = uuid4()
        vector_search_service.repository.get_by_id = AsyncMock(return_value=sample_chunk_without_embedding)
        
        with pytest.raises(InvalidEmbeddingError) as exc_info:
            await vector_search_service.find_similar_chunks(chunk_id=chunk_id)
        
        assert "embedding" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_find_similar_chunks_include_self(self, vector_search_service, mock_db_session, sample_chunk):
        """Test finding similar chunks including self."""
        chunk_id = sample_chunk.id
        vector_search_service.repository.get_by_id = AsyncMock(return_value=sample_chunk)
        
        another_chunk = MagicMock(spec=DocumentChunkModel)
        another_chunk.id = uuid4()
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (sample_chunk, 0.0),
            (another_chunk, 0.4),
        ]
        mock_db_session.execute.return_value = mock_result
        
        results = await vector_search_service.find_similar_chunks(
            chunk_id=chunk_id,
            top_k=5,
            exclude_self=False,
        )
        
        # Should include self
        assert len(results) == 2

    def test_calculate_similarity(self, vector_search_service):
        """Test similarity calculation from distance."""
        # Distance 0 = identical (similarity 1)
        assert vector_search_service._calculate_similarity(0.0) == 1.0
        
        # Distance 1 = orthogonal (similarity 0)
        assert vector_search_service._calculate_similarity(1.0) == 0.0
        
        # Distance 0.5 = similarity 0.5
        assert vector_search_service._calculate_similarity(0.5) == 0.5
        
        # Distance > 1 clamped to 0
        assert vector_search_service._calculate_similarity(1.5) == 0.0
        
        # Distance < 0 clamped to 1
        assert vector_search_service._calculate_similarity(-0.5) == 1.0

    def test_validate_embedding_empty(self, vector_search_service):
        """Test validation of empty embedding."""
        with pytest.raises(InvalidEmbeddingError):
            vector_search_service._validate_embedding([])

    def test_validate_embedding_wrong_dimensions(self, vector_search_service):
        """Test validation of embedding with wrong dimensions."""
        with pytest.raises(InvalidEmbeddingError) as exc_info:
            vector_search_service._validate_embedding([0.1] * 768)
        assert "1536" in str(exc_info.value)

    def test_validate_embedding_non_numeric(self, vector_search_service):
        """Test validation of embedding with non-numeric values."""
        with pytest.raises(InvalidEmbeddingError) as exc_info:
            vector_search_service._validate_embedding(["not", "a", "number"] + [0.1] * 1533)
        assert "index" in str(exc_info.value).lower()

    def test_validate_embedding_infinite(self, vector_search_service):
        """Test validation of embedding with infinite values."""
        embedding = [0.1] * 1536
        embedding[0] = float('inf')
        
        with pytest.raises(InvalidEmbeddingError) as exc_info:
            vector_search_service._validate_embedding(embedding)
        assert "infinite" in str(exc_info.value).lower()


# =============================================================================
# TextSearchService Tests
# =============================================================================

@pytest.mark.unit
class TestTextSearchService:
    """Tests for TextSearchService."""

    @pytest.mark.asyncio
    async def test_search_by_text_success(self, text_search_service, mock_db_session, sample_chunk):
        """Test successful text search."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (sample_chunk, 0.85, None),  # chunk, rank, highlighted
        ]
        mock_db_session.execute.return_value = mock_result
        
        results = await text_search_service.search_by_text(
            query="machine learning",
            top_k=10,
            language="english",
        )
        
        assert len(results) == 1
        assert isinstance(results[0], TextSearchResult)
        assert results[0].chunk == sample_chunk
        assert results[0].rank_score == 0.85
        assert results[0].rank == 1

    @pytest.mark.asyncio
    async def test_search_by_text_with_highlighting(self, text_search_service, mock_db_session, sample_chunk):
        """Test text search with highlighting enabled."""
        highlighted = "This is <mark>sample</mark> content."
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (sample_chunk, 0.9, highlighted),
        ]
        mock_db_session.execute.return_value = mock_result
        
        results = await text_search_service.search_by_text(
            query="sample",
            top_k=10,
            highlight=True,
        )
        
        assert len(results) == 1
        assert results[0].highlighted_content == highlighted
        assert "sample" in results[0].matched_terms

    @pytest.mark.asyncio
    async def test_search_by_text_with_filters(self, text_search_service, mock_db_session, sample_chunk):
        """Test text search with filters."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(sample_chunk, 0.75, None)]
        mock_db_session.execute.return_value = mock_result
        
        job_id = uuid4()
        results = await text_search_service.search_by_text(
            query="test query",
            filters={"job_id": str(job_id), "metadata": {"page": 1}},
        )
        
        assert len(results) == 1
        mock_db_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_text_empty_query_raises_error(self, text_search_service):
        """Test that empty query raises InvalidQueryError."""
        with pytest.raises(InvalidQueryError) as exc_info:
            await text_search_service.search_by_text(query="")
        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_by_text_whitespace_only_query_raises_error(self, text_search_service):
        """Test that whitespace-only query raises InvalidQueryError."""
        with pytest.raises(InvalidQueryError):
            await text_search_service.search_by_text(query="   \t\n  ")

    @pytest.mark.asyncio
    async def test_search_by_text_unsupported_language_raises_error(self, text_search_service):
        """Test that unsupported language raises LanguageNotSupportedError."""
        with pytest.raises(LanguageNotSupportedError) as exc_info:
            await text_search_service.search_by_text(
                query="test",
                language="klingon",
            )
        assert "klingon" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_by_text_database_error(self, text_search_service, mock_db_session):
        """Test that database errors raise TextSearchError."""
        mock_db_session.execute.side_effect = SQLAlchemyError("Query failed")
        
        with pytest.raises(TextSearchError) as exc_info:
            await text_search_service.search_by_text(query="test query")
        assert "database" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_fuzzy_success(self, text_search_service, mock_db_session, sample_chunk):
        """Test successful fuzzy search."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (sample_chunk, 0.65),  # chunk, similarity
        ]
        mock_db_session.execute.return_value = mock_result
        
        results = await text_search_service.search_fuzzy(
            query="accomodation",  # Common misspelling
            similarity_threshold=0.3,
            top_k=10,
        )
        
        assert len(results) == 1
        assert results[0].rank_score == 0.65

    @pytest.mark.asyncio
    async def test_search_fuzzy_too_short_query_raises_error(self, text_search_service):
        """Test that too short query raises InvalidQueryError."""
        with pytest.raises(InvalidQueryError) as exc_info:
            await text_search_service.search_fuzzy(query="ab")  # Less than 3 chars
        assert "3" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_fuzzy_threshold_normalized(self, text_search_service, mock_db_session, sample_chunk):
        """Test that similarity threshold is normalized to 0-1 range."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(sample_chunk, 0.5)]
        mock_db_session.execute.return_value = mock_result
        
        # Test with threshold > 1
        await text_search_service.search_fuzzy(
            query="test",
            similarity_threshold=1.5,
        )
        
        # Test with threshold < 0
        await text_search_service.search_fuzzy(
            query="test",
            similarity_threshold=-0.5,
        )

    def test_build_tsquery_basic(self, text_search_service):
        """Test tsquery building for basic query."""
        query = "machine learning"
        tsquery = text_search_service._build_tsquery(query)
        assert "machine" in tsquery
        assert "learning" in tsquery
        assert "&" in tsquery  # AND operator

    def test_build_tsquery_with_phrases(self, text_search_service):
        """Test tsquery building with quoted phrases."""
        query = '"neural network" architecture'
        tsquery = text_search_service._build_tsquery(query)
        assert "neural" in tsquery
        assert "architecture" in tsquery

    def test_build_tsquery_with_boolean_operators(self, text_search_service):
        """Test tsquery building with boolean operators."""
        query = "machine AND learning OR ai NOT robot"
        tsquery = text_search_service._build_tsquery(query)
        assert "&" in tsquery or "|" in tsquery or "!" in tsquery

    def test_validate_query_empty(self, text_search_service):
        """Test validation of empty query."""
        with pytest.raises(InvalidQueryError):
            text_search_service._validate_query("")

    def test_validate_query_too_short(self, text_search_service):
        """Test validation of too short query."""
        with pytest.raises(InvalidQueryError):
            text_search_service._validate_query("a", min_length=2)

    def test_validate_query_too_long(self, text_search_service):
        """Test validation of too long query."""
        long_query = "x" * 1025
        with pytest.raises(InvalidQueryError):
            text_search_service._validate_query(long_query)

    def test_validate_language_supported(self, text_search_service):
        """Test validation of supported language."""
        # Should not raise
        text_search_service._validate_language("english")
        text_search_service._validate_language("spanish")
        text_search_service._validate_language("simple")

    def test_validate_language_unsupported(self, text_search_service):
        """Test validation of unsupported language."""
        with pytest.raises(LanguageNotSupportedError):
            text_search_service._validate_language("elvish")

    def test_extract_matched_terms(self, text_search_service):
        """Test extraction of matched terms from highlighted content."""
        highlighted = "The <mark>quick</mark> <mark>brown</mark> <mark>fox</mark> jumps over the <mark>quick</mark> dog"
        terms = text_search_service._extract_matched_terms(highlighted)
        
        assert "quick" in terms
        assert "brown" in terms
        assert "fox" in terms
        assert len(terms) == 3  # quick appears twice but should be unique


# =============================================================================
# HybridSearchService Tests
# =============================================================================

@pytest.mark.unit
class TestHybridSearchService:
    """Tests for HybridSearchService."""

    @pytest.mark.asyncio
    async def test_search_with_weighted_sum_fusion(
        self, hybrid_search_service, vector_search_service, text_search_service,
        sample_chunk, mock_db_session, mock_embedder
    ):
        """Test hybrid search with weighted sum fusion."""
        # Setup mock results
        vector_result = SearchResult(
            chunk=sample_chunk,
            similarity_score=0.9,
            rank=1,
        )
        text_result = TextSearchResult(
            chunk=sample_chunk,
            rank_score=0.8,
            rank=1,
        )
        
        # Mock the underlying services
        vector_search_service.search_by_vector = AsyncMock(return_value=[vector_result])
        text_search_service.search_by_text = AsyncMock(return_value=[text_result])
        
        results = await hybrid_search_service.search_with_embedding(
            query_embedding=[0.1] * 1536,
            query_text="test query",
            top_k=10,
            vector_weight=0.7,
            text_weight=0.3,
            fusion_method=FusionMethod.WEIGHTED_SUM,
        )
        
        assert len(results) == 1
        assert results[0].hybrid_score == pytest.approx(0.87, rel=0.01)  # 0.7*0.9 + 0.3*0.8
        assert results[0].vector_score == 0.9
        assert results[0].text_score == 0.8
        assert results[0].fusion_method == "weighted_sum"

    @pytest.mark.asyncio
    async def test_search_with_rrf_fusion(
        self, hybrid_search_service, vector_search_service, text_search_service,
        sample_chunk, mock_embedder
    ):
        """Test hybrid search with RRF fusion."""
        chunk1 = MagicMock(spec=DocumentChunkModel)
        chunk1.id = uuid4()
        chunk2 = MagicMock(spec=DocumentChunkModel)
        chunk2.id = uuid4()
        
        vector_results = [
            SearchResult(chunk=chunk1, similarity_score=0.9, rank=1),
            SearchResult(chunk=chunk2, similarity_score=0.8, rank=2),
        ]
        text_results = [
            TextSearchResult(chunk=chunk2, rank_score=0.85, rank=1),
            TextSearchResult(chunk=chunk1, rank_score=0.75, rank=2),
        ]
        
        vector_search_service.search_by_vector = AsyncMock(return_value=vector_results)
        text_search_service.search_by_text = AsyncMock(return_value=text_results)
        
        results = await hybrid_search_service.search_with_embedding(
            query_embedding=[0.1] * 1536,
            query_text="test query",
            top_k=10,
            fusion_method=FusionMethod.RECIPROCAL_RANK_FUSION,
            rrf_k=60,
        )
        
        assert len(results) == 2
        # RRF score: sum of 1/(k + rank) for each list where chunk appears
        # chunk1: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
        # chunk2: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
        assert all(r.fusion_method == "rrf" for r in results)

    @pytest.mark.asyncio
    async def test_search_invalid_weights_raises_error(self, hybrid_search_service, mock_embedder):
        """Test that invalid weights raise InvalidWeightError."""
        with pytest.raises(InvalidWeightError) as exc_info:
            await hybrid_search_service.search(
                query_text="test",
                embedder=mock_embedder,
                vector_weight=0.8,
                text_weight=0.3,  # Sum = 1.1, not 1.0
            )
        assert "1.0" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_negative_weights_raises_error(self, hybrid_search_service, mock_embedder):
        """Test that negative weights raise InvalidWeightError."""
        with pytest.raises(InvalidWeightError) as exc_info:
            await hybrid_search_service.search_with_embedding(
                query_embedding=[0.1] * 1536,
                query_text="test",
                vector_weight=-0.1,
                text_weight=1.1,
            )
        assert "non-negative" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_invalid_fusion_method_raises_error(self, hybrid_search_service, mock_embedder):
        """Test that invalid fusion method raises InvalidFusionMethodError."""
        with pytest.raises(InvalidFusionMethodError):
            await hybrid_search_service.search_with_embedding(
                query_embedding=[0.1] * 1536,
                query_text="test",
                fusion_method="invalid_method",  # type: ignore
            )

    @pytest.mark.asyncio
    async def test_fallback_to_vector_only(
        self, hybrid_search_service, vector_search_service, text_search_service,
        sample_chunk, mock_embedder
    ):
        """Test fallback to vector-only when text search returns no results."""
        vector_results = [SearchResult(chunk=sample_chunk, similarity_score=0.9, rank=1)]
        
        vector_search_service.search_by_vector = AsyncMock(return_value=vector_results)
        text_search_service.search_by_text = AsyncMock(return_value=[])
        
        results = await hybrid_search_service.search_with_embedding(
            query_embedding=[0.1] * 1536,
            query_text="test",
            fallback_mode="auto",
        )
        
        assert len(results) == 1
        assert results[0].vector_score == 0.9
        assert results[0].text_score is None

    @pytest.mark.asyncio
    async def test_fallback_to_text_only(
        self, hybrid_search_service, vector_search_service, text_search_service,
        sample_chunk, mock_embedder
    ):
        """Test fallback to text-only when vector search returns no results."""
        text_results = [TextSearchResult(chunk=sample_chunk, rank_score=0.8, rank=1)]
        
        vector_search_service.search_by_vector = AsyncMock(return_value=[])
        text_search_service.search_by_text = AsyncMock(return_value=text_results)
        
        results = await hybrid_search_service.search_with_embedding(
            query_embedding=[0.1] * 1536,
            query_text="test",
            fallback_mode="auto",
        )
        
        assert len(results) == 1
        assert results[0].vector_score is None
        assert results[0].text_score == 0.8

    @pytest.mark.asyncio
    async def test_strict_fallback_mode(
        self, hybrid_search_service, vector_search_service, text_search_service,
        sample_chunk, mock_embedder
    ):
        """Test strict fallback mode returns empty when one search fails."""
        vector_results = [SearchResult(chunk=sample_chunk, similarity_score=0.9, rank=1)]
        
        vector_search_service.search_by_vector = AsyncMock(return_value=vector_results)
        text_search_service.search_by_text = AsyncMock(return_value=[])
        
        results = await hybrid_search_service.search_with_embedding(
            query_embedding=[0.1] * 1536,
            query_text="test",
            fallback_mode="strict",
        )
        
        # In strict mode, if one search returns empty, we keep both
        # but since one is empty, results should just have the vector ones
        assert len(results) == 1

    def test_normalize_scores(self, hybrid_search_service, sample_chunk):
        """Test score normalization."""
        results = [
            TextSearchResult(chunk=sample_chunk, rank_score=0.8, rank=1),
        ]
        
        normalized = hybrid_search_service._normalize_scores(results)
        
        # Max score is 0.8, so normalized should be 1.0
        chunk_id = str(sample_chunk.id)
        assert normalized[chunk_id] == 1.0

    def test_normalize_scores_empty(self, hybrid_search_service):
        """Test score normalization with empty results."""
        normalized = hybrid_search_service._normalize_scores([])
        assert normalized == {}

    def test_normalize_scores_all_zero(self, hybrid_search_service, sample_chunk):
        """Test score normalization when all scores are zero."""
        results = [
            TextSearchResult(chunk=sample_chunk, rank_score=0.0, rank=1),
        ]
        
        normalized = hybrid_search_service._normalize_scores(results)
        assert normalized[str(sample_chunk.id)] == 0.0

    def test_validate_fallback_mode(self, hybrid_search_service):
        """Test fallback mode validation."""
        # Valid modes
        for mode in ["auto", "vector", "text", "strict"]:
            hybrid_search_service._validate_fallback_mode(mode)  # Should not raise
        
        # Invalid mode
        with pytest.raises(HybridSearchError):
            hybrid_search_service._validate_fallback_mode("invalid")


# =============================================================================
# EmbeddingService Tests
# =============================================================================

@pytest.mark.unit
class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_cache_get_set(self):
        """Test cache get and set operations."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        
        text = "test text"
        model = "text-embedding-ada-002"
        embedding = [0.1, 0.2, 0.3]
        
        # Set value
        cache.set(text, model, embedding)
        
        # Get value
        cached = cache.get(text, model)
        assert cached == embedding

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = EmbeddingCache()
        
        result = cache.get("nonexistent", "model")
        assert result is None

    def test_cache_expiration(self):
        """Test cache entry expires after TTL."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=0.01)
        
        cache.set("text", "model", [0.1, 0.2])
        
        # Should be available immediately
        assert cache.get("text", "model") is not None
        
        # Wait for expiration
        import time
        time.sleep(0.02)
        
        # Should be expired now
        assert cache.get("text", "model") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_size=2, ttl_seconds=3600)
        
        cache.set("text1", "model", [0.1])
        cache.set("text2", "model", [0.2])
        cache.set("text3", "model", [0.3])  # Should evict text1
        
        assert cache.get("text1", "model") is None
        assert cache.get("text2", "model") is not None
        assert cache.get("text3", "model") is not None

    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = EmbeddingCache()
        
        cache.set("text", "model", [0.1])
        assert cache.get("text", "model") is not None
        
        cache.clear()
        assert cache.get("text", "model") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        
        cache.set("text1", "model", [0.1])
        cache.set("text2", "model", [0.2])
        
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 3600


@pytest.mark.unit
class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock vector store config."""
        config = MagicMock()
        config.enabled = True
        config.embedding.model = "text-embedding-ada-002"
        config.embedding.dimensions = 1536
        config.embedding.batch_size = 10
        config.cache.enabled = True
        config.cache.max_size = 1000
        config.cache.ttl_seconds = 3600
        config.to_litellm_params.return_value = {
            "model": "text-embedding-ada-002",
        }
        return config

    @pytest.mark.asyncio
    async def test_embed_text_success(self, mock_config):
        """Test successful text embedding."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            with patch("src.services.embedding_service.aembedding") as mock_aembedding:
                # Mock the litellm response
                mock_response = MagicMock()
                mock_response.data = [{"embedding": [0.1] * 1536}]
                mock_response.usage = {"prompt_tokens": 10}
                mock_aembedding.return_value = mock_response
                
                service = EmbeddingService(config=mock_config)
                result = await service.embed_text("test text")
                
                assert isinstance(result, EmbeddingResult)
                assert result.text == "test text"
                assert result.embedding == [0.1] * 1536
                assert result.model == "text-embedding-ada-002"
                assert result.dimensions == 1536
                assert result.tokens_used == 10
                assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_embed_text_empty_raises_error(self, mock_config):
        """Test that empty text raises EmbeddingError."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            service = EmbeddingService(config=mock_config)
            
            with pytest.raises(EmbeddingError) as exc_info:
                await service.embed_text("")
            assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_embed_text_dimension_mismatch_raises_error(self, mock_config):
        """Test that dimension mismatch raises EmbeddingDimensionError."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            with patch("src.services.embedding_service.aembedding") as mock_aembedding:
                # Return wrong dimensions
                mock_response = MagicMock()
                mock_response.data = [{"embedding": [0.1] * 768}]  # Wrong size
                mock_aembedding.return_value = mock_response
                
                service = EmbeddingService(config=mock_config)
                
                with pytest.raises(EmbeddingDimensionError) as exc_info:
                    await service.embed_text("test")
                assert "768" in str(exc_info.value)
                assert "1536" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_text_provider_error_raises_error(self, mock_config):
        """Test that provider errors raise EmbeddingProviderError."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            with patch("src.services.embedding_service.aembedding") as mock_aembedding:
                mock_aembedding.side_effect = Exception("API error")
                
                service = EmbeddingService(config=mock_config)
                
                with pytest.raises(EmbeddingProviderError) as exc_info:
                    await service.embed_text("test")
                assert "API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_text_uses_cache(self, mock_config):
        """Test that caching works for embed_text."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            with patch("src.services.embedding_service.aembedding") as mock_aembedding:
                mock_response = MagicMock()
                mock_response.data = [{"embedding": [0.1] * 1536}]
                mock_aembedding.return_value = mock_response
                
                service = EmbeddingService(config=mock_config)
                
                # First call - should hit API
                await service.embed_text("test text")
                assert mock_aembedding.call_count == 1
                
                # Second call - should use cache
                result = await service.embed_text("test text")
                assert mock_aembedding.call_count == 1  # No additional API call
                assert result.latency_ms == 0.0  # Cached result has 0 latency

    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_config):
        """Test successful batch embedding."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            with patch("src.services.embedding_service.aembedding") as mock_aembedding:
                mock_response = MagicMock()
                mock_response.data = [
                    {"embedding": [0.1] * 1536},
                    {"embedding": [0.2] * 1536},
                ]
                mock_aembedding.return_value = mock_response
                
                service = EmbeddingService(config=mock_config)
                results = await service.embed_batch(["text1", "text2"])
                
                assert len(results) == 2
                assert results[0].embedding == [0.1] * 1536
                assert results[1].embedding == [0.2] * 1536

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, mock_config):
        """Test batch embedding with empty list."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            service = EmbeddingService(config=mock_config)
            results = await service.embed_batch([])
            assert results == []

    @pytest.mark.asyncio
    async def test_embed_batch_partial_cache_hit(self, mock_config):
        """Test batch embedding with partial cache hits."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            with patch("src.services.embedding_service.aembedding") as mock_aembedding:
                mock_response = MagicMock()
                mock_response.data = [{"embedding": [0.1] * 1536}]
                mock_aembedding.return_value = mock_response
                
                service = EmbeddingService(config=mock_config)
                
                # First batch - all new
                await service.embed_batch(["text1"])
                assert mock_aembedding.call_count == 1
                
                # Second batch - text1 cached, text2 new
                mock_response.data = [{"embedding": [0.2] * 1536}]
                results = await service.embed_batch(["text1", "text2"])
                
                assert mock_aembedding.call_count == 2  # Only text2 needs API
                assert len(results) == 2

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_config):
        """Test health check when service is healthy."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            with patch("src.services.embedding_service.aembedding") as mock_aembedding:
                mock_response = MagicMock()
                mock_response.data = [{"embedding": [0.1] * 1536}]
                mock_aembedding.return_value = mock_response
                
                service = EmbeddingService(config=mock_config)
                health = await service.health_check()
                
                assert health["healthy"] is True
                assert "latency_ms" in health

    @pytest.mark.asyncio
    async def test_health_check_disabled(self, mock_config):
        """Test health check when service is disabled."""
        mock_config.enabled = False
        
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            service = EmbeddingService(config=mock_config)
            health = await service.health_check()
            
            assert health["healthy"] is False
            assert "disabled" in health["message"].lower()

    @pytest.mark.asyncio
    async def test_health_check_litellm_unavailable(self, mock_config):
        """Test health check when litellm is unavailable."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", False):
            service = EmbeddingService(config=mock_config)
            health = await service.health_check()
            
            assert health["healthy"] is False
            assert "litellm" in health.get("error", "").lower()

    def test_get_cache_stats(self, mock_config):
        """Test getting cache statistics."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            service = EmbeddingService(config=mock_config)
            
            stats = service.get_cache_stats()
            assert stats is not None
            assert "size" in stats

    def test_get_cache_stats_disabled(self, mock_config):
        """Test getting cache stats when caching is disabled."""
        mock_config.cache.enabled = False
        
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            service = EmbeddingService(config=mock_config)
            
            stats = service.get_cache_stats()
            assert stats is None

    def test_clear_cache(self, mock_config):
        """Test clearing the cache."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", True):
            service = EmbeddingService(config=mock_config)
            
            # Add something to cache
            service._cache.set("text", "model", [0.1])
            assert service._cache.get("text", "model") is not None
            
            # Clear cache
            service.clear_cache()
            assert service._cache.get("text", "model") is None

    def test_service_init_litellm_not_installed(self, mock_config):
        """Test that service raises ImportError when litellm is not installed."""
        with patch("src.services.embedding_service.LITELLM_AVAILABLE", False):
            with pytest.raises(ImportError) as exc_info:
                EmbeddingService(config=mock_config)
            assert "litellm" in str(exc_info.value).lower()


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

@pytest.mark.unit
class TestSearchServiceEdgeCases:
    """Tests for edge cases and integration scenarios."""

    @pytest.mark.asyncio
    async def test_vector_search_no_results(self, vector_search_service, mock_db_session):
        """Test vector search returning no results."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db_session.execute.return_value = mock_result
        
        results = await vector_search_service.search_by_vector(
            query_embedding=[0.1] * 1536,
            top_k=10,
        )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_text_search_no_results(self, text_search_service, mock_db_session):
        """Test text search returning no results."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db_session.execute.return_value = mock_result
        
        results = await text_search_service.search_by_text(
            query="xyznonexistent",
            top_k=10,
        )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_search_both_empty(self, hybrid_search_service, vector_search_service, text_search_service):
        """Test hybrid search when both searches return empty."""
        vector_search_service.search_by_vector = AsyncMock(return_value=[])
        text_search_service.search_by_text = AsyncMock(return_value=[])
        
        results = await hybrid_search_service.search_with_embedding(
            query_embedding=[0.1] * 1536,
            query_text="test",
        )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_search_different_chunks(
        self, hybrid_search_service, vector_search_service, text_search_service
    ):
        """Test hybrid search with different chunks in each result set."""
        chunk1 = MagicMock(spec=DocumentChunkModel)
        chunk1.id = uuid4()
        chunk2 = MagicMock(spec=DocumentChunkModel)
        chunk2.id = uuid4()
        
        vector_results = [SearchResult(chunk=chunk1, similarity_score=0.9, rank=1)]
        text_results = [TextSearchResult(chunk=chunk2, rank_score=0.8, rank=1)]
        
        vector_search_service.search_by_vector = AsyncMock(return_value=vector_results)
        text_search_service.search_by_text = AsyncMock(return_value=text_results)
        
        results = await hybrid_search_service.search_with_embedding(
            query_embedding=[0.1] * 1536,
            query_text="test",
            vector_weight=0.7,
            text_weight=0.3,
        )
        
        # Should have both chunks
        assert len(results) == 2
        chunk_ids = {str(r.chunk.id) for r in results}
        assert str(chunk1.id) in chunk_ids
        assert str(chunk2.id) in chunk_ids
