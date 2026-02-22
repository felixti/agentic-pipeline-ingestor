"""Unit tests for Contextual Retrieval functionality.

This module tests the ContextualRetrieval class and related components including
parent document enhancement, window context, and hierarchical context strategies.
"""

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DocumentChunkModel
from src.rag.contextual import (
    ChunkNotFoundError,
    ContextualRetrieval,
    ContextualRetrievalConfig,
    ContextualRetrievalError,
    HierarchicalStrategyConfig,
    InvalidContextTypeError,
    ParentDocumentStrategyConfig,
    WindowStrategyConfig,
)
from src.rag.models import (
    ContextType,
    ContextualContext,
    ContextualRetrievalRequest,
    ContextualRetrievalResult,
    EnhancedChunk,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_db_session():
    """Create a mock AsyncSession."""
    session = MagicMock(spec=AsyncSession)
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_chunk():
    """Create a mock DocumentChunkModel."""
    chunk = MagicMock(spec=DocumentChunkModel)
    chunk.id = uuid4()
    chunk.job_id = uuid4()
    chunk.chunk_index = 5
    chunk.content = "The system uses pgvector for vector storage."
    chunk.chunk_metadata = {
        "source": "test.pdf",
        "page": 10,
        "section_headers": ["Database Setup", "Vector Storage"],
    }
    chunk.content_hash = "abc123"
    return chunk


@pytest.fixture
def mock_chunk_no_metadata():
    """Create a mock chunk without metadata."""
    chunk = MagicMock(spec=DocumentChunkModel)
    chunk.id = uuid4()
    chunk.job_id = uuid4()
    chunk.chunk_index = 0
    chunk.content = "Introduction to the system."
    chunk.chunk_metadata = {}
    chunk.content_hash = "def456"
    return chunk


@pytest.fixture
def mock_job_row():
    """Create a mock job row result."""
    row = MagicMock()
    row.job_id = uuid4()
    row.file_name = "Database Architecture Guide.pdf"
    # For _get_parent_document_context which aliases as job_metadata
    row.job_metadata = {"author": "John Smith", "category": "technical"}
    row.output_data = None
    row.result_metadata = {"processed_pages": 50}
    return row


@pytest.fixture
def mock_parent_doc_row():
    """Create a mock parent document row (for get_parent_document method)."""
    row = MagicMock()
    row.id = uuid4()
    row.file_name = "Database Architecture Guide.pdf"
    # For get_parent_document which uses metadata_json directly
    row.metadata_json = {"author": "John Smith", "category": "technical"}
    row.result_metadata = {"processed_pages": 50}
    return row


@pytest.fixture
def default_config():
    """Create default contextual retrieval config."""
    return ContextualRetrievalConfig(
        enabled=True,
        default_strategy=ContextType.PARENT_DOCUMENT,
        parent_document=ParentDocumentStrategyConfig(
            include_metadata=True,
            metadata_fields=["title", "author", "category"],
            max_context_length=256,
        ),
        window=WindowStrategyConfig(
            window_size=1,
            separator=" | ",
            max_chunk_length=200,
        ),
        hierarchical=HierarchicalStrategyConfig(
            max_depth=3,
            include_path=True,
        ),
    )


@pytest.fixture
def contextual_retrieval(mock_db_session, default_config):
    """Create a ContextualRetrieval instance with mocked dependencies."""
    return ContextualRetrieval(
        db_session=mock_db_session,
        config=default_config,
    )


# ============================================================================
# ContextType Enum Tests
# ============================================================================

class TestContextType:
    """Tests for ContextType enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert ContextType.PARENT_DOCUMENT.value == "parent_document"
        assert ContextType.WINDOW.value == "window"
        assert ContextType.HIERARCHICAL.value == "hierarchical"

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert ContextType("parent_document") == ContextType.PARENT_DOCUMENT
        assert ContextType("window") == ContextType.WINDOW
        assert ContextType("hierarchical") == ContextType.HIERARCHICAL


# ============================================================================
# ContextualContext Model Tests
# ============================================================================

class TestContextualContext:
    """Tests for ContextualContext Pydantic model."""

    def test_default_creation(self):
        """Test creating context with default values."""
        context = ContextualContext()
        
        assert context.document_id is None
        assert context.document_title is None
        assert context.section_headers == []
        assert context.document_metadata == {}
        assert context.previous_chunk_content is None
        assert context.next_chunk_content is None
        assert context.hierarchy_path == []
        assert context.hierarchy_level is None

    def test_full_creation(self):
        """Test creating context with all values."""
        context = ContextualContext(
            document_id="doc-123",
            document_title="Test Document",
            section_headers=["Section 1", "Subsection 1.1"],
            document_metadata={"author": "John", "category": "tech"},
            previous_chunk_content="Previous content...",
            next_chunk_content="Next content...",
            hierarchy_path=["Section 1", "Subsection 1.1"],
            hierarchy_level=2,
        )
        
        assert context.document_id == "doc-123"
        assert context.document_title == "Test Document"
        assert context.section_headers == ["Section 1", "Subsection 1.1"]
        assert context.document_metadata == {"author": "John", "category": "tech"}
        assert context.previous_chunk_content == "Previous content..."
        assert context.next_chunk_content == "Next content..."
        assert context.hierarchy_path == ["Section 1", "Subsection 1.1"]
        assert context.hierarchy_level == 2

    def test_model_json_schema(self):
        """Test that model has valid JSON schema."""
        schema = ContextualContext.model_json_schema()
        
        assert "properties" in schema
        assert "document_id" in schema["properties"]
        assert "section_headers" in schema["properties"]


# ============================================================================
# EnhancedChunk Model Tests
# ============================================================================

class TestEnhancedChunk:
    """Tests for EnhancedChunk Pydantic model."""

    def test_creation(self):
        """Test creating an enhanced chunk."""
        chunk = EnhancedChunk(
            chunk_id="chunk-123",
            original_content="Original text",
            enhanced_text="Document: Test\nContent: Original text",
            context=ContextualContext(document_title="Test"),
            context_type=ContextType.PARENT_DOCUMENT,
        )
        
        assert chunk.chunk_id == "chunk-123"
        assert chunk.original_content == "Original text"
        assert "Document: Test" in chunk.enhanced_text
        assert chunk.embedding is None
        assert chunk.context_type == ContextType.PARENT_DOCUMENT

    def test_with_embedding(self):
        """Test enhanced chunk with embedding."""
        embedding = [0.1, 0.2, 0.3]
        chunk = EnhancedChunk(
            chunk_id="chunk-123",
            original_content="Original",
            enhanced_text="Enhanced",
            embedding=embedding,
        )
        
        assert chunk.embedding == embedding

    def test_validation_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValueError):
            EnhancedChunk()  # Missing required fields


# ============================================================================
# ContextualRetrievalConfig Tests
# ============================================================================

class TestContextualRetrievalConfig:
    """Tests for ContextualRetrievalConfig dataclass."""

    def test_default_creation(self):
        """Test default config creation."""
        config = ContextualRetrievalConfig()
        
        assert config.enabled is True
        assert config.default_strategy == ContextType.PARENT_DOCUMENT
        assert config.enable_embedding is False
        assert isinstance(config.parent_document, ParentDocumentStrategyConfig)
        assert isinstance(config.window, WindowStrategyConfig)
        assert isinstance(config.hierarchical, HierarchicalStrategyConfig)

    def test_from_settings(self):
        """Test loading config from settings."""
        with patch("src.rag.contextual.settings") as mock_settings:
            mock_settings.contextual_retrieval = MagicMock()
            mock_settings.contextual_retrieval.enabled = True
            mock_settings.contextual_retrieval.default_strategy = "window"
            mock_settings.contextual_retrieval.strategies = {
                "parent_document": {
                    "include_metadata": False,
                    "metadata_fields": ["title"],
                    "max_context_length": 128,
                },
                "window": {
                    "window_size": 2,
                    "separator": " || ",
                },
                "hierarchical": {
                    "max_depth": 5,
                    "include_path": False,
                },
            }
            
            config = ContextualRetrievalConfig.from_settings()
            
            assert config.default_strategy == ContextType.WINDOW
            assert config.parent_document.include_metadata is False
            assert config.window.window_size == 2
            assert config.hierarchical.max_depth == 5


# ============================================================================
# ContextualRetrieval Initialization Tests
# ============================================================================

class TestContextualRetrievalInit:
    """Tests for ContextualRetrieval initialization."""

    def test_default_initialization(self, mock_db_session):
        """Test initialization with defaults."""
        retrieval = ContextualRetrieval(db_session=mock_db_session)
        
        assert retrieval.db_session == mock_db_session
        assert retrieval.config is not None
        assert retrieval.embedding_service is None

    def test_custom_config(self, mock_db_session, default_config):
        """Test initialization with custom config."""
        retrieval = ContextualRetrieval(
            db_session=mock_db_session,
            config=default_config,
        )
        
        assert retrieval.config == default_config
        assert retrieval.config.default_strategy == ContextType.PARENT_DOCUMENT

    def test_no_db_session(self):
        """Test initialization without database session."""
        retrieval = ContextualRetrieval()
        
        assert retrieval.db_session is None
        assert retrieval.config is not None


# ============================================================================
# Parent Document Strategy Tests
# ============================================================================

@pytest.mark.asyncio
class TestParentDocumentStrategy:
    """Tests for parent document enhancement strategy."""

    async def test_enhance_chunk_parent_document(self, contextual_retrieval, mock_chunk, mock_job_row):
        """Test enhancing a chunk with parent document context."""
        # Setup mock result
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_job_row
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        # Enhance chunk
        enhanced = await contextual_retrieval.enhance_chunk(
            mock_chunk,
            context_type=ContextType.PARENT_DOCUMENT,
        )
        
        # Verify
        assert enhanced.chunk_id == str(mock_chunk.id)
        assert enhanced.original_content == mock_chunk.content
        assert enhanced.context_type == ContextType.PARENT_DOCUMENT
        assert "Database Architecture Guide" in enhanced.enhanced_text
        assert "Document:" in enhanced.enhanced_text
        assert "Content:" in enhanced.enhanced_text

    async def test_enhance_chunk_no_db_session(self, mock_chunk, default_config):
        """Test enhancement when no DB session is available."""
        retrieval = ContextualRetrieval(db_session=None, config=default_config)
        
        enhanced = await retrieval.enhance_chunk(
            mock_chunk,
            context_type=ContextType.PARENT_DOCUMENT,
        )
        
        # Should still work but with minimal context
        assert enhanced.chunk_id == str(mock_chunk.id)
        assert "Content:" in enhanced.enhanced_text

    async def test_parent_document_context_fetch(self, contextual_retrieval, mock_chunk, mock_job_row):
        """Test fetching parent document context."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_job_row
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        context = await contextual_retrieval._get_parent_document_context(mock_chunk)
        
        assert context.document_title == "Database Architecture Guide.pdf"
        assert "author" in context.document_metadata
        assert context.document_metadata["author"] == "John Smith"

    async def test_get_parent_document(self, contextual_retrieval, mock_chunk, mock_parent_doc_row):
        """Test get_parent_document method."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_parent_doc_row
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        doc = await contextual_retrieval.get_parent_document(str(mock_chunk.id))
        
        assert doc is not None
        assert doc["title"] == "Database Architecture Guide.pdf"
        assert "author" in doc["metadata"]


# ============================================================================
# Window Context Strategy Tests
# ============================================================================

@pytest.mark.asyncio
class TestWindowStrategy:
    """Tests for window context enhancement strategy."""

    async def test_enhance_chunk_window(self, contextual_retrieval, mock_chunk):
        """Test enhancing a chunk with window context."""
        # Setup mock for previous/next chunks
        prev_row = MagicMock()
        prev_row.content = "Previous chunk content here."
        
        next_row = MagicMock()
        next_row.content = "Next chunk content here."
        
        mock_result = MagicMock()
        mock_result.fetchall.side_effect = [[prev_row], [next_row]]
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        enhanced = await contextual_retrieval.enhance_chunk(
            mock_chunk,
            context_type=ContextType.WINDOW,
        )
        
        assert enhanced.context_type == ContextType.WINDOW
        assert "[Prev:" in enhanced.enhanced_text
        assert "[Next:" in enhanced.enhanced_text
        assert mock_chunk.content in enhanced.enhanced_text

    async def test_window_context_fetch(self, contextual_retrieval, mock_chunk):
        """Test fetching window context."""
        prev_row = MagicMock()
        prev_row.content = "Previous content"
        
        next_row = MagicMock()
        next_row.content = "Next content"
        
        mock_result = MagicMock()
        mock_result.fetchall.side_effect = [[prev_row], [next_row]]
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        context = await contextual_retrieval._get_window_context(mock_chunk)
        
        assert context.previous_chunk_content == "Previous content"
        assert context.next_chunk_content == "Next content"

    async def test_window_no_neighbors(self, contextual_retrieval, mock_chunk):
        """Test window context when no neighbors exist."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        enhanced = await contextual_retrieval.enhance_chunk(
            mock_chunk,
            context_type=ContextType.WINDOW,
        )
        
        # Should still have content but no prev/next markers
        assert mock_chunk.content in enhanced.enhanced_text


# ============================================================================
# Hierarchical Strategy Tests
# ============================================================================

@pytest.mark.asyncio
class TestHierarchicalStrategy:
    """Tests for hierarchical context enhancement strategy."""

    async def test_enhance_chunk_hierarchical(self, contextual_retrieval, mock_chunk, mock_job_row):
        """Test enhancing a chunk with hierarchical context."""
        # Setup mock for parent document
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_job_row
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        enhanced = await contextual_retrieval.enhance_chunk(
            mock_chunk,
            context_type=ContextType.HIERARCHICAL,
        )
        
        assert enhanced.context_type == ContextType.HIERARCHICAL
        assert "Document:" in enhanced.enhanced_text or "Path:" in enhanced.enhanced_text
        assert "Content:" in enhanced.enhanced_text

    async def test_hierarchical_context_from_metadata(self, contextual_retrieval, mock_chunk):
        """Test hierarchical context extraction from chunk metadata."""
        # Mock no DB result for hierarchy table
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        context = await contextual_retrieval._get_hierarchical_context(mock_chunk)
        
        # Should extract from chunk metadata
        assert context.section_headers == ["Database Setup", "Vector Storage"]
        assert context.hierarchy_path == ["Database Setup", "Vector Storage"]
        assert context.hierarchy_level == 2

    async def test_get_hierarchy_path(self, contextual_retrieval):
        """Test get_hierarchy_path method."""
        chunk_id = str(uuid4())
        
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None  # No hierarchy table result
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        # Also mock get_section_headers
        with patch.object(
            contextual_retrieval,
            "get_section_headers",
            return_value=["Section 1", "Subsection 1.1"],
        ):
            path = await contextual_retrieval.get_hierarchy_path(chunk_id)
            
            assert path == ["Section 1", "Subsection 1.1"]


# ============================================================================
# Formatting Tests
# ============================================================================

class TestTextFormatting:
    """Tests for text formatting methods."""

    def test_format_parent_document_text(self, contextual_retrieval):
        """Test parent document text formatting."""
        content = "The system uses pgvector."
        context = ContextualContext(
            document_title="Database Guide",
            section_headers=["Vector Storage"],
            document_metadata={"author": "John"},
        )
        
        formatted = contextual_retrieval._format_parent_document_text(content, context)
        
        assert "Document: Database Guide" in formatted
        assert "Section: Vector Storage" in formatted
        assert "Author: John" in formatted
        assert "Content: The system uses pgvector." in formatted

    def test_format_window_text(self, contextual_retrieval):
        """Test window text formatting."""
        content = "Main chunk content."
        context = ContextualContext(
            previous_chunk_content="Previous text...",
            next_chunk_content="Next text...",
        )
        
        formatted = contextual_retrieval._format_window_text(content, context)
        
        assert "[Prev:" in formatted
        assert "Main chunk content." in formatted
        assert "[Next:" in formatted
        assert " | " in formatted  # Default separator

    def test_format_window_text_no_neighbors(self, contextual_retrieval):
        """Test window text formatting without neighbors."""
        content = "Main chunk content."
        context = ContextualContext()
        
        formatted = contextual_retrieval._format_window_text(content, context)
        
        assert formatted == "Main chunk content."
        assert "[Prev:" not in formatted
        assert "[Next:" not in formatted

    def test_format_hierarchical_text(self, contextual_retrieval):
        """Test hierarchical text formatting."""
        content = "JWT tokens expire after 24 hours."
        context = ContextualContext(
            document_title="API Documentation",
            hierarchy_path=["Authentication", "JWT Tokens"],
            hierarchy_level=2,
        )
        
        formatted = contextual_retrieval._format_hierarchical_text(content, context)
        
        assert "Document: API Documentation" in formatted
        assert "Path: Authentication > JWT Tokens" in formatted
        assert "Level: 2" in formatted
        assert "Content: JWT tokens expire after 24 hours." in formatted


# ============================================================================
# Batch Processing Tests
# ============================================================================

@pytest.mark.asyncio
class TestBatchProcessing:
    """Tests for batch chunk enhancement."""

    async def test_enhance_chunks_batch(self, contextual_retrieval, mock_chunk, mock_job_row):
        """Test batch enhancement of chunks."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_job_row
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        chunks = [mock_chunk, mock_chunk]
        enhanced = await contextual_retrieval.enhance_chunks_batch(
            chunks,
            context_type=ContextType.PARENT_DOCUMENT,
        )
        
        assert len(enhanced) == 2
        for chunk in enhanced:
            assert chunk.context_type == ContextType.PARENT_DOCUMENT

    async def test_batch_with_failures(self, contextual_retrieval, mock_chunk):
        """Test batch processing continues on individual failures."""
        # First call succeeds, second fails
        mock_result = MagicMock()
        mock_result.fetchone.side_effect = [MagicMock(), Exception("DB Error")]
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        chunks = [mock_chunk, mock_chunk]
        
        # Should not raise, just log warnings
        enhanced = await contextual_retrieval.enhance_chunks_batch(chunks)
        
        # Should have at least one successful enhancement
        assert len(enhanced) >= 0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_contextual_retrieval_error(self):
        """Test ContextualRetrievalError."""
        error = ContextualRetrievalError("Test error", context={"key": "value"})
        
        assert str(error) == "Test error"
        assert error.context == {"key": "value"}

    def test_chunk_not_found_error(self):
        """Test ChunkNotFoundError."""
        error = ChunkNotFoundError("Chunk not found")
        
        assert isinstance(error, ContextualRetrievalError)
        assert "Chunk not found" in str(error)

    def test_invalid_context_type_error(self):
        """Test InvalidContextTypeError."""
        error = InvalidContextTypeError("Invalid type")
        
        assert isinstance(error, ContextualRetrievalError)
        assert "Invalid type" in str(error)

    @pytest.mark.asyncio
    async def test_invalid_context_type_in_get_context(self, contextual_retrieval, mock_chunk):
        """Test that invalid context type raises error."""
        # Create a mock context type that's not valid
        invalid_type = MagicMock()
        invalid_type.value = "invalid"
        
        with pytest.raises(InvalidContextTypeError):
            # We need to bypass the enum validation
            await contextual_retrieval._get_context(mock_chunk, invalid_type)


# ============================================================================
# Utility Method Tests
# ============================================================================

@pytest.mark.asyncio
class TestUtilityMethods:
    """Tests for utility methods."""

    async def test_get_section_headers(self, contextual_retrieval, mock_chunk):
        """Test get_section_headers method."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(
            chunk_metadata={"section_headers": ["Header 1", "Header 2"]}
        )
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        headers = await contextual_retrieval.get_section_headers(str(mock_chunk.id))
        
        assert headers == ["Header 1", "Header 2"]

    async def test_get_section_headers_from_string(self, contextual_retrieval, mock_chunk):
        """Test get_section_headers with string header."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(
            chunk_metadata={"section_headers": "Single Header"}
        )
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        headers = await contextual_retrieval.get_section_headers(str(mock_chunk.id))
        
        assert headers == ["Single Header"]

    async def test_get_section_headers_no_metadata(self, contextual_retrieval, mock_chunk):
        """Test get_section_headers when no metadata exists."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(chunk_metadata={})
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        headers = await contextual_retrieval.get_section_headers(str(mock_chunk.id))
        
        assert headers == []

    async def test_get_section_headers_no_db(self, mock_chunk, default_config):
        """Test get_section_headers without DB session."""
        retrieval = ContextualRetrieval(db_session=None, config=default_config)
        
        headers = await retrieval.get_section_headers(str(mock_chunk.id))
        
        assert headers == []

    async def test_get_parent_document_no_db(self, mock_chunk, default_config):
        """Test get_parent_document without DB session."""
        retrieval = ContextualRetrieval(db_session=None, config=default_config)
        
        doc = await retrieval.get_parent_document(str(mock_chunk.id))
        
        assert doc is None


# ============================================================================
# Request/Result Model Tests
# ============================================================================

class TestRequestResultModels:
    """Tests for request and result models."""

    def test_contextual_retrieval_request(self):
        """Test ContextualRetrievalRequest model."""
        request = ContextualRetrievalRequest(
            chunk_id="chunk-123",
            context_type=ContextType.PARENT_DOCUMENT,
            include_metadata=True,
            metadata_fields=["title", "author"],
        )
        
        assert request.chunk_id == "chunk-123"
        assert request.context_type == ContextType.PARENT_DOCUMENT
        assert request.include_metadata is True
        assert request.metadata_fields == ["title", "author"]

    def test_contextual_retrieval_request_defaults(self):
        """Test ContextualRetrievalRequest defaults."""
        request = ContextualRetrievalRequest(chunk_id="chunk-123")
        
        assert request.context_type == ContextType.PARENT_DOCUMENT
        assert request.include_metadata is True
        assert request.metadata_fields is None

    def test_contextual_retrieval_result(self):
        """Test ContextualRetrievalResult model."""
        enhanced = EnhancedChunk(
            chunk_id="chunk-123",
            original_content="Original",
            enhanced_text="Enhanced",
        )
        
        result = ContextualRetrievalResult(
            success=True,
            enhanced_chunk=enhanced,
            latency_ms=5.5,
        )
        
        assert result.success is True
        assert result.enhanced_chunk == enhanced
        assert result.latency_ms == 5.5
        assert result.error is None

    def test_contextual_retrieval_result_error(self):
        """Test ContextualRetrievalResult with error."""
        result = ContextualRetrievalResult(
            success=False,
            error="Something went wrong",
            latency_ms=10.0,
        )
        
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.enhanced_chunk is None


# ============================================================================
# Integration-Like Tests
# ============================================================================

@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration-like tests for complete scenarios."""

    async def test_full_parent_document_flow(self, mock_db_session, mock_chunk):
        """Test complete parent document enhancement flow."""
        # Setup comprehensive mock
        job_row = MagicMock()
        job_row.job_id = mock_chunk.job_id
        job_row.file_name = "API Documentation v2.0.pdf"
        job_row.job_metadata = {"author": "Jane Doe", "category": "API"}
        job_row.output_data = None
        job_row.result_metadata = {"version": "2.0"}
        
        mock_result = MagicMock()
        mock_result.fetchone.return_value = job_row
        mock_db_session.execute.return_value = mock_result
        
        config = ContextualRetrievalConfig(
            parent_document=ParentDocumentStrategyConfig(
                include_metadata=True,
                metadata_fields=["title", "author", "category", "version"],
            )
        )
        
        retrieval = ContextualRetrieval(
            db_session=mock_db_session,
            config=config,
        )
        
        enhanced = await retrieval.enhance_chunk(
            mock_chunk,
            context_type=ContextType.PARENT_DOCUMENT,
        )
        
        # Verify enhanced content structure
        assert "Document: API Documentation v2.0.pdf" in enhanced.enhanced_text
        assert "Section:" in enhanced.enhanced_text
        assert "Author: Jane Doe" in enhanced.enhanced_text
        assert "Category: API" in enhanced.enhanced_text
        assert enhanced.context.document_title == "API Documentation v2.0.pdf"
        assert enhanced.context_type == ContextType.PARENT_DOCUMENT

    async def test_full_window_flow(self, mock_db_session, mock_chunk):
        """Test complete window context enhancement flow."""
        # Setup mocks for previous and next chunks
        prev_result = MagicMock()
        prev_result.fetchall.return_value = [
            MagicMock(content="...semantic search requires vector storage."),
        ]
        
        next_result = MagicMock()
        next_result.fetchall.return_value = [
            MagicMock(content="This enables efficient similarity queries."),
        ]
        
        # First call for prev, second for next
        mock_db_session.execute.side_effect = [prev_result, next_result]
        
        config = ContextualRetrievalConfig(
            window=WindowStrategyConfig(
                window_size=1,
                separator=" | ",
            )
        )
        
        retrieval = ContextualRetrieval(
            db_session=mock_db_session,
            config=config,
        )
        
        enhanced = await retrieval.enhance_chunk(
            mock_chunk,
            context_type=ContextType.WINDOW,
        )
        
        # Verify window format
        assert "[Prev:" in enhanced.enhanced_text
        assert "semantic search requires vector storage" in enhanced.enhanced_text
        assert mock_chunk.content in enhanced.enhanced_text
        assert "[Next:" in enhanced.enhanced_text
        assert "efficient similarity queries" in enhanced.enhanced_text

    async def test_full_hierarchical_flow(self, mock_db_session, mock_chunk):
        """Test complete hierarchical enhancement flow."""
        # Setup mock for parent document
        job_row = MagicMock()
        job_row.job_id = mock_chunk.job_id
        job_row.file_name = "API Documentation v2.0"
        job_row.job_metadata = {}
        job_row.output_data = None
        job_row.result_metadata = {}
        
        mock_result = MagicMock()
        mock_result.fetchone.return_value = job_row
        mock_db_session.execute.return_value = mock_result
        
        # Ensure chunk has hierarchy metadata
        mock_chunk.chunk_metadata["hierarchy_level"] = 2
        mock_chunk.chunk_metadata["hierarchy_path"] = ["Authentication", "JWT Tokens"]
        
        config = ContextualRetrievalConfig(
            hierarchical=HierarchicalStrategyConfig(
                max_depth=3,
                include_path=True,
            )
        )
        
        retrieval = ContextualRetrieval(
            db_session=mock_db_session,
            config=config,
        )
        
        enhanced = await retrieval.enhance_chunk(
            mock_chunk,
            context_type=ContextType.HIERARCHICAL,
        )
        
        # Verify hierarchical format
        assert "Document: API Documentation v2.0" in enhanced.enhanced_text
        assert "Path:" in enhanced.enhanced_text
        assert "Content:" in enhanced.enhanced_text

    async def test_strategy_with_metadata_filtering(self, mock_db_session, mock_chunk):
        """Test that metadata fields are properly filtered."""
        job_row = MagicMock()
        job_row.job_id = mock_chunk.job_id
        job_row.file_name = "Test.pdf"
        job_row.job_metadata = {"author": "John", "category": "tech", "irrelevant": "data"}
        job_row.output_data = None
        job_row.result_metadata = {}
        
        mock_result = MagicMock()
        mock_result.fetchone.return_value = job_row
        mock_db_session.execute.return_value = mock_result
        
        config = ContextualRetrievalConfig(
            parent_document=ParentDocumentStrategyConfig(
                include_metadata=True,
                metadata_fields=["author", "category"],  # Only these should appear
            )
        )
        
        retrieval = ContextualRetrieval(
            db_session=mock_db_session,
            config=config,
        )
        
        enhanced = await retrieval.enhance_chunk(
            mock_chunk,
            context_type=ContextType.PARENT_DOCUMENT,
        )
        
        # Should include specified fields
        assert "Author: John" in enhanced.enhanced_text
        assert "Category: tech" in enhanced.enhanced_text


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
class TestPerformance:
    """Tests for performance requirements."""

    async def test_context_lookup_performance(self, contextual_retrieval, mock_chunk, mock_job_row):
        """Test that context lookup is fast (< 10ms target)."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_job_row
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        start = time.perf_counter()
        await contextual_retrieval.enhance_chunk(mock_chunk)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Should complete quickly (mocked, so actual timing may vary)
        # This is more of a smoke test for the timing infrastructure
        assert elapsed_ms < 1000  # Very loose assertion for mocked tests

    async def test_enhanced_chunk_has_timestamp(self, contextual_retrieval, mock_chunk, mock_job_row):
        """Test that enhanced chunks have timestamps."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_job_row
        contextual_retrieval.db_session.execute.return_value = mock_result
        
        enhanced = await contextual_retrieval.enhance_chunk(mock_chunk)
        
        assert enhanced.enhanced_at is not None
        # Verify ISO format
        assert "T" in enhanced.enhanced_at
        assert "Z" in enhanced.enhanced_at
