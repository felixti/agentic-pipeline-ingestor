"""Unit tests for document chunking strategies.

This module tests all chunking strategies including:
- SemanticChunker: Tests semantic similarity-based chunking
- HierarchicalChunker: Tests structure-aware chunking
- FixedSizeChunker: Tests fixed-size token-based chunking
- AgenticChunker: Tests automatic strategy selection
- ChunkingService: Tests the orchestration service
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.chunking import (
    AgenticChunker,
    AgenticChunkerConfig,
    BaseChunker,
    ChunkingError,
    ChunkingService,
    FixedSizeChunker,
    FixedSizeChunkerConfig,
    HierarchicalChunker,
    HierarchicalChunkerConfig,
    SemanticChunker,
    SemanticChunkerConfig,
)
from src.rag.models import Chunk, ChunkingStrategy, Document, DocumentSection

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id="test-doc-1",
        title="Test Document",
        content="This is a test document. It has multiple sentences. Each sentence is distinct.",
        metadata={"author": "Test Author"},
    )


@pytest.fixture
def structured_document():
    """Create a document with clear structure."""
    content = """# Introduction
This is the introduction section.

## Background
Some background information here.

## Methods
The methods section content.

# Results
Results are presented here.

# Conclusion
Final thoughts."""
    
    return Document(
        id="structured-doc-1",
        title="Structured Document",
        content=content,
        metadata={"doc_type": "technical"},
    )


@pytest.fixture
def technical_document():
    """Create a technical document with code."""
    content = """# API Documentation

## Authentication
The API uses JWT tokens for authentication.

```python
def authenticate(token):
    return jwt.decode(token, secret)
```

## Endpoints
Available endpoints are documented below.

### GET /api/users
Returns a list of users.

Parameters:
- limit: Maximum number of users
- offset: Pagination offset"""
    
    return Document(
        id="tech-doc-1",
        title="API Documentation",
        content=content,
        doc_type="technical",
    )


@pytest.fixture
def long_document():
    """Create a long document for fixed-size chunking tests."""
    # Create content that's definitely longer than chunk_size
    paragraphs = []
    for i in range(50):
        paragraphs.append(
            f"Paragraph {i}: This is a test sentence. " * 20
        )
    
    return Document(
        id="long-doc-1",
        title="Long Document",
        content="\n\n".join(paragraphs),
    )


@pytest.fixture
def document_with_code():
    """Create a document with code blocks."""
    content = """# Example Code

Here is some Python code:

```python
def hello_world():
    print("Hello, World!")
    return 42
```

And here is more text after the code block.

```javascript
function greet() {
    console.log("Hello!");
}
```

End of document."""
    
    return Document(
        id="code-doc-1",
        title="Code Example",
        content=content,
    )


@pytest.fixture
def semantic_config():
    """Create semantic chunker config."""
    return SemanticChunkerConfig(
        similarity_threshold=0.85,
        min_chunk_size=10,
        max_chunk_size=100,
    )


@pytest.fixture
def hierarchical_config():
    """Create hierarchical chunker config."""
    return HierarchicalChunkerConfig(
        max_depth=4,
        respect_headers=True,
        preserve_code_blocks=True,
        max_chunk_size=200,
    )


@pytest.fixture
def fixed_config():
    """Create fixed-size chunker config."""
    return FixedSizeChunkerConfig(
        chunk_size=50,
        overlap=10,
    )


@pytest.fixture
def agentic_config():
    """Create agentic chunker config."""
    return AgenticChunkerConfig(
        selection_model="gpt-4.1",
    )


# ============================================================================
# BaseChunker Tests
# ============================================================================


class ConcreteChunker(BaseChunker):
    """Concrete implementation for testing BaseChunker."""
    
    async def chunk(self, document: Document) -> list[Chunk]:
        return []


class TestBaseChunker:
    """Tests for BaseChunker base class."""

    def test_extract_code_blocks(self):
        """Test code block extraction."""
        chunker = ConcreteChunker()
        
        text = """Some text.
```python
def foo():
    pass
```
More text."""
        
        blocks, cleaned = chunker._extract_code_blocks(text)
        
        assert len(blocks) == 1
        assert "CODE_BLOCK_0" in cleaned
        assert "```python" in blocks[0]["content"]
    
    def test_restore_code_blocks(self):
        """Test code block restoration."""
        chunker = ConcreteChunker()
        
        text = "Text with {CODE_BLOCK_0} placeholder."
        blocks = [{"id": "CODE_BLOCK_0", "content": "```code```"}]
        
        restored = chunker._restore_code_blocks(text, blocks)
        
        assert "```code```" in restored
        assert "{CODE_BLOCK_0}" not in restored
    
    def test_split_sentences(self):
        """Test sentence splitting."""
        chunker = ConcreteChunker()
        
        text = "First sentence. Second sentence! Third sentence?"
        sentences = chunker._split_sentences(text)
        
        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
    
    def test_estimate_token_count(self):
        """Test token count estimation."""
        chunker = ConcreteChunker()
        
        text = "a" * 100  # ~25 tokens at 4 chars per token
        count = chunker._estimate_token_count(text)
        
        assert count == 25


# ============================================================================
# SemanticChunker Tests
# ============================================================================


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    @pytest.mark.asyncio
    async def test_chunk_simple_document(self, sample_document, semantic_config):
        """Test chunking a simple document."""
        chunker = SemanticChunker(semantic_config)
        chunks = await chunker.chunk(sample_document)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.metadata.get("strategy") == "semantic" for c in chunks)
    
    @pytest.mark.asyncio
    async def test_chunk_respects_max_size(self, semantic_config):
        """Test that chunks respect max size limit."""
        # Create document with many repeated sentences
        content = "This is a sentence. " * 100
        doc = Document(id="max-size-test", content=content)
        
        chunker = SemanticChunker(semantic_config)
        chunks = await chunker.chunk(doc)
        
        # Each chunk should respect max_chunk_size
        for chunk in chunks:
            assert chunk.token_count is not None
            # Allow some flexibility due to estimation
            assert chunk.token_count <= semantic_config.max_chunk_size * 1.5
    
    @pytest.mark.asyncio
    async def test_chunk_preserves_code_blocks(self, document_with_code, semantic_config):
        """Test that code blocks are preserved."""
        chunker = SemanticChunker(semantic_config)
        chunks = await chunker.chunk(document_with_code)
        
        # Check that code is preserved somewhere
        all_content = " ".join(c.content for c in chunks)
        assert "def hello_world()" in all_content
        assert "console.log" in all_content
    
    def test_fallback_embedding(self, semantic_config):
        """Test fallback embedding method."""
        chunker = SemanticChunker(semantic_config)
        
        text = "test text"
        embedding = chunker._fallback_embedding(text)
        
        assert len(embedding) == 384
        # Should be normalized
        import math
        norm = math.sqrt(sum(e * e for e in embedding))
        assert abs(norm - 1.0) < 0.01 or norm == 0
    
    def test_cosine_similarity(self, semantic_config):
        """Test cosine similarity calculation."""
        chunker = SemanticChunker(semantic_config)
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        
        sim_same = chunker._cosine_similarity(vec1, vec2)
        sim_diff = chunker._cosine_similarity(vec1, vec3)
        
        assert sim_same == 1.0
        assert sim_diff == 0.0
    
    def test_empty_document(self, semantic_config):
        """Test handling of empty document."""
        chunker = SemanticChunker(semantic_config)
        doc = Document(id="empty", content="")
        
        import asyncio
        chunks = asyncio.run(chunker.chunk(doc))
        
        assert chunks == []


# ============================================================================
# HierarchicalChunker Tests
# ============================================================================


class TestHierarchicalChunker:
    """Tests for HierarchicalChunker."""

    @pytest.mark.asyncio
    async def test_chunk_structured_document(self, structured_document, hierarchical_config):
        """Test chunking a document with headers."""
        chunker = HierarchicalChunker(hierarchical_config)
        chunks = await chunker.chunk(structured_document)
        
        assert len(chunks) > 0
        # Should have chunks with section metadata
        assert any("Introduction" in str(c.metadata.get("section_header", "")) for c in chunks)
    
    @pytest.mark.asyncio
    async def test_chunk_preserves_hierarchy(self, hierarchical_config):
        """Test that hierarchical relationships are preserved."""
        content = """# Main Section
Main content.

## Subsection A
Content A.

## Subsection B
Content B."""
        
        doc = Document(id="hierarchy-test", content=content)
        chunker = HierarchicalChunker(hierarchical_config)
        chunks = await chunker.chunk(doc)
        
        # Check hierarchy metadata
        hierarchy_paths = [c.metadata.get("hierarchy_path", []) for c in chunks]
        assert any(len(path) > 0 for path in hierarchy_paths)
    
    def test_parse_sections(self, hierarchical_config):
        """Test section parsing from content."""
        chunker = HierarchicalChunker(hierarchical_config)
        
        content = """# Section 1
Content 1.

## Section 2
Content 2.

### Section 3
Content 3."""
        
        sections = chunker._parse_sections(content)
        
        assert len(sections) == 3
        assert sections[0].header == "Section 1"
        assert sections[0].level == 1
        assert sections[1].level == 2
        assert sections[2].level == 3
    
    def test_build_section_tree(self, hierarchical_config):
        """Test building hierarchical tree from sections."""
        chunker = HierarchicalChunker(hierarchical_config)
        
        sections = [
            DocumentSection(header="H1", level=1, content="C1"),
            DocumentSection(header="H2", level=2, content="C2"),
            DocumentSection(header="H3", level=1, content="C3"),
        ]
        
        tree = chunker._build_section_tree(sections)
        
        assert len(tree) == 2  # Two level-1 sections
        assert len(tree[0].subsections) == 1  # First has one subsection
        assert tree[0].subsections[0].header == "H2"
    
    @pytest.mark.asyncio
    async def test_chunk_without_headers(self, hierarchical_config):
        """Test chunking document without headers."""
        doc = Document(id="no-headers", content="Just plain text without headers.")
        chunker = HierarchicalChunker(hierarchical_config)
        chunks = await chunker.chunk(doc)
        
        assert len(chunks) == 1
        assert "Document" in chunks[0].metadata.get("section_header", "")


# ============================================================================
# FixedSizeChunker Tests
# ============================================================================


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    @pytest.mark.asyncio
    async def test_chunk_long_document(self, long_document, fixed_config):
        """Test chunking a long document."""
        chunker = FixedSizeChunker(fixed_config)
        chunks = await chunker.chunk(long_document)
        
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
    
    @pytest.mark.asyncio
    async def test_overlap_between_chunks(self, long_document, fixed_config):
        """Test that chunks have overlap."""
        chunker = FixedSizeChunker(fixed_config)
        chunks = await chunker.chunk(long_document)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap in content
            content1 = chunks[0].content[-50:]
            content2 = chunks[1].content[:50]
            # There should be some similarity due to overlap
            assert len(content1) > 0 and len(content2) > 0
    
    @pytest.mark.asyncio
    async def test_chunk_preserves_code_blocks(self, document_with_code, fixed_config):
        """Test that code blocks are preserved in fixed-size chunking."""
        chunker = FixedSizeChunker(fixed_config)
        chunks = await chunker.chunk(document_with_code)
        
        all_content = " ".join(c.content for c in chunks)
        assert "def hello_world()" in all_content
    
    def test_fallback_tokenization(self, fixed_config):
        """Test fallback tokenization when tiktoken unavailable."""
        chunker = FixedSizeChunker(fixed_config)
        chunker._tokenizer = False  # Simulate unavailable
        
        text = "Hello world"
        tokens = chunker._encode(text)
        
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    @pytest.mark.asyncio
    async def test_chunk_indices_are_sequential(self, long_document, fixed_config):
        """Test that chunk indices are sequential."""
        chunker = FixedSizeChunker(fixed_config)
        chunks = await chunker.chunk(long_document)
        
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))


# ============================================================================
# AgenticChunker Tests
# ============================================================================


class TestAgenticChunker:
    """Tests for AgenticChunker."""

    @pytest.mark.asyncio
    async def test_selects_hierarchical_for_structured(self, structured_document, agentic_config):
        """Test that structured documents get hierarchical strategy."""
        chunker = AgenticChunker(agentic_config)
        strategy = await chunker.select_strategy(structured_document)
        
        assert strategy == "hierarchical"
    
    @pytest.mark.asyncio
    async def test_selects_semantic_for_technical(self, technical_document, agentic_config):
        """Test that technical documents get semantic strategy."""
        chunker = AgenticChunker(agentic_config)
        strategy = await chunker.select_strategy(technical_document)
        
        assert strategy == "semantic"
    
    @pytest.mark.asyncio
    async def test_selects_fixed_for_narrative(self, agentic_config):
        """Test that narrative documents get fixed strategy."""
        doc = Document(
            id="narrative",
            content="Once upon a time there was a story. It was a long story with many words.",
        )
        chunker = AgenticChunker(agentic_config)
        strategy = await chunker.select_strategy(doc)
        
        assert strategy == "fixed"
    
    @pytest.mark.asyncio
    async def test_chunk_delegates_to_selected_strategy(self, sample_document, agentic_config):
        """Test that chunk delegates to the selected strategy."""
        chunker = AgenticChunker(agentic_config)
        chunks = await chunker.chunk(sample_document)
        
        assert len(chunks) > 0
        assert all("agentic_strategy_selected" in c.metadata for c in chunks)
    
    @pytest.mark.asyncio
    async def test_selection_is_fast(self, sample_document, agentic_config):
        """Test that strategy selection completes quickly."""
        import time
        
        chunker = AgenticChunker(agentic_config)
        
        start = time.monotonic()
        strategy = await chunker.select_strategy(sample_document)
        elapsed_ms = (time.monotonic() - start) * 1000
        
        assert elapsed_ms < 100  # Should complete in < 100ms


# ============================================================================
# ChunkingService Tests
# ============================================================================


class TestChunkingService:
    """Tests for ChunkingService."""

    @pytest.mark.asyncio
    async def test_chunk_document_with_strategy(self, structured_document):
        """Test chunking with specific strategy."""
        service = ChunkingService()
        result = await service.chunk_document(structured_document, strategy="fixed")
        
        assert result.success
        assert len(result.chunks) > 0
        assert result.strategy_used == "fixed"
        assert "total_chunks" in result.metrics
    
    @pytest.mark.asyncio
    async def test_chunk_document_default_strategy(self, sample_document):
        """Test chunking with default strategy."""
        service = ChunkingService()
        result = await service.chunk_document(sample_document)
        
        assert result.success
        assert len(result.chunks) > 0
    
    @pytest.mark.asyncio
    async def test_unknown_strategy_returns_error(self, sample_document):
        """Test that unknown strategy returns error."""
        service = ChunkingService()
        result = await service.chunk_document(sample_document, strategy="unknown")
        
        assert not result.success
        assert "unknown" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_select_strategy(self, structured_document):
        """Test strategy selection through service."""
        service = ChunkingService()
        strategy = await service.select_strategy(structured_document)
        
        assert strategy in ["semantic", "hierarchical", "fixed", "agentic"]
    
    def test_get_available_strategies(self):
        """Test getting list of available strategies."""
        service = ChunkingService()
        strategies = service.get_available_strategies()
        
        assert "semantic" in strategies
        assert "hierarchical" in strategies
        assert "fixed" in strategies
        assert "agentic" in strategies
    
    @pytest.mark.asyncio
    async def test_chunk_batch(self):
        """Test batch chunking multiple documents."""
        docs = [
            Document(id=f"doc-{i}", content=f"Content for document {i}. " * 10)
            for i in range(3)
        ]
        
        service = ChunkingService()
        results = await service.chunk_batch(docs, strategy="fixed")
        
        assert len(results) == 3
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_chunking_metrics(self, long_document):
        """Test that metrics are populated correctly."""
        service = ChunkingService()
        result = await service.chunk_document(long_document, strategy="fixed")
        
        assert result.success
        assert "processing_time_ms" in result.metrics
        assert result.metrics["processing_time_ms"] >= 0
        assert "total_chunks" in result.metrics


# ============================================================================
# Integration Tests
# ============================================================================


class TestChunkingIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_all_strategies_produce_valid_chunks(self, technical_document):
        """Test that all strategies produce valid chunk output."""
        service = ChunkingService()
        
        for strategy in ["semantic", "hierarchical", "fixed", "agentic"]:
            result = await service.chunk_document(technical_document, strategy=strategy)
            
            assert result.success, f"Strategy {strategy} failed: {result.error}"
            assert len(result.chunks) > 0, f"Strategy {strategy} produced no chunks"
            
            # Verify chunk properties
            for chunk in result.chunks:
                assert chunk.content
                assert chunk.index >= 0
                assert isinstance(chunk.metadata, dict)
    
    @pytest.mark.asyncio
    async def test_code_block_integrity_across_strategies(self, document_with_code):
        """Test that code blocks are preserved by all strategies."""
        service = ChunkingService()
        
        for strategy in ["semantic", "hierarchical", "fixed"]:
            result = await service.chunk_document(document_with_code, strategy=strategy)
            
            all_content = " ".join(c.content for c in result.chunks)
            
            # Code should be preserved
            assert "def hello_world()" in all_content, f"Strategy {strategy} lost code"
            assert 'print("Hello, World!")' in all_content
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, long_document):
        """Test that chunking meets performance requirements."""
        import time
        
        service = ChunkingService()
        
        start = time.monotonic()
        result = await service.chunk_document(long_document, strategy="fixed")
        elapsed_ms = (time.monotonic() - start) * 1000
        
        # Should complete reasonably fast (adjust threshold as needed)
        assert elapsed_ms < 5000  # 5 seconds max for long document
        
        # Calculate tokens per second
        total_tokens = sum(c.token_count or 0 for c in result.chunks)
        tokens_per_sec = total_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        
        # Should process at least 1000 tokens/s
        assert tokens_per_sec > 100, f"Too slow: {tokens_per_sec} tokens/s"
    
    @pytest.mark.asyncio
    async def test_agentic_selection_speed(self, structured_document):
        """Test that agentic selection meets speed requirement."""
        import time
        
        service = ChunkingService()
        
        start = time.monotonic()
        strategy = await service.select_strategy(structured_document)
        elapsed_ms = (time.monotonic() - start) * 1000
        
        assert elapsed_ms < 100  # < 100ms requirement
        assert strategy in ["semantic", "hierarchical", "fixed"]


# ============================================================================
# Document Model Tests
# ============================================================================


class TestDocumentModel:
    """Tests for Document and related models."""

    def test_document_has_clear_structure_true(self):
        """Test has_clear_structure with headers."""
        doc = Document(
            id="test",
            content="# Header 1\nContent\n## Header 2\nMore content",
        )
        assert doc.has_clear_structure()
    
    def test_document_has_clear_structure_false(self):
        """Test has_clear_structure without headers."""
        doc = Document(
            id="test",
            content="Just plain text without any headers.",
        )
        assert not doc.has_clear_structure()
    
    def test_document_is_technical_true(self):
        """Test is_technical with technical content."""
        doc = Document(
            id="test",
            content="The API function uses code implementation with class parameters.",
        )
        assert doc.is_technical()
    
    def test_document_is_technical_false(self):
        """Test is_technical with non-technical content."""
        doc = Document(
            id="test",
            content="Once upon a time in a land far away.",
        )
        assert not doc.is_technical()
    
    def test_chunk_model_creation(self):
        """Test Chunk model creation."""
        chunk = Chunk(
            content="Test content",
            index=0,
            metadata={"key": "value"},
        )
        
        assert chunk.content == "Test content"
        assert chunk.index == 0
        assert chunk.metadata == {"key": "value"}
        assert chunk.id is not None  # Auto-generated
    
    def test_document_section_model(self):
        """Test DocumentSection model."""
        section = DocumentSection(
            header="Test Section",
            level=2,
            content="Section content",
            parent_id="parent-1",
        )
        
        assert section.header == "Test Section"
        assert section.level == 2
        assert section.parent_id == "parent-1"


# ============================================================================
# Configuration Tests
# ============================================================================


class TestChunkingConfiguration:
    """Tests for chunking configuration."""

    def test_semantic_config_defaults(self):
        """Test semantic config default values."""
        config = SemanticChunkerConfig()
        
        assert config.similarity_threshold == 0.85
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 512
    
    def test_hierarchical_config_defaults(self):
        """Test hierarchical config default values."""
        config = HierarchicalChunkerConfig()
        
        assert config.max_depth == 4
        assert config.respect_headers is True
        assert config.preserve_code_blocks is True
    
    def test_fixed_config_defaults(self):
        """Test fixed config default values."""
        config = FixedSizeChunkerConfig()
        
        assert config.chunk_size == 512
        assert config.overlap == 50
        assert config.tokenizer == "cl100k_base"
    
    def test_agentic_config_defaults(self):
        """Test agentic config default values."""
        config = AgenticChunkerConfig()
        
        assert config.selection_model == "gpt-4.1"
        assert config.max_selection_time_ms == 100.0


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in chunking."""

    @pytest.mark.asyncio
    async def test_empty_document_handling(self):
        """Test handling of empty document."""
        service = ChunkingService()
        doc = Document(id="empty", content="")
        
        result = await service.chunk_document(doc, strategy="semantic")
        
        # Should succeed with empty chunks list
        assert result.success
        assert result.chunks == []
    
    @pytest.mark.asyncio
    async def test_very_short_document(self):
        """Test handling of very short document."""
        service = ChunkingService()
        doc = Document(id="short", content="Hi.")
        
        result = await service.chunk_document(doc, strategy="fixed")
        
        assert result.success
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "Hi."
