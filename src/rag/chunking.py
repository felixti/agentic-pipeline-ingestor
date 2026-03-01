"""Advanced document chunking strategies with agentic selection.

This module provides multiple chunking strategies for document segmentation:
1. Semantic Chunking - chunks based on semantic similarity boundaries
2. Hierarchical Chunking - respects document structure and sections
3. Fixed-Size Chunking - token-aware fixed-size chunks with overlap
4. Agentic Chunking - intelligently selects the best strategy

All strategies preserve code blocks, tables, and special formatting.

Example:
    >>> from src.rag.chunking import ChunkingService
    >>> service = ChunkingService()
    >>> result = await service.chunk_document(document, strategy="agentic")
    >>> print(f"Created {len(result.chunks)} chunks using {result.strategy_used}")
"""

import asyncio
import hashlib
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from re import Match
from typing import Any

from src.config import settings
from src.observability.logging import get_logger
from src.rag.models import Chunk, ChunkingResult, ChunkingStrategy, Document, DocumentSection

logger = get_logger(__name__)

# =============================================================================
# Exceptions
# =============================================================================


class ChunkingError(Exception):
    """Base exception for chunking errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}


class StrategyNotFoundError(ChunkingError):
    """Raised when an unknown chunking strategy is requested."""

    pass


class TokenizationError(ChunkingError):
    """Raised when tokenization fails."""

    pass


class EmbeddingError(ChunkingError):
    """Raised when embedding computation fails."""

    pass


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class SemanticChunkerConfig:
    """Configuration for semantic chunking.

    Attributes:
        similarity_threshold: Minimum similarity to group sentences (0-1)
        min_chunk_size: Minimum chunk size in tokens
        max_chunk_size: Maximum chunk size in tokens
        embedding_model: Model name for sentence embeddings
    """

    similarity_threshold: float = 0.85
    min_chunk_size: int = 100
    max_chunk_size: int = 512
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class HierarchicalChunkerConfig:
    """Configuration for hierarchical chunking.

    Attributes:
        max_depth: Maximum depth for recursive section processing
        respect_headers: Whether to respect document headers
        preserve_code_blocks: Whether to keep code blocks intact
        max_chunk_size: Maximum chunk size in tokens
    """

    max_depth: int = 4
    respect_headers: bool = True
    preserve_code_blocks: bool = True
    max_chunk_size: int = 512


@dataclass
class FixedSizeChunkerConfig:
    """Configuration for fixed-size chunking.

    Attributes:
        chunk_size: Target chunk size in tokens
        overlap: Number of overlapping tokens between chunks
        tokenizer: Tokenizer encoding name (tiktoken)
    """

    chunk_size: int = 512
    overlap: int = 50
    tokenizer: str = "cl100k_base"


@dataclass
class AgenticChunkerConfig:
    """Configuration for agentic chunking.

    Attributes:
        selection_model: Model for strategy selection
        max_selection_time_ms: Maximum time for strategy selection
    """

    selection_model: str = "gpt-4.1"
    max_selection_time_ms: float = 100.0


@dataclass
class SpecialElementsConfig:
    """Configuration for special element handling.

    Attributes:
        preserve_code_blocks: Whether to preserve code block integrity
        code_block_max_size: Maximum tokens for code blocks
        preserve_table_rows: Whether to preserve table row integrity
        max_rows_per_chunk: Maximum table rows per chunk
    """

    preserve_code_blocks: bool = True
    code_block_max_size: int = 1024
    preserve_table_rows: bool = True
    max_rows_per_chunk: int = 10


# =============================================================================
# Base Chunker
# =============================================================================


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, config: Any | None = None):
        """Initialize the chunker.

        Args:
            config: Optional configuration object
        """
        self.config = config
        self.logger = logger

    @abstractmethod
    async def chunk(self, document: Document) -> list[Chunk]:
        """Chunk a document.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        pass

    def _extract_code_blocks(self, text: str) -> tuple[list[dict[str, Any]], str]:
        """Extract code blocks from text, returning blocks and cleaned text.

        Args:
            text: Input text potentially containing code blocks

        Returns:
            Tuple of (list of code block info, text with placeholders)
        """
        code_blocks: list[dict[str, Any]] = []
        placeholder_pattern = r"```[\w]*\n?[\s\S]*?```"

        def replace_code_block(match: Match[str]) -> str:
            block_id = f"CODE_BLOCK_{len(code_blocks)}"
            code_blocks.append(
                {
                    "id": block_id,
                    "content": match.group(0),
                }
            )
            return f"\n{{{block_id}}}\n"

        cleaned_text = re.sub(placeholder_pattern, replace_code_block, text)
        return code_blocks, cleaned_text

    def _restore_code_blocks(self, text: str, code_blocks: list[dict[str, Any]]) -> str:
        """Restore code blocks in text from placeholders.

        Args:
            text: Text with code block placeholders
            code_blocks: List of code block info

        Returns:
            Text with code blocks restored
        """
        result = text
        for block in code_blocks:
            placeholder = f"{{{block['id']}}}"
            result = result.replace(placeholder, block["content"])
        return result

    def _extract_tables(self, text: str) -> tuple[list[dict[str, Any]], str]:
        """Extract markdown tables from text.

        Args:
            text: Input text potentially containing tables

        Returns:
            Tuple of (list of table info, text with placeholders)
        """
        tables: list[dict[str, Any]] = []
        # Match markdown tables (rows starting with |)
        table_pattern = r"(\|[^\n]+\|\n?)+"

        def replace_table(match: Match[str]) -> str:
            table_id = f"TABLE_{len(tables)}"
            content = match.group(0)
            rows = [r for r in content.split("\n") if r.strip().startswith("|")]
            tables.append(
                {
                    "id": table_id,
                    "content": content,
                    "rows": rows,
                    "row_count": len(rows),
                }
            )
            return f"\n{{{table_id}}}\n"

        cleaned_text = re.sub(table_pattern, replace_table, text)
        return tables, cleaned_text

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting (period, exclamation, question mark)
        # More sophisticated than just split('.') to handle abbreviations
        sentence_endings = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count using simple heuristic.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token on average
        return len(text) // 4


# =============================================================================
# Semantic Chunker
# =============================================================================


class SemanticChunker(BaseChunker):
    """Chunk based on semantic boundaries using sentence embeddings.

    This chunker splits text into sentences and groups them by semantic
    similarity. When similarity drops below threshold, a new chunk is started.

    Example:
        >>> chunker = SemanticChunker()
        >>> chunks = await chunker.chunk(document)
    """

    config: SemanticChunkerConfig

    def __init__(self, config: SemanticChunkerConfig | None = None):
        """Initialize semantic chunker.

        Args:
            config: Optional configuration
        """
        super().__init__(config or SemanticChunkerConfig())
        self._embedding_model: Any = None
        self._embedding_cache: dict[str, list[float]] = {}

    def _get_embedding_model(self) -> Any:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(self.config.embedding_model)
                self.logger.info(
                    "embedding_model_loaded",
                    model=self.config.embedding_model,
                )
            except ImportError:
                self.logger.warning(
                    "sentence_transformers_not_available",
                    message="Using fallback embedding method",
                )
                self._embedding_model = False  # Mark as unavailable
        return self._embedding_model

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        model = self._get_embedding_model()
        if model and model is not False:
            try:
                embedding = model.encode(text, convert_to_numpy=True)
                result: list[float] = embedding.tolist()
                self._embedding_cache[text_hash] = result
                return result
            except Exception as e:
                self.logger.warning(
                    "embedding_computation_failed",
                    error=str(e),
                )

        # Fallback: simple word-based embedding
        return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> list[float]:
        """Simple fallback embedding using word frequency.

        Args:
            text: Input text

        Returns:
            Simple embedding vector
        """
        words = text.lower().split()
        # Create a simple hash-based embedding (384 dimensions like MiniLM)
        embedding = [0.0] * 384
        for i, word in enumerate(words):
            word_hash = hashlib.md5(word.encode()).hexdigest()
            for j in range(16):  # Use 16 values from hash
                idx = (i * 16 + j) % 384
                embedding[idx] += int(word_hash[j * 2 : j * 2 + 2], 16) / 255.0
        # Normalize
        norm = math.sqrt(sum(e * e for e in embedding))
        if norm > 0:
            embedding = [e / norm for e in embedding]
        return embedding

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def _semantic_similarity(
        self,
        current_sentences: list[str],
        new_sentence: str,
    ) -> float:
        """Compute semantic similarity between current chunk and new sentence.

        Args:
            current_sentences: Sentences in current chunk
            new_sentence: New sentence to compare

        Returns:
            Similarity score (0-1)
        """
        current_text = " ".join(current_sentences)
        current_emb = self._compute_embedding(current_text)
        new_emb = self._compute_embedding(new_sentence)
        return self._cosine_similarity(current_emb, new_emb)

    async def chunk(self, document: Document) -> list[Chunk]:
        """Chunk document based on semantic boundaries.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        start_time = time.monotonic()

        # Extract and preserve code blocks
        code_blocks, cleaned_content = self._extract_code_blocks(document.content)

        # Split into sentences
        sentences = self._split_sentences(cleaned_content)

        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_chunk_sentences: list[str] = [sentences[0]]
        current_token_count = self._estimate_token_count(sentences[0])

        for sentence in sentences[1:]:
            sentence_tokens = self._estimate_token_count(sentence)

            # Check if adding this sentence would exceed max size
            if current_token_count + sentence_tokens > self.config.max_chunk_size:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk_sentences)
                chunk_text = self._restore_code_blocks(chunk_text, code_blocks)

                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=len(chunks),
                        token_count=self._estimate_token_count(chunk_text),
                        metadata={
                            "strategy": "semantic",
                            "sentence_count": len(current_chunk_sentences),
                        },
                    )
                )

                current_chunk_sentences = [sentence]
                current_token_count = sentence_tokens
                continue

            # Check semantic similarity
            similarity = self._semantic_similarity(current_chunk_sentences, sentence)

            # If similarity is below threshold and we have min chunk size, create new chunk
            if (
                similarity < self.config.similarity_threshold
                and current_token_count >= self.config.min_chunk_size
            ):
                chunk_text = " ".join(current_chunk_sentences)
                chunk_text = self._restore_code_blocks(chunk_text, code_blocks)

                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=len(chunks),
                        token_count=self._estimate_token_count(chunk_text),
                        metadata={
                            "strategy": "semantic",
                            "sentence_count": len(current_chunk_sentences),
                            "similarity_threshold": self.config.similarity_threshold,
                        },
                    )
                )

                current_chunk_sentences = [sentence]
                current_token_count = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_tokens

        # Add remaining sentences as final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_text = self._restore_code_blocks(chunk_text, code_blocks)

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=len(chunks),
                    token_count=self._estimate_token_count(chunk_text),
                    metadata={
                        "strategy": "semantic",
                        "sentence_count": len(current_chunk_sentences),
                    },
                )
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        self.logger.info(
            "semantic_chunking_completed",
            document_id=document.id,
            chunks_created=len(chunks),
            elapsed_ms=round(elapsed_ms, 2),
        )

        return chunks


# =============================================================================
# Hierarchical Chunker
# =============================================================================


class HierarchicalChunker(BaseChunker):
    """Chunk based on document structure and hierarchy.

    This chunker respects document sections and creates chunks that
    align with the document's hierarchical structure.

    Example:
        >>> chunker = HierarchicalChunker()
        >>> chunks = await chunker.chunk(document)
    """

    config: HierarchicalChunkerConfig

    def __init__(self, config: HierarchicalChunkerConfig | None = None):
        """Initialize hierarchical chunker.

        Args:
            config: Optional configuration
        """
        super().__init__(config or HierarchicalChunkerConfig())

    def _parse_sections(self, content: str) -> list[DocumentSection]:
        """Parse document content into sections based on headers.

        Args:
            content: Document content

        Returns:
            List of document sections
        """
        sections: list[DocumentSection] = []

        # Match markdown-style headers (# ## ###)
        header_pattern = r"^(#{1,6})\s+(.+)$"

        lines = content.split("\n")
        current_section: DocumentSection | None = None
        current_content: list[str] = []

        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                # Save previous section
                if current_section is not None:
                    current_section.content = "\n".join(current_content).strip()
                    sections.append(current_section)

                # Start new section
                hashes = match.group(1)
                title = match.group(2)
                level = len(hashes)

                current_section = DocumentSection(
                    header=title,
                    level=level,
                    content="",
                    metadata={},
                )
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_section is not None:
            current_section.content = "\n".join(current_content).strip()
            sections.append(current_section)

        # If no headers found, treat entire content as single section
        if not sections and content.strip():
            sections.append(
                DocumentSection(
                    header="Document",
                    level=1,
                    content=content.strip(),
                    metadata={},
                )
            )

        return sections

    def _build_section_tree(
        self,
        sections: list[DocumentSection],
    ) -> list[DocumentSection]:
        """Build a tree structure from flat sections.

        Args:
            sections: Flat list of sections

        Returns:
            Hierarchical tree of sections
        """
        if not sections:
            return []

        root: list[DocumentSection] = []
        stack: list[DocumentSection] = []

        for section in sections:
            # Pop sections that are at same or higher level
            while stack and stack[-1].level >= section.level:
                stack.pop()

            if stack:
                # Add as subsection of parent
                parent = stack[-1]
                section.parent_id = parent.header  # Use header as ID
                parent.subsections.append(section)
            else:
                # Add to root
                root.append(section)

            stack.append(section)

        return root

    def _chunk_section(
        self,
        section: DocumentSection,
        parent_path: list[str],
        depth: int = 0,
    ) -> list[Chunk]:
        """Recursively chunk a section and its subsections.

        Args:
            section: Section to chunk
            parent_path: Path of parent section headers
            depth: Current recursion depth

        Returns:
            List of chunks from this section
        """
        if depth >= self.config.max_depth:
            return []

        chunks: list[Chunk] = []
        current_path = parent_path + [section.header]

        # Handle code blocks
        code_blocks: list[dict[str, Any]] = []
        content = section.content
        if self.config.preserve_code_blocks:
            code_blocks, content = self._extract_code_blocks(content)

        # Split content if too large
        max_size = self.config.max_chunk_size
        tokens = self._estimate_token_count(content)

        if tokens <= max_size:
            # Section fits in one chunk
            restored_content = self._restore_code_blocks(content, code_blocks)
            chunks.append(
                Chunk(
                    content=restored_content,
                    index=len(chunks),
                    parent_section_id=section.parent_id,
                    hierarchy_level=section.level,
                    token_count=self._estimate_token_count(restored_content),
                    metadata={
                        "strategy": "hierarchical",
                        "section_header": section.header,
                        "section_level": section.level,
                        "hierarchy_path": current_path,
                    },
                )
            )
        else:
            # Split large section
            sentences = self._split_sentences(content)
            current_chunk: list[str] = []
            current_tokens = 0
            chunk_index = 0

            for sentence in sentences:
                sentence_tokens = self._estimate_token_count(sentence)

                if current_tokens + sentence_tokens > max_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunk_text = self._restore_code_blocks(chunk_text, code_blocks)

                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            index=chunk_index,
                            parent_section_id=section.parent_id,
                            hierarchy_level=section.level,
                            token_count=self._estimate_token_count(chunk_text),
                            metadata={
                                "strategy": "hierarchical",
                                "section_header": section.header,
                                "section_level": section.level,
                                "hierarchy_path": current_path,
                                "part": chunk_index + 1,
                            },
                        )
                    )
                    chunk_index += 1
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens

            # Add remaining content
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_text = self._restore_code_blocks(chunk_text, code_blocks)

                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=chunk_index,
                        parent_section_id=section.parent_id,
                        hierarchy_level=section.level,
                        token_count=self._estimate_token_count(chunk_text),
                        metadata={
                            "strategy": "hierarchical",
                            "section_header": section.header,
                            "section_level": section.level,
                            "hierarchy_path": current_path,
                            "part": chunk_index + 1,
                        },
                    )
                )

        # Process subsections
        for subsection in section.subsections:
            sub_chunks = self._chunk_section(subsection, current_path, depth + 1)
            # Update indices
            for chunk in sub_chunks:
                chunk.index = len(chunks)
            chunks.extend(sub_chunks)

        return chunks

    async def chunk(self, document: Document) -> list[Chunk]:
        """Chunk document based on hierarchical structure.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        start_time = time.monotonic()

        # Use provided sections or parse from content
        if document.sections:
            section_tree = document.sections
        else:
            flat_sections = self._parse_sections(document.content)
            section_tree = self._build_section_tree(flat_sections)

        # Generate chunks
        all_chunks: list[Chunk] = []
        for section in section_tree:
            section_chunks = self._chunk_section(section, [])
            all_chunks.extend(section_chunks)

        # Re-index chunks
        for i, chunk in enumerate(all_chunks):
            chunk.index = i

        elapsed_ms = (time.monotonic() - start_time) * 1000
        self.logger.info(
            "hierarchical_chunking_completed",
            document_id=document.id,
            chunks_created=len(all_chunks),
            elapsed_ms=round(elapsed_ms, 2),
        )

        return all_chunks


# =============================================================================
# Fixed-Size Chunker
# =============================================================================


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with configurable overlap.

    This chunker creates fixed-size chunks using token-aware splitting.

    Example:
        >>> chunker = FixedSizeChunker(chunk_size=512, overlap=50)
        >>> chunks = await chunker.chunk(document)
    """

    config: FixedSizeChunkerConfig

    def __init__(self, config: FixedSizeChunkerConfig | None = None):
        """Initialize fixed-size chunker.

        Args:
            config: Optional configuration
        """
        super().__init__(config or FixedSizeChunkerConfig())
        self._tokenizer: Any = None

    def _get_tokenizer(self) -> Any:
        """Lazy load tiktoken tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken

                self._tokenizer = tiktoken.get_encoding(self.config.tokenizer)
                self.logger.info(
                    "tokenizer_loaded",
                    tokenizer=self.config.tokenizer,
                )
            except ImportError:
                self.logger.warning(
                    "tiktoken_not_available",
                    message="Using character-based token estimation",
                )
                self._tokenizer = False
        return self._tokenizer

    def _encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        tokenizer = self._get_tokenizer()
        if tokenizer and tokenizer is not False:
            try:
                encoded: list[int] = tokenizer.encode(text)
                return encoded
            except Exception as e:
                self.logger.warning(
                    "tokenization_failed",
                    error=str(e),
                )

        # Fallback: character-based estimation
        # Approximate 4 chars per token
        tokens = []
        for i in range(0, len(text), 4):
            tokens.append(hash(text[i : i + 4]) % 100000)
        return tokens

    def _decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        tokenizer = self._get_tokenizer()
        if tokenizer and tokenizer is not False:
            try:
                decoded: str = tokenizer.decode(tokens)
                return decoded
            except Exception as e:
                self.logger.warning(
                    "decoding_failed",
                    error=str(e),
                )

        # Fallback: can't truly decode, so return placeholder
        return f"[Decoded {len(tokens)} tokens]"

    def _find_chunk_boundary(
        self,
        text: str,
        target_pos: int,
    ) -> int:
        """Find a good boundary for chunking near target position.

        Looks for sentence boundaries, then word boundaries.

        Args:
            text: Full text
            target_pos: Target character position

        Returns:
            Character position for clean boundary
        """
        # Search window around target position
        window = 100
        start = max(0, target_pos - window)
        end = min(len(text), target_pos + window)
        search_area = text[start:end]

        # Look for sentence boundary (.!? followed by space and capital)
        for i in range(len(search_area) - 2):
            pos = start + i
            if pos >= target_pos - 20:  # Don't go too far back
                break
            if (
                search_area[i] in ".!?"
                and search_area[i + 1] == " "
                and search_area[i + 2].isupper()
            ):
                return pos + 2  # Include the punctuation and space

        # Look for word boundary (space)
        for i in range(50):
            if target_pos - i > 0 and text[target_pos - i] == " ":
                return target_pos - i
            if target_pos + i < len(text) and text[target_pos + i] == " ":
                return target_pos + i

        # Fallback to exact position
        return target_pos

    async def chunk(self, document: Document) -> list[Chunk]:
        """Chunk document using fixed-size strategy.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        start_time = time.monotonic()

        # Extract code blocks if configured
        code_blocks: list[dict[str, Any]] = []
        content = document.content
        if True:  # Always preserve code blocks
            code_blocks, content = self._extract_code_blocks(content)

        # Tokenize content
        tokens = self._encode(content)

        chunks: list[Chunk] = []
        start_idx = 0
        chunk_index = 0

        while start_idx < len(tokens):
            # Calculate end index
            end_idx = min(start_idx + self.config.chunk_size, len(tokens))

            # Get text for this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self._decode(chunk_tokens)

            # Restore code blocks
            chunk_text = self._restore_code_blocks(chunk_text, code_blocks)

            # Calculate metadata
            char_start = self._token_to_char(start_idx, content)
            char_end = self._token_to_char(end_idx, content)

            chunks.append(
                Chunk(
                    content=chunk_text,
                    index=chunk_index,
                    token_count=len(chunk_tokens),
                    metadata={
                        "strategy": "fixed",
                        "char_start": char_start,
                        "char_end": char_end,
                    },
                )
            )

            if end_idx == len(tokens):
                break

            next_start = end_idx - self.config.overlap
            if next_start <= start_idx:
                next_start = end_idx
            start_idx = next_start

            chunk_index += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000
        self.logger.info(
            "fixed_size_chunking_completed",
            document_id=document.id,
            chunks_created=len(chunks),
            elapsed_ms=round(elapsed_ms, 2),
        )

        return chunks

    def _token_to_char(self, token_idx: int, text: str) -> int:
        """Approximate token index to character position.

        Args:
            token_idx: Token index
            text: Original text

        Returns:
            Character position
        """
        # Rough approximation: 4 chars per token
        return min(token_idx * 4, len(text))


# =============================================================================
# Agentic Chunker
# =============================================================================


class AgenticChunker(BaseChunker):
    """Agentically select and apply the best chunking strategy.

    Analyzes document structure and content to determine the optimal
    chunking strategy, then delegates to the appropriate chunker.

    Example:
        >>> chunker = AgenticChunker()
        >>> chunks = await chunker.chunk(document)  # Automatically selects strategy
    """

    config: AgenticChunkerConfig

    def __init__(self, config: AgenticChunkerConfig | None = None):
        """Initialize agentic chunker.

        Args:
            config: Optional configuration
        """
        super().__init__(config or AgenticChunkerConfig())

        # Initialize sub-chunkers
        self._semantic_chunker = SemanticChunker()
        self._hierarchical_chunker = HierarchicalChunker()
        self._fixed_chunker = FixedSizeChunker()

    async def select_strategy(self, document: Document) -> str:
        """Agentically select the best chunking strategy.

        Analyzes document structure and content to make a fast decision.

        Args:
            document: Document to analyze

        Returns:
            Strategy name ("semantic", "hierarchical", or "fixed")
        """
        start_time = time.monotonic()

        # Check for clear document structure (headers)
        if document.has_clear_structure():
            strategy = "hierarchical"
            reason = "document_has_structure"
        # Check for technical content
        elif document.is_technical():
            strategy = "semantic"
            reason = "technical_content_detected"
        else:
            strategy = "fixed"
            reason = "default_for_mixed_content"

        elapsed_ms = (time.monotonic() - start_time) * 1000

        self.logger.info(
            "strategy_selected",
            document_id=document.id,
            strategy=strategy,
            reason=reason,
            selection_time_ms=round(elapsed_ms, 2),
        )

        return strategy

    async def chunk(self, document: Document) -> list[Chunk]:
        """Chunk document using agentically selected strategy.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        strategy = await self.select_strategy(document)

        # Delegate to appropriate chunker
        if strategy == "semantic":
            chunks = await self._semantic_chunker.chunk(document)
        elif strategy == "hierarchical":
            chunks = await self._hierarchical_chunker.chunk(document)
        else:
            chunks = await self._fixed_chunker.chunk(document)

        # Tag chunks with selected strategy
        for chunk in chunks:
            chunk.metadata["agentic_strategy_selected"] = strategy

        return chunks


# =============================================================================
# Chunking Service
# =============================================================================


class ChunkingService:
    """Service for orchestrating document chunking operations.

    Provides a unified interface for all chunking strategies with
    configuration management and metrics collection.

    Example:
        >>> service = ChunkingService()
        >>>
        >>> # Use specific strategy
        >>> result = await service.chunk_document(doc, strategy="semantic")
        >>>
        >>> # Use agentic selection (default)
        >>> result = await service.chunk_document(doc)  # Automatically selects
    """

    def __init__(self, config: Any | None = None):
        """Initialize chunking service.

        Args:
            config: Optional configuration override
        """
        self.config = config or settings.chunking
        self.logger = logger

        # Initialize strategies
        self._strategies: dict[str, BaseChunker] = {}
        self._init_strategies()

    def _init_strategies(self) -> None:
        """Initialize all chunking strategies with configuration."""
        # Semantic chunker
        semantic_config = SemanticChunkerConfig(
            similarity_threshold=self.config.semantic.get("similarity_threshold", 0.85),
            min_chunk_size=self.config.semantic.get("min_chunk_size", 100),
            max_chunk_size=self.config.semantic.get("max_chunk_size", 512),
            embedding_model=self.config.semantic.get(
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
        )
        self._strategies["semantic"] = SemanticChunker(semantic_config)

        # Hierarchical chunker
        hierarchical_config = HierarchicalChunkerConfig(
            max_depth=self.config.hierarchical.get("max_depth", 4),
            respect_headers=self.config.hierarchical.get("respect_headers", True),
            preserve_code_blocks=self.config.hierarchical.get("preserve_code_blocks", True),
        )
        self._strategies["hierarchical"] = HierarchicalChunker(hierarchical_config)

        # Fixed chunker
        fixed_config = FixedSizeChunkerConfig(
            chunk_size=self.config.fixed.get("chunk_size", 512),
            overlap=self.config.fixed.get("overlap", 50),
            tokenizer=self.config.fixed.get("tokenizer", "cl100k_base"),
        )
        self._strategies["fixed"] = FixedSizeChunker(fixed_config)

        # Agentic chunker
        agentic_config = AgenticChunkerConfig(
            selection_model=self.config.agentic.get("selection_model", "gpt-4.1"),
        )
        self._strategies["agentic"] = AgenticChunker(agentic_config)

    async def chunk_document(
        self,
        document: Document,
        strategy: str | None = None,
    ) -> ChunkingResult:
        """Chunk document using specified strategy.

        Args:
            document: Document to chunk
            strategy: Chunking strategy name (semantic, hierarchical, fixed, agentic)
                     Defaults to config.default_strategy

        Returns:
            ChunkingResult with chunks and metadata
        """
        start_time = time.monotonic()

        # Use specified or default strategy
        strategy = strategy or self.config.default_strategy

        # Validate strategy
        if strategy not in self._strategies:
            self.logger.error(
                "unknown_strategy",
                strategy=strategy,
                available=list(self._strategies.keys()),
            )
            return ChunkingResult(
                success=False,
                error=f"Unknown strategy: {strategy}. Available: {list(self._strategies.keys())}",
            )

        try:
            # Get chunker and process document
            chunker = self._strategies[strategy]
            chunks = await chunker.chunk(document)

            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Calculate metrics
            total_tokens = sum(c.token_count or 0 for c in chunks)

            self.logger.info(
                "chunking_completed",
                document_id=document.id,
                strategy=strategy,
                chunks_created=len(chunks),
                total_tokens=total_tokens,
                elapsed_ms=round(elapsed_ms, 2),
            )

            return ChunkingResult(
                success=True,
                chunks=chunks,
                strategy_used=strategy,
                metrics={
                    "total_chunks": len(chunks),
                    "total_tokens": total_tokens,
                    "processing_time_ms": round(elapsed_ms, 2),
                    "tokens_per_second": round(total_tokens / (elapsed_ms / 1000), 2)
                    if elapsed_ms > 0
                    else 0,
                },
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000

            self.logger.error(
                "chunking_failed",
                document_id=document.id,
                strategy=strategy,
                error=str(e),
                elapsed_ms=round(elapsed_ms, 2),
            )

            return ChunkingResult(
                success=False,
                error=str(e),
                strategy_used=strategy,
                metrics={
                    "processing_time_ms": round(elapsed_ms, 2),
                },
            )

    async def select_strategy(self, document: Document) -> str:
        """Agentically select best chunking strategy.

        Args:
            document: Document to analyze

        Returns:
            Selected strategy name
        """
        agentic = self._strategies.get("agentic")
        if isinstance(agentic, AgenticChunker):
            return await agentic.select_strategy(document)

        # Fallback to default
        return self.config.default_strategy

    def get_available_strategies(self) -> list[str]:
        """Get list of available strategy names.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    async def chunk_batch(
        self,
        documents: list[Document],
        strategy: str | None = None,
    ) -> list[ChunkingResult]:
        """Chunk multiple documents in batch.

        Args:
            documents: List of documents to chunk
            strategy: Chunking strategy (defaults to config.default_strategy)

        Returns:
            List of chunking results
        """
        tasks = [self.chunk_document(doc, strategy) for doc in documents]
        return await asyncio.gather(*tasks)
