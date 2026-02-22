"""Contextual Retrieval service for enhancing chunks with surrounding context.

This module provides contextual retrieval capabilities that enhance document chunks
with surrounding context before embedding. This improves semantic understanding
of chunk boundaries and document structure, leading to better retrieval quality.

Three context strategies are supported:
1. Parent Document Enhancement - Adds document title and metadata
2. Window Context - Includes neighboring chunks (previous/next)
3. Hierarchical Context - Uses document hierarchy (sections/subsections)

Example:
    >>> from src.rag.contextual import ContextualRetrieval
    >>> retrieval = ContextualRetrieval()
    >>> enhanced = await retrieval.enhance_chunk(
    ...     chunk=chunk,
    ...     context_type=ContextType.PARENT_DOCUMENT
    ... )
    >>> print(enhanced.enhanced_text)

    >>> # Batch processing
    >>> enhanced_chunks = await retrieval.enhance_chunks_batch(
    ...     chunks=chunks,
    ...     context_type=ContextType.HIERARCHICAL
    ... )
"""

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db.models import DocumentChunkModel
from src.observability.logging import get_logger
from src.rag.models import (
    ContextType,
    ContextualContext,
    ContextualRetrievalResult,
    EnhancedChunk,
)
from src.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class ContextualRetrievalError(Exception):
    """Base exception for contextual retrieval errors."""
    
    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}


class ChunkNotFoundError(ContextualRetrievalError):
    """Raised when a chunk cannot be found."""
    pass


class DocumentNotFoundError(ContextualRetrievalError):
    """Raised when a parent document cannot be found."""
    pass


class InvalidContextTypeError(ContextualRetrievalError):
    """Raised when an invalid context type is specified."""
    pass


@dataclass
class ParentDocumentStrategyConfig:
    """Configuration for parent document strategy."""
    
    include_metadata: bool = True
    metadata_fields: list[str] = field(default_factory=lambda: ["title", "author", "category"])
    max_context_length: int = 256
    format_template: str = "Document: {title}\n{sections}Content: {content}"


@dataclass
class WindowStrategyConfig:
    """Configuration for window context strategy."""
    
    window_size: int = 1
    separator: str = " | "
    max_chunk_length: int = 200
    format_template: str = "[Prev: {prev}] {content} [Next: {next}]"


@dataclass
class HierarchicalStrategyConfig:
    """Configuration for hierarchical context strategy."""
    
    max_depth: int = 3
    include_path: bool = True
    format_template: str = "{path}\nContent: {content}"


@dataclass
class ContextualRetrievalConfig:
    """Configuration for contextual retrieval."""
    
    enabled: bool = True
    default_strategy: ContextType = ContextType.PARENT_DOCUMENT
    parent_document: ParentDocumentStrategyConfig = field(default_factory=ParentDocumentStrategyConfig)
    window: WindowStrategyConfig = field(default_factory=WindowStrategyConfig)
    hierarchical: HierarchicalStrategyConfig = field(default_factory=HierarchicalStrategyConfig)
    enable_embedding: bool = False  # Whether to generate embeddings for enhanced text
    
    @classmethod
    def from_settings(cls) -> "ContextualRetrievalConfig":
        """Create config from application settings."""
        # Check if contextual_retrieval settings exist
        if hasattr(settings, "contextual_retrieval"):
            cr_settings = settings.contextual_retrieval
            
            # Parse strategy configs if available
            parent_config = ParentDocumentStrategyConfig()
            window_config = WindowStrategyConfig()
            hierarchical_config = HierarchicalStrategyConfig()
            
            if hasattr(cr_settings, "strategies"):
                strategies = cr_settings.strategies
                
                if "parent_document" in strategies:
                    pd = strategies["parent_document"]
                    parent_config.include_metadata = pd.get("include_metadata", True)
                    parent_config.metadata_fields = pd.get("metadata_fields", ["title", "author", "category"])
                    parent_config.max_context_length = pd.get("max_context_length", 256)
                
                if "window" in strategies:
                    wc = strategies["window"]
                    window_config.window_size = wc.get("window_size", 1)
                    window_config.separator = wc.get("separator", " | ")
                
                if "hierarchical" in strategies:
                    hc = strategies["hierarchical"]
                    hierarchical_config.max_depth = hc.get("max_depth", 3)
                    hierarchical_config.include_path = hc.get("include_path", True)
            
            return cls(
                enabled=getattr(cr_settings, "enabled", True),
                default_strategy=ContextType(getattr(cr_settings, "default_strategy", "parent_document")),
                parent_document=parent_config,
                window=window_config,
                hierarchical=hierarchical_config,
            )
        
        return cls()


class ContextualRetrieval:
    """Service for enhancing chunks with contextual information.
    
    This class provides methods to enhance document chunks with surrounding
    context before embedding, improving semantic understanding and retrieval
    quality. Supports three strategies: parent document, window context,
    and hierarchical context.
    
    Example:
        >>> retrieval = ContextualRetrieval(db_session)
        >>> enhanced = await retrieval.enhance_chunk(
        ...     chunk=chunk,
        ...     context_type=ContextType.PARENT_DOCUMENT
        ... )
    """
    
    def __init__(
        self,
        db_session: AsyncSession | None = None,
        config: ContextualRetrievalConfig | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        """Initialize the contextual retrieval service.
        
        Args:
            db_session: Database session for fetching context
            config: Optional configuration override
            embedding_service: Optional embedding service for generating embeddings
        """
        self.db_session = db_session
        self.config = config or ContextualRetrievalConfig.from_settings()
        self.embedding_service = embedding_service
        self.logger = logger
    
    async def enhance_chunk(
        self,
        chunk: DocumentChunkModel,
        context_type: ContextType | None = None,
    ) -> EnhancedChunk:
        """Enhance a chunk with contextual information.
        
        Args:
            chunk: The document chunk to enhance
            context_type: Type of context to add (defaults to config default)
            
        Returns:
            EnhancedChunk with contextual information
            
        Raises:
            InvalidContextTypeError: If context_type is invalid
            ContextualRetrievalError: If enhancement fails
        """
        start_time = time.monotonic()
        context_type = context_type or self.config.default_strategy
        
        try:
            # Get context based on strategy
            context = await self._get_context(chunk, context_type)
            
            # Format enhanced text based on strategy
            enhanced_text = self._format_enhanced_text(
                chunk.content,  # type: ignore[arg-type]
                context,
                context_type,
            )
            
            # Generate embedding if enabled and service available
            embedding: list[float] | None = None
            if self.config.enable_embedding and self.embedding_service:
                try:
                    result = await self.embedding_service.embed_text(enhanced_text)
                    embedding = result.embedding
                except Exception as e:
                    self.logger.warning(
                        "embedding_generation_failed",
                        chunk_id=str(chunk.id),
                        error=str(e),
                    )
            
            latency_ms = (time.monotonic() - start_time) * 1000
            
            self.logger.info(
                "chunk_enhanced",
                chunk_id=str(chunk.id),
                context_type=context_type.value,
                original_length=len(chunk.content),
                enhanced_length=len(enhanced_text),
                latency_ms=round(latency_ms, 2),
            )
            
            return EnhancedChunk(
                chunk_id=str(chunk.id),
                original_content=chunk.content,  # type: ignore[arg-type]
                enhanced_text=enhanced_text,
                context=context,
                embedding=embedding,
                context_type=context_type,
                enhanced_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "chunk_enhancement_failed",
                chunk_id=str(chunk.id),
                context_type=context_type.value,
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )
            raise ContextualRetrievalError(
                f"Failed to enhance chunk {chunk.id}: {e}",
                context={"chunk_id": str(chunk.id), "context_type": context_type.value},
            ) from e
    
    async def enhance_chunks_batch(
        self,
        chunks: list[DocumentChunkModel],
        context_type: ContextType | None = None,
    ) -> list[EnhancedChunk]:
        """Enhance multiple chunks in batch.
        
        Args:
            chunks: List of document chunks to enhance
            context_type: Type of context to add
            
        Returns:
            List of EnhancedChunk objects
        """
        results: list[EnhancedChunk] = []
        
        for chunk in chunks:
            try:
                enhanced = await self.enhance_chunk(chunk, context_type)
                results.append(enhanced)
            except ContextualRetrievalError as e:
                self.logger.warning(
                    "batch_enhancement_chunk_failed",
                    chunk_id=str(chunk.id),
                    error=str(e),
                )
                # Continue with other chunks
        
        self.logger.info(
            "batch_enhancement_completed",
            total_chunks=len(chunks),
            successful=len(results),
            failed=len(chunks) - len(results),
        )
        
        return results
    
    async def _get_context(
        self,
        chunk: DocumentChunkModel,
        context_type: ContextType,
    ) -> ContextualContext:
        """Get contextual information for a chunk based on strategy.
        
        Args:
            chunk: The document chunk
            context_type: Type of context to retrieve
            
        Returns:
            ContextualContext with relevant information
        """
        if context_type == ContextType.PARENT_DOCUMENT:
            return await self._get_parent_document_context(chunk)
        elif context_type == ContextType.WINDOW:
            return await self._get_window_context(chunk)
        elif context_type == ContextType.HIERARCHICAL:
            return await self._get_hierarchical_context(chunk)
        else:
            raise InvalidContextTypeError(f"Unknown context type: {context_type}")
    
    async def _get_parent_document_context(
        self,
        chunk: DocumentChunkModel,
    ) -> ContextualContext:
        """Get parent document context for a chunk.
        
        Args:
            chunk: The document chunk
            
        Returns:
            ContextualContext with parent document information
        """
        context = ContextualContext()
        
        if not self.db_session:
            self.logger.debug("no_db_session_for_parent_context")
            return context
        
        try:
            # Query to get job/document information
            query = text("""
                SELECT 
                    j.id as job_id,
                    j.file_name,
                    j.metadata_json as job_metadata,
                    jr.output_data,
                    jr.result_metadata
                FROM jobs j
                LEFT JOIN job_results jr ON jr.job_id = j.id
                WHERE j.id = :job_id
            """)
            
            result = await self.db_session.execute(
                query, {"job_id": str(chunk.job_id)}
            )
            row = result.fetchone()
            
            if row:
                context.document_id = str(row.job_id)
                
                # Use file_name as title
                if row.file_name:
                    context.document_title = row.file_name
                
                # Extract metadata from job and result
                metadata: dict[str, Any] = {}
                
                if row.job_metadata:
                    metadata.update(row.job_metadata)
                
                if row.result_metadata:
                    metadata.update(row.result_metadata)
                
                # Filter to requested fields if configured
                if self.config.parent_document.metadata_fields:
                    filtered_metadata = {
                        k: v for k, v in metadata.items()
                        if k in self.config.parent_document.metadata_fields
                    }
                    context.document_metadata = filtered_metadata
                else:
                    context.document_metadata = metadata
                
                # Extract section headers from chunk metadata if available
                chunk_metadata: dict[str, Any] = chunk.chunk_metadata  # type: ignore[assignment]
                if chunk_metadata:
                    if "section_headers" in chunk_metadata:
                        headers = chunk_metadata["section_headers"]
                        if isinstance(headers, list):
                            context.section_headers = headers
                        elif isinstance(headers, str):
                            context.section_headers = [headers]
                    
                    if "page" in chunk_metadata:
                        context.document_metadata["page"] = chunk_metadata["page"]
            
            return context
            
        except Exception as e:
            self.logger.warning(
                "parent_context_fetch_failed",
                chunk_id=str(chunk.id),
                error=str(e),
            )
            return context
    
    async def _get_window_context(
        self,
        chunk: DocumentChunkModel,
    ) -> ContextualContext:
        """Get window context (neighboring chunks) for a chunk.
        
        Args:
            chunk: The document chunk
            
        Returns:
            ContextualContext with neighboring chunk information
        """
        context = ContextualContext()
        window_size = self.config.window.window_size
        
        if not self.db_session:
            self.logger.debug("no_db_session_for_window_context")
            return context
        
        try:
            # Get previous chunks
            if window_size > 0:
                prev_query = text("""
                    SELECT content
                    FROM document_chunks
                    WHERE job_id = :job_id
                    AND chunk_index < :chunk_index
                    ORDER BY chunk_index DESC
                    LIMIT :limit
                """)
                
                prev_result = await self.db_session.execute(
                    prev_query,
                    {
                        "job_id": str(chunk.job_id),
                        "chunk_index": chunk.chunk_index,
                        "limit": window_size,
                    },
                )
                prev_rows = prev_result.fetchall()
                
                if prev_rows:
                    # Combine previous chunks in reverse order (closest first)
                    prev_content = " ".join([
                        row.content[:self.config.window.max_chunk_length]
                        for row in reversed(prev_rows)
                    ])
                    context.previous_chunk_content = prev_content
            
            # Get next chunks
            if window_size > 0:
                next_query = text("""
                    SELECT content
                    FROM document_chunks
                    WHERE job_id = :job_id
                    AND chunk_index > :chunk_index
                    ORDER BY chunk_index ASC
                    LIMIT :limit
                """)
                
                next_result = await self.db_session.execute(
                    next_query,
                    {
                        "job_id": str(chunk.job_id),
                        "chunk_index": chunk.chunk_index,
                        "limit": window_size,
                    },
                )
                next_rows = next_result.fetchall()
                
                if next_rows:
                    next_content = " ".join([
                        row.content[:self.config.window.max_chunk_length]
                        for row in next_rows
                    ])
                    context.next_chunk_content = next_content
            
            return context
            
        except Exception as e:
            self.logger.warning(
                "window_context_fetch_failed",
                chunk_id=str(chunk.id),
                error=str(e),
            )
            return context
    
    async def _get_hierarchical_context(
        self,
        chunk: DocumentChunkModel,
    ) -> ContextualContext:
        """Get hierarchical context for a chunk.
        
        This combines parent document context with hierarchy information
        from chunk metadata or document_hierarchy table.
        
        Args:
            chunk: The document chunk
            
        Returns:
            ContextualContext with hierarchical information
        """
        # Start with parent document context
        context = await self._get_parent_document_context(chunk)
        
        if not self.db_session:
            self.logger.debug("no_db_session_for_hierarchical_context")
            return context
        
        try:
            # Try to get hierarchy from document_hierarchy table if it exists
            hierarchy_query = text("""
                SELECT 
                    dh.level,
                    dh.path,
                    parent_h.path as parent_path
                FROM document_hierarchy dh
                LEFT JOIN document_hierarchy parent_h ON dh.parent_id = parent_h.id
                WHERE dh.chunk_id = :chunk_id
                LIMIT 1
            """)
            
            try:
                hierarchy_result = await self.db_session.execute(
                    hierarchy_query, {"chunk_id": str(chunk.id)}
                )
                hierarchy_row = hierarchy_result.fetchone()
                
                if hierarchy_row:
                    context.hierarchy_level = hierarchy_row.level
                    
                    # Parse ltree path
                    if hierarchy_row.path:
                        path_str = str(hierarchy_row.path)
                        context.hierarchy_path = path_str.split(".")
            except Exception as e:
                # Table might not exist yet, fallback to metadata
                self.logger.debug(
                    "hierarchy_table_not_available",
                    error=str(e),
                )
            
            # Fallback: extract hierarchy from chunk metadata
            chunk_metadata: dict[str, Any] = chunk.chunk_metadata  # type: ignore[assignment]
            if not context.hierarchy_path and chunk_metadata:
                if "section_headers" in chunk_metadata:
                    headers = chunk_metadata["section_headers"]
                    if isinstance(headers, list):
                        context.hierarchy_path = headers
                        context.hierarchy_level = len(headers)
                        context.section_headers = headers
                    elif isinstance(headers, str):
                        context.hierarchy_path = [headers]
                        context.hierarchy_level = 1
                        context.section_headers = [headers]
                
                if "hierarchy_level" in chunk_metadata:
                    context.hierarchy_level = chunk_metadata["hierarchy_level"]
                
                if "hierarchy_path" in chunk_metadata:
                    path = chunk_metadata["hierarchy_path"]
                    if isinstance(path, list):
                        context.hierarchy_path = path
                    elif isinstance(path, str):
                        context.hierarchy_path = path.split("/")
            
            return context
            
        except Exception as e:
            self.logger.warning(
                "hierarchical_context_fetch_failed",
                chunk_id=str(chunk.id),
                error=str(e),
            )
            return context
    
    def _format_enhanced_text(
        self,
        content: str,
        context: ContextualContext,
        context_type: ContextType,
    ) -> str:
        """Format enhanced text based on context type.
        
        Args:
            content: Original chunk content
            context: Contextual information
            context_type: Type of context enhancement
            
        Returns:
            Enhanced text with context prepended/appended
        """
        if context_type == ContextType.PARENT_DOCUMENT:
            return self._format_parent_document_text(content, context)
        elif context_type == ContextType.WINDOW:
            return self._format_window_text(content, context)
        elif context_type == ContextType.HIERARCHICAL:
            return self._format_hierarchical_text(content, context)
        else:
            return content  # type: ignore[unreachable]
    
    def _format_parent_document_text(
        self,
        content: str,
        context: ContextualContext,
    ) -> str:
        """Format text with parent document context.
        
        Example:
            Document: Database Architecture Guide
            Section: Vector Storage
            Content: The system uses pgvector for vector storage.
        """
        parts: list[str] = []
        
        # Add document title
        if context.document_title:
            parts.append(f"Document: {context.document_title}")
        
        # Add section headers
        if context.section_headers:
            for i, header in enumerate(context.section_headers, 1):
                indent = "  " * (i - 1)
                parts.append(f"{indent}Section: {header}")
        
        # Add selected metadata if enabled
        if self.config.parent_document.include_metadata and context.document_metadata:
            metadata_parts = []
            for key, value in context.document_metadata.items():
                if key in self.config.parent_document.metadata_fields:
                    metadata_parts.append(f"{key.capitalize()}: {value}")
            if metadata_parts:
                parts.append(" | ".join(metadata_parts))
        
        # Add content
        parts.append(f"Content: {content}")
        
        return "\n".join(parts)
    
    def _format_window_text(
        self,
        content: str,
        context: ContextualContext,
    ) -> str:
        """Format text with window context.
        
        Example:
            [Prev: semantic search requires...] The system uses pgvector... [Next: This enables...]
        """
        parts: list[str] = []
        separator = self.config.window.separator
        
        # Add previous context
        if context.previous_chunk_content:
            prev = context.previous_chunk_content[:100]  # Truncate for brevity
            if len(context.previous_chunk_content) > 100:
                prev += "..."
            parts.append(f"[Prev: {prev}]")
        
        # Add main content
        parts.append(content)
        
        # Add next context
        if context.next_chunk_content:
            next_text = context.next_chunk_content[:100]  # Truncate for brevity
            if len(context.next_chunk_content) > 100:
                next_text += "..."
            parts.append(f"[Next: {next_text}]")
        
        return separator.join(parts)
    
    def _format_hierarchical_text(
        self,
        content: str,
        context: ContextualContext,
    ) -> str:
        """Format text with hierarchical context.
        
        Example:
            Document: API Documentation v2.0
            Path: Authentication > JWT Tokens
            Content: JWT tokens expire after 24 hours.
        """
        parts: list[str] = []
        
        # Add document title
        if context.document_title:
            parts.append(f"Document: {context.document_title}")
        
        # Add hierarchy path
        if context.hierarchy_path and self.config.hierarchical.include_path:
            path_str = " > ".join(context.hierarchy_path)
            parts.append(f"Path: {path_str}")
        
        # Add level indicator
        if context.hierarchy_level is not None:
            parts.append(f"Level: {context.hierarchy_level}")
        
        # Add content
        parts.append(f"Content: {content}")
        
        return "\n".join(parts)
    
    async def get_parent_document(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve parent document information for a chunk.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Parent document information or None if not found
        """
        if not self.db_session:
            return None
        
        try:
            query = text("""
                SELECT 
                    j.id,
                    j.file_name,
                    j.metadata_json,
                    jr.result_metadata
                FROM document_chunks dc
                JOIN jobs j ON dc.job_id = j.id
                LEFT JOIN job_results jr ON jr.job_id = j.id
                WHERE dc.id = :chunk_id
            """)
            
            result = await self.db_session.execute(query, {"chunk_id": chunk_id})
            row = result.fetchone()
            
            if row:
                return {
                    "id": str(row.id),
                    "title": row.file_name,
                    "metadata": {**(row.metadata_json or {}), **(row.result_metadata or {})},
                }
            return None
            
        except Exception as e:
            self.logger.warning(
                "get_parent_document_failed",
                chunk_id=chunk_id,
                error=str(e),
            )
            return None
    
    async def get_section_headers(self, chunk_id: str) -> list[str]:
        """Get section headers that contain a chunk.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            List of section headers
        """
        if not self.db_session:
            return []
        
        try:
            query = text("""
                SELECT chunk_metadata
                FROM document_chunks
                WHERE id = :chunk_id
            """)
            
            result = await self.db_session.execute(query, {"chunk_id": chunk_id})
            row = result.fetchone()
            
            if row and row.chunk_metadata:
                metadata = row.chunk_metadata
                if "section_headers" in metadata:
                    headers = metadata["section_headers"]
                    if isinstance(headers, list):
                        return headers
                    elif isinstance(headers, str):
                        return [headers]
            
            return []
            
        except Exception as e:
            self.logger.warning(
                "get_section_headers_failed",
                chunk_id=chunk_id,
                error=str(e),
            )
            return []
    
    async def get_hierarchy_path(self, chunk_id: str) -> list[str]:
        """Get the hierarchical path for a chunk.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            List representing the hierarchical path
        """
        if not self.db_session:
            return []
        
        try:
            # Try document_hierarchy table first
            query = text("""
                SELECT path
                FROM document_hierarchy
                WHERE chunk_id = :chunk_id
                LIMIT 1
            """)
            
            result = await self.db_session.execute(query, {"chunk_id": chunk_id})
            row = result.fetchone()
            
            if row and row.path:
                path_str = str(row.path)
                return path_str.split(".")
            
            # Fallback to chunk metadata
            headers = await self.get_section_headers(chunk_id)
            return headers
            
        except Exception as e:
            self.logger.warning(
                "get_hierarchy_path_failed",
                chunk_id=chunk_id,
                error=str(e),
            )
            return []
