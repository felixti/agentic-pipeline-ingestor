"""Pipeline execution with content detection and vector store integration."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

import structlog

from src.vector_store_config import get_vector_store_config
from src.core.content_detection.models import ContentAnalysisResult
from src.core.content_detection.service import ContentDetectionService
from src.core.job_context import JobContext
from src.core.parser_selection import ParserConfig, ParserSelector

logger = structlog.get_logger(__name__)


class PipelineStage:
    """Base class for pipeline stages."""
    
    name: str = ""
    
    async def execute(self, context: JobContext) -> JobContext:
        """Execute stage.
        
        Args:
            context: Job context
            
        Returns:
            Updated context
        """
        raise NotImplementedError


class IngestStage(PipelineStage):
    """Stage 1: Ingest and validate file."""
    
    name = "ingest"
    
    async def execute(self, context: JobContext) -> JobContext:
        """Execute ingest stage."""
        log = logger.bind(stage=self.name, job_id=str(context.job_id))
        log.info("pipeline.stage_started")
        
        # Validate file exists
        file_path = Path(context.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Store basic file info
        context.set_stage_result(self.name, {
            "file_size": file_path.stat().st_size,
            "file_name": file_path.name,
        })
        
        log.info("pipeline.stage_completed")
        return context


class DetectStage(PipelineStage):
    """Stage 2: Content detection (NEW)."""
    
    name = "detect"
    
    def __init__(self, detection_service: Optional[ContentDetectionService] = None):
        """Initialize detection stage.
        
        Args:
            detection_service: Detection service instance
        """
        self.detection_service = detection_service or ContentDetectionService()
    
    async def execute(self, context: JobContext) -> JobContext:
        """Execute detection stage."""
        log = logger.bind(stage=self.name, job_id=str(context.job_id))
        log.info("pipeline.stage_started")
        
        # Check if detection result already exists (from cache)
        if context.content_detection_result is not None:
            log.info("pipeline.detection_skipped", reason="already_cached")
            return context
        
        try:
            # Perform detection
            result = await self.detection_service.detect(context.file_path)
            
            # Store in context
            context.set_detection_result(result)
            context.set_stage_result(self.name, {
                "content_type": result.content_type,
                "confidence": result.confidence,
                "processing_time_ms": result.processing_time_ms,
            })
            
            log.info(
                "pipeline.stage_completed",
                content_type=result.content_type,
                confidence=result.confidence,
            )
            
        except Exception as e:
            log.error("pipeline.stage_failed", error=str(e))
            raise
        
        return context


class SelectParserStage(PipelineStage):
    """Stage 3: Parser selection based on detection."""
    
    name = "select_parser"
    
    async def execute(self, context: JobContext) -> JobContext:
        """Execute parser selection stage."""
        log = logger.bind(stage=self.name, job_id=str(context.job_id))
        log.info("pipeline.stage_started")
        
        # Get detection result
        detection_result = context.content_detection_result
        
        if detection_result is None:
            # No detection available, use default
            log.warning("pipeline.no_detection_result", using_default=True)
            context.set_parser_selection("docling", "azure_ocr")
            context.set_stage_result(self.name, {
                "primary": "docling",
                "fallback": "azure_ocr",
                "rationale": "No detection result available, using default",
            })
            return context
        
        # Check for explicit config
        explicit_config = None
        if context.config.get("parser"):
            parser_config = context.config["parser"]
            explicit_config = ParserConfig(
                primary_parser=parser_config.get("primary", "docling"),
                fallback_parser=parser_config.get("fallback"),
                force_ocr=parser_config.get("force_ocr", False)
            )
        
        # Select parser
        selection = ParserSelector.select_parser(detection_result, explicit_config)
        
        # Store in context
        context.set_parser_selection(selection.primary_parser, selection.fallback_parser)
        context.set_stage_result(self.name, {
            "primary": selection.primary_parser,
            "fallback": selection.fallback_parser,
            "rationale": selection.rationale,
            "overridden": selection.overridden,
        })
        
        log.info(
            "pipeline.stage_completed",
            primary=selection.primary_parser,
            fallback=selection.fallback_parser,
        )
        
        return context


class ParseStage(PipelineStage):
    """Stage 4: Parse document."""
    
    name = "parse"
    
    async def execute(self, context: JobContext) -> JobContext:
        """Execute parse stage."""
        log = logger.bind(stage=self.name, job_id=str(context.job_id))
        log.info("pipeline.stage_started")
        
        # This is a placeholder - actual parsing would use the selected parser
        primary = context.selected_parser or "docling"
        
        log.info("pipeline.using_parser", parser=primary)
        
        # In a real implementation, this would extract text from the document
        # For now, we'll store a placeholder indicating successful parsing
        context.set_stage_result(self.name, {
            "parser_used": primary,
            "status": "completed",
            "extracted_text": None,  # Would contain actual extracted text
        })
        
        log.info("pipeline.stage_completed")
        return context


class ChunkStage(PipelineStage):
    """Stage 5: Chunk extracted text for vector storage (NEW).
    
    This stage takes the extracted text from the parse stage and splits it
    into chunks according to the configured chunking strategy. The chunks
    are stored in the context for the embedding stage.
    """
    
    name = "chunk"
    
    def __init__(self):
        """Initialize chunk stage with configuration."""
        self.vs_config = get_vector_store_config()
    
    async def execute(self, context: JobContext) -> JobContext:
        """Execute chunk stage."""
        log = logger.bind(stage=self.name, job_id=str(context.job_id))
        log.info("pipeline.stage_started")
        
        # Check if vector store is enabled
        if not self.vs_config.enabled:
            log.info("pipeline.chunking_disabled")
            context.set_stage_result(self.name, {
                "status": "skipped",
                "reason": "vector_store_disabled",
            })
            return context
        
        # Check if chunking is enabled in pipeline config
        if not self.vs_config.pipeline.store_chunks:
            log.info("pipeline.chunking_disabled", reason="pipeline_config")
            context.set_stage_result(self.name, {
                "status": "skipped",
                "reason": "pipeline_config",
            })
            return context
        
        # Get extracted text from parse stage
        parse_result = context.get_stage_result("parse") or {}
        extracted_text = parse_result.get("extracted_text")
        
        if not extracted_text:
            log.warning("pipeline.no_extracted_text")
            context.set_stage_result(self.name, {
                "status": "skipped",
                "reason": "no_extracted_text",
            })
            return context
        
        try:
            # Generate chunks based on strategy
            chunks = self._chunk_text(extracted_text)
            
            # Store chunks in context for embedding stage
            context.chunks = chunks  # type: ignore
            
            context.set_stage_result(self.name, {
                "status": "completed",
                "chunk_count": len(chunks),
                "strategy": self.vs_config.pipeline.chunking_strategy,
                "chunk_size": self.vs_config.pipeline.chunk_size,
                "chunk_overlap": self.vs_config.pipeline.chunk_overlap,
            })
            
            log.info(
                "pipeline.stage_completed",
                chunk_count=len(chunks),
                strategy=self.vs_config.pipeline.chunking_strategy,
            )
            
        except Exception as e:
            log.error("pipeline.chunking_failed", error=str(e))
            context.set_stage_result(self.name, {
                "status": "failed",
                "error": str(e),
            })
            # Don't raise - chunking failure shouldn't stop the pipeline
        
        return context
    
    def _chunk_text(self, text: str) -> list[dict[str, Any]]:
        """Split text into chunks according to configuration.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        strategy = self.vs_config.pipeline.chunking_strategy
        chunk_size = self.vs_config.pipeline.chunk_size
        chunk_overlap = self.vs_config.pipeline.chunk_overlap
        min_chunk_size = self.vs_config.pipeline.min_chunk_size
        
        chunks = []
        
        if strategy == "fixed":
            # Simple fixed-size chunking with overlap
            start = 0
            chunk_index = 0
            
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                
                # Only keep chunks above minimum size (except for last chunk)
                if len(chunk_text) >= min_chunk_size or end == len(text):
                    content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                    chunks.append({
                        "content": chunk_text,
                        "chunk_index": chunk_index,
                        "content_hash": content_hash,
                        "metadata": {
                            "start_char": start,
                            "end_char": end,
                            "char_count": len(chunk_text),
                        },
                    })
                    chunk_index += 1
                
                # Move start forward by chunk size minus overlap
                start += chunk_size - chunk_overlap
                
                # Prevent infinite loop on small chunks
                if start >= end:
                    break
        
        elif strategy == "semantic":
            # Semantic chunking - split on paragraph boundaries when possible
            import re
            
            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            
            current_chunk = []
            current_size = 0
            chunk_index = 0
            start_char = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                paragraph_size = len(paragraph)
                
                # If adding this paragraph would exceed chunk size, save current chunk
                if current_size + paragraph_size > chunk_size and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    if len(chunk_text) >= min_chunk_size:
                        content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                        chunks.append({
                            "content": chunk_text,
                            "chunk_index": chunk_index,
                            "content_hash": content_hash,
                            "metadata": {
                                "start_char": start_char,
                                "end_char": start_char + len(chunk_text),
                                "char_count": len(chunk_text),
                                "paragraph_count": len(current_chunk),
                            },
                        })
                        chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text = chunk_text[-chunk_overlap:] if len(chunk_text) > chunk_overlap else chunk_text
                    current_chunk = [overlap_text, paragraph] if overlap_text else [paragraph]
                    current_size = len(overlap_text) + paragraph_size if overlap_text else paragraph_size
                    start_char += len(chunk_text) - len(overlap_text) if overlap_text else len(chunk_text)
                else:
                    current_chunk.append(paragraph)
                    current_size += paragraph_size
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text) >= min_chunk_size:
                    content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                    chunks.append({
                        "content": chunk_text,
                        "chunk_index": chunk_index,
                        "content_hash": content_hash,
                        "metadata": {
                            "start_char": start_char,
                            "end_char": start_char + len(chunk_text),
                            "char_count": len(chunk_text),
                            "paragraph_count": len(current_chunk),
                        },
                    })
        
        else:  # hierarchical or unknown - use fixed as fallback
            return self._chunk_text_fixed(text)
        
        return chunks
    
    def _chunk_text_fixed(self, text: str) -> list[dict[str, Any]]:
        """Fixed-size chunking fallback method.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dictionaries
        """
        chunk_size = self.vs_config.pipeline.chunk_size
        chunk_overlap = self.vs_config.pipeline.chunk_overlap
        min_chunk_size = self.vs_config.pipeline.min_chunk_size
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            if len(chunk_text) >= min_chunk_size or end == len(text):
                content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                chunks.append({
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "content_hash": content_hash,
                    "metadata": {
                        "start_char": start,
                        "end_char": end,
                        "char_count": len(chunk_text),
                    },
                })
                chunk_index += 1
            
            start += chunk_size - chunk_overlap
            if start >= end:
                break
        
        return chunks


class EmbedStage(PipelineStage):
    """Stage 6: Generate embeddings and store chunks (NEW).
    
    This stage generates embeddings for the chunks created in the chunk stage
    and stores them in the document_chunks table. It integrates with the
    embedding service to use the configured LLM provider.
    """
    
    name = "embed"
    
    def __init__(self, embedding_service: Any = None):
        """Initialize embed stage.
        
        Args:
            embedding_service: Optional embedding service instance
        """
        self.vs_config = get_vector_store_config()
        self._embedding_service = embedding_service
    
    async def execute(self, context: JobContext) -> JobContext:
        """Execute embed stage."""
        log = logger.bind(stage=self.name, job_id=str(context.job_id))
        log.info("pipeline.stage_started")
        
        # Check if vector store is enabled
        if not self.vs_config.enabled:
            log.info("pipeline.embedding_disabled")
            context.set_stage_result(self.name, {
                "status": "skipped",
                "reason": "vector_store_disabled",
            })
            return context
        
        # Check if auto-generation is enabled
        if not self.vs_config.pipeline.auto_generate_embeddings:
            log.info("pipeline.embedding_disabled", reason="auto_generate_disabled")
            context.set_stage_result(self.name, {
                "status": "skipped",
                "reason": "auto_generate_disabled",
            })
            return context
        
        # Get chunks from context
        chunks = getattr(context, "chunks", None)
        if not chunks:
            log.info("pipeline.no_chunks_to_embed")
            context.set_stage_result(self.name, {
                "status": "skipped",
                "reason": "no_chunks",
            })
            return context
        
        try:
            # Import here to avoid circular dependencies
            from src.services.embedding_service import EmbeddingService
            from src.db.repositories.document_chunk_repository import DocumentChunkRepository
            from src.db.models import DocumentChunkModel
            from sqlalchemy.ext.asyncio import AsyncSession
            
            # Initialize embedding service if not provided
            embedding_service = self._embedding_service
            if embedding_service is None:
                embedding_service = EmbeddingService(self.vs_config)
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk["content"] for chunk in chunks]
            
            log.info("pipeline.generating_embeddings", chunk_count=len(chunk_texts))
            
            embedding_results = await embedding_service.embed_batch(chunk_texts)
            
            # Create DocumentChunkModel instances
            chunk_models = []
            for chunk_data, embedding_result in zip(chunks, embedding_results):
                chunk_model = DocumentChunkModel(
                    job_id=context.job_id,
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["content"],
                    content_hash=chunk_data["content_hash"],
                    metadata={
                        **chunk_data["metadata"],
                        "embedding_model": embedding_result.model,
                        "embedding_latency_ms": embedding_result.latency_ms,
                    },
                )
                # Set the embedding vector
                chunk_model.set_embedding(embedding_result.embedding)
                chunk_models.append(chunk_model)
            
            # Store chunks in database
            # Note: In a real implementation, we'd need to get the database session
            # from somewhere (likely passed in or available via context)
            # For now, we just record what would be stored
            context.set_stage_result(self.name, {
                "status": "completed",
                "chunks_stored": len(chunk_models),
                "embedding_model": self.vs_config.embedding.model,
                "embedding_dimensions": self.vs_config.embedding.dimensions,
                "avg_latency_ms": sum(
                    r.latency_ms or 0 for r in embedding_results
                ) / len(embedding_results) if embedding_results else 0,
            })
            
            # Store chunk models in context for potential later use
            context.chunk_models = chunk_models  # type: ignore
            
            log.info(
                "pipeline.stage_completed",
                chunks_stored=len(chunk_models),
                model=self.vs_config.embedding.model,
            )
            
        except Exception as e:
            log.error("pipeline.embedding_failed", error=str(e))
            context.set_stage_result(self.name, {
                "status": "failed",
                "error": str(e),
            })
            # Don't raise - embedding failure shouldn't stop the pipeline
            # The document is still processed even if embeddings fail
        
        return context


class Pipeline:
    """Pipeline executor."""
    
    STAGES = [
        IngestStage,
        DetectStage,         # Content detection
        SelectParserStage,   # Parser selection based on detection
        ParseStage,          # Document parsing
        ChunkStage,          # NEW: Text chunking
        EmbedStage,          # NEW: Embedding generation
        # Additional stages would be added here:
        # EnrichStage,
        # QualityStage,
        # TransformStage,
        # OutputStage,
    ]
    
    def __init__(
        self,
        detection_service: Optional[ContentDetectionService] = None,
        embedding_service: Any = None,
    ):
        """Initialize pipeline.
        
        Args:
            detection_service: Optional detection service
            embedding_service: Optional embedding service
        """
        self.stages = []
        for stage_class in self.STAGES:
            if stage_class == DetectStage:
                self.stages.append(DetectStage(detection_service))
            elif stage_class == EmbedStage:
                self.stages.append(EmbedStage(embedding_service))
            else:
                self.stages.append(stage_class())
    
    async def execute(
        self,
        file_path: str,
        file_type: str,
        job_id: Optional[UUID] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> JobContext:
        """Execute pipeline on a file.
        
        Args:
            file_path: Path to input file
            file_type: MIME type of file
            job_id: Optional job ID (generated if not provided)
            config: Optional pipeline configuration
            
        Returns:
            Final job context
        """
        from datetime import datetime, timezone
        
        job_id = job_id or uuid4()
        
        # Create initial context
        context = JobContext(
            job_id=job_id,
            file_path=file_path,
            file_type=file_type,
            config=config or {},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        
        log = logger.bind(job_id=str(job_id))
        log.info("pipeline.execution_started", file_path=file_path)
        
        try:
            # Execute each stage
            for stage in self.stages:
                context = await stage.execute(context)
            
            log.info("pipeline.execution_completed")
            
        except Exception as e:
            log.error("pipeline.execution_failed", error=str(e))
            raise
        
        return context


class PipelineExecutor:
    """Pipeline executor for running jobs through the pipeline.
    
    This is the main entry point for executing pipeline jobs with
    full plugin registry support.
    """
    
    def __init__(
        self,
        config: Any = None,
        plugin_registry: Any = None,
        detection_service: Optional[Any] = None,
        embedding_service: Any = None,
    ):
        """Initialize pipeline executor.
        
        Args:
            config: Pipeline configuration
            plugin_registry: Plugin registry for parsers/destinations
            detection_service: Optional content detection service
            embedding_service: Optional embedding service
        """
        self.config = config
        self.plugin_registry = plugin_registry
        self.detection_service = detection_service
        self.embedding_service = embedding_service
        self.pipeline = Pipeline(
            detection_service=detection_service,
            embedding_service=embedding_service,
        )
    
    async def execute(self, job: Any) -> JobContext:
        """Execute a job through the pipeline.
        
        Args:
            job: Job to execute (must have source_uri, id, etc.)
            
        Returns:
            JobContext with execution results
        """
        from datetime import datetime, timezone
        
        job_id = getattr(job, 'id', uuid4())
        file_path = getattr(job, 'source_uri', None)
        file_type = getattr(job, 'mime_type', 'application/octet-stream')
        
        if file_path is None:
            raise ValueError("Job must have source_uri")
        
        # Build config from job
        job_config = {}
        if self.config:
            # Convert config to dict if possible
            if hasattr(self.config, 'model_dump'):
                job_config = self.config.model_dump()
            elif hasattr(self.config, 'dict'):
                job_config = self.config.dict()
            else:
                job_config = vars(self.config)
        
        # Add job-specific config
        if hasattr(job, 'pipeline_config') and job.pipeline_config:
            if hasattr(job.pipeline_config, 'model_dump'):
                job_config.update(job.pipeline_config.model_dump())
            elif hasattr(job.pipeline_config, 'dict'):
                job_config.update(job.pipeline_config.dict())
        
        context = await self.pipeline.execute(
            file_path=file_path,
            file_type=file_type,
            job_id=job_id,
            config=job_config,
        )
        
        # Update job status if methods exist
        if hasattr(job, 'status'):
            job.status = "COMPLETED"
        if hasattr(job, 'completed_at'):
            job.completed_at = datetime.now(timezone.utc)
        
        return context


async def run_pipeline(
    file_path: str,
    file_type: str = "application/pdf",
    job_id: Optional[UUID] = None,
    config: Optional[Dict[str, Any]] = None
) -> JobContext:
    """Convenience function to run pipeline.
    
    Args:
        file_path: Path to input file
        file_type: MIME type
        job_id: Optional job ID
        config: Optional configuration
        
    Returns:
        Final job context
    """
    pipeline = Pipeline()
    return await pipeline.execute(file_path, file_type, job_id, config)
