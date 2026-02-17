"""Pipeline execution module for the Agentic Data Pipeline Ingestor.

This module implements the 7-stage processing pipeline:
1. Ingest - Receive and validate files
2. Detect - Content type detection
3. Parse - Document parsing (Docling/Azure OCR)
4. Enrich - Metadata extraction and entity recognition
5. Quality - Quality assessment
6. Transform - Chunking and embedding generation
7. Output - Route to destinations
"""

import logging
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from src.api.models import (
    ContentDetectionResult,
    Job,
    JobResult,
    JobStatus,
    PipelineConfig,
)
from src.core.decisions import AgenticDecisionEngine
from src.core.detection import ContentDetector
from src.core.quality import QualityAssessor
from src.llm.provider import LLMProvider
from src.observability.metrics import get_metrics_manager
from src.observability.tracing import get_tracer, start_pipeline_stage_span
from src.plugins.base import (
    ParsingResult,
    TransformedData,
    WriteResult,
)
from src.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class PipelineContext:
    """Context object for sharing data between pipeline stages.
    
    Attributes:
        job: The job being processed
        config: Pipeline configuration
        data: Dictionary for stage results
        errors: List of errors encountered
        stage_results: Dictionary of stage execution results
    """

    def __init__(self, job: Job, config: PipelineConfig) -> None:
        """Initialize pipeline context.
        
        Args:
            job: Job being processed
            config: Pipeline configuration
        """
        self.job = job
        self.config = config
        self.data: dict[str, Any] = {}
        self.errors: list[str] = []
        self.stage_results: dict[str, dict[str, Any]] = {}
        self.current_stage: str | None = None

    def set_stage_result(self, stage: str, result: dict[str, Any]) -> None:
        """Set result for a stage.
        
        Args:
            stage: Stage name
            result: Stage result data
        """
        self.stage_results[stage] = result
        self.data[stage] = result

    def get_stage_result(self, stage: str) -> dict[str, Any] | None:
        """Get result for a stage.
        
        Args:
            stage: Stage name
            
        Returns:
            Stage result or None
        """
        return self.stage_results.get(stage)


class PipelineStage(ABC):
    """Abstract base class for pipeline stages.
    
    Each stage in the pipeline must implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stage name."""
        ...

    @abstractmethod
    async def execute(
        self,
        context: PipelineContext,
    ) -> dict[str, Any]:
        """Execute the pipeline stage.
        
        Args:
            context: Pipeline context
            
        Returns:
            Stage result data
        """
        ...

    async def on_success(
        self,
        context: PipelineContext,
        result: dict[str, Any],
    ) -> None:
        """Called when stage executes successfully.
        
        Args:
            context: Pipeline context
            result: Stage result
        """
        pass

    async def on_failure(
        self,
        context: PipelineContext,
        error: Exception,
    ) -> None:
        """Called when stage execution fails.
        
        Args:
            context: Pipeline context
            error: Exception that caused failure
        """
        pass


class IngestStage(PipelineStage):
    """Stage 1: File ingestion and validation.
    
    Responsibilities:
    - Validate file format and size
    - Security scanning
    - Staging to blob storage
    - Create job record
    """

    @property
    def name(self) -> str:
        return "ingest"

    async def execute(
        self,
        context: PipelineContext,
    ) -> dict[str, Any]:
        """Execute ingestion stage."""
        logger.info("ingest_stage_started", job_id=str(context.job.id))

        job = context.job

        # Determine staging path
        staging_dir = Path("/tmp/pipeline/staging") / str(job.id)
        staging_dir.mkdir(parents=True, exist_ok=True)

        staged_path = staging_dir / job.file_name

        # If source is upload, file is already staged
        # Otherwise, download from source
        if job.source_type.value == "upload":
            # File should already be at source_uri
            source_path = Path(job.source_uri)
            if source_path.exists() and source_path != staged_path:
                shutil.copy2(source_path, staged_path)

        # Validate file exists
        if not staged_path.exists():
            raise FileNotFoundError(f"Staged file not found: {staged_path}")

        # Get file info
        file_size = staged_path.stat().st_size
        file_hash = await self._calculate_hash(staged_path)

        result = {
            "staged_path": str(staged_path),
            "file_size": file_size,
            "file_hash": file_hash,
            "validated": True,
        }

        logger.info(
            "ingest_stage_completed",
            job_id=str(context.job.id),
            file_size=file_size,
            file_hash=file_hash[:16],
        )

        return result

    async def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of hash
        """
        import hashlib

        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()


class DetectStage(PipelineStage):
    """Stage 2: Content type detection.
    
    Responsibilities:
    - Analyze file structure
    - Detect scanned vs. text-based PDFs
    - Determine optimal parser strategy
    """

    def __init__(self) -> None:
        """Initialize the detect stage."""
        self.detector = ContentDetector()

    @property
    def name(self) -> str:
        return "detect"

    async def execute(
        self,
        context: PipelineContext,
    ) -> dict[str, Any]:
        """Execute content detection stage."""
        logger.info("detect_stage_started", job_id=str(context.job.id))

        # Get staged file path
        ingest_result = context.get_stage_result("ingest")
        if not ingest_result:
            raise ValueError("Ingest stage must run before detect stage")

        file_path = ingest_result["staged_path"]

        # Perform content detection
        detection_result = await self.detector.detect(
            file_path,
            mime_type=context.job.mime_type,
        )

        result = {
            "detection": detection_result.model_dump(),
            "detected_type": detection_result.detected_type.value,
            "recommended_parser": detection_result.recommended_parser,
            "confidence": detection_result.confidence,
        }

        logger.info(
            "detect_stage_completed",
            job_id=str(context.job.id),
            detected_type=detection_result.detected_type.value,
            confidence=detection_result.confidence,
        )

        return result


class ParseStage(PipelineStage):
    """Stage 3: Document parsing.
    
    Responsibilities:
    - Extract text using primary parser
    - Fallback to secondary parser if needed
    - Handle parsing errors and retries
    """

    def __init__(self, plugin_registry: PluginRegistry | None = None) -> None:
        """Initialize the parse stage.
        
        Args:
            plugin_registry: Plugin registry for accessing parsers
        """
        self.registry = plugin_registry or PluginRegistry()
        self.decision_engine: AgenticDecisionEngine | None = None

    def set_decision_engine(self, engine: AgenticDecisionEngine) -> None:
        """Set the decision engine for intelligent parser selection.
        
        Args:
            engine: Agentic decision engine
        """
        self.decision_engine = engine

    @property
    def name(self) -> str:
        return "parse"

    async def execute(
        self,
        context: PipelineContext,
    ) -> dict[str, Any]:
        """Execute parsing stage."""
        logger.info("parse_stage_started", job_id=str(context.job.id))

        # Get required data
        ingest_result = context.get_stage_result("ingest")
        detect_result = context.get_stage_result("detect")

        if not ingest_result or not detect_result:
            raise ValueError("Ingest and detect stages must run before parse stage")

        file_path = ingest_result["staged_path"]
        detection = ContentDetectionResult(**detect_result["detection"])

        # Get parser config
        parser_config = context.config.parser

        # Use agentic decision if available
        if self.decision_engine:
            selection = await self.decision_engine.select_parser(detection)
            primary_parser = selection.parser
            fallback_parser = selection.fallback_parser
        else:
            primary_parser = parser_config.primary_parser
            fallback_parser = parser_config.fallback_parser

        # Try primary parser
        parse_result = await self._try_parse(
            file_path, primary_parser, parser_config.parser_options
        )

        # Try fallback if primary failed
        if not parse_result.success and fallback_parser:
            logger.warning(
                "primary_parser_failed",
                job_id=str(context.job.id),
                primary=primary_parser,
                fallback=fallback_parser,
                error=parse_result.error,
            )

            parse_result = await self._try_parse(
                file_path, fallback_parser, parser_config.parser_options
            )
            parser_used = fallback_parser if parse_result.success else primary_parser
        else:
            parser_used = primary_parser

        result = {
            "parser_used": parser_used,
            "success": parse_result.success,
            "text": parse_result.text,
            "pages": parse_result.pages,
            "metadata": parse_result.metadata,
            "tables": parse_result.tables,
            "images": parse_result.images,
            "confidence": parse_result.confidence,
            "processing_time_ms": parse_result.processing_time_ms,
        }

        if not parse_result.success:
            result["error"] = parse_result.error

        logger.info(
            "parse_stage_completed",
            job_id=str(context.job.id),
            parser_used=parser_used,
            success=parse_result.success,
            confidence=parse_result.confidence,
        )

        return result

    async def _try_parse(
        self,
        file_path: str,
        parser_id: str,
        options: dict[str, Any],
    ) -> ParsingResult:
        """Try to parse with a specific parser.
        
        Args:
            file_path: Path to file
            parser_id: Parser plugin ID
            options: Parser options
            
        Returns:
            ParsingResult
        """
        try:
            parser = self.registry.get_parser(parser_id)

            if not parser:
                return ParsingResult(
                    success=False,
                    error=f"Parser not found: {parser_id}",
                )

            return await parser.parse(file_path, options)

        except Exception as e:
            logger.error(f"Parser {parser_id} failed: {e}")
            return ParsingResult(
                success=False,
                error=f"Parser error: {e!s}",
            )


class EnrichStage(PipelineStage):
    """Stage 4: Document enrichment.
    
    Responsibilities:
    - Extract metadata
    - Entity recognition
    - Document classification
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        """Initialize the enrich stage.
        
        Args:
            llm_provider: LLM provider for enrichment
        """
        self.llm = llm_provider

    @property
    def name(self) -> str:
        return "enrich"

    async def execute(
        self,
        context: PipelineContext,
    ) -> dict[str, Any]:
        """Execute enrichment stage."""
        logger.info("enrich_stage_started", job_id=str(context.job.id))

        parse_result = context.get_stage_result("parse")
        if not parse_result:
            raise ValueError("Parse stage must run before enrich stage")

        text = parse_result.get("text", "")
        enrichment_config = context.config.enrichment

        entities: list[dict[str, Any]] = []
        classification: str | None = None
        metadata: dict[str, Any] = {}

        # Extract basic metadata
        metadata = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "line_count": len(text.split("\n")),
        }

        # Entity extraction if enabled and LLM available
        if enrichment_config.extract_entities and self.llm and text:
            entities = await self._extract_entities(text, enrichment_config.entity_types)

        # Document classification if enabled
        if enrichment_config.classify_document and self.llm and text:
            classification = await self._classify_document(text)

        result = {
            "entities": entities,
            "classification": classification,
            "metadata": metadata,
        }

        logger.info(
            "enrich_stage_completed",
            job_id=str(context.job.id),
            entities_count=len(entities),
            classification=classification,
        )

        return result

    async def _extract_entities(
        self,
        text: str,
        entity_types: list[str],
    ) -> list[dict[str, Any]]:
        """Extract entities from text using LLM.
        
        Args:
            text: Text to analyze
            entity_types: Types of entities to extract
            
        Returns:
            List of extracted entities
        """
        if not self.llm:
            return []

        try:
            # Sample text if too long
            sample = text[:5000] if len(text) > 5000 else text

            prompt = f"""Extract {', '.join(entity_types)} entities from the following text.
Return a JSON array of objects with "type", "value", and "confidence" fields.

Text:
{sample}

Entities:"""

            response = await self.llm.json_completion(
                prompt=prompt,
                system_prompt="You are a named entity recognition system. Extract entities accurately.",
                max_tokens=500,
            )

            if isinstance(response, list):
                return response
            elif isinstance(response, dict) and "entities" in response:
                return response["entities"]
            else:
                return []

        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    async def _classify_document(self, text: str) -> str | None:
        """Classify document type using LLM.
        
        Args:
            text: Text to classify
            
        Returns:
            Document classification
        """
        if not self.llm:
            return None

        try:
            # Sample text if too long
            sample = text[:3000] if len(text) > 3000 else text

            prompt = f"""Classify the following document into one of these categories:
- contract
- invoice
- report
- letter
- resume
- legal_document
- technical_document
- other

Return only the category name.

Text:
{sample}

Category:"""

            response = await self.llm.simple_completion(
                prompt=prompt,
                system_prompt="You are a document classification system.",
                max_tokens=50,
                temperature=0.1,
            )

            return response.strip().lower()

        except Exception as e:
            logger.warning(f"Document classification failed: {e}")
            return None


class QualityStage(PipelineStage):
    """Stage 5: Quality assessment.
    
    Responsibilities:
    - Calculate quality scores
    - Compare to thresholds
    - Trigger retries if needed
    """

    def __init__(self, decision_engine: AgenticDecisionEngine | None = None) -> None:
        """Initialize the quality stage.
        
        Args:
            decision_engine: Decision engine for retry decisions
        """
        self.assessor = QualityAssessor()
        self.decision_engine = decision_engine

    @property
    def name(self) -> str:
        return "quality"

    async def execute(
        self,
        context: PipelineContext,
    ) -> dict[str, Any]:
        """Execute quality assessment stage."""
        logger.info("quality_stage_started", job_id=str(context.job.id))

        parse_result = context.get_stage_result("parse")
        if not parse_result:
            raise ValueError("Parse stage must run before quality stage")

        # Build parsing result for assessment
        parsing_result = ParsingResult(
            success=parse_result.get("success", False),
            text=parse_result.get("text", ""),
            pages=parse_result.get("pages", []),
            metadata=parse_result.get("metadata", {}),
            tables=parse_result.get("tables", []),
            images=parse_result.get("images", []),
            confidence=parse_result.get("confidence", 0.0),
        )

        # Assess quality
        quality_config = context.config.quality
        quality_score = await self.assessor.assess(parsing_result, quality_config)

        # Check if retry needed
        should_retry = self.assessor.should_retry(
            quality_score, quality_config, context.job.retry_count
        )

        result = {
            "overall_score": quality_score.overall_score,
            "text_quality": quality_score.text_quality,
            "structure_quality": quality_score.structure_quality,
            "ocr_confidence": quality_score.ocr_confidence,
            "completeness": quality_score.completeness,
            "passed": quality_score.passed,
            "should_retry": should_retry,
            "issues": quality_score.issues,
            "recommendations": quality_score.recommendations,
            "threshold": quality_config.min_quality_score,
        }

        logger.info(
            "quality_stage_completed",
            job_id=str(context.job.id),
            score=quality_score.overall_score,
            passed=quality_score.passed,
            should_retry=should_retry,
        )

        return result


class TransformStage(PipelineStage):
    """Stage 6: Document transformation.
    
    Responsibilities:
    - Text chunking
    - Embedding generation (placeholder)
    - Format conversion
    """

    @property
    def name(self) -> str:
        return "transform"

    async def execute(
        self,
        context: PipelineContext,
    ) -> dict[str, Any]:
        """Execute transformation stage."""
        logger.info("transform_stage_started", job_id=str(context.job.id))

        parse_result = context.get_stage_result("parse")
        quality_result = context.get_stage_result("quality")

        if not parse_result:
            raise ValueError("Parse stage must run before transform stage")

        # Skip if quality check failed
        if quality_result and not quality_result.get("passed", True):
            logger.warning(
                "transform_stage_skipped_quality",
                job_id=str(context.job.id),
                score=quality_result.get("overall_score", 0),
            )
            return {
                "chunks": [],
                "skipped": True,
                "reason": "quality_check_failed",
            }

        text = parse_result.get("text", "")
        transform_config = context.config.transformation
        chunking_config = transform_config.chunking

        # Perform chunking if enabled
        chunks: list[dict[str, Any]] = []
        if chunking_config.enabled and text:
            chunks = self._chunk_text(text, chunking_config)
        else:
            # Single chunk with full text
            chunks = [{"content": text, "metadata": {}}]

        result = {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "embeddings": None,  # Placeholder for embedding generation
            "output_format": transform_config.output_format.value,
        }

        logger.info(
            "transform_stage_completed",
            job_id=str(context.job.id),
            chunks=len(chunks),
        )

        return result

    def _chunk_text(
        self,
        text: str,
        config,
    ) -> list[dict[str, Any]]:
        """Chunk text into segments.
        
        Args:
            text: Text to chunk
            config: Chunking configuration
            
        Returns:
            List of chunks
        """
        from src.api.models import ChunkingStrategy

        chunks: list[dict[str, Any]] = []

        if config.strategy == ChunkingStrategy.FIXED:
            chunks = self._fixed_chunking(text, config.chunk_size, config.chunk_overlap)
        elif config.strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._semantic_chunking(text, config.chunk_size)
        else:
            # Default to fixed
            chunks = self._fixed_chunking(text, config.chunk_size, config.chunk_overlap)

        return chunks

    def _fixed_chunking(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[dict[str, Any]]:
        """Fixed-size chunking with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Chunk size in characters
            overlap: Overlap between chunks
            
        Returns:
            List of chunks
        """
        chunks: list[dict[str, Any]] = []
        step = chunk_size - overlap

        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "start_char": i,
                        "end_char": i + len(chunk_text),
                    },
                })

        return chunks

    def _semantic_chunking(
        self,
        text: str,
        chunk_size: int,
    ) -> list[dict[str, Any]]:
        """Semantic chunking by paragraphs.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size
            
        Returns:
            List of chunks
        """
        chunks: list[dict[str, Any]] = []
        paragraphs = text.split("\n\n")

        current_chunk = ""
        start_idx = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": {
                            "start_char": start_idx,
                            "end_char": start_idx + len(current_chunk),
                        },
                    })
                current_chunk = para + "\n\n"
                start_idx += len(current_chunk)

        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {
                    "start_char": start_idx,
                    "end_char": start_idx + len(current_chunk),
                },
            })

        return chunks


class OutputStage(PipelineStage):
    """Stage 7: Output routing.
    
    Responsibilities:
    - Route to configured destinations
    - Apply filters
    - Confirm writes
    """

    def __init__(self, plugin_registry: PluginRegistry | None = None) -> None:
        """Initialize the output stage.
        
        Args:
            plugin_registry: Plugin registry for accessing destinations
        """
        self.registry = plugin_registry or PluginRegistry()

    @property
    def name(self) -> str:
        return "output"

    async def execute(
        self,
        context: PipelineContext,
    ) -> dict[str, Any]:
        """Execute output stage."""
        logger.info("output_stage_started", job_id=str(context.job.id))

        transform_result = context.get_stage_result("transform")
        if not transform_result:
            raise ValueError("Transform stage must run before output stage")

        # Skip if transform was skipped
        if transform_result.get("skipped"):
            return {
                "destinations": [],
                "skipped": True,
                "reason": transform_result.get("reason"),
            }

        # Build transformed data
        data = TransformedData(
            job_id=context.job.id,
            chunks=transform_result.get("chunks", []),
            embeddings=transform_result.get("embeddings"),
            metadata={
                **context.job.model_dump(),
                "pipeline_config": context.config.model_dump(),
            },
            lineage=self._build_lineage(context),
            original_format=context.job.mime_type or "unknown",
            output_format=transform_result.get("output_format", "json"),
        )

        # Write to destinations
        destinations_results: list[dict[str, Any]] = []

        for dest_config in context.job.destinations:
            if not dest_config.enabled:
                continue

            try:
                result = await self._write_to_destination(
                    dest_config.type.value,
                    dest_config.config,
                    data,
                )

                destinations_results.append({
                    "destination_id": dest_config.type.value,
                    "success": result.success,
                    "records_written": result.records_written,
                    "uri": result.destination_uri,
                    "error": result.error,
                })

            except Exception as e:
                logger.error(
                    "destination_write_failed",
                    job_id=str(context.job.id),
                    destination=dest_config.type.value,
                    error=str(e),
                )
                destinations_results.append({
                    "destination_id": dest_config.type.value,
                    "success": False,
                    "error": str(e),
                })

        all_success = all(d.get("success", False) for d in destinations_results)

        result = {
            "destinations": destinations_results,
            "success": all_success,
            "destination_count": len(destinations_results),
        }

        logger.info(
            "output_stage_completed",
            job_id=str(context.job.id),
            destinations=len(destinations_results),
            success=all_success,
        )

        return result

    async def _write_to_destination(
        self,
        destination_type: str,
        dest_config: dict[str, Any],
        data: TransformedData,
    ) -> WriteResult:
        """Write data to a destination.
        
        Args:
            destination_type: Destination type/plugin ID
            dest_config: Destination configuration
            data: Data to write
            
        Returns:
            WriteResult
        """
        try:
            destination = self.registry.get_destination(destination_type)

            if not destination:
                return WriteResult(
                    success=False,
                    error=f"Destination not found: {destination_type}",
                )

            # Initialize if needed
            await destination.initialize(dest_config)

            # Connect and write
            conn = await destination.connect(dest_config)
            result = await destination.write(conn, data)

            return result

        except Exception as e:
            return WriteResult(
                success=False,
                error=f"Destination error: {e!s}",
            )

    def _build_lineage(self, context: PipelineContext) -> dict[str, Any]:
        """Build data lineage information.
        
        Args:
            context: Pipeline context
            
        Returns:
            Lineage dictionary
        """
        lineage: dict[str, Any] = {
            "job_id": str(context.job.id),
            "stages": [],
        }

        for stage_name, stage_data in context.stage_results.items():
            lineage["stages"].append({
                "stage": stage_name,
                "timestamp": datetime.utcnow().isoformat(),
            })

        return lineage


class PipelineExecutor:
    """Executor for the 7-stage processing pipeline.
    
    Manages stage execution, error handling, and context passing.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        plugin_registry: PluginRegistry | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the pipeline executor.
        
        Args:
            config: Pipeline configuration
            plugin_registry: Plugin registry
            llm_provider: LLM provider for agentic decisions
        """
        self.config = config
        self.registry = plugin_registry or PluginRegistry()
        self.llm = llm_provider

        # Create decision engine if LLM available
        self.decision_engine: AgenticDecisionEngine | None = None
        if llm_provider:
            self.decision_engine = AgenticDecisionEngine(llm_provider)

        # Initialize stages
        self.stages: list[PipelineStage] = self._create_stages()

    def _create_stages(self) -> list[PipelineStage]:
        """Create pipeline stages with dependencies.
        
        Returns:
            List of pipeline stages
        """
        parse_stage = ParseStage(self.registry)
        if self.decision_engine:
            parse_stage.set_decision_engine(self.decision_engine)

        return [
            IngestStage(),
            DetectStage(),
            parse_stage,
            EnrichStage(self.llm),
            QualityStage(self.decision_engine),
            TransformStage(),
            OutputStage(self.registry),
        ]

    async def execute(
        self,
        job: Job,
        enabled_stages: list[str] | None = None,
    ) -> PipelineContext:
        """Execute the pipeline for a job.
        
        Args:
            job: Job to process
            enabled_stages: Optional list of stage names to execute
            
        Returns:
            Final pipeline context
        """
        import time

        config = job.pipeline_config or self.config or PipelineConfig(name="default")
        context = PipelineContext(job, config)

        enabled = enabled_stages or config.enabled_stages or [s.name for s in self.stages]

        # Update job status
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()

        # Get tracer and metrics manager
        tracer = get_tracer("pipeline")
        metrics = get_metrics_manager()

        # Record job start in metrics
        metrics.record_job_created(
            source_type=job.source_type.value,
            priority=str(job.priority),
        )
        metrics.record_job_start(str(job.id))

        job_start_time = time.time()

        # Start pipeline trace span
        with tracer.start_as_current_span(
            name="pipeline.execute",
            attributes={
                "job.id": str(job.id),
                "job.source_type": job.source_type.value,
                "job.file_name": job.file_name,
                "job.file_size": job.file_size,
                "job.priority": job.priority,
            },
        ) as pipeline_span:
            try:
                for stage in self.stages:
                    if stage.name not in enabled:
                        logger.debug("skipping_stage", stage=stage.name, job_id=str(job.id))
                        continue

                    context.current_stage = stage.name
                    job.current_stage = stage.name

                    # Start stage span
                    with start_pipeline_stage_span(
                        stage_name=stage.name,
                        job_id=str(job.id),
                    ) as stage_span:
                        try:
                            logger.info(
                                "stage_started",
                                stage=stage.name,
                                job_id=str(job.id),
                            )

                            # Execute stage with timing
                            with metrics.time_stage(stage.name):
                                result = await stage.execute(context)

                            context.set_stage_result(stage.name, result)
                            await stage.on_success(context, result)

                            # Add result attributes to span
                            if isinstance(result, dict):
                                for key, value in result.items():
                                    if isinstance(value, (str, int, float, bool)):
                                        stage_span.set_attribute(f"stage.result.{key}", value)

                            logger.info(
                                "stage_completed",
                                stage=stage.name,
                                job_id=str(job.id),
                            )

                            # Check for retry after quality stage
                            if stage.name == "quality":
                                quality_result = context.get_stage_result("quality")
                                if quality_result and quality_result.get("should_retry"):
                                    if job.retry_count < config.quality.max_retries:
                                        job.retry_count += 1
                                        job.status = JobStatus.RETRYING

                                        # Record retry in metrics
                                        metrics.record_job_completed(
                                            source_type=job.source_type.value,
                                            status="retrying",
                                            job_id=str(job.id),
                                        )

                                        logger.info(
                                            "job_retrying",
                                            job_id=str(job.id),
                                            attempt=job.retry_count,
                                        )
                                        # Break out and let caller handle retry
                                        break

                        except Exception as e:
                            stage_span.set_attribute("error", True)
                            stage_span.set_attribute("error.message", str(e))
                            stage_span.record_exception(e)

                            logger.error(
                                "stage_execution_failed",
                                stage=stage.name,
                                job_id=str(job.id),
                                error=str(e),
                            )
                            await stage.on_failure(context, e)
                            context.errors.append(f"{stage.name}: {e!s}")
                            raise

                # Mark as completed if no errors
                if not context.errors:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.utcnow()

                    # Record job completion
                    metrics.record_job_completed(
                        source_type=job.source_type.value,
                        status="completed",
                        job_id=str(job.id),
                    )

                    # Record quality score if available
                    quality_result = context.get_stage_result("quality")
                    if quality_result:
                        score = quality_result.get("overall_score")
                        parser_used = context.get_stage_result("parse", {}).get("parser_used", "unknown")
                        if score is not None:
                            metrics.record_quality_score(
                                job_id=str(job.id),
                                parser_used=parser_used,
                                score=score,
                            )

                    # Build job result
                    quality_result = context.get_stage_result("quality")
                    output_result = context.get_stage_result("output")
                    parse_result = context.get_stage_result("parse")

                    job.result = JobResult(
                        success=True,
                        quality_score=quality_result.get("overall_score") if quality_result else None,
                        output_format=output_result.get("output_format") if output_result else None,
                        extracted_text=parse_result.get("text")[:1000] if parse_result else None,
                        metadata={
                            "destinations": output_result.get("destinations", []) if output_result else [],
                        },
                    )

                    # Update pipeline span
                    pipeline_span.set_attribute("job.status", "completed")
                    pipeline_span.set_attribute("job.success", True)

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = {
                    "code": "PIPELINE_ERROR",
                    "message": str(e),
                    "failed_stage": context.current_stage,
                }

                # Record job failure
                metrics.record_job_failed(
                    source_type=job.source_type.value,
                    stage=context.current_stage or "unknown",
                    error_type=type(e).__name__,
                )

                # Update pipeline span
                pipeline_span.set_attribute("job.status", "failed")
                pipeline_span.set_attribute("job.success", False)
                pipeline_span.set_attribute("job.failed_stage", context.current_stage)
                pipeline_span.set_attribute("error", True)
                pipeline_span.set_attribute("error.message", str(e))
                pipeline_span.record_exception(e)

                raise

        return context

    def get_stage(self, name: str) -> PipelineStage | None:
        """Get a stage by name.
        
        Args:
            name: Stage name
            
        Returns:
            PipelineStage or None if not found
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None
