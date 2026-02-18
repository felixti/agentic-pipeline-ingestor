"""Pipeline execution with content detection integration."""

from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

import structlog

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
    """Stage 4: Parse document (placeholder)."""
    
    name = "parse"
    
    async def execute(self, context: JobContext) -> JobContext:
        """Execute parse stage."""
        log = logger.bind(stage=self.name, job_id=str(context.job_id))
        log.info("pipeline.stage_started")
        
        # This is a placeholder - actual parsing would use the selected parser
        primary = context.selected_parser or "docling"
        
        log.info("pipeline.using_parser", parser=primary)
        
        context.set_stage_result(self.name, {
            "parser_used": primary,
            "status": "completed"
        })
        
        log.info("pipeline.stage_completed")
        return context


class Pipeline:
    """Pipeline executor."""
    
    STAGES = [
        IngestStage,
        DetectStage,         # NEW: Content detection
        SelectParserStage,   # MODIFIED: Uses detection result
        ParseStage,
        # Additional stages would be added here:
        # EnrichStage,
        # QualityStage,
        # TransformStage,
        # OutputStage,
    ]
    
    def __init__(self, detection_service: Optional[ContentDetectionService] = None):
        """Initialize pipeline.
        
        Args:
            detection_service: Optional detection service
        """
        self.stages = []
        for stage_class in self.STAGES:
            if stage_class == DetectStage:
                self.stages.append(DetectStage(detection_service))
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
    ):
        """Initialize pipeline executor.
        
        Args:
            config: Pipeline configuration
            plugin_registry: Plugin registry for parsers/destinations
            detection_service: Optional content detection service
        """
        self.config = config
        self.plugin_registry = plugin_registry
        self.detection_service = detection_service
        self.pipeline = Pipeline(detection_service=detection_service)
    
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
