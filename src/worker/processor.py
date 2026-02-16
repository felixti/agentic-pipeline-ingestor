"""Job processor for the worker service.

This module handles the actual processing of jobs from the queue,
including pipeline execution and error handling.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.api.models import Job, JobStatus, StageProgress, StageStatus
from src.config import settings
from src.core.engine import OrchestrationEngine
from src.core.pipeline import PipelineExecutor
from src.llm.provider import LLMProvider, load_llm_config
from src.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class JobProcessor:
    """Processor for pipeline jobs.
    
    Handles job retrieval, pipeline execution, and result handling.
    
    Example:
        >>> processor = JobProcessor()
        >>> await processor.start()
        >>> await processor.process_job(job_id)
    """
    
    def __init__(
        self,
        engine: Optional[OrchestrationEngine] = None,
        plugin_registry: Optional[PluginRegistry] = None,
    ) -> None:
        """Initialize the job processor.
        
        Args:
            engine: Orchestration engine
            plugin_registry: Plugin registry
        """
        self.engine = engine
        self.registry = plugin_registry or PluginRegistry()
        self.llm: Optional[LLMProvider] = None
        self._running = False
        self._current_job: Optional[UUID] = None
    
    async def initialize(self) -> None:
        """Initialize the processor with dependencies."""
        logger.info("Initializing job processor...")
        
        # Initialize plugins
        await self._initialize_plugins()
        
        # Initialize LLM if configured
        await self._initialize_llm()
        
        # Initialize engine if not provided
        if self.engine is None:
            self.engine = OrchestrationEngine(
                plugin_registry=self.registry,
                llm_provider=self.llm,
            )
        
        logger.info("Job processor initialized")
    
    async def _initialize_plugins(self) -> None:
        """Initialize all plugins from registry."""
        # Register built-in parsers
        try:
            from src.plugins.parsers.docling_parser import DoclingParser
            from src.plugins.parsers.azure_ocr_parser import AzureOCRParser
            
            docling = DoclingParser()
            await docling.initialize({})
            self.registry.register_parser(docling)
            
            azure_ocr = AzureOCRParser()
            await azure_ocr.initialize({
                "endpoint": getattr(settings, 'AZURE_AI_VISION_ENDPOINT', None),
                "api_key": getattr(settings, 'AZURE_AI_VISION_API_KEY', None),
            })
            self.registry.register_parser(azure_ocr)
            
            logger.info("Registered parser plugins")
            
        except Exception as e:
            logger.warning(f"Failed to register some parser plugins: {e}")
        
        # Register built-in destinations
        try:
            from src.plugins.destinations.cognee import CogneeDestination
            
            cognee = CogneeDestination()
            await cognee.initialize({
                "api_url": getattr(settings, 'COGNEE_API_URL', None),
                "api_key": getattr(settings, 'COGNEE_API_KEY', None),
            })
            self.registry.register_destination(cognee)
            
            logger.info("Registered destination plugins")
            
        except Exception as e:
            logger.warning(f"Failed to register some destination plugins: {e}")
    
    async def _initialize_llm(self) -> None:
        """Initialize LLM provider if configured."""
        try:
            config = load_llm_config()
            if config.routers:
                self.llm = LLMProvider(config)
                logger.info("LLM provider initialized")
            else:
                logger.warning("No LLM routers configured")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM provider: {e}")
    
    async def process_job(self, job_id: UUID) -> Dict[str, Any]:
        """Process a single job.
        
        Args:
            job_id: ID of the job to process
            
        Returns:
            Processing result
        """
        if not self.engine:
            raise RuntimeError("Processor not initialized")
        
        self._current_job = job_id
        
        try:
            logger.info("processing_job", job_id=str(job_id))
            
            # Execute pipeline
            context = await self.engine.process_job(job_id)
            
            # Get job to check final status
            job = await self.engine.get_job(job_id)
            
            result = {
                "job_id": str(job_id),
                "status": job.status.value if job else "unknown",
                "stages_completed": list(context.stage_results.keys()),
                "success": job.status == JobStatus.COMPLETED if job else False,
            }
            
            # Add quality score if available
            quality_result = context.get_stage_result("quality")
            if quality_result:
                result["quality_score"] = quality_result.get("overall_score")
            
            logger.info(
                "job_processing_finished",
                job_id=str(job_id),
                status=result["status"],
                success=result["success"],
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "job_processing_failed",
                job_id=str(job_id),
                error=str(e),
            )
            
            # Update job status to failed
            try:
                await self.engine.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error={"code": "PROCESSING_ERROR", "message": str(e)},
                )
            except Exception as update_error:
                logger.error(
                    "failed_to_update_job_status",
                    job_id=str(job_id),
                    error=str(update_error),
                )
            
            return {
                "job_id": str(job_id),
                "status": "failed",
                "error": str(e),
                "success": False,
            }
        
        finally:
            self._current_job = None
    
    async def process_job_with_retry(
        self,
        job_id: UUID,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Process a job with automatic retry.
        
        Args:
            job_id: Job ID
            max_retries: Maximum retry attempts
            
        Returns:
            Processing result
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await self.process_job(job_id)
                
                if result.get("success"):
                    return result
                
                # Check if we should retry
                if attempt < max_retries:
                    job = await self.engine.get_job(job_id)
                    if job and job.status == JobStatus.RETRYING:
                        logger.info(
                            "retrying_job",
                            job_id=str(job_id),
                            attempt=attempt + 1,
                        )
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                
                return result
                
            except Exception as e:
                last_error = e
                logger.error(
                    "job_attempt_failed",
                    job_id=str(job_id),
                    attempt=attempt,
                    error=str(e),
                )
                
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
        
        # All retries exhausted
        return {
            "job_id": str(job_id),
            "status": "failed",
            "error": f"All retries failed: {last_error}",
            "success": False,
        }
    
    async def shutdown(self) -> None:
        """Shutdown the processor and cleanup resources."""
        logger.info("Shutting down job processor...")
        
        self._running = False
        
        # Shutdown plugins
        for plugin in self.registry._parsers.values():
            try:
                await plugin.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down parser: {e}")
        
        for plugin in self.registry._destinations.values():
            try:
                await plugin.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down destination: {e}")
        
        logger.info("Job processor shutdown complete")
    
    @property
    def is_processing(self) -> bool:
        """Check if processor is currently processing a job."""
        return self._current_job is not None
    
    @property
    def current_job_id(self) -> Optional[UUID]:
        """Get the ID of the currently processing job."""
        return self._current_job
