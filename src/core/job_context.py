"""Job context for pipeline execution."""

from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.core.content_detection.models import ContentAnalysisResult


class JobContext(BaseModel):
    """Context object passed through pipeline stages."""
    
    job_id: UUID = Field(..., description="Job ID")
    file_path: str = Field(..., description="Path to input file")
    file_type: str = Field(..., description="Detected file MIME type")
    
    # Content detection (NEW)
    content_detection_result: Optional[ContentAnalysisResult] = Field(
        default=None,
        description="Content detection result for parser selection"
    )
    
    # Parser selection
    selected_parser: Optional[str] = Field(
        default=None,
        description="Selected primary parser"
    )
    fallback_parser: Optional[str] = Field(
        default=None,
        description="Selected fallback parser"
    )
    
    # Stage results
    stage_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from each pipeline stage"
    )
    
    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline configuration"
    )
    
    # Metadata
    created_at: str = Field(..., description="Creation timestamp")
    priority: int = Field(default=5, description="Job priority (1-10)")
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
    
    def set_detection_result(self, result: ContentAnalysisResult) -> None:
        """Set content detection result.
        
        Args:
            result: Detection result
        """
        self.content_detection_result = result
    
    def get_detection_result(self) -> Optional[ContentAnalysisResult]:
        """Get content detection result.
        
        Returns:
            Detection result if available
        """
        return self.content_detection_result
    
    def set_parser_selection(
        self,
        primary: str,
        fallback: Optional[str] = None
    ) -> None:
        """Set parser selection.
        
        Args:
            primary: Primary parser
            fallback: Optional fallback parser
        """
        self.selected_parser = primary
        self.fallback_parser = fallback
    
    def set_stage_result(self, stage: str, result: Any) -> None:
        """Set result for a pipeline stage.
        
        Args:
            stage: Stage name
            result: Stage result
        """
        self.stage_results[stage] = result
    
    def get_stage_result(self, stage: str) -> Optional[Any]:
        """Get result for a pipeline stage.
        
        Args:
            stage: Stage name
            
        Returns:
            Stage result if available
        """
        return self.stage_results.get(stage)
