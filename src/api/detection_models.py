"""API request/response models for content detection endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

from src.core.content_detection.models import (
    ContentAnalysisResult,
    ContentType,
    ImageStatistics,
    PageAnalysis,
    TextStatistics,
)


class DetectionRequest(BaseModel):
    """Request model for single file detection."""
    
    detailed: bool = Field(
        default=False,
        description="Include detailed per-page analysis in response"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "detailed": True
            }
        }


class DetectionUrlRequest(BaseModel):
    """Request model for URL-based detection."""
    
    url: HttpUrl = Field(..., description="URL of PDF file to analyze")
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional HTTP headers for download"
    )
    detailed: bool = Field(
        default=False,
        description="Include detailed per-page analysis in response"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "url": "https://example.com/document.pdf",
                "headers": {
                    "Authorization": "Bearer token"
                },
                "detailed": True
            }
        }


class BatchDetectionRequest(BaseModel):
    """Request model for batch detection."""
    
    detailed: bool = Field(
        default=False,
        description="Include detailed per-page analysis in response"
    )
    
    @field_validator('detailed')
    @classmethod
    def validate_detailed(cls, v):
        """Validate detailed flag."""
        return v


class DetectionResponseData(BaseModel):
    """Detection result data in API response."""
    
    content_type: ContentType = Field(..., description="Detected content type")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    recommended_parser: str = Field(..., description="Recommended parser")
    alternative_parsers: List[str] = Field(default_factory=list)
    text_statistics: TextStatistics = Field(...)
    image_statistics: ImageStatistics = Field(...)
    page_results: Optional[List[PageAnalysis]] = Field(
        default=None,
        description="Per-page results (only if detailed=True)"
    )
    processing_time_ms: int = Field(...)
    file_hash: str = Field(..., description="SHA-256 hash of analyzed file")
    file_size: int = Field(..., description="File size in bytes")


class DetectionResponse(BaseModel):
    """Standard API response wrapper for detection."""
    
    data: DetectionResponseData = Field(...)
    meta: Dict[str, Any] = Field(...)


class BatchDetectionItem(BaseModel):
    """Single item in batch detection response."""
    
    filename: str = Field(..., description="Original filename")
    content_type: ContentType = Field(...)
    confidence: float = Field(...)
    recommended_parser: str = Field(...)
    processing_time_ms: int = Field(...)
    error: Optional[str] = Field(None, description="Error message if analysis failed")


class BatchDetectionResponseData(BaseModel):
    """Batch detection response data."""
    
    results: List[BatchDetectionItem] = Field(...)
    summary: Dict[str, Any] = Field(...)


class BatchDetectionResponse(BaseModel):
    """API response wrapper for batch detection."""
    
    data: BatchDetectionResponseData = Field(...)
    meta: Dict[str, Any] = Field(...)


class ParserSelectionResponseData(BaseModel):
    """Parser selection response data."""
    
    job_id: str = Field(...)
    detection_result: Optional[Dict[str, Any]] = Field(None)
    selection: Dict[str, Any] = Field(...)
    estimated_processing_time: str = Field(...)
    estimated_cost: float = Field(...)


class ParserSelectionResponse(BaseModel):
    """API response for parser selection endpoint."""
    
    data: ParserSelectionResponseData = Field(...)
    meta: Dict[str, Any] = Field(...)


def create_meta(request_id: str) -> Dict[str, Any]:
    """Create standard response metadata.
    
    Args:
        request_id: Unique request ID
        
    Returns:
        Metadata dictionary
    """
    from datetime import datetime, timezone
    
    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_version": "v1"
    }


def convert_analysis_result_to_response(
    result: ContentAnalysisResult,
    detailed: bool = False
) -> DetectionResponseData:
    """Convert analysis result to API response format.
    
    Args:
        result: Content analysis result
        detailed: Whether to include per-page details
        
    Returns:
        Detection response data
    """
    return DetectionResponseData(
        content_type=result.content_type,
        confidence=result.confidence,
        recommended_parser=result.recommended_parser,
        alternative_parsers=result.alternative_parsers,
        text_statistics=result.text_statistics,
        image_statistics=result.image_statistics,
        page_results=result.page_results if detailed else None,
        processing_time_ms=result.processing_time_ms,
        file_hash=result.file_hash,
        file_size=result.file_size,
    )
