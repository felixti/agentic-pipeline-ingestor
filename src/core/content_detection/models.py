"""Data models for content detection."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Content type classification for PDFs."""
    
    TEXT_BASED = "text_based"
    SCANNED = "scanned"
    MIXED = "mixed"


class TextStatistics(BaseModel):
    """Statistics about text content in a PDF."""
    
    total_pages: int = Field(..., description="Total number of pages")
    total_characters: int = Field(..., description="Total character count")
    has_text_layer: bool = Field(..., description="Whether PDF has extractable text layer")
    font_count: int = Field(..., description="Number of unique fonts")
    encoding: Optional[str] = Field(None, description="Text encoding detected")
    average_chars_per_page: float = Field(..., description="Average characters per page")


class ImageStatistics(BaseModel):
    """Statistics about image content in a PDF."""
    
    total_images: int = Field(..., description="Total number of images")
    image_area_ratio: float = Field(..., description="Ratio of image area to page area", ge=0.0, le=1.0)
    average_dpi: Optional[int] = Field(None, description="Average DPI of images")
    color_pages: int = Field(..., description="Number of pages with color images")
    total_page_area: float = Field(..., description="Total area of all pages in points²")
    total_image_area: float = Field(..., description="Total area of all images in points²")


class PageAnalysis(BaseModel):
    """Analysis results for a single PDF page."""
    
    page_number: int = Field(..., description="Page number (1-indexed)")
    content_type: ContentType = Field(..., description="Detected content type for this page")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)
    text_ratio: float = Field(..., description="Ratio of text area to page area", ge=0.0, le=1.0)
    image_ratio: float = Field(..., description="Ratio of image area to page area", ge=0.0, le=1.0)
    character_count: int = Field(..., description="Number of characters on this page")
    image_count: int = Field(..., description="Number of images on this page")


class ContentAnalysisResult(BaseModel):
    """Complete content analysis result for a PDF."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    file_hash: str = Field(..., description="SHA-256 hash of file content")
    file_size: int = Field(..., description="File size in bytes")
    content_type: ContentType = Field(..., description="Overall content type classification")
    confidence: float = Field(..., description="Overall confidence score (0.0-1.0)", ge=0.0, le=1.0)
    recommended_parser: str = Field(..., description="Recommended parser for this content")
    alternative_parsers: List[str] = Field(default_factory=list, description="Alternative parsers")
    text_statistics: TextStatistics = Field(..., description="Text content statistics")
    image_statistics: ImageStatistics = Field(..., description="Image content statistics")
    page_results: List[PageAnalysis] = Field(..., description="Per-page analysis results")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    
    class Config:
        """Pydantic config."""
        
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "file_hash": "abc123...",
                "file_size": 1048576,
                "content_type": "text_based",
                "confidence": 0.985,
                "recommended_parser": "docling",
                "alternative_parsers": ["azure_ocr"],
                "processing_time_ms": 245,
            }
        }


# Database model for persistence
class ContentDetectionRecord(BaseModel):
    """Database record for content detection results."""
    
    id: UUID = Field(default_factory=uuid4)
    file_hash: str = Field(..., description="SHA-256 hash (unique key)")
    file_size: int = Field(...)
    content_type: ContentType = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommended_parser: str = Field(...)
    alternative_parsers: List[str] = Field(default_factory=list)
    text_statistics: TextStatistics = Field(...)
    image_statistics: ImageStatistics = Field(...)
    page_results: List[PageAnalysis] = Field(...)
    processing_time_ms: int = Field(...)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None)
    access_count: int = Field(default=1)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
