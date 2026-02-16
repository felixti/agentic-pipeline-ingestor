"""Content detection module for analyzing documents.

This module provides intelligent content detection for determining
document types, distinguishing between scanned and text-based PDFs,
and recommending optimal parsing strategies.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.api.models import (
    ContentDetectionResult,
    ContentType,
    DetectionMethod,
    TextStatistics,
)

logger = logging.getLogger(__name__)


@dataclass
class PageAnalysis:
    """Analysis results for a single page."""
    page_number: int
    has_text: bool
    text_length: int
    has_images: bool
    image_count: int
    text_ratio: float


class PDFAnalyzer:
    """Analyzer for PDF documents using PyMuPDF."""
    
    def __init__(self) -> None:
        """Initialize the PDF analyzer."""
        self.fitz = None
        self._load_fitz()
    
    def _load_fitz(self) -> None:
        """Lazy load PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
        except ImportError:
            logger.warning("PyMuPDF not installed. PDF analysis will be limited.")
    
    async def analyze(self, file_path: str) -> Tuple[ContentType, List[PageAnalysis], Dict[str, Any]]:
        """Analyze a PDF file to determine content type.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (content_type, page_analyses, metadata)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid PDF
        """
        if not self.fitz:
            raise RuntimeError("PyMuPDF not available for PDF analysis")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            doc = self.fitz.open(file_path)
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {e}") from e
        
        try:
            page_analyses: List[PageAnalysis] = []
            total_text_chars = 0
            total_images = 0
            pages_with_text = 0
            pages_with_images = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                text_length = len(text.strip())
                has_text = text_length > 0
                
                # Count images
                image_list = page.get_images()
                image_count = len(image_list)
                has_images = image_count > 0
                
                # Calculate text ratio (text chars vs page area)
                page_area = page.rect.width * page.rect.height
                # Approximate: assume ~0.01 chars per unit area for dense text
                expected_chars = page_area * 0.01
                text_ratio = min(text_length / expected_chars, 1.0) if expected_chars > 0 else 0.0
                
                analysis = PageAnalysis(
                    page_number=page_num + 1,
                    has_text=has_text,
                    text_length=text_length,
                    has_images=has_images,
                    image_count=image_count,
                    text_ratio=text_ratio,
                )
                page_analyses.append(analysis)
                
                # Aggregate statistics
                total_text_chars += text_length
                total_images += image_count
                if has_text:
                    pages_with_text += 1
                if has_images:
                    pages_with_images += 1
            
            # Determine content type
            content_type = self._classify_content(
                page_analyses, total_text_chars, total_images
            )
            
            metadata = {
                "page_count": len(doc),
                "total_text_chars": total_text_chars,
                "total_images": total_images,
                "pages_with_text": pages_with_text,
                "pages_with_images": pages_with_images,
                "has_text_layer": pages_with_text > 0,
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
            }
            
            return content_type, page_analyses, metadata
            
        finally:
            doc.close()
    
    def _classify_content(
        self,
        page_analyses: List[PageAnalysis],
        total_text_chars: int,
        total_images: int,
    ) -> ContentType:
        """Classify the content type based on analysis.
        
        Decision matrix from spec:
        - Text ratio > 95% → Text-based PDF → Docling
        - Text ratio < 5%, images > 90% → Scanned PDF → Azure OCR
        - Mixed → Mixed PDF → Docling with OCR fallback
        
        Args:
            page_analyses: List of page analyses
            total_text_chars: Total text characters
            total_images: Total image count
            
        Returns:
            ContentType classification
        """
        if not page_analyses:
            return ContentType.UNKNOWN
        
        total_pages = len(page_analyses)
        pages_with_text = sum(1 for p in page_analyses if p.has_text)
        pages_with_images = sum(1 for p in page_analyses if p.has_images)
        
        text_ratio = pages_with_text / total_pages if total_pages > 0 else 0
        image_ratio = pages_with_images / total_pages if total_pages > 0 else 0
        
        # Decision logic
        if text_ratio > 0.95:
            return ContentType.TEXT_BASED_PDF
        elif text_ratio < 0.05 and image_ratio > 0.90:
            return ContentType.SCANNED_PDF
        elif text_ratio > 0 and image_ratio > 0:
            return ContentType.MIXED_PDF
        elif total_text_chars > 0:
            return ContentType.TEXT_BASED_PDF
        elif total_images > 0:
            return ContentType.SCANNED_PDF
        else:
            return ContentType.UNKNOWN


class ImageAnalyzer:
    """Analyzer for image files."""
    
    def __init__(self) -> None:
        """Initialize the image analyzer."""
        self.pil = None
        self._load_pil()
    
    def _load_pil(self) -> None:
        """Lazy load PIL."""
        try:
            from PIL import Image
            self.pil = Image
        except ImportError:
            logger.warning("Pillow not installed. Image analysis will be limited.")
    
    async def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with image metadata
        """
        if not self.pil:
            raise RuntimeError("Pillow not available for image analysis")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with self.pil.open(file_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                    "has_alpha": img.mode in ("RGBA", "LA", "PA"),
                    "is_grayscale": img.mode in ("L", "LA", "1"),
                }
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return {"error": str(e)}


class ContentDetector:
    """Main content detector for all document types.
    
    This class provides intelligent content detection for:
    - PDFs: Distinguishing scanned vs. text-based
    - Images: Format and content analysis
    - Office documents: Type detection
    
    Example:
        >>> detector = ContentDetector()
        >>> result = await detector.detect("/path/to/document.pdf")
        >>> print(result.detected_type)  # "text_based_pdf"
        >>> print(result.recommended_parser)  # "docling"
    """
    
    # MIME type to content type mapping
    MIME_TYPE_MAP = {
        "application/pdf": ContentType.TEXT_BASED_PDF,  # Default, will be refined
        "image/jpeg": ContentType.IMAGE,
        "image/png": ContentType.IMAGE,
        "image/tiff": ContentType.IMAGE,
        "image/bmp": ContentType.IMAGE,
        "image/gif": ContentType.IMAGE,
        "image/webp": ContentType.IMAGE,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ContentType.OFFICE_DOC,
        "application/msword": ContentType.OFFICE_DOC,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ContentType.OFFICE_SPREADSHEET,
        "application/vnd.ms-excel": ContentType.OFFICE_SPREADSHEET,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ContentType.OFFICE_PRESENTATION,
        "application/vnd.ms-powerpoint": ContentType.OFFICE_PRESENTATION,
        "text/csv": ContentType.CSV,
        "application/zip": ContentType.ARCHIVE,
        "application/x-zip-compressed": ContentType.ARCHIVE,
    }
    
    # Parser recommendations by content type
    PARSER_RECOMMENDATIONS = {
        ContentType.TEXT_BASED_PDF: ("docling", ["azure_ocr"]),
        ContentType.SCANNED_PDF: ("azure_ocr", ["docling"]),
        ContentType.MIXED_PDF: ("docling", ["azure_ocr"]),
        ContentType.OFFICE_DOC: ("docling", []),
        ContentType.OFFICE_SPREADSHEET: ("docling", []),
        ContentType.OFFICE_PRESENTATION: ("docling", []),
        ContentType.IMAGE: ("azure_ocr", ["docling"]),
        ContentType.CSV: ("docling", []),
        ContentType.ARCHIVE: ("docling", []),
        ContentType.UNKNOWN: ("docling", ["azure_ocr"]),
    }
    
    def __init__(self) -> None:
        """Initialize the content detector."""
        self.pdf_analyzer = PDFAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self._magic = None
        self._load_magic()
    
    def _load_magic(self) -> None:
        """Try to load python-magic for MIME type detection."""
        try:
            import magic
            self._magic = magic
        except ImportError:
            logger.debug("python-magic not installed. Using extension-based detection.")
    
    async def detect(
        self,
        file_path: str,
        mime_type: Optional[str] = None,
    ) -> ContentDetectionResult:
        """Detect content type and recommend parser.
        
        Args:
            file_path: Path to the file to analyze
            mime_type: Optional MIME type hint
            
        Returns:
            ContentDetectionResult with detection results
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine MIME type if not provided
        detected_mime = mime_type or self._detect_mime_type(file_path)
        
        # Analyze based on file type
        if detected_mime == "application/pdf":
            return await self._analyze_pdf(file_path)
        elif detected_mime and detected_mime.startswith("image/"):
            return await self._analyze_image(file_path, detected_mime)
        else:
            return self._analyze_by_mime(detected_mime, file_path)
    
    def _detect_mime_type(self, file_path: str) -> Optional[str]:
        """Detect MIME type of file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string or None
        """
        if self._magic:
            try:
                return self._magic.from_file(file_path, mime=True)
            except Exception as e:
                logger.warning(f"Failed to detect MIME type: {e}")
        
        # Fallback to extension-based detection
        path = Path(file_path)
        extension = path.suffix.lower()
        
        EXTENSION_MAP = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".csv": "text/csv",
            ".zip": "application/zip",
        }
        
        return EXTENSION_MAP.get(extension)
    
    async def _analyze_pdf(self, file_path: str) -> ContentDetectionResult:
        """Analyze a PDF file in detail.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ContentDetectionResult
        """
        try:
            content_type, page_analyses, metadata = await self.pdf_analyzer.analyze(file_path)
            
            # Calculate text statistics
            total_chars = metadata.get("total_text_chars", 0)
            page_count = metadata.get("page_count", 0)
            avg_chars = total_chars // page_count if page_count > 0 else 0
            
            text_stats = TextStatistics(
                total_characters=total_chars,
                total_words=total_chars // 5,  # Rough estimate
                text_ratio=metadata.get("pages_with_text", 0) / page_count if page_count > 0 else 0.0,
                avg_chars_per_page=avg_chars,
            )
            
            # Get parser recommendation
            primary_parser, alternative_parsers = self.PARSER_RECOMMENDATIONS.get(
                content_type, ("docling", ["azure_ocr"])
            )
            
            # Calculate confidence based on analysis quality
            confidence = self._calculate_confidence(content_type, page_analyses, metadata)
            
            return ContentDetectionResult(
                detected_type=content_type,
                confidence=confidence,
                detection_method=DetectionMethod.HEURISTIC,
                page_count=page_count,
                has_text_layer=metadata.get("has_text_layer", False),
                has_images=metadata.get("total_images", 0) > 0,
                image_count=metadata.get("total_images", 0),
                text_statistics=text_stats,
                recommended_parser=primary_parser,
                alternative_parsers=alternative_parsers,
                preprocessing_required=False,
            )
            
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            # Return conservative default
            return ContentDetectionResult(
                detected_type=ContentType.TEXT_BASED_PDF,
                confidence=0.5,
                detection_method=DetectionMethod.HEURISTIC,
                recommended_parser="docling",
                alternative_parsers=["azure_ocr"],
                preprocessing_required=False,
            )
    
    async def _analyze_image(
        self,
        file_path: str,
        mime_type: str,
    ) -> ContentDetectionResult:
        """Analyze an image file.
        
        Args:
            file_path: Path to the image file
            mime_type: Detected MIME type
            
        Returns:
            ContentDetectionResult
        """
        try:
            image_info = await self.image_analyzer.analyze(file_path)
            
            primary_parser, alternative_parsers = self.PARSER_RECOMMENDATIONS.get(
                ContentType.IMAGE, ("azure_ocr", ["docling"])
            )
            
            return ContentDetectionResult(
                detected_type=ContentType.IMAGE,
                confidence=0.95,
                detection_method=DetectionMethod.HEURISTIC,
                page_count=1,
                has_text_layer=False,
                has_images=True,
                image_count=1,
                text_statistics=TextStatistics(),
                recommended_parser=primary_parser,
                alternative_parsers=alternative_parsers,
                preprocessing_required=image_info.get("width", 0) > 4000 or image_info.get("height", 0) > 4000,
            )
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return ContentDetectionResult(
                detected_type=ContentType.IMAGE,
                confidence=0.7,
                detection_method=DetectionMethod.HEURISTIC,
                recommended_parser="azure_ocr",
                alternative_parsers=["docling"],
                preprocessing_required=False,
            )
    
    def _analyze_by_mime(
        self,
        mime_type: Optional[str],
        file_path: str,
    ) -> ContentDetectionResult:
        """Analyze file based on MIME type.
        
        Args:
            mime_type: Detected MIME type
            file_path: Path to the file
            
        Returns:
            ContentDetectionResult
        """
        content_type = self.MIME_TYPE_MAP.get(mime_type, ContentType.UNKNOWN)
        
        primary_parser, alternative_parsers = self.PARSER_RECOMMENDATIONS.get(
            content_type, ("docling", ["azure_ocr"])
        )
        
        return ContentDetectionResult(
            detected_type=content_type,
            confidence=0.8 if mime_type else 0.5,
            detection_method=DetectionMethod.HEURISTIC,
            recommended_parser=primary_parser,
            alternative_parsers=alternative_parsers,
            preprocessing_required=False,
        )
    
    def _calculate_confidence(
        self,
        content_type: ContentType,
        page_analyses: List[PageAnalysis],
        metadata: Dict[str, Any],
    ) -> float:
        """Calculate confidence score for detection.
        
        Args:
            content_type: Detected content type
            page_analyses: List of page analyses
            metadata: Document metadata
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not page_analyses:
            return 0.5
        
        # Base confidence
        confidence = 0.9
        
        # Reduce confidence if few pages
        if len(page_analyses) < 3:
            confidence -= 0.1
        
        # Check for edge cases
        pages_with_text = sum(1 for p in page_analyses if p.has_text)
        pages_with_images = sum(1 for p in page_analyses if p.has_images)
        total_pages = len(page_analyses)
        
        # Mixed signals reduce confidence
        text_ratio = pages_with_text / total_pages
        image_ratio = pages_with_images / total_pages
        
        if 0.1 < text_ratio < 0.9 and 0.1 < image_ratio < 0.9:
            # Mixed content - lower confidence
            confidence -= 0.15
        
        # Very short documents have lower confidence
        total_chars = metadata.get("total_text_chars", 0)
        if total_chars < 100:
            confidence -= 0.1
        
        return max(0.5, min(1.0, confidence))


# Convenience function for simple use cases
async def detect_content(
    file_path: str,
    mime_type: Optional[str] = None,
) -> ContentDetectionResult:
    """Detect content type of a file.
    
    Args:
        file_path: Path to the file
        mime_type: Optional MIME type hint
        
    Returns:
        ContentDetectionResult
    """
    detector = ContentDetector()
    return await detector.detect(file_path, mime_type)
