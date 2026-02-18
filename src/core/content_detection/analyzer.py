"""PDF content analysis engine."""

import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
from opentelemetry import trace

from src.core.content_detection.models import (
    ContentAnalysisResult,
    ContentType,
    ImageStatistics,
    PageAnalysis,
    TextStatistics,
)

tracer = trace.get_tracer(__name__)


class TextExtractor:
    """Extracts text statistics from PDF using pdfplumber."""
    
    @staticmethod
    def extract_text_statistics(file_path: Path) -> Tuple[TextStatistics, List[int]]:
        """Extract text statistics from PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (TextStatistics, list of character counts per page)
        """
        total_chars = 0
        font_names = set()
        encodings = set()
        chars_per_page = []
        has_text_layer = False
        
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page in pdf.pages:
                # Extract text
                text = page.extract_text() or ""
                char_count = len(text)
                chars_per_page.append(char_count)
                total_chars += char_count
                
                if char_count > 0:
                    has_text_layer = True
                
                # Extract font information
                for char in page.chars[:1000]:  # Sample first 1000 chars for performance
                    if "fontname" in char:
                        font_names.add(char["fontname"])
                    if "encoding" in char:
                        encodings.add(char["encoding"])
        
        avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
        
        stats = TextStatistics(
            total_pages=total_pages,
            total_characters=total_chars,
            has_text_layer=has_text_layer,
            font_count=len(font_names),
            encoding=next(iter(encodings)) if encodings else None,
            average_chars_per_page=avg_chars_per_page,
        )
        
        return stats, chars_per_page


class ImageAnalyzer:
    """Analyzes image content in PDF using PyMuPDF."""
    
    @staticmethod
    def analyze_images(file_path: Path) -> Tuple[ImageStatistics, List[int], List[float], List[float]]:
        """Analyze images in PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (ImageStatistics, images per page, image ratios per page, text ratios per page)
        """
        doc = fitz.open(file_path)
        
        total_images = 0
        total_page_area = 0.0
        total_image_area = 0.0
        color_pages = 0
        dpis = []
        images_per_page = []
        image_ratios_per_page = []
        text_ratios_per_page = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height
            total_page_area += page_area
            
            # Get images on this page
            image_list = page.get_images(full=True)
            page_image_count = len(image_list)
            images_per_page.append(page_image_count)
            total_images += page_image_count
            
            # Calculate image area for this page
            page_image_area = 0.0
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Calculate area
                if pix.n > 4:  # CMYK: convert to RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                img_area = pix.width * pix.height
                page_image_area += img_area
                
                # Track DPI if available
                if pix.xres > 0 and pix.yres > 0:
                    dpis.append(int((pix.xres + pix.yres) / 2))
                
                # Check if color
                if pix.n >= 3:
                    color_pages += 1
                
                pix = None  # Free memory
            
            total_image_area += page_image_area
            
            # Calculate ratios for this page
            image_ratio = page_image_area / page_area if page_area > 0 else 0.0
            image_ratios_per_page.append(min(image_ratio, 1.0))
            
            # Estimate text ratio (complement of image ratio, adjusted)
            text_ratio = 1.0 - image_ratio if page_image_count == 0 else max(0.0, 0.5 - image_ratio)
            text_ratios_per_page.append(text_ratio)
        
        doc.close()
        
        image_area_ratio = total_image_area / total_page_area if total_page_area > 0 else 0.0
        avg_dpi = int(sum(dpis) / len(dpis)) if dpis else None
        
        stats = ImageStatistics(
            total_images=total_images,
            image_area_ratio=min(image_area_ratio, 1.0),
            average_dpi=avg_dpi,
            color_pages=color_pages,
            total_page_area=total_page_area,
            total_image_area=total_image_area,
        )
        
        return stats, images_per_page, image_ratios_per_page, text_ratios_per_page


class ContentClassifier:
    """Classifies PDF content type based on analysis results."""
    
    # Classification thresholds
    TEXT_RATIO_THRESHOLD = 0.85  # Relaxed from 0.95 for better text detection
    SCANNED_TEXT_MAX = 0.05
    SCANNED_IMAGE_MIN = 0.90
    LOW_CONFIDENCE_THRESHOLD = 0.70
    
    @classmethod
    def classify(
        cls,
        text_stats: TextStatistics,
        image_stats: ImageStatistics,
        chars_per_page: List[int],
        images_per_page: List[int],
    ) -> Tuple[ContentType, float, str, List[str]]:
        """Classify content type and determine recommended parser.
        
        Args:
            text_stats: Text statistics
            image_stats: Image statistics
            chars_per_page: Character count per page
            images_per_page: Image count per page
            
        Returns:
            Tuple of (content_type, confidence, recommended_parser, alternative_parsers)
        """
        # Calculate overall ratios
        total_chars = text_stats.total_characters
        total_area = image_stats.total_page_area
        # Estimate text area: typical page has ~3000 chars, ~600k pt² area
        # So ratio per char is roughly 600k/3000 = 200 pt² per char
        # But we use a simpler heuristic: if has_text_layer and low image ratio -> text-based
        text_area_estimate = total_chars * 200  # Adjusted estimate
        text_ratio = text_area_estimate / total_area if total_area > 0 else 0.0
        image_ratio = image_stats.image_area_ratio
        
        # Apply classification rules
        # Heuristic-based classification using multiple signals
        has_text_layer = text_stats.has_text_layer
        avg_chars_per_page = text_stats.average_chars_per_page
        font_count = text_stats.font_count
        
        # Text-based: has text layer, significant characters, low image ratio
        is_text_based = (
            has_text_layer and 
            avg_chars_per_page > 1000 and 
            image_ratio < 0.15 and
            font_count > 0
        )
        
        # Scanned: very little text, high image ratio
        is_scanned = (
            avg_chars_per_page < 100 and 
            image_ratio > cls.SCANNED_IMAGE_MIN
        )
        
        if is_text_based:
            content_type = ContentType.TEXT_BASED
            confidence = 0.95 + min(0.05, (avg_chars_per_page - 1000) / 50000)
            recommended_parser = "docling"
            alternative_parsers = ["azure_ocr"]
        elif is_scanned:
            content_type = ContentType.SCANNED
            confidence = min(1.0, image_ratio)
            recommended_parser = "azure_ocr"
            alternative_parsers = ["docling"]
        else:
            content_type = ContentType.MIXED
            # Confidence is higher when we're closer to extremes
            # Normalize text_ratio to 0-1 range and calculate distance from center
            normalized_text = min(1.0, text_ratio / 10) if text_ratio > 0 else 0  # text_ratio can be > 1
            confidence = 0.5 + min(0.5, abs(normalized_text - 0.5))
            recommended_parser = "docling"
            alternative_parsers = ["azure_ocr"]
        
        return content_type, confidence, recommended_parser, alternative_parsers
    
    @classmethod
    def classify_page(
        cls,
        page_num: int,
        char_count: int,
        image_count: int,
        text_ratio: float,
        image_ratio: float,
    ) -> PageAnalysis:
        """Classify a single page.
        
        Args:
            page_num: Page number (1-indexed)
            char_count: Character count on page
            image_count: Image count on page
            text_ratio: Text area ratio
            image_ratio: Image area ratio
            
        Returns:
            PageAnalysis for this page
        """
        if text_ratio > cls.TEXT_RATIO_THRESHOLD:
            content_type = ContentType.TEXT_BASED
            confidence = min(1.0, text_ratio)
        elif text_ratio < cls.SCANNED_TEXT_MAX and image_ratio > cls.SCANNED_IMAGE_MIN:
            content_type = ContentType.SCANNED
            confidence = min(1.0, image_ratio)
        else:
            content_type = ContentType.MIXED
            confidence = 0.5 + abs(text_ratio - 0.5)
        
        return PageAnalysis(
            page_number=page_num,
            content_type=content_type,
            confidence=confidence,
            text_ratio=text_ratio,
            image_ratio=image_ratio,
            character_count=char_count,
            image_count=image_count,
        )


class PDFContentAnalyzer:
    """Main analyzer class that orchestrates PDF content detection."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.text_extractor = TextExtractor()
        self.image_analyzer = ImageAnalyzer()
        self.classifier = ContentClassifier()
    
    def analyze(self, file_path: Path) -> ContentAnalysisResult:
        """Analyze PDF content and return detection result.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ContentAnalysisResult with detection details
        """
        with tracer.start_as_current_span("pdf_content_analyzer.analyze") as span:
            start_time = time.time()
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            
            span.set_attribute("pdf.file_hash", file_hash)
            span.set_attribute("pdf.file_size", file_size)
            
            # Extract text statistics
            with tracer.start_as_current_span("pdf.extract_text") as text_span:
                text_stats, chars_per_page = self.text_extractor.extract_text_statistics(file_path)
                text_span.set_attribute("pdf.total_pages", text_stats.total_pages)
                text_span.set_attribute("pdf.total_characters", text_stats.total_characters)
                text_span.set_attribute("pdf.has_text_layer", text_stats.has_text_layer)
            
            # Analyze images
            with tracer.start_as_current_span("pdf.analyze_images") as image_span:
                image_stats, images_per_page, image_ratios, text_ratios = self.image_analyzer.analyze_images(file_path)
                image_span.set_attribute("pdf.total_images", image_stats.total_images)
                image_span.set_attribute("pdf.image_area_ratio", image_stats.image_area_ratio)
            
            # Classify overall content
            with tracer.start_as_current_span("pdf.classify_content") as classify_span:
                content_type, confidence, recommended_parser, alternative_parsers = self.classifier.classify(
                    text_stats, image_stats, chars_per_page, images_per_page
                )
                classify_span.set_attribute("pdf.content_type", content_type.value)
                classify_span.set_attribute("pdf.confidence", confidence)
                classify_span.set_attribute("pdf.recommended_parser", recommended_parser)
            
            # Classify individual pages
            page_results = []
            for i in range(text_stats.total_pages):
                page_num = i + 1
                char_count = chars_per_page[i] if i < len(chars_per_page) else 0
                image_count = images_per_page[i] if i < len(images_per_page) else 0
                text_ratio = text_ratios[i] if i < len(text_ratios) else 0.0
                image_ratio = image_ratios[i] if i < len(image_ratios) else 0.0
                
                page_analysis = self.classifier.classify_page(
                    page_num, char_count, image_count, text_ratio, image_ratio
                )
                page_results.append(page_analysis)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            span.set_attribute("pdf.processing_time_ms", processing_time_ms)
            
            return ContentAnalysisResult(
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
                confidence=confidence,
                recommended_parser=recommended_parser,
                alternative_parsers=alternative_parsers,
                text_statistics=text_stats,
                image_statistics=image_stats,
                page_results=page_results,
                processing_time_ms=processing_time_ms,
            )
    
    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA-256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of SHA-256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def analyze_pdf(file_path: str | Path) -> ContentAnalysisResult:
    """Convenience function to analyze a PDF file.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        ContentAnalysisResult
    """
    analyzer = PDFContentAnalyzer()
    return analyzer.analyze(Path(file_path))
