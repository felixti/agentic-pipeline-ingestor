"""Azure OCR parser plugin for document processing.

This module provides the Azure AI Vision OCR integration for extracting
text from scanned documents, images, and PDFs with poor text layers.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.plugins.base import (
    HealthStatus,
    ParsingResult,
    PluginMetadata,
    PluginType,
    SupportResult,
)
from src.plugins.base import ParserPlugin

logger = logging.getLogger(__name__)


class AzureOCRParser(ParserPlugin):
    """Azure AI Vision OCR parser plugin.
    
    Azure OCR is the fallback parser for scanned documents and images
    where text-based extraction fails. It uses Microsoft's Azure AI
    Vision Read API for high-quality OCR.
    
    Example:
        >>> parser = AzureOCRParser()
        >>> await parser.initialize({
        ...     "endpoint": "https://my-resource.cognitiveservices.azure.com",
        ...     "api_key": "my-api-key"
        ... })
        >>> result = await parser.parse("/path/to/scanned_document.pdf")
        >>> print(result.text)
    """
    
    SUPPORTED_FORMATS = [
        ".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif",
    ]
    
    MIME_TYPE_MAP = {
        "application/pdf": 0.95,
        "image/jpeg": 0.98,
        "image/png": 0.98,
        "image/tiff": 0.95,
        "image/bmp": 0.90,
        "image/gif": 0.85,
    }
    
    def __init__(self) -> None:
        """Initialize the Azure OCR parser."""
        self._client = None
        self._endpoint: Optional[str] = None
        self._api_key: Optional[str] = None
        self._config: Dict[str, Any] = {}
        self._azure_available = False
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="azure_ocr",
            name="Azure AI Vision OCR",
            version="1.0.0",
            type=PluginType.PARSER,
            description="OCR parser using Azure AI Vision Read API for scanned documents",
            author="Pipeline Team",
            supported_formats=self.SUPPORTED_FORMATS,
            requires_auth=True,
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the parser with configuration.
        
        Args:
            config: Parser configuration with:
                - endpoint: Azure AI Vision endpoint
                - api_key: Azure AI Vision API key
                - language: OCR language (default: "en")
        """
        self._config = config
        
        # Get credentials from config or environment
        self._endpoint = config.get("endpoint") or os.getenv("AZURE_AI_VISION_ENDPOINT")
        self._api_key = config.get("api_key") or os.getenv("AZURE_AI_VISION_API_KEY")
        
        # Try to import Azure SDK
        try:
            from azure.ai.vision.imageanalysis import ImageAnalysisClient
            from azure.ai.vision.imageanalysis.models import VisualFeatures
            from azure.core.credentials import AzureKeyCredential
            
            self._azure_available = True
            
            # Initialize client if credentials available
            if self._endpoint and self._api_key:
                self._client = ImageAnalysisClient(
                    endpoint=self._endpoint,
                    credential=AzureKeyCredential(self._api_key),
                )
                logger.info("Azure OCR parser initialized successfully")
            else:
                logger.warning(
                    "Azure OCR credentials not configured. "
                    "Set AZURE_AI_VISION_ENDPOINT and AZURE_AI_VISION_API_KEY "
                    "or pass in config."
                )
                
        except ImportError:
            logger.warning(
                "Azure AI Vision SDK not installed. "
                "Install with: pip install azure-ai-vision"
            )
            self._azure_available = False
    
    async def supports(
        self,
        file_path: str,
        mime_type: Optional[str] = None,
    ) -> SupportResult:
        """Check if this parser supports the given file.
        
        Args:
            file_path: Path to the file
            mime_type: Optional MIME type of the file
            
        Returns:
            SupportResult indicating support status
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Check extension
        if extension in self.SUPPORTED_FORMATS:
            confidence = 0.95
            if extension in (".pdf",):
                # PDFs need conversion to images first
                confidence = 0.85
            return SupportResult(
                supported=True,
                confidence=confidence,
                reason=f"Supported extension: {extension}",
            )
        
        # Check MIME type if provided
        if mime_type and mime_type in self.MIME_TYPE_MAP:
            return SupportResult(
                supported=True,
                confidence=self.MIME_TYPE_MAP[mime_type],
                reason=f"Supported MIME type: {mime_type}",
            )
        
        return SupportResult(
            supported=False,
            confidence=1.0,
            reason=f"Unsupported file format: {extension}",
        )
    
    async def parse(
        self,
        file_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> ParsingResult:
        """Parse a document using Azure OCR.
        
        Args:
            file_path: Path to the file to parse
            options: Parser-specific options:
                - language: OCR language code (default: "en")
                - pages: List of page numbers to process (for PDFs)
                - output_format: "text" or "structured"
            
        Returns:
            ParsingResult containing extracted content
        """
        import time
        
        options = options or {}
        start_time = time.time()
        
        path = Path(file_path)
        if not path.exists():
            return ParsingResult(
                success=False,
                error=f"File not found: {file_path}",
            )
        
        # Check support
        support = await self.supports(file_path)
        if not support.supported:
            return ParsingResult(
                success=False,
                error=support.reason,
            )
        
        # Check if Azure is available
        if not self._azure_available:
            return await self._parse_fallback(file_path, options)
        
        try:
            extension = path.suffix.lower()
            
            if extension == ".pdf":
                result = await self._parse_pdf(file_path, options)
            else:
                result = await self._parse_image(file_path, options)
            
            # Add processing time
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result
            
        except Exception as e:
            logger.error(f"Azure OCR parsing failed: {e}", exc_info=True)
            # Try fallback
            return await self._parse_fallback(file_path, options)
    
    async def _parse_pdf(
        self,
        file_path: str,
        options: Dict[str, Any],
    ) -> ParsingResult:
        """Parse PDF using Azure OCR.
        
        For PDFs, we convert pages to images and OCR each page.
        
        Args:
            file_path: Path to the PDF file
            options: Parser options
            
        Returns:
            ParsingResult
        """
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import io
            
            doc = fitz.open(file_path)
            all_text_parts: List[str] = []
            pages: List[str] = []
            total_confidence = 0.0
            
            # Get target pages
            target_pages = options.get("pages", range(len(doc)))
            if isinstance(target_pages, list):
                target_pages = [p - 1 for p in target_pages]  # Convert to 0-indexed
            
            for page_num in target_pages:
                if page_num >= len(doc):
                    break
                
                page = doc[page_num]
                
                # Render page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Save to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                
                # OCR the image
                page_result = await self._ocr_image_bytes(img_bytes.getvalue(), options)
                
                if page_result.success:
                    pages.append(page_result.text)
                    all_text_parts.append(page_result.text)
                    total_confidence += page_result.confidence
                else:
                    pages.append("")
                
                pix = None  # Free memory
            
            doc.close()
            
            full_text = "\n\n".join(all_text_parts)
            avg_confidence = total_confidence / len(pages) if pages else 0.0
            
            return ParsingResult(
                success=bool(full_text.strip()),
                text=full_text,
                pages=pages,
                format=".pdf",
                parser_used="azure_ocr",
                confidence=avg_confidence,
            )
            
        except ImportError as e:
            return ParsingResult(
                success=False,
                error=f"PDF processing requires PyMuPDF and Pillow: {e}",
            )
        except Exception as e:
            return ParsingResult(
                success=False,
                error=f"PDF OCR failed: {e}",
            )
    
    async def _parse_image(
        self,
        file_path: str,
        options: Dict[str, Any],
    ) -> ParsingResult:
        """Parse image using Azure OCR.
        
        Args:
            file_path: Path to the image file
            options: Parser options
            
        Returns:
            ParsingResult
        """
        try:
            with open(file_path, "rb") as f:
                image_data = f.read()
            
            return await self._ocr_image_bytes(image_data, options)
            
        except Exception as e:
            return ParsingResult(
                success=False,
                error=f"Image OCR failed: {e}",
            )
    
    async def _ocr_image_bytes(
        self,
        image_data: bytes,
        options: Dict[str, Any],
    ) -> ParsingResult:
        """Perform OCR on image bytes using Azure AI Vision.
        
        Args:
            image_data: Raw image bytes
            options: OCR options
            
        Returns:
            ParsingResult
        """
        if not self._client:
            return ParsingResult(
                success=False,
                error="Azure OCR client not initialized",
            )
        
        try:
            from azure.ai.vision.imageanalysis.models import VisualFeatures
            
            # Call Azure Read API
            result = self._client.analyze(
                image_data=image_data,
                visual_features=[VisualFeatures.READ],
            )
            
            # Extract text from result
            text_parts: List[str] = []
            confidence_sum = 0.0
            word_count = 0
            
            if result.read and result.read.blocks:
                for block in result.read.blocks:
                    for line in block.lines:
                        text_parts.append(line.text)
                        # Approximate confidence from words if available
                        if hasattr(line, 'words'):
                            for word in line.words:
                                if hasattr(word, 'confidence'):
                                    confidence_sum += word.confidence
                                    word_count += 1
            
            full_text = "\n".join(text_parts)
            
            # Calculate average confidence
            avg_confidence = confidence_sum / word_count if word_count > 0 else 0.85
            
            return ParsingResult(
                success=bool(full_text.strip()),
                text=full_text,
                pages=[full_text],
                format=".ocr",
                parser_used="azure_ocr",
                confidence=avg_confidence,
            )
            
        except Exception as e:
            logger.error(f"Azure OCR API call failed: {e}")
            return ParsingResult(
                success=False,
                error=f"Azure OCR API error: {e}",
            )
    
    async def _parse_fallback(
        self,
        file_path: str,
        options: Dict[str, Any],
    ) -> ParsingResult:
        """Fallback parsing when Azure is unavailable.
        
        Tries to use Tesseract OCR if available.
        
        Args:
            file_path: Path to the file
            options: Parser options
            
        Returns:
            ParsingResult
        """
        logger.warning("Azure OCR unavailable, trying Tesseract fallback")
        
        try:
            import pytesseract
            from PIL import Image
            
            path = Path(file_path)
            extension = path.suffix.lower()
            
            if extension == ".pdf":
                # Convert PDF to images
                try:
                    import fitz
                    from pdf2image import convert_from_path
                    images = convert_from_path(file_path)
                except ImportError:
                    return ParsingResult(
                        success=False,
                        error="PDF OCR requires pdf2image. Install with: pip install pdf2image",
                    )
            else:
                images = [Image.open(file_path)]
            
            pages: List[str] = []
            language = options.get("language", "eng")
            
            for img in images:
                text = pytesseract.image_to_string(img, lang=language)
                pages.append(text)
            
            full_text = "\n\n".join(pages)
            
            return ParsingResult(
                success=bool(full_text.strip()),
                text=full_text,
                pages=pages,
                format=extension,
                parser_used="azure_ocr-fallback-tesseract",
                confidence=0.7,  # Lower confidence for fallback
            )
            
        except ImportError as e:
            return ParsingResult(
                success=False,
                error=f"OCR fallback requires pytesseract and Pillow: {e}",
            )
        except Exception as e:
            return ParsingResult(
                success=False,
                error=f"Fallback OCR failed: {e}",
            )
    
    async def health_check(self, config: Optional[Dict[str, Any]] = None) -> HealthStatus:
        """Check the health of the parser.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus indicating parser health
        """
        if not self._azure_available:
            return HealthStatus.DEGRADED
        
        if not self._client:
            return HealthStatus.UNHEALTHY
        
        # Try a simple test call
        try:
            # Test with a simple 1x1 pixel image
            import io
            from PIL import Image
            
            img = Image.new("RGB", (1, 1), color="white")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            
            result = await self._ocr_image_bytes(img_bytes.getvalue(), {})
            
            if result.success or "client not initialized" not in result.error.lower():
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.UNHEALTHY
                
        except Exception as e:
            logger.warning(f"Azure OCR health check failed: {e}")
            return HealthStatus.DEGRADED
