"""Docling parser plugin for document processing.

This module provides the Docling parser integration for extracting
text and structure from PDF, DOCX, PPTX, and XLSX files.
"""

import logging
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


class DoclingParser(ParserPlugin):
    """Docling parser plugin for PDF, Office, and image documents.
    
    Docling is the primary parser for structured document types
    including PDFs with text layers, Word documents, PowerPoint
    presentations, and Excel spreadsheets.
    
    Example:
        >>> parser = DoclingParser()
        >>> await parser.initialize({})
        >>> result = await parser.parse("/path/to/document.pdf")
        >>> print(result.text)
    """
    
    SUPPORTED_FORMATS = [
        ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
        ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif",
    ]
    
    MIME_TYPE_MAP = {
        "application/pdf": 0.95,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": 0.95,
        "application/msword": 0.90,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": 0.95,
        "application/vnd.ms-excel": 0.90,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": 0.95,
        "application/vnd.ms-powerpoint": 0.90,
        "image/jpeg": 0.80,
        "image/png": 0.80,
        "image/tiff": 0.80,
        "image/bmp": 0.70,
        "image/gif": 0.70,
    }
    
    def __init__(self) -> None:
        """Initialize the Docling parser."""
        self._docling_available = False
        self._document_converter = None
        self._config: Dict[str, Any] = {}
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="docling",
            name="Docling Parser",
            version="1.0.0",
            type=PluginType.PARSER,
            description="Primary parser for PDF, Office docs, and images using Docling",
            author="Pipeline Team",
            supported_formats=self.SUPPORTED_FORMATS,
            requires_auth=False,
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the parser with configuration.
        
        Args:
            config: Parser configuration options
        """
        self._config = config
        
        # Try to import docling
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.document import ConversionResult
            
            self._document_converter = DocumentConverter()
            self._docling_available = True
            logger.info("Docling parser initialized successfully")
            
        except ImportError:
            logger.warning(
                "Docling not installed. Parser will use fallback methods. "
                "Install with: pip install docling"
            )
            self._docling_available = False
    
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
            if extension in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif"):
                # Images have lower confidence for Docling
                confidence = 0.80
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
        """Parse a document and extract content.
        
        Args:
            file_path: Path to the file to parse
            options: Parser-specific options
            
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
        
        # Check support first
        support = await self.supports(file_path)
        if not support.supported:
            return ParsingResult(
                success=False,
                error=support.reason,
            )
        
        try:
            if self._docling_available and self._document_converter:
                result = await self._parse_with_docling(file_path, options)
            else:
                result = await self._parse_fallback(file_path, options)
            
            # Add processing time
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result
            
        except Exception as e:
            logger.error(f"Docling parsing failed: {e}", exc_info=True)
            return ParsingResult(
                success=False,
                error=f"Parsing failed: {str(e)}",
            )
    
    async def _parse_with_docling(
        self,
        file_path: str,
        options: Dict[str, Any],
    ) -> ParsingResult:
        """Parse using Docling library.
        
        Args:
            file_path: Path to the file
            options: Parser options
            
        Returns:
            ParsingResult
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import ConversionResult
        
        # Convert document
        result = self._document_converter.convert(file_path)
        
        # Extract text
        full_text = result.document.export_to_markdown()
        
        # Extract pages (approximate from markdown sections)
        pages = self._extract_pages(result)
        
        # Extract metadata
        metadata = self._extract_metadata(result)
        
        # Extract tables
        tables = self._extract_tables(result)
        
        # Calculate confidence based on extraction quality
        confidence = self._calculate_confidence(result, full_text)
        
        return ParsingResult(
            success=True,
            text=full_text,
            pages=pages,
            metadata=metadata,
            format=Path(file_path).suffix.lower(),
            parser_used="docling",
            confidence=confidence,
            tables=tables,
        )
    
    async def _parse_fallback(
        self,
        file_path: str,
        options: Dict[str, Any],
    ) -> ParsingResult:
        """Fallback parsing using available libraries.
        
        Args:
            file_path: Path to the file
            options: Parser options
            
        Returns:
            ParsingResult
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == ".pdf":
            return await self._parse_pdf_fallback(file_path)
        elif extension in (".txt", ".md", ".rst"):
            return await self._parse_text_fallback(file_path)
        else:
            return ParsingResult(
                success=False,
                error=f"Fallback parsing not available for {extension}. "
                      "Install docling for full support.",
            )
    
    async def _parse_pdf_fallback(self, file_path: str) -> ParsingResult:
        """Parse PDF using PyMuPDF as fallback.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ParsingResult
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            pages: List[str] = []
            full_text = ""
            
            for page in doc:
                text = page.get_text()
                pages.append(text)
                full_text += text + "\n"
            
            metadata = dict(doc.metadata) if doc.metadata else {}
            
            doc.close()
            
            # Calculate simple confidence based on text extraction
            confidence = 0.8 if full_text.strip() else 0.3
            
            return ParsingResult(
                success=True,
                text=full_text.strip(),
                pages=pages,
                metadata=metadata,
                format=".pdf",
                parser_used="docling-fallback-pymupdf",
                confidence=confidence,
            )
            
        except ImportError:
            return ParsingResult(
                success=False,
                error="PyMuPDF not available for PDF fallback parsing",
            )
    
    async def _parse_text_fallback(self, file_path: str) -> ParsingResult:
        """Parse text files as fallback.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            ParsingResult
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            return ParsingResult(
                success=True,
                text=content,
                pages=[content],
                format=Path(file_path).suffix.lower(),
                parser_used="docling-fallback-text",
                confidence=0.95,
            )
            
        except Exception as e:
            return ParsingResult(
                success=False,
                error=f"Text parsing failed: {e}",
            )
    
    def _extract_pages(self, result: Any) -> List[str]:
        """Extract page texts from Docling result.
        
        Args:
            result: Docling ConversionResult
            
        Returns:
            List of page texts
        """
        pages: List[str] = []
        
        try:
            # Try to get pages from document
            if hasattr(result, 'document') and hasattr(result.document, 'pages'):
                for page in result.document.pages:
                    if hasattr(page, 'text'):
                        pages.append(page.text)
                    elif hasattr(page, 'export_to_text'):
                        pages.append(page.export_to_text())
        except Exception as e:
            logger.warning(f"Failed to extract pages: {e}")
        
        # Fallback: split markdown by headers
        if not pages:
            full_text = result.document.export_to_markdown()
            # Split by page-like separators or headers
            import re
            pages = re.split(r'\n#{1,2}\s+', full_text)
            pages = [p.strip() for p in pages if p.strip()]
        
        return pages
    
    def _extract_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract metadata from Docling result.
        
        Args:
            result: Docling ConversionResult
            
        Returns:
            Metadata dictionary
        """
        metadata: Dict[str, Any] = {}
        
        try:
            if hasattr(result, 'document'):
                doc = result.document
                
                # Try to get various metadata fields
                if hasattr(doc, 'name'):
                    metadata['title'] = doc.name
                
                if hasattr(doc, 'origin') and doc.origin:
                    origin = doc.origin
                    if hasattr(origin, 'mimetype'):
                        metadata['mime_type'] = origin.mimetype
                    if hasattr(origin, 'filename'):
                        metadata['filename'] = origin.filename
                
                # Extract document properties if available
                if hasattr(doc, 'properties') and doc.properties:
                    for key, value in doc.properties.items():
                        metadata[key] = value
                        
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
    
    def _extract_tables(self, result: Any) -> List[Dict[str, Any]]:
        """Extract tables from Docling result.
        
        Args:
            result: Docling ConversionResult
            
        Returns:
            List of table dictionaries
        """
        tables: List[Dict[str, Any]] = []
        
        try:
            if hasattr(result, 'document') and hasattr(result.document, 'tables'):
                for table in result.document.tables:
                    table_data: Dict[str, Any] = {}
                    
                    if hasattr(table, 'data'):
                        table_data['data'] = table.data
                    
                    if hasattr(table, 'export_to_dataframe'):
                        try:
                            df = table.export_to_dataframe()
                            table_data['rows'] = df.values.tolist()
                            table_data['columns'] = df.columns.tolist()
                        except Exception:
                            pass
                    
                    tables.append(table_data)
                    
        except Exception as e:
            logger.warning(f"Failed to extract tables: {e}")
        
        return tables
    
    def _calculate_confidence(self, result: Any, extracted_text: str) -> float:
        """Calculate confidence score for the extraction.
        
        Args:
            result: Docling ConversionResult
            extracted_text: The extracted text
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        confidence = 0.85  # Base confidence for Docling
        
        # Adjust based on text extraction
        if not extracted_text or len(extracted_text.strip()) < 10:
            confidence -= 0.3
        
        # Check for document structure
        if hasattr(result, 'document'):
            doc = result.document
            
            # Bonus for having structure
            if hasattr(doc, 'pages') and doc.pages:
                confidence += 0.05
            
            if hasattr(doc, 'tables') and doc.tables:
                confidence += 0.05
        
        return min(1.0, max(0.0, confidence))
    
    async def health_check(self, config: Optional[Dict[str, Any]] = None) -> HealthStatus:
        """Check the health of the parser.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus indicating parser health
        """
        if not self._docling_available:
            return HealthStatus.DEGRADED
        
        if not self._document_converter:
            return HealthStatus.UNHEALTHY
        
        return HealthStatus.HEALTHY
