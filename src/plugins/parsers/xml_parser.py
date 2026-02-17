"""XML parser plugin with XPath extraction and schema validation.

This module provides XML parsing capabilities with support for
XPath queries, schema validation, and namespace handling.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.plugins.base import (
    HealthStatus,
    ParserPlugin,
    ParsingResult,
    PluginMetadata,
    PluginType,
    SupportResult,
)

logger = logging.getLogger(__name__)


class XMLParser(ParserPlugin):
    """XML parser plugin with XPath extraction capabilities.
    
    This parser handles XML files with support for:
    - Namespace handling
    - XPath extraction
    - Schema validation (XSD, DTD)
    - Large file processing
    - CDATA preservation
    
    Example:
        >>> parser = XMLParser()
        >>> await parser.initialize({})
        >>> result = await parser.parse("/path/to/data.xml")
        >>> print(result.text)
    """

    SUPPORTED_FORMATS = [".xml", ".xhtml", ".svg", ".rss", ".atom", ".wsdl", ".xslt"]

    MIME_TYPE_MAP = {
        "application/xml": 1.0,
        "text/xml": 1.0,
        "application/xhtml+xml": 1.0,
        "image/svg+xml": 1.0,
        "application/rss+xml": 1.0,
        "application/atom+xml": 1.0,
    }

    def __init__(self) -> None:
        """Initialize the XML parser."""
        self._config: dict[str, Any] = {}

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="xml",
            name="XML Parser",
            version="1.0.0",
            type=PluginType.PARSER,
            description="XML parser with XPath extraction and schema validation support",
            author="Pipeline Team",
            supported_formats=self.SUPPORTED_FORMATS,
            requires_auth=False,
        )

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the parser with configuration.
        
        Args:
            config: Parser configuration options including:
                - encoding: File encoding (default: utf-8)
                - namespaces: Namespace mapping dict
                - xsd_path: Path to XSD schema for validation
                - extract_cdata: Whether to extract CDATA content (default: True)
                - preserve_whitespace: Whether to preserve whitespace (default: False)
                - xpath_queries: Dict of named XPath queries for extraction
                - chunk_by: Element tag to chunk by (e.g., "item", "record")
                - max_depth: Maximum parsing depth (default: unlimited)
        """
        self._config = {
            "encoding": config.get("encoding", "utf-8"),
            "namespaces": config.get("namespaces", {}),
            "xsd_path": config.get("xsd_path"),
            "extract_cdata": config.get("extract_cdata", True),
            "preserve_whitespace": config.get("preserve_whitespace", False),
            "xpath_queries": config.get("xpath_queries", {}),
            "chunk_by": config.get("chunk_by"),
            "max_depth": config.get("max_depth"),
        }
        logger.info("XML parser initialized")

    async def supports(
        self,
        file_path: str,
        mime_type: str | None = None,
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
            return SupportResult(
                supported=True,
                confidence=0.95,
                reason=f"Supported extension: {extension}",
            )

        # Check MIME type if provided
        if mime_type and mime_type in self.MIME_TYPE_MAP:
            return SupportResult(
                supported=True,
                confidence=self.MIME_TYPE_MAP[mime_type],
                reason=f"Supported MIME type: {mime_type}",
            )

        # Try to detect XML content
        confidence = await self._detect_xml_content(file_path)
        if confidence > 0.7:
            return SupportResult(
                supported=True,
                confidence=confidence,
                reason="Detected XML content",
            )

        return SupportResult(
            supported=False,
            confidence=1.0,
            reason=f"Unsupported file format: {extension}",
        )

    async def _detect_xml_content(self, file_path: str) -> float:
        """Detect if a file contains XML content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        try:
            with open(file_path, encoding=self._config["encoding"], errors="ignore") as f:
                # Read first 1KB
                sample = f.read(1024).strip()

                if sample.startswith("<?xml"):
                    return 1.0

                if sample.startswith("<") and ">" in sample:
                    # Check for XML-like structure
                    if sample.count("<") == sample.count(">"):
                        return 0.8
                    return 0.6

                return 0.0

        except Exception:
            return 0.0

    async def parse(
        self,
        file_path: str,
        options: dict[str, Any] | None = None,
    ) -> ParsingResult:
        """Parse an XML file and extract content.
        
        Args:
            file_path: Path to the file to parse
            options: Parser-specific options that override config
            
        Returns:
            ParsingResult containing extracted content
        """
        import time

        opts = {**self._config, **(options or {})}
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
            result = await self._parse_xml(file_path, opts)
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result

        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            return ParsingResult(
                success=False,
                error=f"XML parse error: {e!s}",
            )
        except Exception as e:
            logger.error(f"XML parsing failed: {e}", exc_info=True)
            return ParsingResult(
                success=False,
                error=f"Parsing failed: {e!s}",
            )

    async def _parse_xml(
        self,
        file_path: str,
        options: dict[str, Any],
    ) -> ParsingResult:
        """Parse XML file with full feature support.
        
        Args:
            file_path: Path to the file
            options: Parsing options
            
        Returns:
            ParsingResult
        """
        path = Path(file_path)
        encoding = options.get("encoding", "utf-8")

        # Parse the XML
        tree = ET.parse(file_path, parser=ET.XMLParser(encoding=encoding))
        root = tree.getroot()

        # Extract namespaces
        namespaces = self._extract_namespaces(root)

        # Validate against XSD if provided
        validation_result = None
        xsd_path = options.get("xsd_path")
        if xsd_path:
            validation_result = await self._validate_with_xsd(file_path, xsd_path)

        # Extract text content
        full_text = self._extract_text(root, options.get("preserve_whitespace", False))

        # Execute XPath queries if provided
        xpath_results: dict[str, list[str]] = {}
        xpath_queries = options.get("xpath_queries", {})
        if xpath_queries:
            for name, query in xpath_queries.items():
                try:
                    elements = root.findall(query, namespaces)
                    xpath_results[name] = [self._element_to_text(elem) for elem in elements]
                except Exception as e:
                    logger.warning(f"XPath query '{name}' failed: {e}")
                    xpath_results[name] = []

        # Create chunks based on configuration
        chunk_by = options.get("chunk_by")
        if chunk_by:
            chunks = self._create_chunks_by_element(root, chunk_by, namespaces)
        else:
            chunks = self._create_chunks_by_size(root, 5000)

        # Extract metadata
        metadata = {
            "root_tag": root.tag,
            "namespaces": namespaces,
            "encoding": encoding,
            "file_size_bytes": path.stat().st_size,
        }

        if validation_result:
            metadata["validation"] = validation_result

        if xpath_results:
            metadata["xpath_results"] = {k: len(v) for k, v in xpath_results.items()}

        # Calculate confidence
        confidence = 0.95
        if validation_result and not validation_result.get("valid", True):
            confidence = 0.7

        return ParsingResult(
            success=True,
            text=full_text,
            pages=chunks or [full_text],
            metadata=metadata,
            format=path.suffix.lower(),
            parser_used="xml",
            confidence=confidence,
        )

    def _extract_namespaces(self, root: ET.Element) -> dict[str, str]:
        """Extract namespace mappings from the XML.
        
        Args:
            root: Root element
            
        Returns:
            Dictionary of prefix -> namespace URI
        """
        namespaces: dict[str, str] = {}

        # Get namespaces from root element
        for elem in [root] + list(root.iter()):
            for key, value in elem.attrib.items():
                if key.startswith("xmlns"):
                    if ":" in key:
                        prefix = key.split(":", 1)[1]
                        namespaces[prefix] = value
                    else:
                        namespaces[""] = value  # Default namespace

        return namespaces

    def _extract_text(self, element: ET.Element, preserve_whitespace: bool = False) -> str:
        """Extract all text content from an element.
        
        Args:
            element: XML element
            preserve_whitespace: Whether to preserve whitespace
            
        Returns:
            Extracted text
        """
        texts = []

        if element.text:
            text = element.text
            if not preserve_whitespace:
                text = " ".join(text.split())
            if text:
                texts.append(text)

        for child in element:
            child_text = self._extract_text(child, preserve_whitespace)
            if child_text:
                texts.append(child_text)

            if child.tail:
                tail = child.tail
                if not preserve_whitespace:
                    tail = " ".join(tail.split())
                if tail:
                    texts.append(tail)

        return " ".join(texts) if texts else ""

    def _element_to_text(self, element: ET.Element, indent: int = 0) -> str:
        """Convert an element to readable text format.
        
        Args:
            element: XML element
            indent: Indentation level
            
        Returns:
            Text representation
        """
        lines = []
        indent_str = "  " * indent

        # Get tag name without namespace
        tag = element.tag
        if "}" in tag:
            tag = tag.split("}", 1)[1]

        # Build attribute string
        attrs = ""
        if element.attrib:
            attr_parts = [f'{k}="{v}"' for k, v in element.attrib.items()]
            attrs = " " + " ".join(attr_parts)

        # Get text content
        text = element.text.strip() if element.text else ""

        if not list(element) and not text:
            # Self-closing or empty
            lines.append(f"{indent_str}<{tag}{attrs} />")
        elif not list(element):
            # Element with text only
            lines.append(f"{indent_str}<{tag}{attrs}>{text}</{tag}>")
        else:
            # Element with children
            lines.append(f"{indent_str}<{tag}{attrs}>")
            if text:
                lines.append(f"{indent_str}  {text}")

            for child in element:
                lines.append(self._element_to_text(child, indent + 1))

            lines.append(f"{indent_str}</{tag}>")

        return "\n".join(lines)

    def _create_chunks_by_element(
        self,
        root: ET.Element,
        tag_name: str,
        namespaces: dict[str, str],
    ) -> list[str]:
        """Create chunks by grouping elements with a specific tag.
        
        Args:
            root: Root element
            tag_name: Tag name to chunk by
            namespaces: Namespace mappings
            
        Returns:
            List of chunk strings
        """
        chunks = []

        # Find all elements with the given tag
        # Handle both namespaced and non-namespaced tags
        elements = []

        # Try with namespaces
        for ns_prefix, ns_uri in namespaces.items():
            if ns_prefix:
                xpath = f".//{ns_prefix}:{tag_name}"
            else:
                xpath = f".//{tag_name}"
            try:
                found = root.findall(xpath, namespaces)
                elements.extend(found)
            except Exception:
                pass

        # Try without namespace
        if not elements:
            elements = root.findall(f".//{tag_name}")

        for elem in elements:
            chunk_text = self._element_to_text(elem)
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def _create_chunks_by_size(self, root: ET.Element, max_size: int) -> list[str]:
        """Create chunks by limiting size.
        
        Args:
            root: Root element
            max_size: Maximum chunk size in characters
            
        Returns:
            List of chunk strings
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for child in root:
            child_text = self._element_to_text(child)
            child_size = len(child_text)

            if current_size + child_size > max_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [child_text]
                current_size = child_size
            else:
                current_chunk.append(child_text)
                current_size += child_size

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    async def _validate_with_xsd(
        self,
        xml_path: str,
        xsd_path: str,
    ) -> dict[str, Any]:
        """Validate XML against XSD schema.
        
        Args:
            xml_path: Path to XML file
            xsd_path: Path to XSD schema file
            
        Returns:
            Validation result dictionary
        """
        try:
            from lxml import etree

            # Parse XSD
            with open(xsd_path, "rb") as f:
                schema_root = etree.XML(f.read())
            schema = etree.XMLSchema(schema_root)

            # Parse XML
            with open(xml_path, "rb") as f:
                xml_doc = etree.parse(f)

            # Validate
            is_valid = schema.validate(xml_doc)

            result = {
                "valid": is_valid,
                "schema": xsd_path,
            }

            if not is_valid:
                result["errors"] = [str(error) for error in schema.error_log]

            return result

        except ImportError:
            logger.warning("lxml not available for XSD validation")
            return {"valid": True, "warning": "XSD validation skipped - lxml not installed"}
        except Exception as e:
            logger.error(f"XSD validation failed: {e}")
            return {"valid": False, "error": str(e)}

    async def execute_xpath(
        self,
        file_path: str,
        xpath: str,
        namespaces: dict[str, str] | None = None,
    ) -> list[str]:
        """Execute an XPath query on an XML file.
        
        Args:
            file_path: Path to XML file
            xpath: XPath query string
            namespaces: Optional namespace mappings
            
        Returns:
            List of matching text content
        """
        tree = ET.parse(file_path)
        root = tree.getroot()

        ns = namespaces or self._extract_namespaces(root)

        try:
            elements = root.findall(xpath, ns)
            return [self._extract_text(elem) for elem in elements]
        except Exception as e:
            logger.error(f"XPath query failed: {e}")
            return []

    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Check the health of the parser.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus indicating parser health
        """
        # XML parsing uses standard library, always healthy
        return HealthStatus.HEALTHY
