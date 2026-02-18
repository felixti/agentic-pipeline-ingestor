"""Unit tests for XML parser plugin."""

import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.plugins.base import HealthStatus, ParsingResult, SupportResult
from src.plugins.parsers.xml_parser import XMLParser


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def parser():
    """Create an initialized XML parser."""
    p = XMLParser()
    await p.initialize({})
    return p


# ============================================================================
# XMLParser Class Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParser:
    """Tests for XMLParser class."""

    def test_init(self):
        """Test parser initialization."""
        parser = XMLParser()
        assert parser._config == {}

    def test_metadata(self):
        """Test parser metadata."""
        parser = XMLParser()
        metadata = parser.metadata

        assert metadata.id == "xml"
        assert metadata.name == "XML Parser"
        assert metadata.version == "1.0.0"
        assert ".xml" in metadata.supported_formats
        assert ".xhtml" in metadata.supported_formats
        assert ".svg" in metadata.supported_formats

    @pytest.mark.asyncio
    async def test_initialize_default_config(self):
        """Test initialization with default config."""
        parser = XMLParser()
        await parser.initialize({})

        assert parser._config["encoding"] == "utf-8"
        assert parser._config["namespaces"] == {}
        assert parser._config["xsd_path"] is None
        assert parser._config["extract_cdata"] is True
        assert parser._config["preserve_whitespace"] is False
        assert parser._config["xpath_queries"] == {}
        assert parser._config["chunk_by"] is None
        assert parser._config["max_depth"] is None

    @pytest.mark.asyncio
    async def test_initialize_custom_config(self):
        """Test initialization with custom config."""
        parser = XMLParser()
        await parser.initialize({
            "encoding": "latin-1",
            "namespaces": {"ns": "http://example.com"},
            "xsd_path": "/path/to/schema.xsd",
            "extract_cdata": False,
            "preserve_whitespace": True,
            "xpath_queries": {"title": ".//title"},
            "chunk_by": "item",
            "max_depth": 10,
        })

        assert parser._config["encoding"] == "latin-1"
        assert parser._config["namespaces"] == {"ns": "http://example.com"}
        assert parser._config["xsd_path"] == "/path/to/schema.xsd"
        assert parser._config["extract_cdata"] is False
        assert parser._config["preserve_whitespace"] is True
        assert parser._config["xpath_queries"] == {"title": ".//title"}
        assert parser._config["chunk_by"] == "item"
        assert parser._config["max_depth"] == 10


# ============================================================================
# Supports Method Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserSupports:
    """Tests for XML parser supports method."""

    @pytest.mark.asyncio
    async def test_supports_xml_extension(self):
        """Test support for .xml files."""
        parser = XMLParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file.xml")

        assert isinstance(result, SupportResult)
        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_xhtml_extension(self):
        """Test support for .xhtml files."""
        parser = XMLParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file.xhtml")

        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_svg_extension(self):
        """Test support for .svg files."""
        parser = XMLParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file.svg")

        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_mime_type_xml(self):
        """Test support with XML MIME type."""
        parser = XMLParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file", mime_type="application/xml")

        assert result.supported is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_supports_mime_type_svg(self):
        """Test support with SVG MIME type."""
        parser = XMLParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file", mime_type="image/svg+xml")

        assert result.supported is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_supports_detected_xml_content_with_declaration(self, tmp_path):
        """Test support detection for files with XML declaration."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.txt"
        xml_file.write_text('<?xml version="1.0"?><root><item>test</item></root>')

        result = await parser.supports(str(xml_file))

        assert result.supported is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_supports_detected_xml_content_without_declaration(self, tmp_path):
        """Test support detection for XML-like content without declaration."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.txt"
        xml_file.write_text('<root><item>test</item></root>')

        result = await parser.supports(str(xml_file))

        assert result.supported is True
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_supports_rejects_non_xml_content(self, tmp_path):
        """Test rejection of non-XML content."""
        parser = XMLParser()
        await parser.initialize({})

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is just plain text without any XML tags")

        result = await parser.supports(str(txt_file))

        assert result.supported is False

    @pytest.mark.asyncio
    async def test_supports_unsupported_extension(self):
        """Test rejection of unsupported file extensions."""
        parser = XMLParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file.pdf")

        assert result.supported is False
        assert result.confidence == 1.0


# ============================================================================
# Parse Simple XML Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserParseSimple:
    """Tests for parsing simple XML content."""

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        parser = XMLParser()
        await parser.initialize({})

        result = await parser.parse("/nonexistent/file.xml")

        assert isinstance(result, ParsingResult)
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_parse_unsupported_format(self, tmp_path):
        """Test parsing unsupported file format."""
        parser = XMLParser()
        await parser.initialize({})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("not a pdf")

        result = await parser.parse(str(pdf_file))

        assert result.success is False

    @pytest.mark.asyncio
    async def test_parse_simple_xml(self, tmp_path):
        """Test parsing simple XML file."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
<root>
    <name>Alice</name>
    <age>30</age>
</root>"""
        xml_file.write_text(xml_content)

        result = await parser.parse(str(xml_file))

        assert result.success is True
        assert "Alice" in result.text
        assert "30" in result.text
        assert result.metadata["root_tag"] == "root"
        assert result.format == ".xml"
        assert result.parser_used == "xml"

    @pytest.mark.asyncio
    async def test_parse_empty_xml(self, tmp_path):
        """Test parsing empty XML file."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_file.write_text('<?xml version="1.0"?><root></root>')

        result = await parser.parse(str(xml_file))

        assert result.success is True
        assert result.metadata["root_tag"] == "root"


# ============================================================================
# Parse Nested XML Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserParseNested:
    """Tests for parsing nested XML structures."""

    @pytest.mark.asyncio
    async def test_parse_nested_xml(self, tmp_path):
        """Test parsing nested XML elements."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
<library>
    <book>
        <title>Python Programming</title>
        <author>John Doe</author>
        <chapters>
            <chapter>Introduction</chapter>
            <chapter>Advanced Topics</chapter>
        </chapters>
    </book>
</library>"""
        xml_file.write_text(xml_content)

        result = await parser.parse(str(xml_file))

        assert result.success is True
        assert "Python Programming" in result.text
        assert "John Doe" in result.text
        assert "Introduction" in result.text
        assert "Advanced Topics" in result.text

    @pytest.mark.asyncio
    async def test_parse_deeply_nested_xml(self, tmp_path):
        """Test parsing deeply nested XML."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
<a>
    <b>
        <c>
            <d>
                <e>Deep Value</e>
            </d>
        </c>
    </b>
</a>"""
        xml_file.write_text(xml_content)

        result = await parser.parse(str(xml_file))

        assert result.success is True
        assert "Deep Value" in result.text

    @pytest.mark.asyncio
    async def test_parse_xml_with_mixed_content(self, tmp_path):
        """Test parsing XML with mixed text and elements."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
<article>
    <title>Sample Article</title>
    <body>
        This is <b>bold</b> and <i>italic</i> text.
    </body>
</article>"""
        xml_file.write_text(xml_content)

        result = await parser.parse(str(xml_file))

        assert result.success is True
        assert "Sample Article" in result.text
        assert "bold" in result.text
        assert "italic" in result.text


# ============================================================================
# Parse XML with Attributes Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserParseAttributes:
    """Tests for parsing XML with attributes."""

    @pytest.mark.asyncio
    async def test_parse_xml_with_attributes(self, tmp_path):
        """Test parsing XML elements with attributes."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
<catalog>
    <book id="bk101" genre="fiction">
        <author>Gambardella, Matthew</author>
        <title>XML Developer's Guide</title>
        <price currency="USD">44.95</price>
    </book>
</catalog>"""
        xml_file.write_text(xml_content)

        result = await parser.parse(str(xml_file))

        assert result.success is True
        assert "Gambardella, Matthew" in result.text
        assert "XML Developer's Guide" in result.text
        assert "44.95" in result.text

    @pytest.mark.asyncio
    async def test_element_to_text_with_attributes(self):
        """Test converting element with attributes to text."""
        parser = XMLParser()
        element = ET.Element("item")
        element.set("id", "123")
        element.set("name", "test")
        element.text = "Content"

        text = parser._element_to_text(element)

        assert 'id="123"' in text
        assert 'name="test"' in text
        assert "Content" in text


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserErrorHandling:
    """Tests for XML parser error handling."""

    @pytest.mark.asyncio
    async def test_parse_invalid_xml(self, tmp_path):
        """Test parsing invalid XML content."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<root><unclosed>")

        result = await parser.parse(str(xml_file))

        assert result.success is False
        assert "parse error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_parse_malformed_xml(self, tmp_path):
        """Test parsing malformed XML."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<root><child></different>")

        result = await parser.parse(str(xml_file))

        assert result.success is False

    @pytest.mark.asyncio
    async def test_parse_empty_file(self, tmp_path):
        """Test parsing empty file."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_file.write_text("")

        result = await parser.parse(str(xml_file))

        assert result.success is False


# ============================================================================
# XPath Query Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserXPathQueries:
    """Tests for XPath query execution."""

    @pytest.mark.asyncio
    async def test_parse_with_xpath_queries(self, tmp_path):
        """Test parsing with XPath queries."""
        parser = XMLParser()
        await parser.initialize({
            "xpath_queries": {
                "titles": ".//title",
                "authors": ".//author"
            }
        })

        xml_file = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
<library>
    <book>
        <title>Book One</title>
        <author>Author A</author>
    </book>
    <book>
        <title>Book Two</title>
        <author>Author B</author>
    </book>
</library>"""
        xml_file.write_text(xml_content)

        result = await parser.parse(str(xml_file))

        assert result.success is True
        assert "xpath_results" in result.metadata
        assert result.metadata["xpath_results"]["titles"] == 2
        assert result.metadata["xpath_results"]["authors"] == 2

    @pytest.mark.asyncio
    async def test_execute_xpath_query(self, tmp_path):
        """Test executing XPath query on XML file."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
<root>
    <item id="1">First</item>
    <item id="2">Second</item>
    <item id="3">Third</item>
</root>"""
        xml_file.write_text(xml_content)

        results = await parser.execute_xpath(str(xml_file), ".//item")

        assert len(results) == 3
        assert "First" in results
        assert "Second" in results
        assert "Third" in results


# ============================================================================
# Namespace Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserNamespaces:
    """Tests for XML namespace handling."""

    @pytest.mark.asyncio
    async def test_extract_namespaces(self):
        """Test extracting namespaces from XML element attributes."""
        parser = XMLParser()
        
        # Create an element with xmlns attributes manually
        # (ElementTree strips xmlns from attrib during parsing)
        root = ET.Element("root")
        root.set("xmlns", "http://default.ns")
        root.set("xmlns:custom", "http://custom.ns")

        namespaces = parser._extract_namespaces(root)

        # Check that namespaces are extracted from attrib
        assert "" in namespaces  # Default namespace
        assert namespaces[""] == "http://default.ns"
        assert "custom" in namespaces
        assert namespaces["custom"] == "http://custom.ns"

    @pytest.mark.asyncio
    async def test_parse_with_namespaces(self, tmp_path):
        """Test parsing XML with namespaces."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
<root xmlns="http://example.com" xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Title</dc:title>
    <dc:creator>Test Creator</dc:creator>
</root>"""
        xml_file.write_text(xml_content)

        result = await parser.parse(str(xml_file))

        assert result.success is True
        assert "namespaces" in result.metadata


# ============================================================================
# Schema Validation Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserSchemaValidation:
    """Tests for XSD schema validation."""

    @pytest.mark.asyncio
    async def test_validate_with_xsd_not_installed(self, tmp_path):
        """Test XSD validation when lxml is not available."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<root></root>")

        with patch.dict("sys.modules", {"lxml": None}):
            result = await parser._validate_with_xsd(str(xml_file), "/fake/schema.xsd")

        assert result["valid"] is True
        assert "warning" in result
        assert "lxml not installed" in result["warning"]

    @pytest.mark.asyncio
    async def test_parse_with_xsd_validation(self, tmp_path):
        """Test parsing with XSD validation option."""
        parser = XMLParser()

        # Create a temporary XSD file
        xsd_file = tmp_path / "schema.xsd"
        xsd_content = """<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
    <xs:element name="root" type="xs:string"/>
</xs:schema>"""
        xsd_file.write_text(xsd_content)

        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<root>test</root>")

        # Initialize with xsd_path
        await parser.initialize({"xsd_path": str(xsd_file)})

        with patch.object(parser, '_validate_with_xsd') as mock_validate:
            mock_validate.return_value = {"valid": True, "schema": str(xsd_file)}
            result = await parser.parse(str(xml_file))

        assert result.success is True


# ============================================================================
# Chunking Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserChunking:
    """Tests for XML content chunking."""

    @pytest.mark.asyncio
    async def test_create_chunks_by_element(self, tmp_path):
        """Test chunking by element tag."""
        parser = XMLParser()
        await parser.initialize({"chunk_by": "item"})

        xml_file = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
<root>
    <item>Content 1</item>
    <item>Content 2</item>
    <item>Content 3</item>
</root>"""
        xml_file.write_text(xml_content)

        result = await parser.parse(str(xml_file))

        assert result.success is True
        assert len(result.pages) == 3

    @pytest.mark.asyncio
    async def test_create_chunks_by_size(self, tmp_path):
        """Test chunking by size."""
        parser = XMLParser()
        await parser.initialize({})

        xml_file = tmp_path / "test.xml"
        # Create large XML with many elements
        items = "\n".join([f"    <item>Item {i} content here</item>" for i in range(100)])
        xml_content = f"""<?xml version="1.0"?>
<root>
{items}
</root>"""
        xml_file.write_text(xml_content)

        result = await parser.parse(str(xml_file))

        assert result.success is True
        assert len(result.pages) >= 1


# ============================================================================
# Text Extraction Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserTextExtraction:
    """Tests for XML text extraction."""

    def test_extract_text_simple(self):
        """Test extracting text from simple element."""
        parser = XMLParser()
        element = ET.Element("root")
        element.text = "Simple text"

        text = parser._extract_text(element)

        assert text == "Simple text"

    def test_extract_text_with_children(self):
        """Test extracting text from element with children."""
        parser = XMLParser()
        root = ET.Element("root")
        root.text = "Root text"

        child = ET.SubElement(root, "child")
        child.text = "Child text"
        child.tail = "Tail text"

        text = parser._extract_text(root)

        assert "Root text" in text
        assert "Child text" in text
        assert "Tail text" in text

    def test_extract_text_preserve_whitespace(self):
        """Test extracting text with whitespace preservation."""
        parser = XMLParser()
        element = ET.Element("root")
        element.text = "  Multiple   spaces  "

        text_preserved = parser._extract_text(element, preserve_whitespace=True)
        text_normalized = parser._extract_text(element, preserve_whitespace=False)

        # When preserve_whitespace is True, whitespace should be preserved
        # When False, multiple spaces should be collapsed
        assert "Multiple   spaces" in text_preserved
        assert "Multiple" in text_normalized
        # Verify whitespace handling differs between modes
        assert len(text_preserved) >= len(text_normalized)


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserHealthCheck:
    """Tests for XML parser health check."""

    @pytest.mark.asyncio
    async def test_health_check_always_healthy(self):
        """Test that health check always returns healthy."""
        parser = XMLParser()
        await parser.initialize({})

        health = await parser.health_check()

        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_with_config(self):
        """Test health check with config parameter."""
        parser = XMLParser()
        await parser.initialize({})

        health = await parser.health_check({"some": "config"})

        assert health == HealthStatus.HEALTHY


# ============================================================================
# Options Override Tests
# ============================================================================

@pytest.mark.unit
class TestXMLParserOptionsOverride:
    """Tests for parse options override."""

    @pytest.mark.asyncio
    async def test_parse_with_options_override(self, tmp_path):
        """Test parsing with options override."""
        parser = XMLParser()
        await parser.initialize({"encoding": "utf-8"})

        xml_file = tmp_path / "test.xml"
        xml_file.write_text('<?xml version="1.0"?><root>test</root>')

        # Override with different options
        result = await parser.parse(str(xml_file), options={"preserve_whitespace": True})

        assert result.success is True
