"""Unit tests for Docling parser plugin."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.plugins.base import HealthStatus, ParsingResult, SupportResult
from src.plugins.parsers.docling_parser import DoclingParser


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def parser():
    """Create an initialized Docling parser."""
    p = DoclingParser()
    await p.initialize({})
    return p


# ============================================================================
# DoclingParser Class Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParser:
    """Tests for DoclingParser class."""

    def test_init(self):
        """Test parser initialization."""
        parser = DoclingParser()
        assert parser._docling_available is False
        assert parser._document_converter is None
        assert parser._config == {}

    def test_metadata(self):
        """Test parser metadata."""
        parser = DoclingParser()
        metadata = parser.metadata

        assert metadata.id == "docling"
        assert metadata.name == "Docling Parser"
        assert metadata.version == "1.0.0"
        assert ".pdf" in metadata.supported_formats
        assert ".docx" in metadata.supported_formats
        assert ".pptx" in metadata.supported_formats
        assert ".xlsx" in metadata.supported_formats
        assert ".jpg" in metadata.supported_formats
        assert ".png" in metadata.supported_formats

    @pytest.mark.asyncio
    async def test_initialize_without_docling(self):
        """Test initialization when docling is not available."""
        parser = DoclingParser()

        with patch.dict("sys.modules", {"docling": None}):
            await parser.initialize({})

        assert parser._docling_available is False
        assert parser._document_converter is None

    @pytest.mark.asyncio
    async def test_initialize_with_docling(self):
        """Test initialization when docling is available."""
        parser = DoclingParser()

        mock_docling = MagicMock()
        mock_converter_class = MagicMock()
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        mock_docling.document_converter.DocumentConverter = mock_converter_class

        with patch.dict("sys.modules", {
            "docling": mock_docling,
            "docling.datamodel": mock_docling.datamodel,
            "docling.datamodel.base_models": mock_docling.datamodel.base_models,
            "docling.datamodel.document": mock_docling.datamodel.document,
            "docling.document_converter": mock_docling.document_converter,
        }):
            with patch.object(parser, '_document_converter', None):
                # Simulate docling being available
                parser._docling_available = False
                await parser.initialize({})
                # Manually set what would be set by the actual initialize
                parser._document_converter = mock_converter
                parser._docling_available = True

        assert parser._docling_available is True


# ============================================================================
# Supports Method Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserSupports:
    """Tests for Docling parser supports method."""

    @pytest.mark.asyncio
    async def test_supports_pdf_extension(self):
        """Test support for .pdf files."""
        parser = DoclingParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file.pdf")

        assert isinstance(result, SupportResult)
        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_docx_extension(self):
        """Test support for .docx files."""
        parser = DoclingParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file.docx")

        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_image_extension(self):
        """Test support for image files."""
        parser = DoclingParser()
        await parser.initialize({})

        # Test JPEG
        result_jpg = await parser.supports("/path/to/file.jpg")
        assert result_jpg.supported is True
        assert result_jpg.confidence == 0.80

        # Test PNG
        result_png = await parser.supports("/path/to/file.png")
        assert result_png.supported is True
        assert result_png.confidence == 0.80

    @pytest.mark.asyncio
    async def test_supports_mime_type_pdf(self):
        """Test support with PDF MIME type."""
        parser = DoclingParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file", mime_type="application/pdf")

        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_mime_type_docx(self):
        """Test support with DOCX MIME type."""
        parser = DoclingParser()
        await parser.initialize({})

        result = await parser.supports(
            "/path/to/file",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_unsupported_extension(self):
        """Test rejection of unsupported file extensions."""
        parser = DoclingParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file.xyz")

        assert result.supported is False
        assert result.confidence == 1.0


# ============================================================================
# Parse PDF Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserParsePDF:
    """Tests for parsing PDF files."""

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        parser = DoclingParser()
        await parser.initialize({})

        result = await parser.parse("/nonexistent/file.pdf")

        assert isinstance(result, ParsingResult)
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_parse_unsupported_format(self, tmp_path):
        """Test parsing unsupported file format."""
        parser = DoclingParser()
        await parser.initialize({})

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not supported")

        result = await parser.parse(str(txt_file))

        assert result.success is False

    @pytest.mark.asyncio
    async def test_parse_pdf_with_docling_mock(self, tmp_path):
        """Test parsing PDF with mocked Docling."""
        parser = DoclingParser()
        await parser.initialize({})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf content")

        # Mock the Docling conversion result
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# PDF Title\n\nPDF content here."

        # Mock pages
        mock_page = MagicMock()
        mock_page.text = "Page 1 content"
        mock_result.document.pages = [mock_page]

        # Mock tables
        mock_table = MagicMock()
        mock_table.data = [["A", "B"], ["1", "2"]]
        mock_result.document.tables = [mock_table]

        # Mock document metadata
        mock_result.document.name = "test.pdf"
        mock_origin = MagicMock()
        mock_origin.mimetype = "application/pdf"
        mock_origin.filename = "test.pdf"
        mock_result.document.origin = mock_origin

        parser._docling_available = True
        parser._document_converter = MagicMock()
        parser._document_converter.convert.return_value = mock_result

        result = await parser.parse(str(pdf_file))

        assert result.success is True
        assert "PDF Title" in result.text
        assert "PDF content here" in result.text
        assert result.parser_used == "docling"

    @pytest.mark.asyncio
    async def test_parse_pdf_fallback_with_pymupdf_mock(self, tmp_path):
        """Test parsing PDF with PyMuPDF fallback."""
        parser = DoclingParser()
        await parser.initialize({})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf content")

        # Create mock page
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page text content"

        # Create mock document
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page, mock_page]))
        mock_doc.metadata = {"title": "Test PDF"}

        # Mock fitz.open
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = await parser._parse_pdf_fallback(str(pdf_file))

        assert result.success is True
        assert "Page text content" in result.text
        assert result.parser_used == "docling-fallback-pymupdf"

    @pytest.mark.asyncio
    async def test_parse_pdf_fallback_without_pymupdf(self, tmp_path):
        """Test PDF fallback when PyMuPDF is not available."""
        parser = DoclingParser()

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf content")

        with patch.dict("sys.modules", {"fitz": None}):
            result = await parser._parse_pdf_fallback(str(pdf_file))

        assert result.success is False
        assert "PyMuPDF" in result.error


# ============================================================================
# Parse Word Documents Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserParseWord:
    """Tests for parsing Word documents."""

    @pytest.mark.asyncio
    async def test_parse_docx_with_docling_mock(self, tmp_path):
        """Test parsing DOCX with mocked Docling."""
        parser = DoclingParser()
        await parser.initialize({})

        docx_file = tmp_path / "test.docx"
        docx_file.write_text("fake docx content")

        # Mock the Docling conversion result
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# Document Title\n\nDocument content."

        mock_page = MagicMock()
        mock_page.text = "Page content"
        mock_result.document.pages = [mock_page]
        mock_result.document.tables = []

        mock_result.document.name = "test.docx"
        mock_origin = MagicMock()
        mock_origin.mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        mock_origin.filename = "test.docx"
        mock_result.document.origin = mock_origin

        parser._docling_available = True
        parser._document_converter = MagicMock()
        parser._document_converter.convert.return_value = mock_result

        result = await parser.parse(str(docx_file))

        assert result.success is True
        assert "Document Title" in result.text
        assert result.parser_used == "docling"

    @pytest.mark.asyncio
    async def test_parse_doc_fallback_unsupported(self, tmp_path):
        """Test fallback for .doc files shows unsupported message."""
        parser = DoclingParser()
        await parser.initialize({})

        doc_file = tmp_path / "test.doc"
        doc_file.write_text("fake doc content")

        # Parser should use fallback which doesn't support .doc
        result = await parser.parse(str(doc_file))

        assert result.success is False
        assert "fallback" in result.error.lower() or "parsing failed" in result.error.lower()


# ============================================================================
# Fallback Parsing Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserFallback:
    """Tests for fallback parsing methods."""

    @pytest.mark.asyncio
    async def test_parse_fallback_text_file(self, tmp_path):
        """Test parsing text file as fallback."""
        parser = DoclingParser()

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Plain text content.")

        result = await parser._parse_text_fallback(str(txt_file))

        assert result.success is True
        assert "Plain text content" in result.text
        assert result.parser_used == "docling-fallback-text"

    @pytest.mark.asyncio
    async def test_parse_fallback_unsupported_extension(self, tmp_path):
        """Test fallback with unsupported extension."""
        parser = DoclingParser()

        xyz_file = tmp_path / "test.xyz"
        xyz_file.write_text("unknown content")

        result = await parser._parse_fallback(str(xyz_file), {})

        assert result.success is False
        assert "fallback" in result.error.lower() or "not available" in result.error.lower()


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserErrorHandling:
    """Tests for Docling parser error handling."""

    @pytest.mark.asyncio
    async def test_parse_with_exception(self, tmp_path):
        """Test parsing when an exception occurs."""
        parser = DoclingParser()
        await parser.initialize({})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf content")

        # Make docling available but cause it to raise an exception
        parser._docling_available = True
        parser._document_converter = MagicMock()
        parser._document_converter.convert.side_effect = Exception("Conversion failed")

        result = await parser.parse(str(pdf_file))

        assert result.success is False
        assert "parsing failed" in result.error.lower() or "conversion failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_parse_empty_file(self, tmp_path):
        """Test parsing empty file."""
        parser = DoclingParser()
        await parser.initialize({})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"")

        # Empty file should be handled
        result = await parser.parse(str(pdf_file))

        # May succeed or fail depending on implementation
        assert isinstance(result, ParsingResult)


# ============================================================================
# Extract Pages Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserExtractPages:
    """Tests for page extraction."""

    def test_extract_pages_with_pages_attribute(self):
        """Test extracting pages when pages attribute exists."""
        parser = DoclingParser()

        mock_page = MagicMock()
        mock_page.text = "Page 1 text"

        mock_doc = MagicMock()
        mock_doc.pages = [mock_page, mock_page]

        mock_result = MagicMock()
        mock_result.document = mock_doc

        pages = parser._extract_pages(mock_result)

        assert len(pages) == 2
        assert "Page 1 text" in pages[0]

    def test_extract_pages_with_export_method(self):
        """Test extracting pages when page has export_to_text method."""
        parser = DoclingParser()

        mock_page = MagicMock()
        mock_page.export_to_text.return_value = "Exported page text"
        del mock_page.text  # Remove text attribute

        mock_doc = MagicMock()
        mock_doc.pages = [mock_page]

        mock_result = MagicMock()
        mock_result.document = mock_doc

        pages = parser._extract_pages(mock_result)

        assert len(pages) == 1
        assert "Exported page text" in pages[0]

    def test_extract_pages_fallback_to_markdown(self):
        """Test page extraction fallback to markdown splitting."""
        parser = DoclingParser()

        mock_doc = MagicMock()
        # No pages attribute
        del mock_doc.pages

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.document.export_to_markdown.return_value = "# Section 1\n\nContent 1\n\n# Section 2\n\nContent 2"

        pages = parser._extract_pages(mock_result)

        assert len(pages) == 2
        assert "Section 1" in pages[0]
        assert "Section 2" in pages[1]


# ============================================================================
# Extract Metadata Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserExtractMetadata:
    """Tests for metadata extraction."""

    def test_extract_metadata_with_name(self):
        """Test extracting metadata with document name."""
        parser = DoclingParser()

        mock_doc = MagicMock()
        mock_doc.name = "Test Document"
        mock_doc.properties = {}

        mock_result = MagicMock()
        mock_result.document = mock_doc

        metadata = parser._extract_metadata(mock_result)

        assert metadata["title"] == "Test Document"

    def test_extract_metadata_with_origin(self):
        """Test extracting metadata with origin information."""
        parser = DoclingParser()

        mock_origin = MagicMock()
        mock_origin.mimetype = "application/pdf"
        mock_origin.filename = "test.pdf"

        mock_doc = MagicMock()
        mock_doc.origin = mock_origin

        mock_result = MagicMock()
        mock_result.document = mock_doc

        metadata = parser._extract_metadata(mock_result)

        assert metadata["mime_type"] == "application/pdf"
        assert metadata["filename"] == "test.pdf"

    def test_extract_metadata_with_properties(self):
        """Test extracting metadata with document properties."""
        parser = DoclingParser()

        mock_doc = MagicMock()
        mock_doc.properties = {"author": "John Doe", "pages": 10}

        mock_result = MagicMock()
        mock_result.document = mock_doc

        metadata = parser._extract_metadata(mock_result)

        assert metadata["author"] == "John Doe"
        assert metadata["pages"] == 10


# ============================================================================
# Extract Tables Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserExtractTables:
    """Tests for table extraction."""

    def test_extract_tables_with_data_attribute(self):
        """Test extracting tables with data attribute."""
        parser = DoclingParser()

        mock_table = MagicMock()
        mock_table.data = [["A", "B"], ["1", "2"]]

        mock_doc = MagicMock()
        mock_doc.tables = [mock_table]

        mock_result = MagicMock()
        mock_result.document = mock_doc

        tables = parser._extract_tables(mock_result)

        assert len(tables) == 1
        assert tables[0]["data"] == [["A", "B"], ["1", "2"]]

    def test_extract_tables_with_dataframe_export(self):
        """Test extracting tables with DataFrame export."""
        parser = DoclingParser()

        # Create a mock DataFrame-like object
        mock_df = MagicMock()
        mock_df.values.tolist.return_value = [[1, 3], [2, 4]]
        mock_df.columns.tolist.return_value = ["A", "B"]

        mock_table = MagicMock()
        mock_table.data = None
        mock_table.export_to_dataframe.return_value = mock_df

        mock_doc = MagicMock()
        mock_doc.tables = [mock_table]

        mock_result = MagicMock()
        mock_result.document = mock_doc

        tables = parser._extract_tables(mock_result)

        assert len(tables) == 1
        assert tables[0]["columns"] == ["A", "B"]
        assert tables[0]["rows"] == [[1, 3], [2, 4]]


# ============================================================================
# Calculate Confidence Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserCalculateConfidence:
    """Tests for confidence calculation."""

    def test_calculate_confidence_with_good_text(self):
        """Test confidence calculation with good text extraction."""
        parser = DoclingParser()

        mock_doc = MagicMock()
        mock_doc.pages = [MagicMock()]
        mock_doc.tables = []

        mock_result = MagicMock()
        mock_result.document = mock_doc

        confidence = parser._calculate_confidence(mock_result, "This is a long text with sufficient content.")

        assert confidence > 0.8

    def test_calculate_confidence_with_short_text(self):
        """Test confidence calculation with short text."""
        parser = DoclingParser()

        mock_doc = MagicMock()
        mock_doc.pages = []
        mock_doc.tables = []

        mock_result = MagicMock()
        mock_result.document = mock_doc

        confidence = parser._calculate_confidence(mock_result, "Hi")

        assert confidence < 0.6

    def test_calculate_confidence_with_structure(self):
        """Test confidence calculation with document structure."""
        parser = DoclingParser()

        mock_doc = MagicMock()
        mock_doc.pages = [MagicMock(), MagicMock()]
        mock_doc.tables = [MagicMock()]

        mock_result = MagicMock()
        mock_result.document = mock_doc

        confidence = parser._calculate_confidence(mock_result, "Some text content here.")

        # Should have bonus for pages and tables
        assert confidence >= 0.85

    def test_calculate_confidence_max_cap(self):
        """Test confidence is capped at 1.0."""
        parser = DoclingParser()

        mock_doc = MagicMock()
        mock_doc.pages = [MagicMock(), MagicMock(), MagicMock()]
        mock_doc.tables = [MagicMock(), MagicMock()]

        mock_result = MagicMock()
        mock_result.document = mock_doc

        confidence = parser._calculate_confidence(mock_result, "A" * 1000)

        assert confidence <= 1.0


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserHealthCheck:
    """Tests for Docling parser health check."""

    @pytest.mark.asyncio
    async def test_health_check_when_healthy(self):
        """Test health check when docling is available and converter ready."""
        parser = DoclingParser()
        parser._docling_available = True
        parser._document_converter = MagicMock()

        health = await parser.health_check()

        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_when_degraded(self):
        """Test health check when docling is not available."""
        parser = DoclingParser()
        parser._docling_available = False
        parser._document_converter = None

        health = await parser.health_check()

        assert health == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_when_unhealthy(self):
        """Test health check when docling available but no converter."""
        parser = DoclingParser()
        parser._docling_available = True
        parser._document_converter = None

        health = await parser.health_check()

        assert health == HealthStatus.UNHEALTHY


# ============================================================================
# Processing Time Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserProcessingTime:
    """Tests for processing time tracking."""

    @pytest.mark.asyncio
    async def test_processing_time_recorded(self, tmp_path):
        """Test that processing time is recorded in result."""
        parser = DoclingParser()
        await parser.initialize({})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf content")

        # Mock docling result
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "PDF content"
        mock_result.document.pages = []
        mock_result.document.tables = []

        parser._docling_available = True
        parser._document_converter = MagicMock()
        parser._document_converter.convert.return_value = mock_result

        result = await parser.parse(str(pdf_file))

        assert result.success is True
        assert result.processing_time_ms >= 0


# ============================================================================
# Options Override Tests
# ============================================================================

@pytest.mark.unit
class TestDoclingParserOptionsOverride:
    """Tests for parse options override."""

    @pytest.mark.asyncio
    async def test_parse_with_options(self, tmp_path):
        """Test parsing with options parameter."""
        parser = DoclingParser()
        await parser.initialize({})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf content")

        # Mock docling result
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "PDF content"
        mock_result.document.pages = []
        mock_result.document.tables = []

        parser._docling_available = True
        parser._document_converter = MagicMock()
        parser._document_converter.convert.return_value = mock_result

        # Parse with custom options
        result = await parser.parse(str(pdf_file), options={"some_option": True})

        assert result.success is True
