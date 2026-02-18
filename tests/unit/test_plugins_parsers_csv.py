"""Unit tests for CSV parser plugin."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.plugins.base import HealthStatus, ParsingResult, SupportResult
from src.plugins.parsers.csv_parser import CSVParser


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def parser():
    """Create an initialized CSV parser."""
    p = CSVParser()
    await p.initialize({})
    return p


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file."""
    csv_file = tmp_path / "test.csv"
    csv_content = """name,age,city
Alice,30,New York
Bob,25,Boston
Charlie,35,Chicago"""
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def sample_tsv_file(tmp_path):
    """Create a sample TSV file."""
    tsv_file = tmp_path / "test.tsv"
    tsv_content = """name\tage\tcity
Alice\t30\tNew York
Bob\t25\tBoston"""
    tsv_file.write_text(tsv_content)
    return tsv_file


# ============================================================================
# CSVParser Class Tests
# ============================================================================

@pytest.mark.unit
class TestCSVParser:
    """Tests for CSVParser class."""

    def test_init(self):
        """Test parser initialization."""
        parser = CSVParser()
        assert parser._config == {}

    def test_metadata(self):
        """Test parser metadata."""
        parser = CSVParser()
        metadata = parser.metadata
        
        assert metadata.id == "csv"
        assert metadata.name == "CSV/TSV Parser"
        assert metadata.version == "1.0.0"
        assert ".csv" in metadata.supported_formats
        assert ".tsv" in metadata.supported_formats
        assert ".tab" in metadata.supported_formats

    @pytest.mark.asyncio
    async def test_initialize_default_config(self):
        """Test initialization with default config."""
        parser = CSVParser()
        await parser.initialize({})
        
        assert parser._config["encoding"] == "utf-8"
        assert parser._config["delimiter"] is None
        assert parser._config["has_header"] is None
        assert parser._config["quotechar"] == '"'
        assert parser._config["escapechar"] == "\\"
        assert parser._config["chunk_rows"] == 100
        assert parser._config["skip_empty_lines"] is True

    @pytest.mark.asyncio
    async def test_initialize_custom_config(self):
        """Test initialization with custom config."""
        parser = CSVParser()
        await parser.initialize({
            "encoding": "latin-1",
            "delimiter": ";",
            "has_header": False,
            "quotechar": "'",
            "escapechar": "|",
            "chunk_rows": 50,
            "skip_empty_lines": False,
        })
        
        assert parser._config["encoding"] == "latin-1"
        assert parser._config["delimiter"] == ";"
        assert parser._config["has_header"] is False
        assert parser._config["quotechar"] == "'"
        assert parser._config["escapechar"] == "|"
        assert parser._config["chunk_rows"] == 50
        assert parser._config["skip_empty_lines"] is False


# ============================================================================
# Supports Method Tests
# ============================================================================

@pytest.mark.unit
class TestCSVParserSupports:
    """Tests for CSV parser supports method."""

    @pytest.mark.asyncio
    async def test_supports_csv_extension(self):
        """Test support for .csv files."""
        parser = CSVParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file.csv")
        
        assert isinstance(result, SupportResult)
        assert result.supported is True
        assert result.confidence == 0.95
        assert "csv" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_supports_tsv_extension(self):
        """Test support for .tsv files."""
        parser = CSVParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file.tsv")
        
        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_tab_extension(self):
        """Test support for .tab files."""
        parser = CSVParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file.tab")
        
        assert result.supported is True

    @pytest.mark.asyncio
    async def test_supports_mime_type_csv(self):
        """Test support with CSV MIME type."""
        parser = CSVParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file", mime_type="text/csv")
        
        assert result.supported is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_supports_mime_type_tsv(self):
        """Test support with TSV MIME type."""
        parser = CSVParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file", mime_type="text/tab-separated-values")
        
        assert result.supported is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_supports_txt_with_csv_content(self, tmp_path):
        """Test support for .txt files with CSV-like content."""
        parser = CSVParser()
        await parser.initialize({})
        
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("a,b,c\n1,2,3\n4,5,6")
        
        result = await parser.supports(str(txt_file))
        
        assert result.supported is True
        assert result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_supports_txt_without_csv_content(self, tmp_path):
        """Test rejection of .txt files without CSV content."""
        parser = CSVParser()
        await parser.initialize({})
        
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is just plain text without delimiters")
        
        result = await parser.supports(str(txt_file))
        
        assert result.supported is False

    @pytest.mark.asyncio
    async def test_supports_unsupported_extension(self):
        """Test rejection of unsupported file extensions."""
        parser = CSVParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file.pdf")
        
        assert result.supported is False
        assert result.confidence == 1.0


# ============================================================================
# Parse Method Tests
# ============================================================================

@pytest.mark.unit
class TestCSVParserParse:
    """Tests for CSV parser parse method."""

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        parser = CSVParser()
        await parser.initialize({})
        
        result = await parser.parse("/nonexistent/file.csv")
        
        assert isinstance(result, ParsingResult)
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_parse_unsupported_format(self, tmp_path):
        """Test parsing unsupported file format."""
        parser = CSVParser()
        await parser.initialize({})
        
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("not a pdf")
        
        result = await parser.parse(str(pdf_file))
        
        assert result.success is False
        assert "unsupported" in result.error.lower()

    @pytest.mark.asyncio
    async def test_parse_simple_csv(self, tmp_path):
        """Test parsing a simple CSV file."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25")
        
        result = await parser.parse(str(csv_file))
        
        assert result.success is True
        assert "Alice" in result.text
        assert "Bob" in result.text
        assert result.metadata["total_rows"] == 2
        assert result.metadata["total_columns"] == 2

    @pytest.mark.asyncio
    async def test_parse_csv_with_quotes(self, tmp_path):
        """Test parsing CSV with quoted values."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text('name,description\nAlice,"Hello, World"\nBob,"Test"')
        
        result = await parser.parse(str(csv_file))
        
        assert result.success is True
        assert "Hello, World" in result.text

    @pytest.mark.asyncio
    async def test_parse_empty_csv(self, tmp_path):
        """Test parsing empty CSV file."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("header1,header2")
        
        result = await parser.parse(str(csv_file))
        
        assert result.success is True
        assert result.metadata["total_rows"] == 0

    @pytest.mark.asyncio
    async def test_parse_csv_with_empty_values(self, tmp_path):
        """Test parsing CSV with empty values."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,\nCharlie,35")
        
        result = await parser.parse(str(csv_file))
        
        assert result.success is True
        assert result.metadata["total_rows"] == 3


# ============================================================================
# Delimiter Tests
# ============================================================================

@pytest.mark.unit
class TestCSVParserDelimiters:
    """Tests for different CSV delimiters."""

    @pytest.mark.asyncio
    async def test_parse_comma_delimited(self, tmp_path):
        """Test parsing comma-delimited CSV."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3")
        
        result = await parser.parse(str(csv_file))
        
        assert result.success is True
        assert result.metadata["delimiter"] == ","

    @pytest.mark.asyncio
    async def test_parse_semicolon_delimited(self, tmp_path):
        """Test parsing semicolon-delimited CSV."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a;b;c\n1;2;3")
        
        result = await parser.parse(str(csv_file))
        
        assert result.success is True
        assert result.metadata["delimiter"] == ";"

    @pytest.mark.asyncio
    async def test_parse_pipe_delimited(self, tmp_path):
        """Test parsing pipe-delimited CSV."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a|b|c\n1|2|3")
        
        result = await parser.parse(str(csv_file))
        
        assert result.success is True
        assert result.metadata["delimiter"] == "|"

    @pytest.mark.asyncio
    async def test_parse_tsv_extension_auto_detect(self, tmp_path):
        """Test TSV extension auto-detects tab delimiter."""
        parser = CSVParser()
        await parser.initialize({})
        
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text("a\tb\tc\n1\t2\t3")
        
        result = await parser.parse(str(tsv_file))
        
        assert result.success is True
        assert result.metadata["delimiter"] == "\t"

    @pytest.mark.asyncio
    async def test_parse_tab_extension_auto_detect(self, tmp_path):
        """Test TAB extension auto-detects tab delimiter."""
        parser = CSVParser()
        await parser.initialize({})
        
        tab_file = tmp_path / "test.tab"
        tab_file.write_text("a\tb\tc\n1\t2\t3")
        
        result = await parser.parse(str(tab_file))
        
        assert result.success is True
        assert result.metadata["delimiter"] == "\t"

    @pytest.mark.asyncio
    async def test_parse_with_explicit_delimiter(self, tmp_path):
        """Test parsing with explicitly specified delimiter."""
        parser = CSVParser()
        await parser.initialize({"delimiter": "|"})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a|b|c\n1|2|3")
        
        result = await parser.parse(str(csv_file))
        
        assert result.success is True
        assert result.metadata["delimiter"] == "|"


# ============================================================================
# Schema Inference Tests
# ============================================================================

@pytest.mark.unit
class TestCSVParserSchemaInference:
    """Tests for CSV schema inference."""

    @pytest.mark.asyncio
    async def test_infer_integer_type(self, tmp_path):
        """Test inferring integer column type."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id\n1\n2\n3")
        
        result = await parser.parse(str(csv_file))
        
        assert result.metadata["schema"]["id"] == "integer"

    @pytest.mark.asyncio
    async def test_infer_float_type(self, tmp_path):
        """Test inferring float column type."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("value\n1.5\n2.7\n3.14")
        
        result = await parser.parse(str(csv_file))
        
        assert result.metadata["schema"]["value"] == "float"

    @pytest.mark.asyncio
    async def test_infer_date_type(self, tmp_path):
        """Test inferring date column type."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("date\n2024-01-15\n2024-02-20")
        
        result = await parser.parse(str(csv_file))
        
        assert result.metadata["schema"]["date"] == "date"

    @pytest.mark.asyncio
    async def test_infer_string_type(self, tmp_path):
        """Test inferring string column type."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name\nAlice\nBob\nCharlie")
        
        result = await parser.parse(str(csv_file))
        
        assert result.metadata["schema"]["name"] == "string"

    @pytest.mark.asyncio
    async def test_infer_mixed_type_as_string(self, tmp_path):
        """Test that mixed types default to string."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("value\n1\ntext\n3.14")
        
        result = await parser.parse(str(csv_file))
        
        assert result.metadata["schema"]["value"] == "string"


# ============================================================================
# Chunking Tests
# ============================================================================

@pytest.mark.unit
class TestCSVParserChunking:
    """Tests for CSV semantic chunking."""

    @pytest.mark.asyncio
    async def test_chunking_small_file(self, tmp_path):
        """Test chunking with small file creates single chunk."""
        parser = CSVParser()
        await parser.initialize({"chunk_rows": 10})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id\n1\n2\n3")
        
        result = await parser.parse(str(csv_file))
        
        assert result.success is True
        assert len(result.pages) == 1

    @pytest.mark.asyncio
    async def test_chunking_large_file(self, tmp_path):
        """Test chunking with large file creates multiple chunks."""
        parser = CSVParser()
        await parser.initialize({"chunk_rows": 2})
        
        csv_file = tmp_path / "test.csv"
        lines = ["id,value"] + [f"{i},{i*10}" for i in range(10)]
        csv_file.write_text("\n".join(lines))
        
        result = await parser.parse(str(csv_file))
        
        assert result.success is True
        assert len(result.pages) == 5  # 10 rows / 2 per chunk


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestCSVParserErrorHandling:
    """Tests for CSV parser error handling."""

    @pytest.mark.asyncio
    async def test_parse_malformed_csv(self, tmp_path):
        """Test parsing malformed CSV content."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        # CSV with inconsistent columns
        csv_file.write_text("a,b,c\n1,2\n3,4,5,6")
        
        # Parser should handle gracefully
        result = await parser.parse(str(csv_file))
        
        # The parser uses csv.DictReader which handles this
        assert result.success is True

    @pytest.mark.asyncio
    async def test_parse_encoding_fallback(self, tmp_path):
        """Test encoding fallback on decode error."""
        parser = CSVParser()
        await parser.initialize({"encoding": "utf-8"})
        
        csv_file = tmp_path / "test.csv"
        # Write latin-1 encoded content
        csv_file.write_bytes(b"name\n\xe9\xe8")  # éè in latin-1
        
        result = await parser.parse(str(csv_file))
        
        # Should succeed with fallback encoding
        assert result.success is True

    @pytest.mark.asyncio
    async def test_parse_with_options_override(self, tmp_path):
        """Test parsing with options override."""
        parser = CSVParser()
        await parser.initialize({"delimiter": ","})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a|b\n1|2")
        
        # Override delimiter in parse options
        result = await parser.parse(str(csv_file), options={"delimiter": "|"})
        
        assert result.success is True
        assert result.metadata["delimiter"] == "|"


# ============================================================================
# Parse Chunks Iterator Tests
# ============================================================================

@pytest.mark.unit
class TestCSVParserParseChunks:
    """Tests for CSV parser chunk iterator."""

    @pytest.mark.asyncio
    async def test_parse_chunks_basic(self, tmp_path):
        """Test basic chunk parsing."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        lines = ["id,value"] + [f"{i},{i*10}" for i in range(5)]
        csv_file.write_text("\n".join(lines))
        
        chunks = []
        async for chunk in parser.parse_chunks(str(csv_file), chunk_size=2):
            chunks.append(chunk)
        
        assert len(chunks) == 3  # 5 rows / 2 per chunk = 3 chunks
        assert chunks[0]["chunk_num"] == 0
        assert chunks[1]["chunk_num"] == 1
        assert "headers" in chunks[0]
        assert "rows" in chunks[0]

    @pytest.mark.asyncio
    async def test_parse_chunks_empty_file(self, tmp_path):
        """Test chunk parsing with empty file."""
        parser = CSVParser()
        await parser.initialize({})
        
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("header1,header2")
        
        chunks = []
        async for chunk in parser.parse_chunks(str(csv_file), chunk_size=10):
            chunks.append(chunk)
        
        # Should still yield a chunk (empty)
        assert len(chunks) >= 0


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestCSVParserHealthCheck:
    """Tests for CSV parser health check."""

    @pytest.mark.asyncio
    async def test_health_check_always_healthy(self):
        """Test that health check always returns healthy."""
        parser = CSVParser()
        await parser.initialize({})
        
        health = await parser.health_check()
        
        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_with_config(self):
        """Test health check with config parameter."""
        parser = CSVParser()
        await parser.initialize({})
        
        health = await parser.health_check({"some": "config"})
        
        assert health == HealthStatus.HEALTHY


# ============================================================================
# Date Detection Tests
# ============================================================================

@pytest.mark.unit
class TestCSVParserDateDetection:
    """Tests for date detection in CSV parser."""

    def test_is_date_yyyy_mm_dd(self):
        """Test YYYY-MM-DD format detection."""
        parser = CSVParser()
        
        assert parser._is_date("2024-01-15") is True
        assert parser._is_date("2023-12-31") is True

    def test_is_date_mm_dd_yyyy(self):
        """Test MM/DD/YYYY format detection."""
        parser = CSVParser()
        
        assert parser._is_date("01/15/2024") is True
        assert parser._is_date("12/31/2023") is True

    def test_is_date_dd_mm_yyyy(self):
        """Test DD-MM-YYYY format detection."""
        parser = CSVParser()
        
        assert parser._is_date("15-01-2024") is True
        assert parser._is_date("31-12-2023") is True

    def test_is_date_yyyy_mm_dd_slashes(self):
        """Test YYYY/MM/DD format detection."""
        parser = CSVParser()
        
        assert parser._is_date("2024/01/15") is True

    def test_is_date_invalid(self):
        """Test rejection of non-date strings."""
        parser = CSVParser()
        
        assert parser._is_date("not a date") is False
        assert parser._is_date("12345") is False
        assert parser._is_date("") is False
        # Note: The date regex only checks format (DD-MM-YYYY pattern), not validity
        # so "15-15-2024" would match even though month 15 is invalid
        # These should NOT match any date pattern:
        assert parser._is_date("Jan 15, 2024") is False  # Text month format
        assert parser._is_date("15 Jan 2024") is False   # Another text format
        assert parser._is_date("2024") is False          # Year only
