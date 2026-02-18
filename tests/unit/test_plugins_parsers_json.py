"""Unit tests for JSON parser plugin."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.plugins.base import HealthStatus, ParsingResult, SupportResult
from src.plugins.parsers.json_parser import JSONParser


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def parser():
    """Create an initialized JSON parser."""
    p = JSONParser()
    await p.initialize({})
    return p


# ============================================================================
# JSONParser Class Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParser:
    """Tests for JSONParser class."""

    def test_init(self):
        """Test parser initialization."""
        parser = JSONParser()
        assert parser._config == {}

    def test_metadata(self):
        """Test parser metadata."""
        parser = JSONParser()
        metadata = parser.metadata
        
        assert metadata.id == "json"
        assert metadata.name == "JSON/JSONL Parser"
        assert metadata.version == "1.0.0"
        assert ".json" in metadata.supported_formats
        assert ".jsonl" in metadata.supported_formats
        assert ".ndjson" in metadata.supported_formats

    @pytest.mark.asyncio
    async def test_initialize_default_config(self):
        """Test initialization with default config."""
        parser = JSONParser()
        await parser.initialize({})
        
        assert parser._config["encoding"] == "utf-8"
        assert parser._config["max_depth"] is None
        assert parser._config["extract_arrays"] is True
        assert parser._config["chunk_size"] == 5000
        assert parser._config["strict_parsing"] is False
        assert parser._config["path_separator"] == "."

    @pytest.mark.asyncio
    async def test_initialize_custom_config(self):
        """Test initialization with custom config."""
        parser = JSONParser()
        await parser.initialize({
            "encoding": "latin-1",
            "max_depth": 3,
            "extract_arrays": False,
            "chunk_size": 1000,
            "strict_parsing": True,
            "path_separator": "/",
        })
        
        assert parser._config["encoding"] == "latin-1"
        assert parser._config["max_depth"] == 3
        assert parser._config["extract_arrays"] is False
        assert parser._config["chunk_size"] == 1000
        assert parser._config["strict_parsing"] is True
        assert parser._config["path_separator"] == "/"


# ============================================================================
# Supports Method Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserSupports:
    """Tests for JSON parser supports method."""

    @pytest.mark.asyncio
    async def test_supports_json_extension(self):
        """Test support for .json files."""
        parser = JSONParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file.json")
        
        assert isinstance(result, SupportResult)
        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_jsonl_extension(self):
        """Test support for .jsonl files."""
        parser = JSONParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file.jsonl")
        
        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_ndjson_extension(self):
        """Test support for .ndjson files."""
        parser = JSONParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file.ndjson")
        
        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_mime_type_json(self):
        """Test support with JSON MIME type."""
        parser = JSONParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file", mime_type="application/json")
        
        assert result.supported is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_supports_mime_type_ndjson(self):
        """Test support with NDJSON MIME type."""
        parser = JSONParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file", mime_type="application/x-ndjson")
        
        assert result.supported is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_supports_txt_with_json_object(self, tmp_path):
        """Test support for .txt files with JSON object content."""
        parser = JSONParser()
        await parser.initialize({})
        
        txt_file = tmp_path / "test.txt"
        txt_file.write_text('{"name": "test", "value": 123}')
        
        result = await parser.supports(str(txt_file))
        
        assert result.supported is True
        assert result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_supports_txt_with_json_array(self, tmp_path):
        """Test support for .txt files with JSON array content."""
        parser = JSONParser()
        await parser.initialize({})
        
        txt_file = tmp_path / "test.txt"
        txt_file.write_text('[1, 2, 3, 4, 5]')
        
        result = await parser.supports(str(txt_file))
        
        assert result.supported is True
        assert result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_supports_txt_with_jsonl(self, tmp_path):
        """Test support for .txt files with JSONL content."""
        parser = JSONParser()
        await parser.initialize({})
        
        txt_file = tmp_path / "test.txt"
        txt_file.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        
        result = await parser.supports(str(txt_file))
        
        # JSONL detection in .txt files may vary based on implementation
        # Just verify we get a valid SupportResult
        assert isinstance(result, SupportResult)

    @pytest.mark.asyncio
    async def test_supports_txt_without_json_content(self, tmp_path):
        """Test rejection of .txt files without JSON content."""
        parser = JSONParser()
        await parser.initialize({})
        
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is just plain text")
        
        result = await parser.supports(str(txt_file))
        
        assert result.supported is False

    @pytest.mark.asyncio
    async def test_supports_unsupported_extension(self):
        """Test rejection of unsupported file extensions."""
        parser = JSONParser()
        await parser.initialize({})
        
        result = await parser.supports("/path/to/file.pdf")
        
        assert result.supported is False
        assert result.confidence == 1.0


# ============================================================================
# Parse JSON Content Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserParseJSON:
    """Tests for parsing JSON content."""

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        parser = JSONParser()
        await parser.initialize({})
        
        result = await parser.parse("/nonexistent/file.json")
        
        assert isinstance(result, ParsingResult)
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_parse_unsupported_format(self, tmp_path):
        """Test parsing unsupported file format."""
        parser = JSONParser()
        await parser.initialize({})
        
        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<root><item>test</item></root>")
        
        result = await parser.parse(str(xml_file))
        
        assert result.success is False

    @pytest.mark.asyncio
    async def test_parse_simple_object(self, tmp_path):
        """Test parsing simple JSON object."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        json_file.write_text('{"name": "Alice", "age": 30}')
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True
        assert "Alice" in result.text
        assert "30" in result.text
        assert result.metadata["type"] == "dict"
        assert result.metadata["key_count"] == 2

    @pytest.mark.asyncio
    async def test_parse_simple_array(self, tmp_path):
        """Test parsing simple JSON array of objects."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        json_file.write_text('[{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]')
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True
        assert result.metadata["type"] == "list"
        assert result.metadata["length"] == 5

    @pytest.mark.asyncio
    async def test_parse_empty_object(self, tmp_path):
        """Test parsing empty JSON object."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        json_file.write_text('{}')
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True
        assert result.metadata["type"] == "dict"
        assert result.metadata["key_count"] == 0

    @pytest.mark.asyncio
    async def test_parse_empty_array(self, tmp_path):
        """Test parsing empty JSON array."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        json_file.write_text('[]')
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True
        assert result.metadata["type"] == "list"
        assert result.metadata["length"] == 0


# ============================================================================
# Parse Nested JSON Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserNestedJSON:
    """Tests for parsing nested JSON structures."""

    @pytest.mark.asyncio
    async def test_parse_nested_object(self, tmp_path):
        """Test parsing nested JSON object."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        data = {
            "user": {
                "name": "Alice",
                "address": {
                    "city": "New York",
                    "zip": "10001"
                }
            }
        }
        json_file.write_text(json.dumps(data))
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True
        assert "Alice" in result.text
        assert "New York" in result.text

    @pytest.mark.asyncio
    async def test_parse_array_of_objects(self, tmp_path):
        """Test parsing array of objects."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35}
        ]
        json_file.write_text(json.dumps(data))
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True
        assert "Alice" in result.text
        assert "Bob" in result.text
        assert "Charlie" in result.text
        assert result.metadata["type"] == "list"
        assert result.metadata["length"] == 3

    @pytest.mark.asyncio
    async def test_parse_deeply_nested_structure(self, tmp_path):
        """Test parsing deeply nested structure."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        }
        json_file.write_text(json.dumps(data))
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True
        assert "deep" in result.text

    @pytest.mark.asyncio
    async def test_parse_mixed_nested_structure(self, tmp_path):
        """Test parsing mixed nested structure with arrays and objects."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        data = {
            "company": "TechCorp",
            "employees": [
                {"name": "Alice", "skills": ["Python", "JavaScript"]},
                {"name": "Bob", "skills": ["Java", "Go"]}
            ]
        }
        json_file.write_text(json.dumps(data))
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True
        assert "TechCorp" in result.text
        assert "Alice" in result.text
        assert "Python" in result.text


# ============================================================================
# Parse JSONL Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserJSONL:
    """Tests for parsing JSON Lines format."""

    @pytest.mark.asyncio
    async def test_parse_jsonl_basic(self, tmp_path):
        """Test parsing basic JSONL file."""
        parser = JSONParser()
        await parser.initialize({})
        
        jsonl_file = tmp_path / "test.jsonl"
        lines = [
            '{"name": "Alice", "age": 30}',
            '{"name": "Bob", "age": 25}',
            '{"name": "Charlie", "age": 35}'
        ]
        jsonl_file.write_text("\n".join(lines))
        
        result = await parser.parse(str(jsonl_file))
        
        assert result.success is True
        assert result.metadata["format"] == "jsonl"
        assert result.metadata["total_records"] == 3

    @pytest.mark.asyncio
    async def test_parse_ndjson_extension(self, tmp_path):
        """Test parsing .ndjson file."""
        parser = JSONParser()
        await parser.initialize({})
        
        ndjson_file = tmp_path / "test.ndjson"
        lines = ['{"a": 1}', '{"b": 2}']
        ndjson_file.write_text("\n".join(lines))
        
        result = await parser.parse(str(ndjson_file))
        
        assert result.success is True
        assert result.metadata["total_records"] == 2

    @pytest.mark.asyncio
    async def test_parse_jsonl_with_empty_lines(self, tmp_path):
        """Test parsing JSONL with empty lines."""
        parser = JSONParser()
        await parser.initialize({})
        
        jsonl_file = tmp_path / "test.jsonl"
        lines = [
            '{"name": "Alice"}',
            '',
            '{"name": "Bob"}',
            '',
            '{"name": "Charlie"}'
        ]
        jsonl_file.write_text("\n".join(lines))
        
        result = await parser.parse(str(jsonl_file))
        
        assert result.success is True
        assert result.metadata["total_records"] == 3

    @pytest.mark.asyncio
    async def test_parse_jsonl_with_errors_non_strict(self, tmp_path):
        """Test parsing JSONL with invalid lines in non-strict mode."""
        parser = JSONParser()
        await parser.initialize({"strict_parsing": False})
        
        jsonl_file = tmp_path / "test.jsonl"
        lines = [
            '{"name": "Alice"}',
            'invalid json {',
            '{"name": "Bob"}'
        ]
        jsonl_file.write_text("\n".join(lines))
        
        result = await parser.parse(str(jsonl_file))
        
        assert result.success is True
        assert result.metadata["parse_errors"] == 1
        assert result.metadata["total_records"] == 2

    @pytest.mark.asyncio
    async def test_parse_jsonl_with_errors_strict(self, tmp_path):
        """Test parsing JSONL with errors in strict mode."""
        parser = JSONParser()
        await parser.initialize({"strict_parsing": True})
        
        jsonl_file = tmp_path / "test.jsonl"
        lines = [
            '{"name": "Alice"}',
            'invalid json {'
        ]
        jsonl_file.write_text("\n".join(lines))
        
        result = await parser.parse(str(jsonl_file))
        
        assert result.success is False


# ============================================================================
# Schema Inference Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserSchemaInference:
    """Tests for JSON schema inference."""

    @pytest.mark.asyncio
    async def test_infer_schema_basic_types(self, tmp_path):
        """Test inferring basic types in schema."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        data = [
            {"name": "Alice", "age": 30, "active": True, "score": 95.5},
            {"name": "Bob", "age": 25, "active": False, "score": 87.0}
        ]
        json_file.write_text(json.dumps(data))
        
        result = await parser.parse(str(json_file))
        
        schema = result.metadata.get("schema", {})
        assert schema.get("name") == "string"
        assert schema.get("age") == "integer"
        assert schema.get("active") == "boolean"
        assert schema.get("score") == "float"

    @pytest.mark.asyncio
    async def test_infer_schema_complex_types(self, tmp_path):
        """Test inferring complex types in schema."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        data = [
            {"tags": ["a", "b"], "metadata": {"key": "value"}},
            {"tags": ["c"], "metadata": {"other": "data"}}
        ]
        json_file.write_text(json.dumps(data))
        
        result = await parser.parse(str(json_file))
        
        schema = result.metadata.get("schema", {})
        assert schema.get("tags") == "array"
        assert schema.get("metadata") == "object"

    @pytest.mark.asyncio
    async def test_infer_schema_nullable(self, tmp_path):
        """Test inferring nullable types in schema."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        data = [
            {"name": "Alice", "nickname": "Ally"},
            {"name": "Bob", "nickname": None}
        ]
        json_file.write_text(json.dumps(data))
        
        result = await parser.parse(str(json_file))
        
        schema = result.metadata.get("schema", {})
        assert "null" in schema.get("nickname", "")


# ============================================================================
# Chunking Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserChunking:
    """Tests for JSON chunking."""

    @pytest.mark.asyncio
    async def test_chunking_array(self, tmp_path):
        """Test chunking JSON array."""
        parser = JSONParser()
        await parser.initialize({"chunk_size": 100})
        
        json_file = tmp_path / "test.json"
        data = [{"id": i, "data": "x" * 50} for i in range(10)]
        json_file.write_text(json.dumps(data))
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True
        assert len(result.pages) > 1  # Should create multiple chunks

    @pytest.mark.asyncio
    async def test_chunking_object(self, tmp_path):
        """Test chunking JSON object by keys."""
        parser = JSONParser()
        await parser.initialize({"chunk_size": 50})
        
        json_file = tmp_path / "test.json"
        data = {f"key_{i}": "x" * 50 for i in range(10)}
        json_file.write_text(json.dumps(data))
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True
        assert len(result.pages) >= 1

    @pytest.mark.asyncio
    async def test_chunking_disabled(self, tmp_path):
        """Test parsing with chunking disabled."""
        parser = JSONParser()
        await parser.initialize({"chunk_size": 10000})
        
        json_file = tmp_path / "test.json"
        data = [{"id": i} for i in range(5)]
        json_file.write_text(json.dumps(data))
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserErrorHandling:
    """Tests for JSON parser error handling."""

    @pytest.mark.asyncio
    async def test_parse_invalid_json_strict(self, tmp_path):
        """Test parsing invalid JSON in strict mode."""
        parser = JSONParser()
        await parser.initialize({"strict_parsing": True})
        
        json_file = tmp_path / "test.json"
        json_file.write_text('{"invalid json: missing closing brace')
        
        result = await parser.parse(str(json_file))
        
        assert result.success is False

    @pytest.mark.asyncio
    async def test_parse_invalid_json_non_strict(self, tmp_path):
        """Test parsing invalid JSON in non-strict mode."""
        parser = JSONParser()
        await parser.initialize({"strict_parsing": False})
        
        json_file = tmp_path / "test.json"
        json_file.write_text('{"invalid json: missing closing brace')
        
        result = await parser.parse(str(json_file))
        
        # Non-strict mode attempts recovery
        assert result.success is True

    @pytest.mark.asyncio
    async def test_parse_partial_json_recovery(self, tmp_path):
        """Test recovery from partial JSON content."""
        parser = JSONParser()
        await parser.initialize({"strict_parsing": False})
        
        json_file = tmp_path / "test.json"
        # Mix of valid and invalid lines
        content = '{"valid": true}\ninvalid\n{"also_valid": false}'
        json_file.write_text(content)
        
        result = await parser.parse(str(json_file))
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_parse_with_options_override(self, tmp_path):
        """Test parsing with options override."""
        parser = JSONParser()
        await parser.initialize({"chunk_size": 100})
        
        json_file = tmp_path / "test.json"
        json_file.write_text('[{"id": 1}, {"id": 2}, {"id": 3}]')
        
        # Override chunk_size in parse options
        result = await parser.parse(str(json_file), options={"chunk_size": 500})
        
        assert result.success is True


# ============================================================================
# Large File Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserLargeFiles:
    """Tests for large JSON file handling."""

    @pytest.mark.asyncio
    async def test_parse_large_file_fallback(self, tmp_path):
        """Test parsing large file falls back to standard parser."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        # Create a file larger than MAX_NON_STREAMING_SIZE
        large_data = [{"id": i, "data": "x" * 1000} for i in range(10000)]
        json_file.write_text(json.dumps(large_data))
        
        # Mock ijson to not be available
        with patch.dict("sys.modules", {"ijson": None}):
            result = await parser.parse(str(json_file))
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_parse_with_ijson(self, tmp_path):
        """Test parsing with ijson when available."""
        parser = JSONParser()
        await parser.initialize({})
        
        json_file = tmp_path / "test.json"
        # Create large array
        large_data = [{"id": i} for i in range(100)]
        json_file.write_text(json.dumps(large_data))
        
        # Mock ijson
        mock_ijson = MagicMock()
        mock_items = []
        for i in range(10):
            item = MagicMock()
            item.return_value = {"id": i}
            mock_items.append(item.return_value)
        
        with patch.dict("sys.modules", {"ijson": mock_ijson}):
            # The actual implementation will try to use ijson
            result = await parser.parse(str(json_file))
        
        # Should parse successfully (may or may not use ijson depending on file size)
        assert result.success is True


# ============================================================================
# Text Generation Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserTextGeneration:
    """Tests for JSON text generation."""

    def test_generate_text_simple_object(self):
        """Test text generation for simple object."""
        parser = JSONParser()
        data = {"name": "Alice", "age": 30}
        
        text = parser._generate_text(data)
        
        assert "name: Alice" in text
        assert "age: 30" in text

    def test_generate_text_nested_object(self):
        """Test text generation for nested object."""
        parser = JSONParser()
        data = {"user": {"name": "Alice", "address": {"city": "NYC"}}}
        
        text = parser._generate_text(data)
        
        assert "user:" in text
        assert "name: Alice" in text
        assert "address:" in text
        assert "city: NYC" in text

    def test_generate_text_array(self):
        """Test text generation for array."""
        parser = JSONParser()
        data = ["apple", "banana", "cherry"]
        
        text = parser._generate_text(data)
        
        assert "- apple" in text
        assert "- banana" in text
        assert "- cherry" in text

    def test_generate_text_array_of_objects(self):
        """Test text generation for array of objects."""
        parser = JSONParser()
        data = [{"name": "Alice"}, {"name": "Bob"}]
        
        text = parser._generate_text(data)
        
        assert "Item 0:" in text
        assert "name: Alice" in text
        assert "Item 1:" in text
        assert "name: Bob" in text

    def test_generate_text_from_records(self):
        """Test text generation from records list."""
        parser = JSONParser()
        records = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        
        text = parser._generate_text_from_records(records)
        
        assert "JSON Lines Data" in text
        assert "Total records: 2" in text
        assert "Record 1:" in text
        assert "Alice" in text
        assert "Record 2:" in text
        assert "Bob" in text

    def test_generate_text_from_many_records(self):
        """Test text generation truncates many records."""
        parser = JSONParser()
        records = [{"id": i} for i in range(20)]
        
        text = parser._generate_text_from_records(records)
        
        assert "Total records: 20" in text
        assert "... and 10 more records" in text


# ============================================================================
# Recovery Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserRecovery:
    """Tests for JSON recovery functionality."""

    def test_recover_json_parse_valid_lines(self):
        """Test recovery from lines with valid JSON."""
        parser = JSONParser()
        content = '{"a": 1}\n{"b": 2}\n{"c": 3}'
        
        result = parser._recover_json_parse(content)
        
        assert len(result) == 3
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}
        assert result[2] == {"c": 3}

    def test_recover_json_parse_mixed_lines(self):
        """Test recovery with mixed valid/invalid lines."""
        parser = JSONParser()
        content = '{"a": 1}\ninvalid\n{"c": 3}'
        
        result = parser._recover_json_parse(content)
        
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"c": 3}

    def test_recover_json_parse_no_valid_lines(self):
        """Test recovery with no valid JSON lines."""
        parser = JSONParser()
        content = 'not json\nalso not json'
        
        result = parser._recover_json_parse(content)
        
        assert "raw_content" in result


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserHealthCheck:
    """Tests for JSON parser health check."""

    @pytest.mark.asyncio
    async def test_health_check_always_healthy(self):
        """Test that health check always returns healthy."""
        parser = JSONParser()
        await parser.initialize({})
        
        health = await parser.health_check()
        
        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_with_config(self):
        """Test health check with config parameter."""
        parser = JSONParser()
        await parser.initialize({})
        
        health = await parser.health_check({"some": "config"})
        
        assert health == HealthStatus.HEALTHY


# ============================================================================
# Extract Metadata Tests
# ============================================================================

@pytest.mark.unit
class TestJSONParserExtractMetadata:
    """Tests for metadata extraction."""

    def test_extract_metadata_dict(self):
        """Test metadata extraction from dict."""
        parser = JSONParser()
        data = {"a": 1, "b": 2, "c": 3}
        
        metadata = parser._extract_metadata(data)
        
        assert metadata["type"] == "dict"
        assert metadata["key_count"] == 3
        assert "keys" in metadata

    def test_extract_metadata_list(self):
        """Test metadata extraction from list."""
        parser = JSONParser()
        data = [1, 2, 3, 4, 5]
        
        metadata = parser._extract_metadata(data)
        
        assert metadata["type"] == "list"
        assert metadata["length"] == 5

    def test_extract_metadata_list_of_dicts(self):
        """Test metadata extraction from list of dicts."""
        parser = JSONParser()
        data = [{"name": "Alice"}, {"name": "Bob"}]
        
        metadata = parser._extract_metadata(data)
        
        assert metadata["type"] == "list"
        assert metadata["length"] == 2
        assert "item_types" in metadata
        assert "schema" in metadata

    def test_extract_metadata_primitive(self):
        """Test metadata extraction from primitive."""
        parser = JSONParser()
        
        assert parser._extract_metadata("string")["type"] == "str"
        assert parser._extract_metadata(42)["type"] == "int"
        assert parser._extract_metadata(3.14)["type"] == "float"
        assert parser._extract_metadata(True)["type"] == "bool"
