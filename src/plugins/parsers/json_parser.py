"""JSON/JSONL parser plugin for structured data processing.

This module provides JSON and JSON Lines parsing capabilities with support for
nested structures, array processing, and streaming large files.
"""

import json
import logging
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


class JSONParser(ParserPlugin):
    """JSON/JSONL parser plugin for structured data.
    
    This parser handles JSON files and JSON Lines (JSONL) format,
    supporting nested structures and array processing. Large files
    are processed with streaming to manage memory usage.
    
    Example:
        >>> parser = JSONParser()
        >>> await parser.initialize({})
        >>> result = await parser.parse("/path/to/data.json")
        >>> print(result.text)
    """

    SUPPORTED_FORMATS = [".json", ".jsonl", ".ndjson"]

    MIME_TYPE_MAP = {
        "application/json": 1.0,
        "application/x-ndjson": 1.0,
        "application/jsonlines": 1.0,
        "text/plain": 0.3,
    }

    # Default maximum file size for non-streaming (10MB)
    MAX_NON_STREAMING_SIZE = 10 * 1024 * 1024

    def __init__(self) -> None:
        """Initialize the JSON parser."""
        self._config: dict[str, Any] = {}

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="json",
            name="JSON/JSONL Parser",
            version="1.0.0",
            type=PluginType.PARSER,
            description="Parser for JSON and JSON Lines (JSONL) files with nested structure support",
            author="Pipeline Team",
            supported_formats=self.SUPPORTED_FORMATS,
            requires_auth=False,
        )

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the parser with configuration.
        
        Args:
            config: Parser configuration options including:
                - encoding: File encoding (default: utf-8)
                - max_depth: Maximum nesting depth to flatten (default: unlimited)
                - extract_arrays: Whether to extract arrays as separate chunks (default: True)
                - chunk_size: Target chunk size in characters (default: 5000)
                - strict_parsing: Whether to fail on invalid JSON (default: False)
                - path_separator: Separator for flattened keys (default: ".")
        """
        self._config = {
            "encoding": config.get("encoding", "utf-8"),
            "max_depth": config.get("max_depth"),
            "extract_arrays": config.get("extract_arrays", True),
            "chunk_size": config.get("chunk_size", 5000),
            "strict_parsing": config.get("strict_parsing", False),
            "path_separator": config.get("path_separator", "."),
        }
        logger.info("JSON parser initialized")

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
        if extension in (".json", ".jsonl", ".ndjson"):
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

        # Try to detect JSON content in .txt files
        if extension == ".txt":
            confidence = await self._detect_json_content(file_path)
            if confidence > 0.7:
                return SupportResult(
                    supported=True,
                    confidence=confidence,
                    reason="Detected JSON content in text file",
                )

        return SupportResult(
            supported=False,
            confidence=1.0,
            reason=f"Unsupported file format: {extension}",
        )

    async def _detect_json_content(self, file_path: str) -> float:
        """Detect if a text file contains JSON content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        try:
            with open(file_path, encoding=self._config["encoding"], errors="ignore") as f:
                content = f.read(4096).strip()

                if not content:
                    return 0.0

                # Check for JSON-like structure
                if (content.startswith("{") and content.endswith("}")) or \
                   (content.startswith("[") and content.endswith("]")):
                    try:
                        json.loads(content)
                        return 0.9
                    except json.JSONDecodeError:
                        # Partial match
                        return 0.5

                # Check for JSONL format
                lines = content.split("\n")[:3]
                if all(line.strip() and (line.strip().startswith("{") or line.strip().startswith("["))
                       for line in lines if line.strip()):
                    return 0.7

                return 0.0

        except Exception:
            return 0.0

    async def parse(
        self,
        file_path: str,
        options: dict[str, Any] | None = None,
    ) -> ParsingResult:
        """Parse a JSON/JSONL file and extract content.
        
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
            # Determine file type and size
            extension = path.suffix.lower()
            file_size = path.stat().st_size

            if extension in (".jsonl", ".ndjson"):
                result = await self._parse_jsonl(file_path, opts)
            elif file_size > self.MAX_NON_STREAMING_SIZE:
                result = await self._parse_large_json(file_path, opts)
            else:
                result = await self._parse_json(file_path, opts)

            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result

        except Exception as e:
            logger.error(f"JSON parsing failed: {e}", exc_info=True)
            return ParsingResult(
                success=False,
                error=f"Parsing failed: {e!s}",
            )

    async def _parse_json(
        self,
        file_path: str,
        options: dict[str, Any],
    ) -> ParsingResult:
        """Parse a standard JSON file.
        
        Args:
            file_path: Path to the file
            options: Parsing options
            
        Returns:
            ParsingResult
        """
        path = Path(file_path)
        encoding = options.get("encoding", "utf-8")

        with open(file_path, encoding=encoding, errors="replace") as f:
            content = f.read()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            if options.get("strict_parsing", False):
                raise
            # Try to recover - parse line by line
            logger.warning(f"JSON parse error, attempting recovery: {e}")
            data = self._recover_json_parse(content)

        # Generate text representation
        full_text = self._generate_text(data)

        # Extract metadata
        metadata = self._extract_metadata(data)
        metadata["file_size_bytes"] = path.stat().st_size

        # Create chunks
        chunks = self._create_chunks(data, options.get("chunk_size", 5000))

        # Calculate confidence
        confidence = 0.95 if data else 0.5

        return ParsingResult(
            success=True,
            text=full_text,
            pages=chunks or [full_text],
            metadata=metadata,
            format=".json",
            parser_used="json",
            confidence=confidence,
        )

    async def _parse_jsonl(
        self,
        file_path: str,
        options: dict[str, Any],
    ) -> ParsingResult:
        """Parse a JSON Lines file.
        
        Args:
            file_path: Path to the file
            options: Parsing options
            
        Returns:
            ParsingResult
        """
        path = Path(file_path)
        encoding = options.get("encoding", "utf-8")
        chunk_size = options.get("chunk_size", 5000)

        all_records: list[dict[str, Any]] = []
        chunks: list[str] = []
        current_chunk: list[dict[str, Any]] = []
        current_chunk_size = 0

        line_count = 0
        error_count = 0

        with open(file_path, encoding=encoding, errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                line_count += 1

                try:
                    record = json.loads(line)
                    all_records.append(record)
                    current_chunk.append(record)

                    # Estimate chunk size
                    record_size = len(json.dumps(record))
                    current_chunk_size += record_size

                    if current_chunk_size >= chunk_size:
                        chunk_text = self._format_chunk(current_chunk)
                        chunks.append(chunk_text)
                        current_chunk = []
                        current_chunk_size = 0

                except json.JSONDecodeError as e:
                    error_count += 1
                    if options.get("strict_parsing", False):
                        raise
                    logger.warning(f"Line {line_num}: JSON decode error: {e}")

        # Add remaining records
        if current_chunk:
            chunk_text = self._format_chunk(current_chunk)
            chunks.append(chunk_text)

        # Generate full text
        full_text = self._generate_text_from_records(all_records)

        # Extract metadata
        metadata = {
            "format": "jsonl",
            "total_records": len(all_records),
            "lines_parsed": line_count,
            "parse_errors": error_count,
            "file_size_bytes": path.stat().st_size,
        }

        # Add schema info from first record if available
        if all_records:
            metadata["schema"] = self._infer_schema(all_records[:100])

        confidence = 0.95 if error_count == 0 else max(0.5, 1.0 - (error_count / max(line_count, 1)))

        return ParsingResult(
            success=True,
            text=full_text,
            pages=chunks or [full_text],
            metadata=metadata,
            format=".jsonl",
            parser_used="json",
            confidence=confidence,
        )

    async def _parse_large_json(
        self,
        file_path: str,
        options: dict[str, Any],
    ) -> ParsingResult:
        """Parse a large JSON file using streaming ijson if available.
        
        Args:
            file_path: Path to the file
            options: Parsing options
            
        Returns:
            ParsingResult
        """
        try:
            import ijson
            return await self._parse_with_ijson(file_path, options)
        except ImportError:
            logger.warning("ijson not available, falling back to standard parsing")
            return await self._parse_json(file_path, options)

    async def _parse_with_ijson(
        self,
        file_path: str,
        options: dict[str, Any],
    ) -> ParsingResult:
        """Parse large JSON using ijson streaming.
        
        Args:
            file_path: Path to the file
            options: Parsing options
            
        Returns:
            ParsingResult
        """
        import ijson

        path = Path(file_path)
        encoding = options.get("encoding", "utf-8")

        chunks: list[str] = []
        current_chunk: list[Any] = []
        current_chunk_size = 0
        chunk_target = options.get("chunk_size", 5000)

        total_items = 0

        with open(file_path, "rb") as f:
            # Try to parse as array of objects
            try:
                for item in ijson.items(f, "item"):
                    total_items += 1
                    current_chunk.append(item)
                    current_chunk_size += len(json.dumps(item))

                    if current_chunk_size >= chunk_target:
                        chunk_text = self._format_chunk(current_chunk)
                        chunks.append(chunk_text)
                        current_chunk = []
                        current_chunk_size = 0

                # Add remaining items
                if current_chunk:
                    chunk_text = self._format_chunk(current_chunk)
                    chunks.append(chunk_text)

                full_text = f"JSON Array Data\nTotal items: {total_items}\n\n"
                full_text += "\n\n".join(chunks[:20])  # Preview first 20 chunks

                metadata = {
                    "format": "json",
                    "total_items": total_items,
                    "streaming": True,
                    "file_size_bytes": path.stat().st_size,
                }

                return ParsingResult(
                    success=True,
                    text=full_text,
                    pages=chunks,
                    metadata=metadata,
                    format=".json",
                    parser_used="json-streaming",
                    confidence=0.9,
                )

            except Exception as e:
                logger.warning(f"ijson streaming failed: {e}, falling back")
                return await self._parse_json(file_path, options)

    def _recover_json_parse(self, content: str) -> Any:
        """Attempt to recover from JSON parse errors.
        
        Args:
            content: Raw content string
            
        Returns:
            Parsed data (best effort)
        """
        # Try to find valid JSON subset
        lines = content.strip().split("\n")
        valid_objects = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                valid_objects.append(obj)
            except json.JSONDecodeError:
                pass

        if valid_objects:
            return valid_objects

        # Last resort: return as string
        return {"raw_content": content}

    def _generate_text(self, data: Any, indent: int = 0) -> str:
        """Generate human-readable text from JSON data.
        
        Args:
            data: Parsed JSON data
            indent: Current indentation level
            
        Returns:
            Text representation
        """
        lines = []
        indent_str = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{indent_str}{key}:")
                    lines.append(self._generate_text(value, indent + 1))
                else:
                    lines.append(f"{indent_str}{key}: {value}")

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{indent_str}Item {i}:")
                    lines.append(self._generate_text(item, indent + 1))
                else:
                    lines.append(f"{indent_str}- {item}")

        else:
            lines.append(f"{indent_str}{data}")

        return "\n".join(lines)

    def _generate_text_from_records(self, records: list[dict[str, Any]]) -> str:
        """Generate text from a list of records.
        
        Args:
            records: List of record dictionaries
            
        Returns:
            Text representation
        """
        lines = ["JSON Lines Data", f"Total records: {len(records)}\n"]

        # Show first 10 records
        for i, record in enumerate(records[:10]):
            lines.append(f"Record {i + 1}:")
            for key, value in record.items():
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, ensure_ascii=False)[:200]
                    if len(json.dumps(value, ensure_ascii=False)) > 200:
                        value_str += "..."
                else:
                    value_str = str(value)[:200]
                lines.append(f"  {key}: {value_str}")
            lines.append("")

        if len(records) > 10:
            lines.append(f"... and {len(records) - 10} more records")

        return "\n".join(lines)

    def _create_chunks(self, data: Any, chunk_size: int) -> list[str]:
        """Create semantic chunks from JSON data.
        
        Args:
            data: Parsed JSON data
            chunk_size: Target chunk size in characters
            
        Returns:
            List of chunk strings
        """
        chunks: list[str] = []

        if isinstance(data, list):
            # Split array into chunks
            current_chunk = []
            current_size = 0

            for item in data:
                item_text = json.dumps(item, ensure_ascii=False)
                item_size = len(item_text)

                if current_size + item_size > chunk_size and current_chunk:
                    chunks.append(self._format_chunk(current_chunk))
                    current_chunk = [item]
                    current_size = item_size
                else:
                    current_chunk.append(item)
                    current_size += item_size

            if current_chunk:
                chunks.append(self._format_chunk(current_chunk))

        elif isinstance(data, dict):
            # For objects, create chunks by top-level keys
            current_chunk_keys = []
            current_size = 0

            for key, value in data.items():
                item_text = json.dumps({key: value}, ensure_ascii=False)
                item_size = len(item_text)

                if current_size + item_size > chunk_size and current_chunk_keys:
                    chunk_data = {k: data[k] for k in current_chunk_keys}
                    chunks.append(self._format_chunk(chunk_data))
                    current_chunk_keys = [key]
                    current_size = item_size
                else:
                    current_chunk_keys.append(key)
                    current_size += item_size

            if current_chunk_keys:
                chunk_data = {k: data[k] for k in current_chunk_keys}
                chunks.append(self._format_chunk(chunk_data))

        return chunks

    def _format_chunk(self, data: Any) -> str:
        """Format data as a readable chunk.
        
        Args:
            data: Data to format
            
        Returns:
            Formatted text
        """
        if isinstance(data, list):
            return self._generate_text_from_records(data)
        elif isinstance(data, dict):
            return self._generate_text(data)
        else:
            return str(data)

    def _extract_metadata(self, data: Any) -> dict[str, Any]:
        """Extract metadata from JSON data.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            Metadata dictionary
        """
        metadata: dict[str, Any] = {
            "type": type(data).__name__,
        }

        if isinstance(data, list):
            metadata["length"] = len(data)
            if data:
                metadata["item_types"] = list(set(type(item).__name__ for item in data[:100]))
                if isinstance(data[0], dict):
                    metadata["schema"] = self._infer_schema(data[:100])

        elif isinstance(data, dict):
            metadata["keys"] = list(data.keys())
            metadata["key_count"] = len(data.keys())

        return metadata

    def _infer_schema(self, records: list[dict[str, Any]]) -> dict[str, str]:
        """Infer schema from record sample.
        
        Args:
            records: Sample of records
            
        Returns:
            Schema dictionary mapping keys to types
        """
        if not records:
            return {}

        schema: dict[str, str] = {}
        all_keys = set()

        for record in records:
            all_keys.update(record.keys())

        for key in all_keys:
            types = set()
            for record in records:
                if key in record:
                    value = record[key]
                    if value is None:
                        types.add("null")
                    elif isinstance(value, bool):
                        types.add("boolean")
                    elif isinstance(value, int):
                        types.add("integer")
                    elif isinstance(value, float):
                        types.add("float")
                    elif isinstance(value, str):
                        types.add("string")
                    elif isinstance(value, list):
                        types.add("array")
                    elif isinstance(value, dict):
                        types.add("object")

            schema[key] = "|".join(sorted(types)) if types else "unknown"

        return schema

    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Check the health of the parser.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus indicating parser health
        """
        # JSON parsing uses standard library, always healthy
        return HealthStatus.HEALTHY
