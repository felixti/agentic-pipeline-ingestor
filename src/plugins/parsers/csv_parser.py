"""CSV/TSV parser plugin for structured data processing.

This module provides CSV and TSV parsing capabilities with support for
large file streaming, schema detection, and semantic chunking.
"""

import csv
import io
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from src.plugins.base import (
    HealthStatus,
    ParsingResult,
    PluginMetadata,
    PluginType,
    SupportResult,
)
from src.plugins.base import ParserPlugin

logger = logging.getLogger(__name__)


class CSVParser(ParserPlugin):
    """CSV/TSV parser plugin for structured tabular data.
    
    This parser handles CSV, TSV, and other delimited text files,
    supporting large files through streaming and providing schema
    detection for automatic type inference.
    
    Example:
        >>> parser = CSVParser()
        >>> await parser.initialize({})
        >>> result = await parser.parse("/path/to/data.csv")
        >>> print(result.text)
    """
    
    SUPPORTED_FORMATS = [".csv", ".tsv", ".tab", ".txt"]
    
    MIME_TYPE_MAP = {
        "text/csv": 1.0,
        "text/tab-separated-values": 1.0,
        "text/plain": 0.5,
    }
    
    # Default chunk size for semantic chunking (number of rows)
    DEFAULT_CHUNK_ROWS = 100
    
    def __init__(self) -> None:
        """Initialize the CSV parser."""
        self._config: Dict[str, Any] = {}
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="csv",
            name="CSV/TSV Parser",
            version="1.0.0",
            type=PluginType.PARSER,
            description="Parser for CSV, TSV, and delimited text files with streaming support",
            author="Pipeline Team",
            supported_formats=self.SUPPORTED_FORMATS,
            requires_auth=False,
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the parser with configuration.
        
        Args:
            config: Parser configuration options including:
                - delimiter: Field delimiter (auto-detect if not specified)
                - encoding: File encoding (default: utf-8)
                - has_header: Whether file has header row (auto-detect if not specified)
                - quotechar: Quote character (default: ")
                - escapechar: Escape character (default: \\
                - chunk_rows: Number of rows per semantic chunk (default: 100)
                - skip_empty_lines: Whether to skip empty lines (default: True)
        """
        self._config = {
            "delimiter": config.get("delimiter", None),  # None = auto-detect
            "encoding": config.get("encoding", "utf-8"),
            "has_header": config.get("has_header", None),  # None = auto-detect
            "quotechar": config.get("quotechar", '"'),
            "escapechar": config.get("escapechar", "\\"),
            "chunk_rows": config.get("chunk_rows", self.DEFAULT_CHUNK_ROWS),
            "skip_empty_lines": config.get("skip_empty_lines", True),
        }
        logger.info("CSV parser initialized")
    
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
        if extension in (".csv", ".tsv", ".tab"):
            return SupportResult(
                supported=True,
                confidence=0.95,
                reason=f"Supported extension: {extension}",
            )
        
        # Check MIME type if provided
        if mime_type:
            if mime_type in self.MIME_TYPE_MAP:
                return SupportResult(
                    supported=True,
                    confidence=self.MIME_TYPE_MAP[mime_type],
                    reason=f"Supported MIME type: {mime_type}",
                )
        
        # Try to detect CSV content in .txt files
        if extension == ".txt":
            confidence = await self._detect_csv_content(file_path)
            if confidence > 0.7:
                return SupportResult(
                    supported=True,
                    confidence=confidence,
                    reason="Detected CSV-like content in text file",
                )
        
        return SupportResult(
            supported=False,
            confidence=1.0,
            reason=f"Unsupported file format: {extension}",
        )
    
    async def _detect_csv_content(self, file_path: str) -> float:
        """Detect if a text file contains CSV-like content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        try:
            with open(file_path, 'r', encoding=self._config["encoding"], errors='ignore') as f:
                # Read first few lines
                sample_lines = []
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    sample_lines.append(line.strip())
                
                if len(sample_lines) < 2:
                    return 0.0
                
                # Try common delimiters
                delimiters = [',', '\t', ';', '|']
                for delim in delimiters:
                    counts = [line.count(delim) for line in sample_lines]
                    if len(set(counts)) == 1 and counts[0] > 0:
                        # Consistent delimiter count across lines
                        return 0.8
                
                return 0.0
                
        except Exception:
            return 0.0
    
    async def parse(
        self,
        file_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> ParsingResult:
        """Parse a CSV/TSV file and extract content.
        
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
            result = await self._parse_csv(file_path, opts)
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result
            
        except Exception as e:
            logger.error(f"CSV parsing failed: {e}", exc_info=True)
            return ParsingResult(
                success=False,
                error=f"Parsing failed: {str(e)}",
            )
    
    async def _parse_csv(
        self,
        file_path: str,
        options: Dict[str, Any],
    ) -> ParsingResult:
        """Parse CSV file with streaming support.
        
        Args:
            file_path: Path to the file
            options: Parsing options
            
        Returns:
            ParsingResult
        """
        path = Path(file_path)
        encoding = options.get("encoding", "utf-8")
        
        # Detect delimiter if not specified
        delimiter = options.get("delimiter")
        if not delimiter:
            delimiter = await self._detect_delimiter(file_path, encoding)
        
        # Parse the file
        all_rows: List[Dict[str, Any]] = []
        headers: List[str] = []
        chunks: List[str] = []
        
        chunk_size = options.get("chunk_rows", self.DEFAULT_CHUNK_ROWS)
        current_chunk_rows: List[Dict[str, Any]] = []
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                # Use csv.DictReader for structured parsing
                reader = csv.DictReader(
                    f,
                    delimiter=delimiter,
                    quotechar=options.get("quotechar", '"'),
                    escapechar=options.get("escapechar", "\\"),
                )
                
                headers = reader.fieldnames or []
                
                for row in reader:
                    # Convert empty strings to None for better processing
                    clean_row = {k: (v if v else None) for k, v in row.items()}
                    all_rows.append(clean_row)
                    current_chunk_rows.append(clean_row)
                    
                    # Create semantic chunks
                    if len(current_chunk_rows) >= chunk_size:
                        chunk_text = self._format_chunk(current_chunk_rows, headers)
                        chunks.append(chunk_text)
                        current_chunk_rows = []
                
                # Add remaining rows
                if current_chunk_rows:
                    chunk_text = self._format_chunk(current_chunk_rows, headers)
                    chunks.append(chunk_text)
        
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning(f"Encoding issue with {encoding}, trying latin-1")
            return await self._parse_csv(file_path, {**options, "encoding": "latin-1"})
        
        # Generate full text representation
        full_text = self._generate_text(all_rows, headers, path.suffix.lower())
        
        # Infer schema from data
        schema = self._infer_schema(all_rows, headers)
        
        # Extract metadata
        metadata = {
            "delimiter": delimiter,
            "encoding": encoding,
            "headers": headers,
            "total_rows": len(all_rows),
            "total_columns": len(headers),
            "schema": schema,
            "file_size_bytes": path.stat().st_size,
        }
        
        # Calculate confidence based on parsing success
        confidence = 0.95 if all_rows else 0.5
        
        return ParsingResult(
            success=True,
            text=full_text,
            pages=chunks if chunks else [full_text],
            metadata=metadata,
            format=path.suffix.lower(),
            parser_used="csv",
            confidence=confidence,
        )
    
    async def _detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect the delimiter used in the CSV file.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            
        Returns:
            Detected delimiter character
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Default based on extension
        if extension == ".tsv" or extension == ".tab":
            return '\t'
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                # Read sample
                sample = f.read(8192)
                
                # Try sniffer
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    return dialect.delimiter
                except csv.Error:
                    pass
                
                # Fallback: count occurrences
                delimiters = {',': 0, '\t': 0, ';': 0, '|': 0}
                for delim in delimiters:
                    delimiters[delim] = sample.count(delim)
                
                # Return most common
                if any(delimiters.values()):
                    return max(delimiters, key=delimiters.get)
        
        except Exception as e:
            logger.warning(f"Failed to detect delimiter: {e}")
        
        return ','  # Default to comma
    
    def _format_chunk(self, rows: List[Dict[str, Any]], headers: List[str]) -> str:
        """Format a chunk of rows as readable text.
        
        Args:
            rows: List of row dictionaries
            headers: Column headers
            
        Returns:
            Formatted text representation
        """
        lines = []
        
        for i, row in enumerate(rows):
            lines.append(f"Row {i + 1}:")
            for header in headers:
                value = row.get(header)
                if value:
                    lines.append(f"  {header}: {value}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_text(
        self,
        rows: List[Dict[str, Any]],
        headers: List[str],
        format_ext: str,
    ) -> str:
        """Generate full text representation of the CSV data.
        
        Args:
            rows: All row data
            headers: Column headers
            format_ext: File extension
            
        Returns:
            Full text representation
        """
        lines = []
        
        # Add header info
        lines.append(f"CSV Data ({format_ext})")
        lines.append(f"Total rows: {len(rows)}")
        lines.append(f"Columns: {', '.join(headers)}")
        lines.append("")
        
        # Add data preview (first 100 rows)
        preview_rows = rows[:100]
        lines.append("Data Preview:")
        lines.append("")
        
        for i, row in enumerate(preview_rows):
            lines.append(f"Row {i + 1}:")
            for header in headers:
                value = row.get(header)
                if value:
                    lines.append(f"  {header}: {value}")
            lines.append("")
        
        if len(rows) > 100:
            lines.append(f"... and {len(rows) - 100} more rows")
        
        return "\n".join(lines)
    
    def _infer_schema(
        self,
        rows: List[Dict[str, Any]],
        headers: List[str],
    ) -> Dict[str, str]:
        """Infer data types for each column.
        
        Args:
            rows: Sample rows to analyze
            headers: Column headers
            
        Returns:
            Dictionary mapping column names to inferred types
        """
        schema: Dict[str, str] = {}
        
        for header in headers:
            values = [row.get(header) for row in rows[:100] if row.get(header)]
            
            if not values:
                schema[header] = "unknown"
                continue
            
            # Try to infer type
            types_found = set()
            for value in values:
                val_str = str(value).strip()
                
                # Try integer
                try:
                    int(val_str)
                    types_found.add("integer")
                    continue
                except ValueError:
                    pass
                
                # Try float
                try:
                    float(val_str)
                    types_found.add("float")
                    continue
                except ValueError:
                    pass
                
                # Try date
                if self._is_date(val_str):
                    types_found.add("date")
                    continue
                
                # Default to string
                types_found.add("string")
            
            # Determine most specific type
            if "string" in types_found:
                schema[header] = "string"
            elif "date" in types_found:
                schema[header] = "date"
            elif "float" in types_found:
                schema[header] = "float"
            elif "integer" in types_found:
                schema[header] = "integer"
            else:
                schema[header] = "unknown"
        
        return schema
    
    def _is_date(self, value: str) -> bool:
        """Check if a string value looks like a date.
        
        Args:
            value: String to check
            
        Returns:
            True if it looks like a date
        """
        import re
        
        # Common date patterns
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return True
        
        return False
    
    async def parse_chunks(
        self,
        file_path: str,
        chunk_size: int,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Parse CSV in chunks for memory-efficient processing.
        
        Args:
            file_path: Path to the file
            chunk_size: Number of rows per chunk
            options: Parser options
            
        Yields:
            Chunk dictionaries with rows and metadata
        """
        opts = {**self._config, **(options or {})}
        path = Path(file_path)
        encoding = opts.get("encoding", "utf-8")
        
        # Detect delimiter
        delimiter = opts.get("delimiter")
        if not delimiter:
            delimiter = await self._detect_delimiter(file_path, encoding)
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            reader = csv.DictReader(
                f,
                delimiter=delimiter,
                quotechar=opts.get("quotechar", '"'),
            )
            
            headers = reader.fieldnames or []
            chunk_num = 0
            current_chunk: List[Dict[str, Any]] = []
            
            for row in reader:
                current_chunk.append({k: (v if v else None) for k, v in row.items()})
                
                if len(current_chunk) >= chunk_size:
                    yield {
                        "chunk_num": chunk_num,
                        "headers": headers,
                        "rows": current_chunk,
                        "text": self._format_chunk(current_chunk, headers),
                    }
                    chunk_num += 1
                    current_chunk = []
            
            # Yield remaining rows
            if current_chunk:
                yield {
                    "chunk_num": chunk_num,
                    "headers": headers,
                    "rows": current_chunk,
                    "text": self._format_chunk(current_chunk, headers),
                }
    
    async def health_check(self, config: Optional[Dict[str, Any]] = None) -> HealthStatus:
        """Check the health of the parser.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus indicating parser health
        """
        # CSV parsing uses only standard library, always healthy
        return HealthStatus.HEALTHY
