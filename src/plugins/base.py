"""Plugin system base classes and interfaces.

This module defines the abstract base classes for the plugin system,
including SourcePlugin, ParserPlugin, and DestinationPlugin.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, BinaryIO, Dict, List, Optional, Protocol, Union
from uuid import UUID


class PluginType(str, Enum):
    """Type of plugin."""
    SOURCE = "source"
    PARSER = "parser"
    DESTINATION = "destination"


class HealthStatus(str, Enum):
    """Health status of a plugin or connection."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PluginMetadata:
    """Metadata for a plugin.
    
    Attributes:
        id: Unique plugin identifier (e.g., "docling", "azure_blob")
        name: Human-readable plugin name
        version: Plugin version in semver format
        type: Plugin type (source, parser, destination)
        description: Brief description of the plugin
        author: Plugin author or organization
        supported_formats: List of supported file formats/extensions
        config_schema: JSON Schema for plugin configuration
        requires_auth: Whether the plugin requires authentication
    """
    id: str
    name: str
    version: str
    type: PluginType
    description: str = ""
    author: str = ""
    supported_formats: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    requires_auth: bool = False


@dataclass
class Connection:
    """Connection handle for source/destination plugins.
    
    Attributes:
        id: Unique connection identifier
        plugin_id: ID of the plugin that created this connection
        config: Connection configuration
        connected_at: Timestamp when connection was established
        metadata: Additional connection metadata
        is_open: Whether the connection is currently open
    """
    id: UUID
    plugin_id: str
    config: Dict[str, Any] = field(default_factory=dict)
    connected_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_open: bool = True


@dataclass
class SourceFile:
    """Represents a file from a source.
    
    Attributes:
        path: File path or identifier in the source system
        name: File name
        size: File size in bytes
        mime_type: MIME type of the file
        modified_at: Last modification timestamp
        created_at: Creation timestamp
        metadata: Additional file metadata
    """
    path: str
    name: str
    size: Optional[int] = None
    mime_type: Optional[str] = None
    modified_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedFile:
    """File retrieved from a source.
    
    Attributes:
        source_file: Original source file information
        content: File content as bytes or file-like object
        content_hash: Hash of the file content (e.g., SHA-256)
        retrieved_at: Timestamp when file was retrieved
        local_path: Optional local file path if saved to disk
    """
    source_file: SourceFile
    content: Union[bytes, BinaryIO]
    content_hash: str = ""
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    local_path: Optional[str] = None


@dataclass
class ParsingResult:
    """Result of document parsing.
    
    Attributes:
        success: Whether parsing was successful
        text: Extracted text content
        pages: List of page contents (one per page)
        metadata: Extracted document metadata
        format: Original document format
        parser_used: Name of the parser that produced this result
        processing_time_ms: Time taken to parse the document
        confidence: Confidence score for the extraction (0.0 - 1.0)
        error: Error message if parsing failed
        images: List of extracted images
        tables: List of extracted tables
        attachments: List of extracted attachments
    """
    success: bool
    text: str = ""
    pages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    format: str = ""
    parser_used: str = ""
    processing_time_ms: int = 0
    confidence: float = 0.0
    error: Optional[str] = None
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    attachments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TransformedData:
    """Data prepared for destination output.
    
    Attributes:
        job_id: ID of the job that produced this data
        chunks: List of text chunks
        embeddings: Optional embeddings for the chunks
        metadata: Document metadata
        lineage: Processing lineage information
        original_format: Original document format
        output_format: Requested output format
    """
    job_id: UUID
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    original_format: str = ""
    output_format: str = "json"


@dataclass
class WriteResult:
    """Result of writing data to a destination.
    
    Attributes:
        success: Whether the write was successful
        destination_id: ID of the destination
        destination_uri: URI where data was written
        records_written: Number of records written
        bytes_written: Number of bytes written
        processing_time_ms: Time taken to write
        error: Error message if write failed
        metadata: Additional result metadata
    """
    success: bool
    destination_id: str = ""
    destination_uri: str = ""
    records_written: int = 0
    bytes_written: int = 0
    processing_time_ms: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of configuration validation.
    
    Attributes:
        valid: Whether the configuration is valid
        errors: List of validation error messages
        warnings: List of validation warnings
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SupportResult:
    """Result of checking if a parser supports a file.
    
    Attributes:
        supported: Whether the file is supported
        confidence: Confidence in support detection
        reason: Explanation if not supported
    """
    supported: bool
    confidence: float = 1.0
    reason: Optional[str] = None


# ============================================================================
# Abstract Base Classes
# ============================================================================

class BasePlugin(ABC):
    """Base class for all plugins.
    
    All plugins must inherit from this class and implement
    the required abstract methods.
    """
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.
        
        Returns:
            PluginMetadata containing plugin information
        """
        ...
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration.
        
        This method is called once when the plugin is loaded.
        Override to perform any necessary setup.
        
        Args:
            config: Plugin configuration dictionary
        """
        pass
    
    async def health_check(self, config: Optional[Dict[str, Any]] = None) -> HealthStatus:
        """Check the health of the plugin.
        
        Args:
            config: Optional configuration for the health check
            
        Returns:
            HealthStatus indicating plugin health
        """
        return HealthStatus.HEALTHY
    
    async def shutdown(self) -> None:
        """Shutdown the plugin and cleanup resources.
        
        This method is called when the plugin is being unloaded.
        Override to perform any necessary cleanup.
        """
        pass


class SourcePlugin(BasePlugin, ABC):
    """Abstract base class for source plugins.
    
    Source plugins provide connectivity to data sources such as
    S3, Azure Blob Storage, SharePoint, or local filesystem.
    """
    
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> Connection:
        """Establish a connection to the source.
        
        Args:
            config: Connection configuration (credentials, endpoints, etc.)
            
        Returns:
            Connection handle for subsequent operations
            
        Raises:
            ConnectionError: If connection fails
        """
        ...
    
    @abstractmethod
    async def list_files(
        self,
        conn: Connection,
        path: str,
        recursive: bool = False,
        pattern: Optional[str] = None,
    ) -> List[SourceFile]:
        """List files in the source.
        
        Args:
            conn: Connection handle from connect()
            path: Path or prefix to list
            recursive: Whether to list recursively
            pattern: Optional glob pattern to filter files
            
        Returns:
            List of SourceFile objects
            
        Raises:
            ConnectionError: If connection is invalid
        """
        ...
    
    @abstractmethod
    async def get_file(
        self,
        conn: Connection,
        path: str,
        download_to: Optional[str] = None,
    ) -> RetrievedFile:
        """Retrieve a file from the source.
        
        Args:
            conn: Connection handle from connect()
            path: Path to the file
            download_to: Optional local path to save the file
            
        Returns:
            RetrievedFile containing file content and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ConnectionError: If connection fails
        """
        ...
    
    async def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate source configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        return ValidationResult(valid=True)
    
    async def test_connection(self, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Test if a connection can be established.
        
        Args:
            config: Configuration to test
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            conn = await self.connect(config)
            await self.health_check(config)
            # Close the test connection
            conn.is_open = False
            return True, None
        except Exception as e:
            return False, str(e)
    
    async def authorize(
        self,
        user_id: str,
        resource: str,
        action: str,
        config: Dict[str, Any],
    ) -> bool:
        """Check if user is authorized to access a resource.
        
        This method can be overridden by plugins to implement
        source-specific authorization (e.g., delegating to SharePoint
        permissions, using IAM policies for S3, etc.).
        
        Args:
            user_id: User identifier
            resource: Resource path/identifier
            action: Action to perform (read, write, delete, etc.)
            config: Source configuration
            
        Returns:
            True if user is authorized
        """
        # Default implementation: allow all authenticated users to read
        if action == "read":
            return True
        
        # For write operations, default to requiring explicit permission
        return False
    
    async def check_permission(
        self,
        user_id: str,
        resource: str,
        action: str,
        config: Dict[str, Any],
    ) -> bool:
        """Check user permission for a resource action.
        
        Convenience method that calls authorize() and can be extended
        with additional permission logic.
        
        Args:
            user_id: User identifier
            resource: Resource path/identifier
            action: Action to perform
            config: Source configuration
            
        Returns:
            True if user has permission
        """
        return await self.authorize(user_id, resource, action, config)


class ParserPlugin(BasePlugin, ABC):
    """Abstract base class for parser plugins.
    
    Parser plugins extract text and structure from documents
    using various parsing strategies (Docling, Azure OCR, etc.).
    """
    
    @abstractmethod
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
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ParseError: If parsing fails
        """
        ...
    
    @abstractmethod
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
        ...
    
    async def preprocess(
        self,
        file_path: str,
        steps: List[str],
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Preprocess a file before parsing.
        
        Args:
            file_path: Path to the file
            steps: List of preprocessing steps to apply
            options: Preprocessing options
            
        Returns:
            Path to the preprocessed file
        """
        # Default implementation returns original file
        return file_path
    
    async def get_quality_score(self, result: ParsingResult) -> float:
        """Calculate quality score for parsing result.
        
        Args:
            result: Parsing result to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not result.success:
            return 0.0
        return result.confidence


class DestinationPlugin(BasePlugin, ABC):
    """Abstract base class for destination plugins.
    
    Destination plugins route processed data to output systems
    such as Cognee, vector databases, webhooks, etc.
    """
    
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> Connection:
        """Establish a connection to the destination.
        
        Args:
            config: Connection configuration
            
        Returns:
            Connection handle for subsequent operations
        """
        ...
    
    @abstractmethod
    async def write(
        self,
        conn: Connection,
        data: TransformedData,
    ) -> WriteResult:
        """Write transformed data to the destination.
        
        Args:
            conn: Connection handle from connect()
            data: Transformed data to write
            
        Returns:
            WriteResult containing operation status
        """
        ...
    
    async def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate destination configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation status
        """
        return ValidationResult(valid=True)
    
    async def test_connection(self, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Test if a connection can be established.
        
        Args:
            config: Configuration to test
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            conn = await self.connect(config)
            await self.health_check(config)
            conn.is_open = False
            return True, None
        except Exception as e:
            return False, str(e)


# ============================================================================
# Plugin Capabilities Protocols
# ============================================================================

class StreamingSource(Protocol):
    """Protocol for sources that support streaming large files."""
    
    async def stream_file(
        self,
        conn: Connection,
        path: str,
        chunk_size: int = 8192,
    ) -> BinaryIO:
        """Stream a file from the source.
        
        Args:
            conn: Connection handle
            path: File path
            chunk_size: Size of chunks to stream
            
        Returns:
            File-like object for streaming
        """
        ...


class AsyncParser(Protocol):
    """Protocol for parsers with async chunk processing."""
    
    async def parse_chunks(
        self,
        file_path: str,
        chunk_size: int,
        options: Optional[Dict[str, Any]] = None,
    ) -> ParsingResult:
        """Parse a document in chunks for memory efficiency.
        
        Args:
            file_path: Path to the file
            chunk_size: Maximum chunk size
            options: Parser options
            
        Returns:
            ParsingResult containing all extracted content
        """
        ...


class BatchDestination(Protocol):
    """Protocol for destinations that support batch writes."""
    
    async def write_batch(
        self,
        conn: Connection,
        data_list: List[TransformedData],
    ) -> List[WriteResult]:
        """Write multiple data items in a batch.
        
        Args:
            conn: Connection handle
            data_list: List of transformed data items
            
        Returns:
            List of write results
        """
        ...
