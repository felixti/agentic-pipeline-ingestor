"""Plugin system for extensible data processing."""

from src.plugins.base import (
    SourcePlugin,
    ParserPlugin,
    DestinationPlugin,
    PluginMetadata,
    Connection,
    RetrievedFile,
    ParsingResult,
    WriteResult,
    ValidationResult,
    HealthStatus,
)
from src.plugins.registry import PluginRegistry

__all__ = [
    "SourcePlugin",
    "ParserPlugin",
    "DestinationPlugin",
    "PluginMetadata",
    "Connection",
    "RetrievedFile",
    "ParsingResult",
    "WriteResult",
    "ValidationResult",
    "HealthStatus",
    "PluginRegistry",
]
