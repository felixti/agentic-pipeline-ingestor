"""Plugin system for extensible data processing."""

from src.plugins.base import (
    Connection,
    DestinationPlugin,
    HealthStatus,
    ParserPlugin,
    ParsingResult,
    PluginMetadata,
    RetrievedFile,
    SourcePlugin,
    ValidationResult,
    WriteResult,
)
from src.plugins.registry import PluginRegistry

__all__ = [
    "Connection",
    "DestinationPlugin",
    "HealthStatus",
    "ParserPlugin",
    "ParsingResult",
    "PluginMetadata",
    "PluginRegistry",
    "RetrievedFile",
    "SourcePlugin",
    "ValidationResult",
    "WriteResult",
]
