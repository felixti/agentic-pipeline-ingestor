"""Unit tests for the plugin system."""

import pytest
from unittest.mock import AsyncMock

from src.plugins.base import (
    PluginMetadata,
    PluginType,
    SourcePlugin,
    ParserPlugin,
    DestinationPlugin,
    HealthStatus,
    ValidationResult,
)
from src.plugins.registry import PluginRegistry, get_registry, reset_registry


# ============================================================================
# Test Fixtures
# ============================================================================

class MockSourcePlugin(SourcePlugin):
    """Mock source plugin for testing."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="mock_source",
            name="Mock Source",
            version="1.0.0",
            type=PluginType.SOURCE,
        )
    
    async def connect(self, config: dict):
        from src.plugins.base import Connection
        from uuid import uuid4
        return Connection(id=uuid4(), plugin_id="mock_source")
    
    async def list_files(self, conn, path: str, **kwargs):
        return []
    
    async def get_file(self, conn, path: str, **kwargs):
        from src.plugins.base import SourceFile, RetrievedFile
        return RetrievedFile(
            source_file=SourceFile(path=path, name="test.txt"),
            content=b"test content",
        )


class MockParserPlugin(ParserPlugin):
    """Mock parser plugin for testing."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="mock_parser",
            name="Mock Parser",
            version="1.0.0",
            type=PluginType.PARSER,
            supported_formats=["txt"],
        )
    
    async def parse(self, file_path: str, options: dict = None):
        from src.plugins.base import ParsingResult
        return ParsingResult(
            success=True,
            text="parsed content",
            parser_used="mock_parser",
        )
    
    async def supports(self, file_path: str, mime_type: str = None):
        from src.plugins.base import SupportResult
        return SupportResult(supported=True)


class MockDestinationPlugin(DestinationPlugin):
    """Mock destination plugin for testing."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="mock_destination",
            name="Mock Destination",
            version="1.0.0",
            type=PluginType.DESTINATION,
        )
    
    async def connect(self, config: dict):
        from src.plugins.base import Connection
        from uuid import uuid4
        return Connection(id=uuid4(), plugin_id="mock_destination")
    
    async def write(self, conn, data):
        from src.plugins.base import WriteResult
        return WriteResult(success=True)


# ============================================================================
# Plugin Registry Tests
# ============================================================================

@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    reset_registry()
    return get_registry()


class TestPluginRegistry:
    """Tests for PluginRegistry."""
    
    def test_register_source_plugin(self, registry):
        """Test registering a source plugin."""
        plugin = MockSourcePlugin()
        registry.register(plugin)
        
        assert "mock_source" in registry.sources
        assert registry.get_source("mock_source") is plugin
    
    def test_register_parser_plugin(self, registry):
        """Test registering a parser plugin."""
        plugin = MockParserPlugin()
        registry.register(plugin)
        
        assert "mock_parser" in registry.parsers
        assert registry.get_parser("mock_parser") is plugin
    
    def test_register_destination_plugin(self, registry):
        """Test registering a destination plugin."""
        plugin = MockDestinationPlugin()
        registry.register(plugin)
        
        assert "mock_destination" in registry.destinations
        assert registry.get_destination("mock_destination") is plugin
    
    def test_register_duplicate_raises_error(self, registry):
        """Test that registering duplicate plugin raises error."""
        plugin = MockSourcePlugin()
        registry.register(plugin)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(plugin)
    
    def test_unregister_plugin(self, registry):
        """Test unregistering a plugin."""
        plugin = MockSourcePlugin()
        registry.register(plugin)
        registry.unregister("mock_source")
        
        assert "mock_source" not in registry.sources
        assert registry.get_source("mock_source") is None
    
    def test_unregister_nonexistent_raises_error(self, registry):
        """Test that unregistering nonexistent plugin raises error."""
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("nonexistent")
    
    def test_get_metadata(self, registry):
        """Test getting plugin metadata."""
        plugin = MockSourcePlugin()
        registry.register(plugin)
        
        metadata = registry.get_metadata("mock_source")
        assert metadata is not None
        assert metadata.id == "mock_source"
        assert metadata.name == "Mock Source"
    
    def test_list_plugins(self, registry):
        """Test listing plugins."""
        registry.register(MockSourcePlugin())
        registry.register(MockParserPlugin())
        registry.register(MockDestinationPlugin())
        
        assert len(registry.list_sources()) == 1
        assert len(registry.list_parsers()) == 1
        assert len(registry.list_destinations()) == 1
        assert len(registry.list_all()) == 3
    
    def test_list_returns_copies(self, registry):
        """Test that list methods return copies."""
        plugin = MockSourcePlugin()
        registry.register(plugin)
        
        sources = registry.sources
        sources.pop("mock_source")
        
        # Original should be unchanged
        assert "mock_source" in registry.sources


# ============================================================================
# Plugin Base Class Tests
# ============================================================================

class TestPluginBaseClasses:
    """Tests for base plugin classes."""
    
    def test_source_plugin_metadata(self):
        """Test source plugin metadata."""
        plugin = MockSourcePlugin()
        metadata = plugin.metadata
        
        assert metadata.type == PluginType.SOURCE
        assert metadata.id == "mock_source"
    
    def test_parser_plugin_metadata(self):
        """Test parser plugin metadata."""
        plugin = MockParserPlugin()
        metadata = plugin.metadata
        
        assert metadata.type == PluginType.PARSER
        assert "txt" in metadata.supported_formats
    
    def test_destination_plugin_metadata(self):
        """Test destination plugin metadata."""
        plugin = MockDestinationPlugin()
        metadata = plugin.metadata
        
        assert metadata.type == PluginType.DESTINATION
    
    def test_default_health_check(self):
        """Test default health check returns healthy."""
        plugin = MockSourcePlugin()
        
        import asyncio
        result = asyncio.run(plugin.health_check())
        
        assert result == HealthStatus.HEALTHY
    
    def test_default_validation_result(self):
        """Test default validation returns valid."""
        plugin = MockSourcePlugin()
        
        import asyncio
        result = asyncio.run(plugin.validate_config({}))
        
        assert result.valid is True
        assert len(result.errors) == 0


# ============================================================================
# Async Tests
# ============================================================================

@pytest.mark.asyncio
class TestAsyncPluginOperations:
    """Tests for async plugin operations."""
    
    async def test_initialize_plugin(self, registry):
        """Test initializing a plugin."""
        plugin = MockSourcePlugin()
        registry.register(plugin)
        
        await registry.initialize_plugin("mock_source", {"key": "value"})
        
        assert "mock_source" in registry._initialized
    
    async def test_initialize_all_plugins(self, registry):
        """Test initializing all plugins."""
        registry.register(MockSourcePlugin())
        registry.register(MockParserPlugin())
        
        failures = await registry.initialize_all()
        
        assert len(failures) == 0
        assert len(registry._initialized) == 2
    
    async def test_health_check_plugin(self, registry):
        """Test health checking a plugin."""
        plugin = MockSourcePlugin()
        registry.register(plugin)
        await registry.initialize_plugin("mock_source")
        
        is_healthy = await registry.health_check("mock_source")
        
        assert is_healthy is True
    
    async def test_shutdown_all_plugins(self, registry):
        """Test shutting down all plugins."""
        registry.register(MockSourcePlugin())
        await registry.initialize_all()
        
        await registry.shutdown_all()
        
        assert len(registry._initialized) == 0
