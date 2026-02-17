"""Plugin registry for managing plugin lifecycle and discovery.

This module provides a centralized registry for managing plugins,
including registration, discovery, and lifecycle management.
"""

import logging
from typing import Any, TypeVar

from src.plugins.base import (
    BasePlugin,
    DestinationPlugin,
    ParserPlugin,
    PluginMetadata,
    SourcePlugin,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BasePlugin)


class PluginRegistry:
    """Central registry for managing plugins.
    
    The plugin registry maintains a catalog of all available plugins
    and manages their lifecycle (initialization, health checks, shutdown).
    
    Example:
        >>> registry = PluginRegistry()
        >>> registry.register(MySourcePlugin())
        >>> plugin = registry.get_source("azure_blob")
    """

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        self._sources: dict[str, SourcePlugin] = {}
        self._parsers: dict[str, ParserPlugin] = {}
        self._destinations: dict[str, DestinationPlugin] = {}
        self._metadata: dict[str, PluginMetadata] = {}
        self._initialized: set[str] = set()

    @property
    def sources(self) -> dict[str, SourcePlugin]:
        """Get all registered source plugins.
        
        Returns:
            Dictionary mapping plugin IDs to source plugins
        """
        return self._sources.copy()

    @property
    def parsers(self) -> dict[str, ParserPlugin]:
        """Get all registered parser plugins.
        
        Returns:
            Dictionary mapping plugin IDs to parser plugins
        """
        return self._parsers.copy()

    @property
    def destinations(self) -> dict[str, DestinationPlugin]:
        """Get all registered destination plugins.
        
        Returns:
            Dictionary mapping plugin IDs to destination plugins
        """
        return self._destinations.copy()

    def register_parser(self, plugin: ParserPlugin) -> None:
        """Register a parser plugin.
        
        Args:
            plugin: Parser plugin to register
        """
        self._parsers[plugin.metadata.id] = plugin
        self._metadata[plugin.metadata.id] = plugin.metadata
        logger.info(f"Registered parser plugin: {plugin.metadata.id}")

    def register_destination(self, plugin: DestinationPlugin) -> None:
        """Register a destination plugin.
        
        Args:
            plugin: Destination plugin to register
        """
        self._destinations[plugin.metadata.id] = plugin
        self._metadata[plugin.metadata.id] = plugin.metadata
        logger.info(f"Registered destination plugin: {plugin.metadata.id}")

    def register(self, plugin: BasePlugin) -> None:
        """Register a plugin with the registry.
        
        Args:
            plugin: Plugin instance to register
            
        Raises:
            ValueError: If a plugin with the same ID is already registered
            TypeError: If the plugin is not a valid plugin type
        """
        metadata = plugin.metadata
        plugin_id = metadata.id

        if plugin_id in self._metadata:
            raise ValueError(f"Plugin '{plugin_id}' is already registered")

        # Store based on plugin type
        if isinstance(plugin, SourcePlugin):
            self._sources[plugin_id] = plugin
        elif isinstance(plugin, ParserPlugin):
            self._parsers[plugin_id] = plugin
        elif isinstance(plugin, DestinationPlugin):
            self._destinations[plugin_id] = plugin
        else:
            raise TypeError(f"Unknown plugin type: {type(plugin)}")

        self._metadata[plugin_id] = metadata
        logger.info(f"Registered {metadata.type.value} plugin: {plugin_id} v{metadata.version}")

    def unregister(self, plugin_id: str) -> None:
        """Unregister a plugin from the registry.
        
        Args:
            plugin_id: ID of the plugin to unregister
            
        Raises:
            KeyError: If the plugin is not registered
        """
        if plugin_id not in self._metadata:
            raise KeyError(f"Plugin '{plugin_id}' is not registered")

        metadata = self._metadata[plugin_id]

        # Shutdown if initialized
        if plugin_id in self._initialized:
            self._shutdown_plugin(plugin_id)

        # Remove from appropriate dictionary
        if plugin_id in self._sources:
            del self._sources[plugin_id]
        elif plugin_id in self._parsers:
            del self._parsers[plugin_id]
        elif plugin_id in self._destinations:
            del self._destinations[plugin_id]

        del self._metadata[plugin_id]
        logger.info(f"Unregistered {metadata.type.value} plugin: {plugin_id}")

    def get_source(self, plugin_id: str) -> SourcePlugin | None:
        """Get a source plugin by ID.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Source plugin instance or None if not found
        """
        return self._sources.get(plugin_id)

    def get_parser(self, plugin_id: str) -> ParserPlugin | None:
        """Get a parser plugin by ID.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Parser plugin instance or None if not found
        """
        return self._parsers.get(plugin_id)

    def get_destination(self, plugin_id: str) -> DestinationPlugin | None:
        """Get a destination plugin by ID.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Destination plugin instance or None if not found
        """
        return self._destinations.get(plugin_id)

    def get_metadata(self, plugin_id: str) -> PluginMetadata | None:
        """Get metadata for a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Plugin metadata or None if not found
        """
        return self._metadata.get(plugin_id)

    def list_sources(self) -> list[PluginMetadata]:
        """List all registered source plugins.
        
        Returns:
            List of source plugin metadata
        """
        return [
            self._metadata[plugin_id]
            for plugin_id in self._sources.keys()
        ]

    def list_parsers(self) -> list[PluginMetadata]:
        """List all registered parser plugins.
        
        Returns:
            List of parser plugin metadata
        """
        return [
            self._metadata[plugin_id]
            for plugin_id in self._parsers.keys()
        ]

    def list_destinations(self) -> list[PluginMetadata]:
        """List all registered destination plugins.
        
        Returns:
            List of destination plugin metadata
        """
        return [
            self._metadata[plugin_id]
            for plugin_id in self._destinations.keys()
        ]

    def list_all(self) -> list[PluginMetadata]:
        """List all registered plugins.
        
        Returns:
            List of all plugin metadata
        """
        return list(self._metadata.values())

    async def initialize_plugin(
        self,
        plugin_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a specific plugin.
        
        Args:
            plugin_id: ID of the plugin to initialize
            config: Configuration dictionary for the plugin
            
        Raises:
            KeyError: If the plugin is not registered
        """
        if plugin_id not in self._metadata:
            raise KeyError(f"Plugin '{plugin_id}' is not registered")

        if plugin_id in self._initialized:
            logger.debug(f"Plugin '{plugin_id}' is already initialized")
            return

        plugin = (
            self._sources.get(plugin_id)
            or self._parsers.get(plugin_id)
            or self._destinations.get(plugin_id)
        )

        if plugin is None:
            raise KeyError(f"Plugin '{plugin_id}' not found in any category")

        config = config or {}
        await plugin.initialize(config)
        self._initialized.add(plugin_id)
        logger.info(f"Initialized plugin: {plugin_id}")

    async def initialize_all(
        self,
        configs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Exception]:
        """Initialize all registered plugins.
        
        Args:
            configs: Dictionary mapping plugin IDs to their configurations
            
        Returns:
            Dictionary of plugin IDs to exceptions for failed initializations
        """
        configs = configs or {}
        failures: dict[str, Exception] = {}

        for plugin_id in self._metadata.keys():
            try:
                await self.initialize_plugin(plugin_id, configs.get(plugin_id, {}))
            except Exception as e:
                logger.error(f"Failed to initialize plugin '{plugin_id}': {e}")
                failures[plugin_id] = e

        return failures

    async def shutdown_plugin(self, plugin_id: str) -> None:
        """Shutdown a specific plugin.
        
        Args:
            plugin_id: ID of the plugin to shutdown
            
        Raises:
            KeyError: If the plugin is not registered
        """
        if plugin_id not in self._metadata:
            raise KeyError(f"Plugin '{plugin_id}' is not registered")

        self._shutdown_plugin(plugin_id)

    def _shutdown_plugin(self, plugin_id: str) -> None:
        """Internal method to shutdown a plugin."""
        plugin = (
            self._sources.get(plugin_id)
            or self._parsers.get(plugin_id)
            or self._destinations.get(plugin_id)
        )

        if plugin and plugin_id in self._initialized:
            import asyncio
            try:
                asyncio.run(plugin.shutdown())
            except Exception as e:
                logger.error(f"Error during shutdown of '{plugin_id}': {e}")
            self._initialized.discard(plugin_id)
            logger.info(f"Shutdown plugin: {plugin_id}")

    async def shutdown_all(self) -> None:
        """Shutdown all initialized plugins."""
        for plugin_id in list(self._initialized):
            self._shutdown_plugin(plugin_id)

    async def health_check(self, plugin_id: str) -> bool:
        """Check health of a specific plugin.
        
        Args:
            plugin_id: ID of the plugin to check
            
        Returns:
            True if healthy, False otherwise
        """
        if plugin_id not in self._initialized:
            return False

        plugin = (
            self._sources.get(plugin_id)
            or self._parsers.get(plugin_id)
            or self._destinations.get(plugin_id)
        )

        if plugin is None:
            return False

        try:
            from src.plugins.base import HealthStatus
            status = await plugin.health_check()
            return status == HealthStatus.HEALTHY
        except Exception as e:
            logger.error(f"Health check failed for '{plugin_id}': {e}")
            return False

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all initialized plugins.
        
        Returns:
            Dictionary mapping plugin IDs to health status
        """
        results: dict[str, bool] = {}
        for plugin_id in self._initialized:
            results[plugin_id] = await self.health_check(plugin_id)
        return results


# Global registry instance
_global_registry: PluginRegistry | None = None


def get_registry() -> PluginRegistry:
    """Get the global plugin registry instance.
    
    Returns:
        Global PluginRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _global_registry
    _global_registry = None
