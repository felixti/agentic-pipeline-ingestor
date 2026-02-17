"""Plugin discovery and loading mechanisms.

This module provides mechanisms for discovering and loading plugins
from various sources including entry points, modules, and file paths.
"""

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, TypeVar

from src.plugins.base import BasePlugin, DestinationPlugin, ParserPlugin, SourcePlugin
from src.plugins.registry import PluginRegistry, get_registry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BasePlugin)


class PluginLoader:
    """Loader for discovering and loading plugins.
    
    The plugin loader discovers plugins from various sources:
    - Built-in plugins (src/plugins/builtins/)
    - Entry points (for pip-installed plugins)
    - Module paths (for development)
    - File paths (for external plugins)
    
    Example:
        >>> loader = PluginLoader()
        >>> loader.load_builtin_plugins()
        >>> loader.load_from_module("my_package.plugins")
    """

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        """Initialize the plugin loader.
        
        Args:
            registry: Plugin registry to load plugins into.
                     If None, uses the global registry.
        """
        self.registry = registry or get_registry()
        self._loaded_modules: set[str] = set()

    def load_builtin_plugins(self) -> int:
        """Load all built-in plugins.
        
        Returns:
            Number of plugins loaded
        """
        # Built-in plugins will be in src/plugins/builtins/
        builtins_path = Path(__file__).parent / "builtins"
        if not builtins_path.exists():
            logger.debug("No builtins directory found")
            return 0

        count = 0
        for file_path in builtins_path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                module = self._load_module_from_file(file_path)
                count += self._register_plugins_from_module(module)
            except Exception as e:
                logger.error(f"Failed to load builtin plugin from {file_path}: {e}")

        logger.info(f"Loaded {count} built-in plugins")
        return count

    def load_from_module(self, module_name: str) -> int:
        """Load plugins from a Python module.
        
        Args:
            module_name: Fully qualified module name (e.g., "my_package.plugins")
            
        Returns:
            Number of plugins loaded
        """
        if module_name in self._loaded_modules:
            logger.debug(f"Module {module_name} already loaded")
            return 0

        try:
            module = importlib.import_module(module_name)
            self._loaded_modules.add(module_name)
            count = self._register_plugins_from_module(module)
            logger.info(f"Loaded {count} plugins from module {module_name}")
            return count
        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")
            return 0

    def load_from_file(self, file_path: str) -> int:
        """Load plugins from a Python file.
        
        Args:
            file_path: Path to the Python file containing plugins
            
        Returns:
            Number of plugins loaded
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Plugin file not found: {file_path}")
            return 0

        try:
            module = self._load_module_from_file(path)
            count = self._register_plugins_from_module(module)
            logger.info(f"Loaded {count} plugins from file {file_path}")
            return count
        except Exception as e:
            logger.error(f"Failed to load plugins from {file_path}: {e}")
            return 0

    def load_from_directory(self, directory: str) -> int:
        """Load all plugins from a directory.
        
        Args:
            directory: Path to directory containing plugin files
            
        Returns:
            Number of plugins loaded
        """
        path = Path(directory)
        if not path.is_dir():
            logger.error(f"Plugin directory not found: {directory}")
            return 0

        count = 0
        for file_path in path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                module = self._load_module_from_file(file_path)
                count += self._register_plugins_from_module(module)
            except Exception as e:
                logger.error(f"Failed to load plugin from {file_path}: {e}")

        logger.info(f"Loaded {count} plugins from directory {directory}")
        return count

    def load_from_entry_points(self, group: str = "pipeline_ingestor.plugins") -> int:
        """Load plugins from package entry points.
        
        This allows pip-installed packages to register plugins.
        
        Args:
            group: Entry point group name
            
        Returns:
            Number of plugins loaded
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            # Python < 3.10
            try:
                from importlib_metadata import entry_points
            except ImportError:
                logger.warning("entry_points not available, skipping entry point plugins")
                return 0

        count = 0
        try:
            # Python 3.10+ uses groups parameter
            eps = entry_points(group=group)
        except TypeError:
            # Older versions
            all_eps = entry_points()
            eps = all_eps.get(group, [])

        for ep in eps:
            try:
                plugin_class = ep.load()
                if self._is_valid_plugin_class(plugin_class):
                    instance = plugin_class()
                    self.registry.register(instance)
                    count += 1
            except Exception as e:
                logger.error(f"Failed to load entry point plugin {ep.name}: {e}")

        logger.info(f"Loaded {count} plugins from entry points")
        return count

    def _load_module_from_file(self, file_path: Path) -> Any:
        """Load a Python module from a file path.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Loaded module
        """
        module_name = f"_pipeline_plugin_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _register_plugins_from_module(self, module: Any) -> int:
        """Find and register all plugin classes from a module.
        
        Args:
            module: Python module to scan
            
        Returns:
            Number of plugins registered
        """
        count = 0

        for name in dir(module):
            obj = getattr(module, name)

            if not inspect.isclass(obj):
                continue

            if not self._is_valid_plugin_class(obj):
                continue

            try:
                instance = obj()
                self.registry.register(instance)
                count += 1
            except Exception as e:
                logger.error(f"Failed to instantiate plugin {name}: {e}")

        return count

    def _is_valid_plugin_class(self, cls: type[Any]) -> bool:
        """Check if a class is a valid plugin class.
        
        Args:
            cls: Class to check
            
        Returns:
            True if the class is a valid plugin class
        """
        # Must be a concrete class (not abstract)
        if inspect.isabstract(cls):
            return False

        # Must inherit from one of the base plugin classes
        valid_bases = (SourcePlugin, ParserPlugin, DestinationPlugin)
        return issubclass(cls, valid_bases) and cls not in valid_bases


class AutoDiscoveryPluginLoader(PluginLoader):
    """Plugin loader with automatic discovery.
    
    This loader automatically discovers plugins from:
    1. Built-in plugins
    2. Entry points
    3. Configured plugin directories
    """

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        plugin_dirs: list[str] | None = None,
    ) -> None:
        """Initialize the auto-discovery loader.
        
        Args:
            registry: Plugin registry
            plugin_dirs: Additional directories to scan for plugins
        """
        super().__init__(registry)
        self.plugin_dirs = plugin_dirs or []

    def load_all(self) -> dict[str, int]:
        """Load plugins from all sources.
        
        Returns:
            Dictionary mapping source names to counts
        """
        results: dict[str, int] = {}

        # Load built-in plugins
        results["builtins"] = self.load_builtin_plugins()

        # Load from entry points
        results["entry_points"] = self.load_from_entry_points()

        # Load from configured directories
        dir_count = 0
        for directory in self.plugin_dirs:
            dir_count += self.load_from_directory(directory)
        results["directories"] = dir_count

        total = sum(results.values())
        logger.info(f"Auto-discovery loaded {total} total plugins")

        return results


def load_plugins(
    plugin_dirs: list[str] | None = None,
    modules: list[str] | None = None,
    files: list[str] | None = None,
) -> dict[str, int]:
    """Convenience function to load plugins from multiple sources.
    
    Args:
        plugin_dirs: Directories to scan for plugins
        modules: Module names to import
        files: File paths to load
        
    Returns:
        Dictionary mapping source to count of loaded plugins
    """
    loader = AutoDiscoveryPluginLoader(plugin_dirs=plugin_dirs or [])
    results = loader.load_all()

    # Load from specified modules
    if modules:
        module_count = 0
        for module in modules:
            module_count += loader.load_from_module(module)
        results["modules"] = module_count

    # Load from specified files
    if files:
        file_count = 0
        for file_path in files:
            file_count += loader.load_from_file(file_path)
        results["files"] = file_count

    return results
