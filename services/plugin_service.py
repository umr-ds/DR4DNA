# -*- coding: utf-8 -*-
"""
Plugin service for DR4DNA.

This service manages plugin lifecycle, loading, and execution.
"""

import importlib
import logging
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from exceptions import (
    PluginCompatibilityError,
    PluginException,
    PluginExecutionError,
    PluginLoadError,
)
from logging_config import get_logger
from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from state import AppState

logger = get_logger(__name__)


class PluginMetadata:
    """Metadata for a plugin."""

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        compatible_file_types: Optional[List[str]] = None,
    ):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.compatible_file_types = compatible_file_types or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "compatible_file_types": self.compatible_file_types,
        }


class PluginService:
    """
    Service class for plugin management.

    This class handles plugin discovery, loading, lifecycle management,
    and execution. It provides error isolation and detailed logging.

    Attributes:
        state: Application state instance
        plugin_dir: Directory containing plugins
        loaded_plugins: Dictionary of loaded plugin instances
        plugin_metadata: Dictionary of plugin metadata
    """

    def __init__(self, state: Optional[AppState] = None, plugin_dir: Optional[Path] = None):
        """
        Initialize the plugin service.

        Args:
            state: Application state instance
            plugin_dir: Directory containing plugins
        """
        from state import get_app_state

        self.state = state or get_app_state()
        self.plugin_dir = plugin_dir or Path(__file__).parent.parent / "repair_algorithms"
        self.loaded_plugins: Dict[str, FileSpecificRepair] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self._plugin_classes: Dict[str, Type[FileSpecificRepair]] = {}

        logger.info(f"PluginService initialized with plugin_dir={self.plugin_dir}")

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in the plugin directory.

        Returns:
            List of plugin class names
        """
        plugin_names = []

        try:
            if not self.plugin_dir.exists():
                logger.warning(f"Plugin directory does not exist: {self.plugin_dir}")
                return plugin_names

            # Look for Python files in plugin directory
            for py_file in self.plugin_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                module_name = py_file.stem

                try:
                    # Import the module
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                    # Look for plugin classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, FileSpecificRepair)
                            and attr is not FileSpecificRepair
                        ):
                            plugin_names.append(attr_name)
                            self._plugin_classes[attr_name] = attr
                            logger.debug(f"Discovered plugin: {attr_name}")

                except Exception as e:
                    logger.warning(f"Error discovering plugin in {py_file}: {e}")

            logger.info(f"Discovered {len(plugin_names)} plugins")

        except Exception as e:
            logger.error(f"Error discovering plugins: {e}")

        return plugin_names

    def load_plugin(
        self, plugin_class: Type[FileSpecificRepair], semi_automatic_solver: Any
    ) -> Optional[FileSpecificRepair]:
        """
        Load a plugin instance.

        Args:
            plugin_class: Plugin class to instantiate
            semi_automatic_solver: Semi-automatic solver instance

        Returns:
            Loaded plugin instance or None if loading failed
        """
        plugin_name = plugin_class.__name__

        try:
            logger.info(f"Loading plugin: {plugin_name}")

            # Create plugin instance
            plugin_instance = plugin_class(
                semi_automatic_solver=semi_automatic_solver, chunk_tag=self.state.get_chunk_tag()
            )

            # Store metadata
            self.plugin_metadata[plugin_name] = PluginMetadata(
                name=plugin_name, description=plugin_class.__doc__ or ""
            )

            # Store instance
            self.loaded_plugins[plugin_name] = plugin_instance

            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return plugin_instance

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            logger.debug(traceback.format_exc())
            return None

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of plugin to unload

        Returns:
            True if plugin was unloaded successfully
        """
        try:
            if plugin_name in self.loaded_plugins:
                plugin = self.loaded_plugins[plugin_name]

                # Call cleanup if available
                if hasattr(plugin, "cleanup"):
                    try:
                        plugin.cleanup()
                    except Exception as e:
                        logger.warning(f"Error during plugin cleanup: {e}")

                del self.loaded_plugins[plugin_name]
                logger.info(f"Unloaded plugin: {plugin_name}")
                return True

            logger.warning(f"Plugin not found: {plugin_name}")
            return False

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    def activate_plugin(self, plugin_name: str, file_type: Optional[str] = None) -> bool:
        """
        Activate a plugin if compatible.

        Args:
            plugin_name: Name of plugin to activate
            file_type: Current file type (for compatibility check)

        Returns:
            True if plugin was activated
        """
        try:
            if plugin_name not in self.loaded_plugins:
                logger.warning(f"Plugin not loaded: {plugin_name}")
                return False

            plugin = self.loaded_plugins[plugin_name]

            # Check compatibility
            if file_type is not None:
                try:
                    if not plugin.is_compatible(file_type):
                        logger.info(
                            f"Plugin {plugin_name} not compatible with file type: {file_type}"
                        )
                        return False
                except Exception as e:
                    logger.warning(f"Error checking plugin compatibility: {e}")

            # Activate plugin
            plugin.on_load()
            plugin.active = True

            logger.info(f"Activated plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error activating plugin {plugin_name}: {e}")
            return False

    def execute_plugin_operation(self, plugin_name: str, operation: str, *args, **kwargs) -> Any:
        """
        Execute a plugin operation with error handling.

        Args:
            plugin_name: Name of plugin
            operation: Operation to execute (method name)
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result or error dictionary

        Raises:
            PluginExecutionError: If operation fails
        """
        if plugin_name not in self.loaded_plugins:
            raise PluginExecutionError(
                plugin_name,
                operation,
                original_error=Exception(f"Plugin not loaded: {plugin_name}"),
            )

        plugin = self.loaded_plugins[plugin_name]

        try:
            if not hasattr(plugin, operation):
                raise PluginExecutionError(
                    plugin_name,
                    operation,
                    original_error=AttributeError(f"Operation not found: {operation}"),
                )

            method = getattr(plugin, operation)

            if not callable(method):
                raise PluginExecutionError(
                    plugin_name,
                    operation,
                    original_error=TypeError(f"Operation is not callable: {operation}"),
                )

            logger.debug(f"Executing {plugin_name}.{operation}")
            result = method(*args, **kwargs)
            logger.debug(f"Completed {plugin_name}.{operation}")

            return result

        except PluginExecutionError:
            raise
        except Exception as e:
            logger.error(f"Plugin operation failed: {plugin_name}.{operation}: {e}")
            logger.debug(traceback.format_exc())
            raise PluginExecutionError(plugin_name, operation, original_error=e)

    def get_plugin_ui_elements(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get UI elements for a plugin.

        Args:
            plugin_name: Name of plugin

        Returns:
            Dictionary of UI elements
        """
        try:
            if plugin_name not in self.loaded_plugins:
                return {}

            plugin = self.loaded_plugins[plugin_name]
            return plugin.get_ui_elements()

        except Exception as e:
            logger.error(f"Error getting UI elements for {plugin_name}: {e}")
            return {}

    def get_compatible_plugins(self, file_type: str) -> List[str]:
        """
        Get list of plugins compatible with a file type.

        Args:
            file_type: File type to check

        Returns:
            List of compatible plugin names
        """
        compatible = []

        for plugin_name, plugin in self.loaded_plugins.items():
            try:
                if plugin.is_compatible(file_type):
                    compatible.append(plugin_name)
            except Exception as e:
                logger.warning(f"Error checking compatibility for {plugin_name}: {e}")

        logger.debug(f"Compatible plugins for {file_type}: {compatible}")
        return compatible

    def get_all_plugins_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all loaded plugins.

        Returns:
            List of plugin information dictionaries
        """
        info = []

        for plugin_name, plugin in self.loaded_plugins.items():
            plugin_info = {
                "name": plugin_name,
                "active": plugin.active,
                "class": plugin.__class__.__name__,
            }

            if plugin_name in self.plugin_metadata:
                plugin_info.update(self.plugin_metadata[plugin_name].to_dict())

            info.append(plugin_info)

        return info

    def update_all_plugins_gepp(self, gepp):
        """
        Update GEPP for all active plugins.

        Args:
            gepp: New GEPP instance
        """
        for plugin_name, plugin in self.loaded_plugins.items():
            if plugin.active:
                try:
                    plugin.update_gepp(gepp)
                    logger.debug(f"Updated GEPP for plugin: {plugin_name}")
                except Exception as e:
                    logger.warning(f"Error updating GEPP for {plugin_name}: {e}")

    def update_all_plugins_chunk_tag(self, chunk_tag: List[int]):
        """
        Update chunk tag for all compatible plugins.

        Args:
            chunk_tag: New chunk tag list
        """
        for plugin_name, plugin in self.loaded_plugins.items():
            if plugin.active:
                try:
                    plugin.update_chunk_tag(chunk_tag)
                    logger.debug(f"Updated chunk tag for plugin: {plugin_name}")
                except Exception as e:
                    logger.warning(f"Error updating chunk tag for {plugin_name}: {e}")
