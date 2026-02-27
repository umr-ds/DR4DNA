# -*- coding: utf-8 -*-
"""
Configuration management for DR4DNA.

This module provides centralized configuration management with:
- Environment variable support
- Configuration validation
- Type-safe access
- Default values
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from exceptions import ConfigurationError


class LogLevel(Enum):
    """Logging level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False
    threaded: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    rotation_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    rotation_backup_count: int = 5
    use_json_format: bool = False


@dataclass
class PluginConfig:
    """Plugin configuration."""

    plugin_dir: Path = field(default_factory=lambda: Path(__file__).parent / "repair_algorithms")
    auto_load: bool = True
    timeout_seconds: int = 30
    max_memory_mb: int = 512


@dataclass
class PerformanceConfig:
    """Performance-related configuration."""

    max_permutations: int = 100
    cache_enabled: bool = True
    cache_max_size: int = 1000
    worker_threads: int = 4


@dataclass
class AppConfig:
    """
    Main application configuration.

    This class provides a centralized configuration system with validation
    and environment variable support.
    """

    # Application metadata
    app_name: str = "DR4DNA"
    version: str = "1.0.0"
    environment: str = "development"

    # Sub-configurations
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Working directory
    working_dir: Path = field(default_factory=lambda: Path(__file__).parent / "working_dir")

    # Feature flags
    enable_analytics: bool = False
    enable_error_reporting: bool = True

    @staticmethod
    def _set_nested_attr(config_section, key, value):
        """Safely set attribute on config section if it exists."""
        if hasattr(config_section, key):
            setattr(config_section, key, value)

    @staticmethod
    def _parse_bool(value):
        """Parse boolean from string."""
        return value.lower() in ("true", "1", "yes")

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Create configuration from environment variables.

        Environment variables are prefixed with DR4DNA_ and use uppercase.
        For nested config, use double underscore.

        Examples:
            DR4DNA_SERVER_HOST=0.0.0.0
            DR4DNA_SERVER_PORT=8080
            DR4DNA_LOGGING_LEVEL=DEBUG
            DR4DNA_PLUGINS_AUTO_LOAD=false

        Returns:
            AppConfig instance populated from environment
        """
        config = cls()

        # Server config
        if host := os.getenv("DR4DNA_SERVER_HOST"):
            config.server.host = host
        if port := os.getenv("DR4DNA_SERVER_PORT"):
            config.server.port = int(port)
        if debug := os.getenv("DR4DNA_SERVER_DEBUG"):
            config.server.debug = cls._parse_bool(debug)

        # Logging config
        if level := os.getenv("DR4DNA_LOGGING_LEVEL"):
            try:
                config.logging.level = LogLevel(level.upper())
            except ValueError:
                raise ConfigurationError(
                    f"Invalid log level: {level}", config_key="logging.level", invalid_value=level
                )
        if log_to_file := os.getenv("DR4DNA_LOGGING_TO_FILE"):
            config.logging.log_to_file = cls._parse_bool(log_to_file)
        if log_dir := os.getenv("DR4DNA_LOGGING_DIR"):
            config.logging.log_dir = Path(log_dir)

        # Plugin config
        if plugin_dir := os.getenv("DR4DNA_PLUGINS_DIR"):
            config.plugins.plugin_dir = Path(plugin_dir)
        if auto_load := os.getenv("DR4DNA_PLUGINS_AUTO_LOAD"):
            config.plugins.auto_load = cls._parse_bool(auto_load)
        if timeout := os.getenv("DR4DNA_PLUGINS_TIMEOUT"):
            config.plugins.timeout_seconds = int(timeout)

        # Performance config
        if max_perms := os.getenv("DR4DNA_PERFORMANCE_MAX_PERMUTATIONS"):
            config.performance.max_permutations = int(max_perms)
        if cache_enabled := os.getenv("DR4DNA_PERFORMANCE_CACHE_ENABLED"):
            config.performance.cache_enabled = cls._parse_bool(cache_enabled)
        if workers := os.getenv("DR4DNA_PERFORMANCE_WORKER_THREADS"):
            config.performance.worker_threads = int(workers)

        # Working directory
        if working_dir := os.getenv("DR4DNA_WORKING_DIR"):
            config.working_dir = Path(working_dir)

        # Feature flags
        if analytics := os.getenv("DR4DNA_ENABLE_ANALYTICS"):
            config.enable_analytics = cls._parse_bool(analytics)
        if error_reporting := os.getenv("DR4DNA_ENABLE_ERROR_REPORTING"):
            config.enable_error_reporting = cls._parse_bool(error_reporting)

        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            AppConfig instance
        """
        config = cls()

        # Handle nested configurations using common helper
        cls._apply_dict_section(config.server, data.get("server"))
        cls._apply_dict_section(config.logging, data.get("logging"), {"level": LogLevel})
        cls._apply_dict_section(config.plugins, data.get("plugins"), {"plugin_dir": Path})
        cls._apply_dict_section(config.performance, data.get("performance"))

        # Handle top-level config
        top_level_converters = {"working_dir": Path}
        for key in ["working_dir", "enable_analytics", "enable_error_reporting"]:
            if key in data:
                value = data[key]
                converter = top_level_converters.get(key, lambda x: x)
                setattr(config, key, converter(value) if isinstance(value, str) else value)

        return config

    @classmethod
    def _apply_dict_section(cls, section, data, converters=None):
        """
        Apply dictionary values to a config section.

        Args:
            section: Config section object to update
            data: Dictionary of values to apply
            converters: Optional dict mapping keys to converter functions
        """
        if not data:
            return

        converters = converters or {}
        for key, value in data.items():
            # Apply type conversion if specified
            if key in converters and isinstance(value, str):
                value = converters[key](value)
            cls._set_nested_attr(section, key, value)

    def validate(self):
        """
        Validate configuration values.

        Raises:
            ConfigurationError: If validation fails
        """
        # Validate server config
        if not 1 <= self.server.port <= 65535:
            raise ConfigurationError(
                f"Invalid server port: {self.server.port}",
                config_key="server.port",
                invalid_value=self.server.port,
            )

        # Validate plugin config
        if not self.plugins.plugin_dir.exists():
            raise ConfigurationError(
                f"Plugin directory does not exist: {self.plugins.plugin_dir}",
                config_key="plugins.plugin_dir",
                invalid_value=str(self.plugins.plugin_dir),
            )

        if self.plugins.timeout_seconds < 1:
            raise ConfigurationError(
                f"Invalid plugin timeout: {self.plugins.timeout_seconds}",
                config_key="plugins.timeout_seconds",
                invalid_value=self.plugins.timeout_seconds,
            )

        # Validate performance config
        if self.performance.max_permutations < 1:
            raise ConfigurationError(
                f"Invalid max permutations: {self.performance.max_permutations}",
                config_key="performance.max_permutations",
                invalid_value=self.performance.max_permutations,
            )

        if self.performance.worker_threads < 1:
            raise ConfigurationError(
                f"Invalid worker threads: {self.performance.worker_threads}",
                config_key="performance.worker_threads",
                invalid_value=self.performance.worker_threads,
            )

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.working_dir.mkdir(parents=True, exist_ok=True)
        if self.logging.log_to_file:
            self.logging.log_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "app_name": self.app_name,
            "version": self.version,
            "environment": self.environment,
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "debug": self.server.debug,
            },
            "logging": {
                "level": self.logging.level.value,
                "log_to_file": self.logging.log_to_file,
                "log_dir": str(self.logging.log_dir),
            },
            "plugins": {
                "plugin_dir": str(self.plugins.plugin_dir),
                "auto_load": self.plugins.auto_load,
                "timeout_seconds": self.plugins.timeout_seconds,
            },
            "performance": {
                "max_permutations": self.performance.max_permutations,
                "cache_enabled": self.performance.cache_enabled,
                "worker_threads": self.performance.worker_threads,
            },
            "working_dir": str(self.working_dir),
            "enable_analytics": self.enable_analytics,
            "enable_error_reporting": self.enable_error_reporting,
        }


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the global configuration instance.

    Returns:
        Global AppConfig instance
    """
    global _config
    if _config is None:
        _config = AppConfig.from_env()
        _config.validate()
        _config.ensure_directories()
    return _config


def set_config(config: AppConfig):
    """
    Set the global configuration instance.

    Args:
        config: New AppConfig instance
    """
    global _config
    _config = config


def reload_config():
    """Reload configuration from environment."""
    global _config
    _config = AppConfig.from_env()
    _config.validate()
    _config.ensure_directories()
    return _config
