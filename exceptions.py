# -*- coding: utf-8 -*-
"""
Custom exceptions for DR4DNA application.

Provides specific exception types for better error handling and reporting.
"""
import typing
from typing import Any, Optional


class DR4DNAError(Exception):
    """Base exception for all DR4DNA errors."""

    pass


class ApplicationNotInitializedError(DR4DNAError):
    """Raised when the application is accessed before initialization."""

    def __init__(self, component: str = "Application"):
        """Initialize with the component name that is not initialized."""
        self.component = component
        super().__init__(f"{component} not initialized!")


class SolverNotInitializedError(ApplicationNotInitializedError):
    """Raised when the solver is accessed before initialization."""

    def __init__(self):
        """Initialize SolverNotInitializedError."""
        super().__init__("Solver")


class PluginManagerNotInitializedError(ApplicationNotInitializedError):
    """Raised when the plugin manager is accessed before initialization."""

    def __init__(self):
        """Initialize PluginManagerNotInitializedError."""
        super().__init__("PluginManager")


class ValidationError(DR4DNAError):
    """Raised when data validation fails."""

    pass


class PacketIDError(DR4DNAError):
    """Raised when there's an error with packet ID validation or usage."""

    pass


class ChunkTagError(DR4DNAError):
    """Raised when there's an error with chunk tagging."""

    pass


class RepairError(DR4DNAError):
    """Raised when a repair operation fails."""

    pass


class RepairException(RepairError):
    """Base exception for repair-related errors (alias for RepairError)."""

    pass


class RepairValidationError(RepairError):
    """Raised when a repair operation validation fails."""

    def __init__(self, message: str, **kwargs: typing.Any):
        """Initialize RepairValidationError with message and optional details."""
        self.details = kwargs
        super().__init__(message)


class NoSolutionError(DR4DNAError):
    """Raised when no solution can be found."""

    pass


class DecoderException(DR4DNAError):
    """Base exception for decoder-related errors."""

    pass


class DecodeError(DecoderException):
    """Raised when decoding fails."""

    def __init__(self, message: str, **kwargs: typing.Any):
        """Initialize DecodeError with message and optional details."""
        self.details = kwargs
        super().__init__(message)


class MultipleSolutionsError(DR4DNAError):
    """Raised when multiple solutions are found but only one is expected."""

    pass


class PluginError(DR4DNAError):
    """Raised when a plugin operation fails."""

    pass


class PluginCompatibilityError(PluginError):
    """Raised when a plugin is not compatible with the current file type."""

    pass


class FileParseError(DR4DNAError):
    """Raised when file parsing fails."""

    def __init__(self, file_type: str, message: str):
        """Initialize FileParseError with file type and error message."""
        self.file_type = file_type
        super().__init__(f"Failed to parse {file_type} file: {message}")


class StateError(DR4DNAError):
    """Raised when there's an error with application state."""

    pass


class ConfigurationError(DR4DNAError):
    """
    Raised when there's an error with application configuration.

    Attributes:
        message: Error message
        config_key: The configuration key that caused the error (optional)
        invalid_value: The invalid value that was provided (optional)
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        invalid_value: Optional[Any] = None,
    ):
        """Initialize ConfigurationError with message and optional details."""
        self.message = message
        self.config_key = config_key
        self.invalid_value = invalid_value
        super().__init__(message)


class PluginExecutionError(PluginError):
    """Raised when a plugin execution fails."""

    def __init__(
        self, plugin_name: str, operation: str, original_error: Optional[Exception] = None
    ):
        """Initialize PluginExecutionError with plugin name, operation, and optional original error."""
        self.plugin_name = plugin_name
        self.operation = operation
        self.original_error = original_error
        message = f"Plugin execution failed: {plugin_name}.{operation}"
        if original_error:
            message += f" - {original_error}"
        super().__init__(message)


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""

    pass


class FileIOException(DR4DNAError):
    """Raised when file I/O operations fail."""

    def __init__(self, message: str, filepath: Optional[str] = None):
        """Initialize FileIOException with message and optional file path."""
        self.filepath = filepath
        full_message = message
        if filepath:
            full_message += f" (file: {filepath})"
        super().__init__(full_message)


class DataIntegrityError(DR4DNAError):
    """Raised when data integrity checks fail."""

    pass
