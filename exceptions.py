# -*- coding: utf-8 -*-
"""
Custom exceptions for DR4DNA application.

Provides specific exception types for better error handling and reporting.
"""


class DR4DNAError(Exception):
    """Base exception for all DR4DNA errors."""

    pass


class ApplicationNotInitializedError(DR4DNAError):
    """Raised when the application is accessed before initialization."""

    def __init__(self, component: str = "Application"):
        self.component = component
        super().__init__(f"{component} not initialized!")


class SolverNotInitializedError(ApplicationNotInitializedError):
    """Raised when the solver is accessed before initialization."""

    def __init__(self):
        super().__init__("Solver")


class PluginManagerNotInitializedError(ApplicationNotInitializedError):
    """Raised when the plugin manager is accessed before initialization."""

    def __init__(self):
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


class NoSolutionError(DR4DNAError):
    """Raised when no solution can be found."""

    pass


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
        self.file_type = file_type
        super().__init__(f"Failed to parse {file_type} file: {message}")


class StateError(DR4DNAError):
    """Raised when there's an error with application state."""

    pass
