# -*- coding: utf-8 -*-
"""
Custom exception hierarchy for DR4DNA.

This module provides a comprehensive set of exceptions for better error handling
and debugging throughout the application.
"""


class DR4DNAException(Exception):
    """Base exception for all DR4DNA-specific errors."""
    
    def __init__(self, message, error_code=None, details=None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
    
    def to_dict(self):
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ConfigurationException(DR4DNAException):
    """Raised when there's an issue with application configuration."""
    
    def __init__(self, message, config_key=None, invalid_value=None):
        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            details={"config_key": config_key, "invalid_value": invalid_value}
        )


class PluginException(DR4DNAException):
    """Base exception for plugin-related errors."""
    
    def __init__(self, message, plugin_name=None, error_code="PLUGIN_ERROR"):
        super().__init__(
            message,
            error_code=error_code,
            details={"plugin_name": plugin_name}
        )


class PluginLoadError(PluginException):
    """Raised when a plugin fails to load."""
    
    def __init__(self, plugin_name, original_error=None):
        super().__init__(
            f"Failed to load plugin '{plugin_name}': {str(original_error)}",
            plugin_name=plugin_name,
            error_code="PLUGIN_LOAD_ERROR"
        )
        self.original_error = original_error


class PluginExecutionError(PluginException):
    """Raised when a plugin execution fails."""
    
    def __init__(self, plugin_name, operation, original_error=None):
        super().__init__(
            f"Plugin '{plugin_name}' failed during {operation}: {str(original_error)}",
            plugin_name=plugin_name,
            error_code="PLUGIN_EXECUTION_ERROR"
        )
        self.original_error = original_error
        self.operation = operation


class PluginCompatibilityError(PluginException):
    """Raised when a plugin is incompatible with the current state."""
    
    def __init__(self, plugin_name, reason=None):
        super().__init__(
            f"Plugin '{plugin_name}' is incompatible: {reason or 'unknown reason'}",
            plugin_name=plugin_name,
            error_code="PLUGIN_COMPATIBILITY_ERROR"
        )


class DecoderException(DR4DNAException):
    """Base exception for decoder-related errors."""
    
    def __init__(self, message, error_code="DECODER_ERROR"):
        super().__init__(message, error_code=error_code)


class DecodeError(DecoderException):
    """Raised when decoding fails."""
    
    def __init__(self, message, chunk_id=None, packet_id=None):
        super().__init__(
            message,
            error_code="DECODE_ERROR",
            details={"chunk_id": chunk_id, "packet_id": packet_id}
        )


class RepairException(DR4DNAException):
    """Base exception for repair-related errors."""
    
    def __init__(self, message, error_code="REPAIR_ERROR"):
        super().__init__(message, error_code=error_code)


class RepairValidationError(RepairException):
    """Raised when repair validation fails."""
    
    def __init__(self, message, chunk_id=None, invalid_data=None):
        super().__init__(
            message,
            error_code="REPAIR_VALIDATION_ERROR",
            details={"chunk_id": chunk_id, "invalid_data": invalid_data}
        )


class StateException(DR4DNAException):
    """Base exception for state management errors."""
    
    def __init__(self, message, error_code="STATE_ERROR"):
        super().__init__(message, error_code=error_code)


class StateValidationError(StateException):
    """Raised when state validation fails."""
    
    def __init__(self, message, state_key=None, expected_type=None):
        super().__init__(
            message,
            error_code="STATE_VALIDATION_ERROR",
            details={"state_key": state_key, "expected_type": expected_type}
        )


class FileIOException(DR4DNAException):
    """Raised when file I/O operations fail."""
    
    def __init__(self, message, filepath=None, operation=None):
        super().__init__(
            message,
            error_code="FILE_IO_ERROR",
            details={"filepath": filepath, "operation": operation}
        )


class DataIntegrityException(DR4DNAException):
    """Raised when data integrity checks fail."""
    
    def __init__(self, message, expected=None, actual=None):
        super().__init__(
            message,
            error_code="DATA_INTEGRITY_ERROR",
            details={"expected": expected, "actual": actual}
        )
