# -*- coding: utf-8 -*-
"""
Logging configuration for DR4DNA.

This module provides centralized logging setup with support for:
- Console and file logging
- Log rotation
- Structured logging for production
- Different log levels for different components
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


# Log format templates
SIMPLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] "
    "- %(funcName)s() - %(message)s"
)
JSON_FORMAT = (
    '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", '
    '"file": "%(filename)s", "line": %(lineno)d, "function": "%(funcName)s", '
    '"message": "%(message)s"}'
)

# Default log directory
DEFAULT_LOG_DIR = Path(__file__).parent / "logs"


class LogFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color and sys.stdout.isatty()
    
    def format(self, record):
        # Add color to levelname
        if self.use_color and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    rotation_max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    rotation_backup_count: int = 5,
    use_json_format: bool = False,
) -> logging.Logger:
    """
    Set up logging for the DR4DNA application.
    
    Args:
        level: Logging level (default: INFO)
        log_dir: Directory for log files (default: ./logs)
        log_to_file: Enable file logging (default: True)
        log_to_console: Enable console logging (default: True)
        rotation_max_bytes: Max size of log file before rotation
        rotation_backup_count: Number of backup log files to keep
        use_json_format: Use JSON format for logs (useful for production)
    
    Returns:
        Root logger for the application
    """
    # Create logger
    logger = logging.getLogger("dr4dna")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Choose format
    if use_json_format:
        formatter = logging.Formatter(JSON_FORMAT)
    else:
        formatter = LogFormatter(DETAILED_FORMAT, use_color=log_to_console)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        if log_dir is None:
            log_dir = DEFAULT_LOG_DIR
        
        # Create log directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "dr4dna.log"
        
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=rotation_max_bytes,
                backupCount=rotation_backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Also create an error-only log file
            error_handler = logging.handlers.RotatingFileHandler(
                log_dir / "dr4dna_errors.log",
                maxBytes=rotation_max_bytes,
                backupCount=rotation_backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
            
        except (PermissionError, OSError) as e:
            # If we can't write to log file, just log to console
            logger.warning(f"Could not create log file at {log_file}: {e}")
    
    # Log startup message
    logger.info(f"DR4DNA logging initialized (level={logging.getLevelName(level)})")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module/component.
    
    Args:
        name: Name of the module/component (e.g., 'app', 'plugins.upload_repair')
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"dr4dna.{name}")


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls and their execution time.
    
    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            pass
    
    Args:
        logger: Logger to use (if None, uses module logger)
    """
    import functools
    import time
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            func_name = func.__name__
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.debug(f"{func_name} completed in {elapsed_time:.4f}s")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"{func_name} failed after {elapsed_time:.4f}s: {e}")
                raise
        
        return wrapper
    
    return decorator


# Initialize default logging when module is imported
# This ensures logging is available even if setup_logging isn't explicitly called
_default_logger = None


def get_default_logger() -> logging.Logger:
    """Get or create the default logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger
