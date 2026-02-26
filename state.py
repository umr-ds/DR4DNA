# -*- coding: utf-8 -*-
"""
Application State Management for DR4DNA.

This module provides a centralized state management system that eliminates
the need for global variables and provides thread-safe access to shared state.

Example:
    >>> from state import get_app_state, initialize_app_state
    >>> state = get_app_state()
    >>> if not state.is_initialized():
    ...     initialize_app_state(solver, checksum_format)
    >>> solver = state.get_solver()  # Raises SolverNotInitializedError if None
"""

import threading
import typing
from dataclasses import dataclass, field

from exceptions import PluginManagerNotInitializedError, SolverNotInitializedError
from logger import get_logger
from repair_algorithms.PluginManager import PluginManager
from semi_automatic_reconstruction_toolkit import SemiAutomaticReconstructionToolkit

logger = get_logger(__name__)


@dataclass
class AppState:
    """
    Centralized application state container.

    This class holds all shared state that needs to be accessed across
    different callbacks and handlers. It provides a clean interface for
    state management and eliminates the need for global variables.

    Attributes:
        semi_automatic_solver: Main solver instance for DNA data reconstruction
        common_packets: List of common packets identified during analysis
        chunk_tag: Tags for each chunk (0=unknown, 1=invalid, 2=valid, 3=undecoded)
        column_tag: Tags for each column indicating correctness
        content_updated: Flag indicating if content has been updated
        show_canvas: Flag indicating if canvas should be displayed
        checksum_len_format: Format string for checksum length
        plugin_manager: Manager for all loaded plugins
    """

    semi_automatic_solver: typing.Optional[SemiAutomaticReconstructionToolkit] = None
    common_packets: list = field(default_factory=list)
    chunk_tag: list = field(default_factory=list)
    column_tag: list = field(default_factory=list)
    content_updated: bool = False
    show_canvas: bool = False
    checksum_len_format: typing.Optional[str] = None
    plugin_manager: typing.Optional[PluginManager] = None

    # Thread lock for thread-safe state modifications
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def initialize(
        self,
        solver: SemiAutomaticReconstructionToolkit,
        checksum_len_format: typing.Optional[str] = None,
    ):
        """
        Initialize the application state with a solver instance.

        Args:
            solver: The SemiAutomaticReconstructionToolkit instance
            checksum_len_format: Optional checksum length format string
        """
        with self._lock:
            self.semi_automatic_solver = solver
            self.checksum_len_format = checksum_len_format
            self.common_packets = []
            self.chunk_tag = [0 for _ in range(len(solver.decoder.GEPP.b))]
            self.column_tag = [0 for _ in range(solver.decoder.GEPP.b.shape[1])]
            self.content_updated = False
            self.show_canvas = False
            # Initialize plugin manager
            if self.plugin_manager is None:
                self.plugin_manager = PluginManager()
            self.plugin_manager.plugin_instances.clear()

    def get_solver(self) -> SemiAutomaticReconstructionToolkit:
        """
        Get the solver instance, raising an error if not initialized.

        Returns:
            The SemiAutomaticReconstructionToolkit instance

        Raises:
            SolverNotInitializedError: If the solver has not been initialized
        """
        if self.semi_automatic_solver is None:
            logger.error("Solver accessed before initialization")
            raise SolverNotInitializedError()
        return self.semi_automatic_solver

    def is_initialized(self) -> bool:
        """Check if the application state has been initialized."""
        return self.semi_automatic_solver is not None

    def get_plugin_manager(self) -> PluginManager:
        """
        Get the plugin manager, raising an error if not initialized.

        Returns:
            The PluginManager instance

        Raises:
            PluginManagerNotInitializedError: If the plugin manager has not been initialized
        """
        if self.plugin_manager is None:
            logger.error("PluginManager accessed before initialization")
            raise PluginManagerNotInitializedError()
        return self.plugin_manager

    def update_common_packets(self, packets: list):
        """Thread-safe update of common packets."""
        with self._lock:
            self.common_packets = packets

    def update_chunk_tag(self, tag: list):
        """Thread-safe update of chunk tags."""
        with self._lock:
            self.chunk_tag = tag

    def get_chunk_tag(self) -> list:
        """Thread-safe access to chunk tags."""
        with self._lock:
            return self.chunk_tag.copy()

    def update_column_tag(self, tag: list):
        """Thread-safe update of column tags."""
        with self._lock:
            self.column_tag = tag

    def get_column_tag(self) -> list:
        """Thread-safe access to column tags."""
        with self._lock:
            return self.column_tag.copy()

    def mark_content_updated(self):
        """Mark content as updated."""
        with self._lock:
            self.content_updated = True

    def is_content_updated(self) -> bool:
        """Check if content has been updated."""
        with self._lock:
            return self.content_updated

    def reset_content_updated(self):
        """Reset the content updated flag."""
        with self._lock:
            self.content_updated = False


# Global singleton instance - this is the ONLY global we need
# All code should access state through this instance
app_state = AppState()


def get_app_state() -> AppState:
    """
    Get the global application state instance.

    Returns:
        The AppState singleton instance
    """
    return app_state


def initialize_app_state(
    solver: SemiAutomaticReconstructionToolkit, checksum_len_format: typing.Optional[str] = None
):
    """
    Initialize the global application state.

    Args:
        solver: The SemiAutomaticReconstructionToolkit instance
        checksum_len_format: Optional checksum length format string
    """
    app_state.initialize(solver, checksum_len_format)
