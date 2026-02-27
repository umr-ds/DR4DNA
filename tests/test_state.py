# -*- coding: utf-8 -*-
"""
Tests for the state management module.

This module contains unit tests for the AppState class and related functions.
"""

import pytest

from exceptions import PluginManagerNotInitializedError, SolverNotInitializedError
from state import AppState, get_app_state, initialize_app_state


class TestAppState:
    """Test cases for AppState class."""

    def test_initial_state(self):
        """Test that initial state has correct default values."""
        state = AppState()

        assert state.semi_automatic_solver is None
        assert state.common_packets == []
        assert state.chunk_tag == []
        assert state.column_tag == []
        assert state.content_updated is False
        assert state.show_canvas is False
        assert state.checksum_len_format is None
        assert state.plugin_manager is None

    def test_is_initialized_false(self):
        """Test is_initialized returns False when solver is None."""
        state = AppState()
        assert state.is_initialized() is False

    def test_get_solver_raises_when_none(self):
        """Test get_solver raises error when solver is None."""
        state = AppState()

        with pytest.raises(SolverNotInitializedError):
            state.get_solver()

    def test_get_plugin_manager_raises_when_none(self):
        """Test get_plugin_manager raises error when plugin_manager is None."""
        state = AppState()

        with pytest.raises(PluginManagerNotInitializedError):
            state.get_plugin_manager()

    def test_update_common_packets(self):
        """Test updating common packets."""
        state = AppState()
        packets = [True, False, True]

        state.update_common_packets(packets)

        assert state.common_packets == packets

    def test_update_chunk_tag(self):
        """Test updating chunk tags."""
        state = AppState()
        tag = [0, 1, 2, 0]

        state.update_chunk_tag(tag)

        assert state.chunk_tag == tag

    def test_get_chunk_tag_returns_copy(self):
        """Test that get_chunk_tag returns a copy, not the original."""
        state = AppState()
        original_tag = [0, 1, 2]
        state.update_chunk_tag(original_tag)

        returned_tag = state.get_chunk_tag()
        returned_tag[0] = 999  # Modify returned tag

        # Original should be unchanged
        assert state.chunk_tag[0] == 0
        assert returned_tag[0] == 999

    def test_update_column_tag(self):
        """Test updating column tags."""
        state = AppState()
        tag = [1, 0, 1]

        state.update_column_tag(tag)

        assert state.column_tag == tag

    def test_mark_content_updated(self):
        """Test marking content as updated."""
        state = AppState()

        state.mark_content_updated()

        assert state.is_content_updated() is True

    def test_reset_content_updated(self):
        """Test resetting content updated flag."""
        state = AppState()
        state.mark_content_updated()

        state.reset_content_updated()

        assert state.is_content_updated() is False

    def test_initialize_sets_all_fields(self, mock_solver):
        """Test that initialize sets all state fields correctly."""
        state = AppState()
        checksum_format = "I"

        state.initialize(mock_solver, checksum_format)

        assert state.semi_automatic_solver is mock_solver
        assert state.checksum_len_format == checksum_format
        assert state.common_packets == []
        assert len(state.chunk_tag) == 10
        assert len(state.column_tag) == 8
        assert state.content_updated is False
        assert state.show_canvas is False

    def test_initialize_creates_plugin_manager(self, mock_solver):
        """Test that initialize creates a PluginManager instance."""
        state = AppState()

        state.initialize(mock_solver)

        assert state.plugin_manager is not None

    def test_is_initialized_true_after_initialize(self, mock_solver):
        """Test is_initialized returns True after initialization."""
        state = AppState()

        state.initialize(mock_solver)

        assert state.is_initialized() is True

    def test_get_solver_returns_solver_after_initialize(self, mock_solver):
        """Test get_solver returns the solver after initialization."""
        state = AppState()

        state.initialize(mock_solver)

        assert state.get_solver() is mock_solver

    def test_get_plugin_manager_returns_manager_after_initialize(self, mock_solver):
        """Test get_plugin_manager returns the manager after initialization."""
        state = AppState()

        state.initialize(mock_solver)

        assert state.get_plugin_manager() is state.plugin_manager


class TestGlobalStateFunctions:
    """Test cases for global state functions."""

    def test_get_app_state_returns_singleton(self):
        """Test that get_app_state returns the singleton instance."""
        state1 = get_app_state()
        state2 = get_app_state()

        assert state1 is state2

    def test_initialize_app_state_calls_initialize(self, mock_solver):
        """Test that initialize_app_state calls initialize on the singleton."""

        # This should not raise
        initialize_app_state(mock_solver, "I")

        state = get_app_state()
        assert state.semi_automatic_solver is mock_solver
        assert state.checksum_len_format == "I"


class TestAppStateThreadSafety:
    """Test cases for AppState thread safety."""

    def test_concurrent_updates(self):
        """Test that concurrent updates don't cause race conditions."""
        import threading
        import time

        state = AppState()
        errors = []

        def update_tag(thread_id):
            try:
                for i in range(100):
                    state.update_chunk_tag([thread_id, i])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_tag, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
