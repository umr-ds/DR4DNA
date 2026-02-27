# -*- coding: utf-8 -*-
"""
Pytest fixtures and configuration for DR4DNA tests.

This module provides shared fixtures and utilities for the test suite.
"""

from unittest.mock import Mock

import numpy as np
import pytest


class MockGEPP:
    """Mock GEPP class with numpy arrays."""

    def __init__(self):
        self.b = np.zeros((10, 8), dtype=np.uint8)
        self.A = np.zeros((10, 8), dtype=np.uint8)
        self.chunk_to_used_packets = np.ones((10, 10), dtype=bool)


class MockDecoder:
    """Mock Decoder class."""

    def __init__(self):
        self.GEPP = MockGEPP()
        self.use_headerchunk = False
        self.number_of_chunks = 10


class MockSolver:
    """Mock Solver class."""

    def __init__(self):
        self.decoder = MockDecoder()
        self.multi_error_packets_mode = False
        self.headerChunk = None


@pytest.fixture
def mock_solver():
    """Create a mock SemiAutomaticReconstructionToolkit instance."""
    return MockSolver()


@pytest.fixture
def mock_plugin_manager():
    """Create a mock PluginManager instance."""
    manager = Mock()
    manager.plugin_instances = []
    manager.get_plugins.return_value = []
    return manager


@pytest.fixture
def sample_chunk_tag():
    """Create a sample chunk tag list."""
    return [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


@pytest.fixture
def sample_column_tag():
    """Create a sample column tag list."""
    return [0, 1, 0, 1, 0, 1, 0, 1]


@pytest.fixture
def sample_bytes():
    """Create sample byte data for testing."""
    return bytes([0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x00, 0x57, 0x6F, 0x72, 0x6C, 0x64])


@pytest.fixture
def sample_text():
    """Create sample text data for testing."""
    return "Hello World"


@pytest.fixture
def mock_dash_context():
    """Create a mock Dash callback context."""
    ctx = Mock()
    ctx.triggered = [{"prop_id": "button.n_clicks", "value": 1}]
    ctx.triggered_id = {"type": "button", "index": 0}
    ctx.inputs = {}
    ctx.states = {}
    return ctx


@pytest.fixture
def mock_callback_response():
    """Create a mock callback response tuple."""
    from dash import no_update

    return (
        "Info message",
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        None,  # canvas_image_content
        None,  # kaitai_view
    )


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "plugin: mark test as plugin-specific")
