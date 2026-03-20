# -*- coding: utf-8 -*-
"""
Shared type definitions for DR4DNA repair plugins.

This module provides TypedDict definitions and other type aliases
used across multiple repair plugin modules.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union


# ============================================================================
# Dash Callback Context Types
# ============================================================================

class DashCallbackContext(TypedDict, total=False):
    """
    Dash callback context structure.
    
    Note: This is a simplified representation. Actual Dash context has more fields.
    """
    triggered: List[Dict[str, str]]
    triggered_id: Union[str, Dict[str, Any]]
    args_grouping: Dict[str, Any]
    states_grouping: Dict[str, Any]
    inputs_grouping: Dict[str, Any]


# ============================================================================
# Plugin Callback Kwargs
# ============================================================================

class PluginCallbackKwargs(TypedDict, total=False):
    """
    Standard keyword arguments for plugin callbacks.
    
    These kwargs are commonly passed to plugin callback methods by the
    callback handler infrastructure.
    """
    c_ctx: DashCallbackContext
    chunk_tag: List[int]


class RepairCallbackKwargs(PluginCallbackKwargs, total=False):
    """
    Keyword arguments for repair-related callbacks.
    
    Extends PluginCallbackKwargs with repair-specific parameters.
    """
    # Repair operation parameters
    repair_id: int
    corrected_row: int
    corrected_value: List[int]
    hex_value: str
    
    # Canvas/image parameters
    canvas_json: Dict[str, Any]
    canvas_data: Dict[str, Any]
    image_content: str
    
    # Configuration parameters
    width: int
    height: int
    num_repair: int
    num_columns_to_repair: int
    
    # File content
    repaired_content: bytes
    fill_row_content: Optional[bytes]


class ShuffleCallbackKwargs(PluginCallbackKwargs, total=False):
    """
    Keyword arguments for shuffle-based repair callbacks.
    """
    num_shuffle: int
    permutations: int


class MetadataCallbackKwargs(PluginCallbackKwargs, total=False):
    """
    Keyword arguments for metadata repair callbacks.
    """
    no_columns_to_repair: int
    metadata_dict: Dict[str, Any]


class BMPRepairCallbackKwargs(PluginCallbackKwargs, total=False):
    """
    Keyword arguments for BMP file repair callbacks.
    """
    # Image dimensions
    image_width: int
    image_height: int
    
    # Canvas data
    canvas_json: Dict[str, Any]
    
    # Repair parameters
    num_repair: int


class ZipRepairCallbackKwargs(PluginCallbackKwargs, total=False):
    """
    Keyword arguments for ZIP file repair callbacks.
    """
    # Kaitai viewer toggle
    show_kaitai: bool
    
    # Repair parameters
    corrected_row: int
    corrected_value: List[int]


class UploadCallbackKwargs(PluginCallbackKwargs, total=False):
    """
    Keyword arguments for file upload callbacks.
    """
    # Upload data
    contents: Optional[str]
    filename: Optional[str]
    last_modified: Optional[float]
    
    # Download data
    download_data: bytes
    download_filename: str


class LanguageToolCallbackKwargs(PluginCallbackKwargs, total=False):
    """
    Keyword arguments for language tool text repair callbacks.
    """
    # Language detection
    language: Optional[str]
    
    # Inspection parameters
    no_inspect_chunks: int
    
    # Repair parameters
    no_columns_to_repair: int


class CountRequiredTagsCallbackKwargs(PluginCallbackKwargs, total=False):
    """
    Keyword arguments for count required tags analysis callbacks.
    """
    # Analysis parameters
    no_inspect_packet: int
    no_permutations: int
    inspect_num: int


# ============================================================================
# Plugin Return Types
# ============================================================================

class PluginCallbackResult(TypedDict, total=False):
    """
    Standard return type for plugin callbacks.
    
    This TypedDict uses total=False to make all fields optional,
    allowing plugins to return only the fields they need to update.
    """
    # Status messages
    info: str
    
    # View updates
    refresh_view: bool
    update_b: bool
    update_gepp: bool
    
    # Canvas updates
    image_content: str
    canvas_image_content: str
    canvas_data: Dict[str, Any]
    
    # Kaitai viewer
    kaitai_content: str
    kaitai_view: Any
    
    # Repair data
    repair_variations: Dict[str, Any]
    repair_for_each_packet: Dict[str, Any]
    
    # Download data
    download: bytes
    filename: str
    
    # Chunk/column tags
    chunk_tag: List[int]
    column_tag: List[int]
    
    # Special flags
    updates_canvas: bool
    generate_all: bool
    correctness_function: str
    repair_list: List[Any]
    
    # Height/width for canvas
    height: int
    width: int


# ============================================================================
# UI Element Types
# ============================================================================

class ButtonUIElement(TypedDict, total=False):
    """UI element configuration for a button."""
    type: str  # 'button'
    text: str
    callback: Any  # Callable
    updates_b: bool


class IntInputUIElement(TypedDict, total=False):
    """UI element configuration for an integer input."""
    type: str  # 'int'
    text: str
    default: int
    callback: Any  # Callable
    updates_b: bool


class SelectUIElement(TypedDict, total=False):
    """UI element configuration for a select dropdown."""
    type: str  # 'select'
    text: str
    options: List[Dict[str, str]]
    default: str
    callback: Any  # Callable
    updates_b: bool


class TextInputUIElement(TypedDict, total=False):
    """UI element configuration for a text input."""
    type: str  # 'text'
    text: str
    default: str
    callback: Any  # Callable
    updates_b: bool


class ToggleUIElement(TypedDict, total=False):
    """UI element configuration for a toggle switch."""
    type: str  # 'toggle'
    text: str
    default: bool
    callback: Any  # Callable
    updates_b: bool


# Union type for any UI element
UIElement = Union[
    ButtonUIElement,
    IntInputUIElement,
    SelectUIElement,
    TextInputUIElement,
    ToggleUIElement,
]


# ============================================================================
# File Type Detection
# ============================================================================

class FileTypeSignature(TypedDict):
    """File type signature with magic bytes and offset."""
    name: str
    magic: bytes
    offset: int


# Common file type signatures
FILE_TYPE_SIGNATURES: List[FileTypeSignature] = [
    {"name": "ZIP", "magic": b"PK\x03\x04", "offset": 0},
    {"name": "ZIP", "magic": b"PK\x05\x06", "offset": 0},  # Empty ZIP
    {"name": "BMP", "magic": b"BM", "offset": 0},
    {"name": "PNG", "magic": b"\x89PNG\r\n\x1a\n", "offset": 0},
    {"name": "JPEG", "magic": b"\xff\xd8\xff", "offset": 0},
    {"name": "GIF", "magic": b"GIF87a", "offset": 0},
    {"name": "GIF", "magic": b"GIF89a", "offset": 0},
]


# ============================================================================
# Helper Type Aliases
# ============================================================================

# Chunk tag values: 0=unknown, 1=invalid, 2=valid, 3=undecoded
ChunkTagValue = int

# Column tag values: integer count of errors
ColumnTagValue = int

# Packet ID
PacketId = int

# Chunk ID
ChunkId = int

# GEPP matrix row index
RowIndex = int

# GEPP matrix column index
ColumnIndex = int


# ============================================================================
# Error Matrix Types
# ============================================================================

class ErrorRegion(TypedDict):
    """Description of an error region in the data."""
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    error_count: int


class ErrorAnalysisResult(TypedDict):
    """Result of error analysis."""
    error_matrix: Any  # numpy.ndarray
    incorrect_rows: List[int]
    correct_rows: List[int]
    incorrect_columns: List[int]
    error_regions: List[ErrorRegion]
