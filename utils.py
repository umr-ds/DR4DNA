# -*- coding: utf-8 -*-
"""
Utility functions for DR4DNA application.

Provides common helper functions used across multiple modules.
"""

import itertools
from typing import List, Union


def filter_nonprintable(text: str) -> str:
    """
    Remove non-printable characters from text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with non-printable characters removed
    """
    nonprintable = itertools.chain(
        range(0x00, 0x20),
        range(0x7f, 0xa0)
    )
    return text.translate({character: None for character in nonprintable})


def bytes_to_printable_string(data: bytes) -> str:
    """
    Convert bytes to a printable string representation.
    
    Non-printable characters are replaced with '.'.
    
    Args:
        data: Input bytes
        
    Returns:
        Printable string representation
    """
    return "".join([chr(b) if 32 <= b <= 127 else "." for b in data])


def bytes_to_hex_string(data: bytes, separator: str = " ") -> str:
    """
    Convert bytes to a hex string representation.
    
    Args:
        data: Input bytes
        separator: Separator between hex values (default: space)
        
    Returns:
        Hex string representation
    """
    return separator.join([f"{b:02x}" for b in data])


def hex_to_bytes(hex_string: str) -> bytes:
    """
    Convert a hex string to bytes.
    
    Args:
        hex_string: Hex string (with or without spaces)
        
    Returns:
        Bytes object
    """
    return bytes.fromhex(hex_string.replace(" ", ""))


def is_printable_char(value: int) -> bool:
    """
    Check if a byte value represents a printable character.
    
    Args:
        value: Byte value (0-255)
        
    Returns:
        True if printable, False otherwise
    """
    return 32 <= value <= 127


def format_packet_id(packet_id: int, prefix: str = "#") -> str:
    """
    Format a packet ID for display.
    
    Args:
        packet_id: Packet ID number
        prefix: Prefix string (default: "#")
        
    Returns:
        Formatted packet ID string
    """
    return f"{prefix}{packet_id}"


def format_chunk_id(chunk_id: int, width: int = 8) -> str:
    """
    Format a chunk ID for display.
    
    Args:
        chunk_id: Chunk ID number
        width: Width for zero-padding (default: 8)
        
    Returns:
        Formatted chunk ID string
    """
    return str(chunk_id).zfill(width)


def validate_packet_id(packet_id: Union[str, int], max_value: int) -> tuple:
    """
    Validate a packet ID input.
    
    Args:
        packet_id: Packet ID to validate (string or int)
        max_value: Maximum allowed value
        
    Returns:
        Tuple of (is_valid: bool, packet_id: int or None, error_message: str or None)
    """
    try:
        packet_id_int = int(packet_id)
        if packet_id_int < 0 or packet_id_int > max_value:
            return False, None, "Packet ID out of range!"
        return True, packet_id_int, None
    except (ValueError, TypeError):
        return False, None, "Packet ID must be a number!"


def create_no_update_tuple(size: int = 14) -> tuple:
    """
    Create a tuple of dash.no_update values.
    
    Args:
        size: Size of the tuple (default: 14)
        
    Returns:
        Tuple of no_update values
        
    Note:
        Import dash inside function to avoid circular imports
    """
    from dash import no_update
    return tuple([no_update] * size)
