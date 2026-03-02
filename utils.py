# -*- coding: utf-8 -*-
"""
Utility functions for DR4DNA application.

Provides common helper functions used across multiple modules.
"""

import itertools
from typing import Union


def filter_nonprintable(text: str) -> str:
    """
    Remove non-printable characters from text.

    Filters out control characters (0x00-0x20) and extended ASCII
    non-printable characters (0x7F-0xA0).

    Args:
        text: Input text string

    Returns:
        Text with non-printable characters removed

    Example:
        >>> filter_nonprintable("Hello\\x00World")
        'HelloWorld'
    """
    nonprintable = itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0))
    return text.translate(dict.fromkeys(nonprintable, None))


def bytes_to_printable_string(data: bytes) -> str:
    """
    Convert bytes to a printable string representation.

    Non-printable characters (outside ASCII 32-127) are replaced with '.'.

    Args:
        data: Input bytes

    Returns:
        Printable string representation with non-printable chars as '.'

    Example:
        >>> bytes_to_printable_string(b'Hello\\x00World')
        'Hello.World'
    """
    return "".join([chr(b) if 32 <= b <= 127 else "." for b in data])


def bytes_to_hex_string(data: bytes, separator: str = " ") -> str:
    """
    Convert bytes to a hex string representation.

    Args:
        data: Input bytes
        separator: Separator between hex values (default: space)

    Returns:
        Hex string representation with two-digit hex values

    Example:
        >>> bytes_to_hex_string(b'AB')
        '41 42'
        >>> bytes_to_hex_string(b'AB', separator='-')
        '41-42'
    """
    return separator.join([f"{b:02x}" for b in data])


def hex_to_bytes(hex_string: str) -> bytes:
    """
    Convert a hex string to bytes.

    Args:
        hex_string: Hex string (with or without spaces)

    Returns:
        Bytes object

    Example:
        >>> hex_to_bytes("41 42")
        b'AB'
        >>> hex_to_bytes("4142")
        b'AB'
    """
    return bytes.fromhex(hex_string.replace(" ", ""))


def is_printable_char(value: int) -> bool:
    """
    Check if a byte value represents a printable character.

    Printable characters are in the ASCII range 32-127 (space through tilde).

    Args:
        value: Byte value (0-255)

    Returns:
        True if printable, False otherwise

    Example:
        >>> is_printable_char(65)  # 'A'
        True
        >>> is_printable_char(0)   # null
        False
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

    Example:
        >>> format_packet_id(5)
        '#5'
        >>> format_packet_id(5, prefix='Packet ')
        'Packet 5'
    """
    return f"{prefix}{packet_id}"


def format_chunk_id(chunk_id: int, width: int = 8) -> str:
    """
    Format a chunk ID for display.

    Args:
        chunk_id: Chunk ID number
        width: Width for zero-padding (default: 8)

    Returns:
        Formatted chunk ID string with zero-padding

    Example:
        >>> format_chunk_id(5)
        '00000005'
        >>> format_chunk_id(5, width=4)
        '0005'
    """
    return str(chunk_id).zfill(width)


def validate_packet_id(packet_id: Union[str, int], max_value: int) -> tuple:
    """
    Validate a packet ID input.

    Checks if the packet ID is a valid integer within the allowed range.

    Args:
        packet_id: Packet ID to validate (string or int)
        max_value: Maximum allowed value for packet ID

    Returns:
        Tuple of (is_valid: bool, packet_id: int or None, error_message: str or None)
            - is_valid: True if packet ID is valid
            - packet_id: Converted integer value if valid, None otherwise
            - error_message: Error description if invalid, None if valid

    Example:
        >>> validate_packet_id(5, 100)
        (True, 5, None)
        >>> validate_packet_id("abc", 100)
        (False, None, "Packet ID must be a number!")
        >>> validate_packet_id(150, 100)
        (False, None, "Packet ID out of range!")
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

    Used for Dash callbacks to indicate that a particular output should
    not be updated. This is useful when a callback only needs to update
    some of its declared outputs.

    Args:
        size: Size of the tuple (default: 14)

    Returns:
        Tuple of no_update values

    Note:
        Import dash inside function to avoid circular imports

    Example:
        >>> # In a callback with 5 outputs, update only the 3rd:
        >>> from dash import no_update
        >>> result = (no_update, no_update, new_value, no_update, no_update)
        >>> # Or use this function:
        >>> updates = create_no_update_tuple(5)
    """
    from dash import no_update

    return tuple([no_update] * size)
