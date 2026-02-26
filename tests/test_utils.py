# -*- coding: utf-8 -*-
"""
Tests for utility functions.

This module contains unit tests for the utility functions in utils.py.
"""

from unittest.mock import Mock, patch

import pytest

from utils import (
    bytes_to_hex_string,
    bytes_to_printable_string,
    create_no_update_tuple,
    filter_nonprintable,
    format_chunk_id,
    format_packet_id,
    hex_to_bytes,
    is_printable_char,
    validate_packet_id,
)


class TestFilterNonprintable:
    """Test cases for filter_nonprintable function."""

    def test_removes_control_characters(self):
        """Test that control characters are removed."""
        text = "Hello\x00World\x1f"
        result = filter_nonprintable(text)
        assert result == "HelloWorld"

    def test_removes_extended_control_characters(self):
        """Test that extended control characters are removed."""
        text = "Hello\x7fWorld\x9f"
        result = filter_nonprintable(text)
        assert result == "HelloWorld"

    def test_keeps_printable_characters(self):
        """Test that printable characters are kept."""
        text = "Hello World! 123"
        result = filter_nonprintable(text)
        assert result == "Hello World! 123"

    def test_empty_string(self):
        """Test with empty string."""
        result = filter_nonprintable("")
        assert result == ""

    def test_only_nonprintable(self):
        """Test with only non-printable characters."""
        text = "\x00\x01\x02\x7f\x80\x9f"
        result = filter_nonprintable(text)
        assert result == ""


class TestBytesToPrintableString:
    """Test cases for bytes_to_printable_string function."""

    def test_printable_bytes(self):
        """Test with printable bytes."""
        data = b"Hello"
        result = bytes_to_printable_string(data)
        assert result == "Hello"

    def test_non_printable_bytes(self):
        """Test with non-printable bytes."""
        data = bytes([72, 0, 119, 31])  # H\x00w\x1f
        result = bytes_to_printable_string(data)
        assert result == "H.w."

    def test_mixed_bytes(self):
        """Test with mixed printable and non-printable bytes."""
        data = bytes([72, 101, 108, 108, 111, 0, 87, 111, 114, 108, 100])
        result = bytes_to_printable_string(data)
        assert result == "Hello.World"

    def test_empty_bytes(self):
        """Test with empty bytes."""
        result = bytes_to_printable_string(b"")
        assert result == ""


class TestBytesToHexString:
    """Test cases for bytes_to_hex_string function."""

    def test_simple_bytes(self):
        """Test with simple bytes."""
        data = bytes([0x48, 0x65, 0x6C, 0x6C, 0x6F])
        result = bytes_to_hex_string(data)
        assert result == "48 65 6c 6c 6f"

    def test_custom_separator(self):
        """Test with custom separator."""
        data = bytes([0x48, 0x65])
        result = bytes_to_hex_string(data, separator=",")
        assert result == "48,65"

    def test_empty_bytes(self):
        """Test with empty bytes."""
        result = bytes_to_hex_string(b"")
        assert result == ""

    def test_no_separator(self):
        """Test with no separator."""
        data = bytes([0x48, 0x65])
        result = bytes_to_hex_string(data, separator="")
        assert result == "4865"


class TestHexToBytes:
    """Test cases for hex_to_bytes function."""

    def test_simple_hex(self):
        """Test with simple hex string."""
        hex_string = "48 65 6c 6c 6f"
        result = hex_to_bytes(hex_string)
        assert result == b"Hello"

    def test_hex_without_spaces(self):
        """Test with hex string without spaces."""
        hex_string = "48656c6c6f"
        result = hex_to_bytes(hex_string)
        assert result == b"Hello"

    def test_empty_hex(self):
        """Test with empty hex string."""
        result = hex_to_bytes("")
        assert result == b""

    def test_mixed_case_hex(self):
        """Test with mixed case hex string."""
        hex_string = "48 6E 6c 6C 6f"
        result = hex_to_bytes(hex_string)
        assert result == bytes([0x48, 0x6E, 0x6C, 0x6C, 0x6F])


class TestIsPrintableChar:
    """Test cases for is_printable_char function."""

    def test_printable_chars(self):
        """Test with printable characters."""
        assert is_printable_char(65) is True  # 'A'
        assert is_printable_char(97) is True  # 'a'
        assert is_printable_char(32) is True  # ' '
        assert is_printable_char(126) is True  # '~'
        assert is_printable_char(127) is True  # DEL (included in printable range)

    def test_non_printable_chars(self):
        """Test with non-printable characters."""
        assert is_printable_char(0) is False  # NULL
        assert is_printable_char(31) is False  # Control
        assert is_printable_char(128) is False  # Extended
        assert is_printable_char(159) is False  # Extended control

    def test_boundary_values(self):
        """Test boundary values."""
        assert is_printable_char(31) is False
        assert is_printable_char(32) is True
        assert is_printable_char(127) is True
        assert is_printable_char(128) is False


class TestFormatPacketId:
    """Test cases for format_packet_id function."""

    def test_default_prefix(self):
        """Test with default prefix."""
        result = format_packet_id(5)
        assert result == "#5"

    def test_custom_prefix(self):
        """Test with custom prefix."""
        result = format_packet_id(5, prefix="P")
        assert result == "P5"

    def test_no_prefix(self):
        """Test with no prefix."""
        result = format_packet_id(5, prefix="")
        assert result == "5"

    def test_zero_id(self):
        """Test with zero ID."""
        result = format_packet_id(0)
        assert result == "#0"


class TestFormatChunkId:
    """Test cases for format_chunk_id function."""

    def test_default_width(self):
        """Test with default width."""
        result = format_chunk_id(5)
        assert result == "00000005"

    def test_custom_width(self):
        """Test with custom width."""
        result = format_chunk_id(5, width=4)
        assert result == "0005"

    def test_zero_id(self):
        """Test with zero ID."""
        result = format_chunk_id(0)
        assert result == "00000000"

    def test_large_id(self):
        """Test with large ID."""
        result = format_chunk_id(12345)
        assert result == "00012345"


class TestValidatePacketId:
    """Test cases for validate_packet_id function."""

    def test_valid_int(self):
        """Test with valid integer."""
        is_valid, packet_id, error = validate_packet_id(5, 10)
        assert is_valid is True
        assert packet_id == 5
        assert error is None

    def test_valid_string(self):
        """Test with valid string."""
        is_valid, packet_id, error = validate_packet_id("5", 10)
        assert is_valid is True
        assert packet_id == 5
        assert error is None

    def test_out_of_range(self):
        """Test with out of range value."""
        is_valid, packet_id, error = validate_packet_id(15, 10)
        assert is_valid is False
        assert packet_id is None
        assert error is not None

    def test_negative_value(self):
        """Test with negative value."""
        is_valid, packet_id, error = validate_packet_id(-1, 10)
        assert is_valid is False
        assert packet_id is None
        assert error is not None

    def test_invalid_string(self):
        """Test with invalid string."""
        is_valid, packet_id, error = validate_packet_id("abc", 10)
        assert is_valid is False
        assert packet_id is None
        assert error is not None

    def test_none_value(self):
        """Test with None value."""
        is_valid, packet_id, error = validate_packet_id(None, 10)
        assert is_valid is False
        assert packet_id is None
        assert error is not None


class TestCreateNoUpdateTuple:
    """Test cases for create_no_update_tuple function."""

    def test_default_size(self):
        """Test with default size."""
        from dash import no_update

        result = create_no_update_tuple()
        assert len(result) == 14
        assert all(x is no_update for x in result)

    def test_custom_size(self):
        """Test with custom size."""
        from dash import no_update

        result = create_no_update_tuple(5)
        assert len(result) == 5
        assert all(x is no_update for x in result)

    def test_zero_size(self):
        """Test with zero size."""
        result = create_no_update_tuple(0)
        assert len(result) == 0
        assert result == ()
