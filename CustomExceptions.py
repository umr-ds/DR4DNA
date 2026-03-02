"""
Custom exception classes for handling out-of-bounds and invalid data errors.

Provides specialized exception types for parsing and validation errors in
DNA data processing.
"""


class CustomOutOfBoundsException(Exception):
    """
    Exception raised when data is outside expected bounds.

    Used when parsing data structures where filename or extra data lengths
    don't match expected values.

    Attributes:
        filename_pos: Position where filename parsing failed
        extra_pos: Position where extra data parsing failed
        expected_filenmae_len: Expected filename length (if known)
        expected_extra_len: Expected extra data length (if known)
    """

    def __init__(
        self, message, filename_pos, extra_pos, expected_filenmae_len=None, expected_extra_len=None
    ):
        """
        Initialize CustomOutOfBoundsException with error details.

        Args:
            message: Error message describing the exception
            filename_pos: Position where filename parsing failed
            extra_pos: Position where extra data parsing failed
            expected_filenmae_len: Expected filename length (optional)
            expected_extra_len: Expected extra data length (optional)
        """
        self.filename_pos = filename_pos
        self.extra_pos = extra_pos
        self.expected_filenmae_len = expected_filenmae_len
        self.expected_extra_len = expected_extra_len
        super().__init__(message)


class InvalidDataException(Exception):
    """
    Exception raised when data is invalid.

    Used when parsed data doesn't match expected format or contains
    invalid values.

    Attributes:
        error_pos: Position where the error occurred
        expected: Expected value or format (if known)
    """

    def __init__(self, message, error_pos, expected=None):
        """
        Initialize InvalidDataException with error details.

        Args:
            message: Error message describing the invalid data
            error_pos: Position where the error occurred
            expected: Expected value or format (optional)
        """
        self.error_pos = error_pos
        self.expected = expected
        super().__init__(message)
