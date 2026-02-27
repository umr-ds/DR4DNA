"""Custom exception classes for handling out-of-bounds and invalid data errors."""


class CustomOutOfBoundsException(Exception):
    """Exception raised when data is outside expected bounds."""

    def __init__(
        self, message, filename_pos, extra_pos, expected_filenmae_len=None, expected_extra_len=None
    ):
        """Initialize CustomOutOfBoundsException with error details."""
        self.filename_pos = filename_pos
        self.extra_pos = extra_pos
        self.expected_filenmae_len = expected_filenmae_len
        self.expected_extra_len = expected_extra_len
        super().__init__(message)


class InvalidDataException(Exception):
    """Exception raised when data is invalid."""

    def __init__(self, message, error_pos, expected=None):
        """Initialize InvalidDataException with error details."""
        self.error_pos = error_pos
        self.expected = expected
        super().__init__(message)
