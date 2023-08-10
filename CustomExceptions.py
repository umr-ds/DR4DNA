class CustomOutOfBoundsException(Exception):
    def __init__(self, message, filename_pos, extra_pos, expected_filenmae_len=None, expected_extra_len=None):
        self.filename_pos = filename_pos
        self.extra_pos = extra_pos
        self.expected_filenmae_len = expected_filenmae_len
        self.expected_extra_len = expected_extra_len
        super().__init__(message)


class InvalidDataException(Exception):
    def __init__(self, message, error_pos, expected=None):
        self.error_pos = error_pos
        self.expected = expected
        super().__init__(message)
