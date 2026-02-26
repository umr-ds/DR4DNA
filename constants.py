# -*- coding: utf-8 -*-
"""
Application constants for DR4DNA.

Centralizes all magic numbers and constant values for better maintainability.
"""

# Color configurations
COLOR_CORRECT_BUTTON = {'backgroundColor': 'green'}
COLOR_INCORRECT_BUTTON = {'backgroundColor': 'red'}
COLOR_CORRECT_COLORBLIND = {'backgroundColor': '#84CE73'}
COLOR_INCORRECT_COLORBLIND = {'backgroundColor': 'brown'}
COLOR_LIGHT_RED_BUTTON = {'backgroundColor': 'lightcoral'}
COLOR_YELLOW_BUTTON = {'backgroundColor': 'yellow'}
COLOR_WHITE_BUTTON = {'backgroundColor': 'white'}

# Chunk tag states
CHUNK_TAG_UNKNOWN = 0
CHUNK_TAG_INVALID = 1
CHUNK_TAG_VALID = 2
CHUNK_TAG_UNDECODED = 3

# Display constants
MAX_COLUMN_DISPLAY_VALUE = 255
COLUMN_COLOR_MULTIPLIER_THRESHOLD = 5
COLUMN_COLOR_MAX_VALUE = 255

# File format constants
LAST_CHUNK_LEN_FORMAT = "I"
DEFAULT_CHECKSUM_LEN_FORMAT = None

# UI constants
EXTERNAL_STYLESHEETS = ["https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css"]
META_TAGS = [{"name": "viewport", "content": "width=device-width, initial-scale=1"}]

# Callback response constants
CALLBACK_RESPONSE_SIZE = 14

# Algorithm constants
DEFAULT_NUM_SHUFFLES = 5
MAX_PERMUTATION_ATTEMPTS_MULTIPLIER = 100
DEFAULT_NUM_INSPECT_CHUNKS = 10
DEFAULT_NUM_INSPECT_PACKETS = 10
DEFAULT_NUM_PERMUTATIONS = 5

# ZIP file constants
ZIP_VERSION_VALUES = [513, 1027, 1541, 2055]
ZIP_MONTH_MIN = 1
ZIP_MONTH_MAX = 12
ZIP_PARSE_ITERATIONS = 50

# BMP file constants
BMP_FILE_TYPES = ["BM", "BA", "CI", "CP", "IC", "PT"]
BMP_DEFAULT_REPAIR_BYTES = 2

# Metadata repair constants
METADATA_MAX_REPEATS_MULTIPLIER = 2
METADATA_SPECIAL_ROWS_LIMIT = 20

# Rank calculation messages
RANK_STATUS_DETECTABLE = ": Erroneous packet detectable!"
RANK_STATUS_PARTIAL_RECOVERY = "augmented rank ({}) < number of chunks ({}), but partial recovery might be possible."
RANK_STATUS_SOLVABLE = "LES seems solvable."
RANK_STATUS_AMBIGUOUS = ": {} Either all packets are correct or the corrupt packet is not linear dependent in the LES. This will be a tough one."

# Error messages
ERROR_APP_NOT_INITIALIZED = "Application not initialized!"
ERROR_SOLVER_NONE = "Solver is None."
ERROR_PLUGIN_MANAGER_NONE = "PluginManager is None."
ERROR_NO_LANGUAGE_SELECTED = "No language selected. Detect language first!"
ERROR_NO_VIABLE_SOLUTION = "Found no viable solution, try multi error mode!"
ERROR_CALCULATE_CORRUPT_PACKET_FIRST = "Calculate the corrupt packet(s) using 'Find corrupt packet by shuffling' first."
ERROR_SINGLE_PACKET_MODE = "More than one packet still possible!"
ERROR_REPAIR_INVALID_CHUNK = "Repair only possible for rows tagged as invalid. Additionally, a single corrupt packet should be identified."
ERROR_NO_SOLUTION_WITHOUT_CORRUPT = "No solution without the corrupt packet(s) found."
ERROR_CHOSEN_PACKET_INVALID = "Chosen packet is not a number or not in range!"
ERROR_CALCULATE_CORRUPT_PACKETS_FIRST = "Calculate corrupt packets first!"
ERROR_NO_DIFFERING_SOLUTIONS = "No differing solutions found!"
ERROR_MULTIPLE_CORRUPT_PACKETS = "Found multiple corrupt packets: {}. You might want to increase the number of permutations."
ERROR_LINEAR_DEPENDENT = "Found multiple possible corrupt packets for the same diff: {}. This indicates that the errors are linearly dependent!"
ERROR_NO_CHUNK_MATCHING = "Could not find a chunk matching the corrupt packet. This usually only happens if the corrupt packet was not used for any row."
ERROR_NO_CORRUPT_PACKET_FOUND = "Found no corrupt packet: The LES seems to be correct (or the corrupt packet cannot be described by a linear combination of other packets)."
ERROR_MULTI_ERROR_MODE_REQUIRED = "Found multiple diffs between solutions. This indicates multiple corrupt packets. Turn on Multi-Error Mode to find them."
ERROR_INCREASE_PERMUTATIONS = " Only partial solutions found, try increasing the number of permutations or perform multi-file automatic repair :"

# Working directories
WORKING_DIR_REORDERED_SOLUTION = "reordered_solution"

# Non-printable character ranges
NON_PRINTABLE_RANGE_START = 0x00
NON_PRINTABLE_RANGE_END = 0x20
NON_PRINTABLE_EXTENDED_START = 0x7f
NON_PRINTABLE_EXTENDED_END = 0xa0
PRINTABLE_CHAR_MIN = 32
PRINTABLE_CHAR_MAX = 127
