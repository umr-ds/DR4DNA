r"""
Multi-Version Coder for NOREC4DNA Encoded Files.

This module provides comprehensive functionality for encoding file updates into DNA
sequences, enabling efficient multi-version support in NOREC4DNA-encoded files.

Key Features:
    1. Differential Encoding: Only encode changes between file versions
    2. Version String Insertion: Embed version identifiers in DNA packets
    3. Chunk-Level Diff: Calculate differences at the chunk level
    4. Optimal Packet Selection: Scan seeds to find best packets for changes
    5. Error Probability Calculation: Evaluate and select low-error packets
    6. Packet Pair Support: Split changes across multiple packets when needed
    7. Multi-Version Pool Management: Handle pools with existing versions

This module consolidates the functionality previously in file_update_coding.py
with a cleaner, class-based API.

Example Usage (Class-based API):
    >>> from MultiVersionCoder import MultiVersionCoder
    >>> from ConfigWorker import ConfigReadAndExecute
    >>>
    >>> # Load configuration from existing pool
    >>> config = ConfigReadAndExecute("existing_pool.ini")
    >>> coder = MultiVersionCoder(config)
    >>>
    >>> # Check existing versions
    >>> max_version = coder.get_max_version_in_pool()
    >>> print(f"Pool contains versions 0 to {max_version}")
    >>>
    >>> # Encode new version
    >>> with open("updated_file.bin", "rb") as f:
    ...     new_data = f.read()
    >>> version_num, packets = coder.encode_new_version(new_data)
    >>> coder.save_updated_pool("updated_pool.fasta")

Example Usage (Function-based API):
    >>> from MultiVersionCoder import (
    ...     find_affected_chunks,
    ...     get_current_file_version,
    ...     generate_dna_version_string,
    ... )
    >>>
    >>> # Get current version from pool
    >>> current = get_current_file_version(solver)
    >>> new_version = current + 1
    >>>
    >>> # Calculate diff
    >>> diff, changed = find_affected_chunks(solver, new_file_data)
    >>>
    >>> # Generate version string
    >>> version_string = generate_dna_version_string(new_version)
"""

import argparse
import logging
import os
import struct
import typing
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from NOREC4DNA.metadata_coding import encoder_from_decoder
from NOREC4DNA.norec4dna.HeaderChunk import HeaderChunk

try:
    import imagehash

    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    logger = logging.getLogger(__name__)
    logger.warning("imagehash not available, perceptual hashing disabled")

from norec4dna.helper.helper import xor_with_seed
from norec4dna.helper.helper_cpu_single_core import should_drop_packet, xor_numpy
from norec4dna.helper.quaternary2Bin import tranlate_quat_to_byte
from norec4dna.helper.RU10Helper import choose_packet_numbers, from_true_false_list, int31
from norec4dna.rules.FastDNARules import FastDNARules

from NOREC4DNA.ConfigWorker import ConfigReadAndExecute
from NOREC4DNA.invivo_window_decoder import load_fasta
from NOREC4DNA.norec4dna import RU10Encoder
from NOREC4DNA.norec4dna.helper.bin2Quaternary import byte2QUATS
from NOREC4DNA.norec4dna.RU10Packet import RU10Packet
from repair_algorithms.utils.select_numbers import select_numbers
from semi_automatic_reconstruction_toolkit import SemiAutomaticReconstructionToolkit

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.error(os.getcwd())

# ============================================================================
# Core Functions (Function-based API)
# ============================================================================

os.chdir(str(Path(__file__).parent.absolute()))

logger.error(os.getcwd())


def find_affected_chunks(
    semiautomatic_solver: SemiAutomaticReconstructionToolkit,
    new_file: typing.Optional[bytes] = None,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Calculate diff between existing chunks and new file.

    This function compares the current decoded file state with a new file version
    to identify which chunks have changed. It returns both the raw difference array
    and the indices of differing chunks.

    Args:
        semiautomatic_solver: SemiAutomaticReconstructionToolkit instance with
            decoder state.
        new_file: New file bytes to compare against. If None, creates artificial
            changes at positions 200 and 500 for testing.

    Returns:
        A tuple containing:
        - diff: Numpy array of differences between old and new chunks
        - differing_rows: Array of chunk indices that have changed

    Raises:
        ValueError: If file size doesn't match expected chunk layout.

    Note:
        Currently requires the new file to have equal length or be shorter than
        the original. Index is defined WITHOUT header row as the header must
        be changed for any changed content.

    Example:
        >>> diff, changed = find_affected_chunks(solver, new_file_data)
        >>> print(f"Changed chunks: {changed}")
    """
    semiautomatic_solver.decoder.solve()
    # Get current file content from decoder
    current_file_bytes = semiautomatic_solver.get_file_as_bytes(include_padding=True)

    # For testing/development: create artificial changes if no new_file provided
    if new_file is None:
        #new_file = bytearray(current_file_bytes)
        #if len(new_file) > 500:
        #    new_file[500] = 0x00
        #    new_file[200] = 0x00
        logger.error("[!] new_file was empty - skipping!")
        quit()

    # Validate file size - new file must not be larger than current
    if len(new_file) > len(current_file_bytes):
        # TODO: implement diff /binarydiff format for this case!
        raise ValueError(
            f"New file ({len(new_file)} bytes) is larger than current file "
            f"({len(current_file_bytes)} bytes). File growth not yet supported."
        )

    # Pad new_file to match current file size if smaller
    if len(new_file) < len(current_file_bytes):
        logger.debug(f"Padding new file from {len(new_file)} to {len(current_file_bytes)} bytes")
        new_file_padded = bytearray(new_file)
        new_file_padded.extend(b"\x00" * (len(current_file_bytes) - len(new_file)))
        new_file = bytes(new_file_padded)

    # Get chunk size from GEPP
    chunk_size = semiautomatic_solver.decoder.GEPP.b.shape[1]

    # Create compatible numpy array with correct chunk sizes from bytes
    # Skip header chunk (index 0) as we compare data chunks only
    new_array = np.frombuffer(new_file, dtype=np.uint8).reshape(-1, chunk_size)

    # Calculate difference between old and new chunks (skip header chunk at index 0)
    # GEPP.b[1:] contains data chunks 0 to n-1 (where n is number_of_chunks - 1)
    # FIXME: somewhere the last-chunk padding gets filled with random data.
    #  Further, the diff is 2 rows longer than required?!?!
    semiautomatic_solver.decoder.populate_header_chunk()
    semiautomatic_solver.decoder.GEPP.b[-1][
        semiautomatic_solver.decoder.headerChunk.get_last_chunk_length() :
    ] = 0
    diff = np.bitwise_xor(semiautomatic_solver.decoder.GEPP.b[1:semiautomatic_solver.decoder.number_of_chunks], new_array)
    diff = np.vstack((np.zeros_like(diff[0]), diff))
    # Find indices of differing rows (chunks)
    differing_rows = np.nonzero(np.any(diff != 0, axis=1))[0]

    logger.info(f"Found {len(differing_rows)} changed chunks")

    return diff, differing_rows


def generate_dna_version_string(fileversion: int = 0, base_length: int = 8) -> str:
    """
    Generate a DNA version string for embedding in packets.

    Creates a DNA sequence that encodes the file version number followed by
    a magic string marker. The version is encoded in the 3 bases immediately
    preceding the magic string.

    Args:
        fileversion: Version number to encode (must be in range [0, 63]).
        base_length: Length of version encoding in bases (default: 8).

    Returns:
        DNA version string (e.g., "AAAGAGCCAGTGAGTCGTA" for version 0).

    Raises:
        RuntimeError: If fileversion is >= 64 (exceeds 6-bit encoding capacity).
        ValueError: If fileversion is negative.

    Example:
        >>> v0 = generate_dna_version_string(0)
        >>> print(v0)  # "AAAGAGCCAGTGAGTCGTA"
        >>> v1 = generate_dna_version_string(1)
        >>> print(v1)  # "AACGAGCCAGTGAGTCGTA"
    """
    if fileversion < 0:
        raise ValueError("fileversion must be non-negative")
    if fileversion >= 64:
        raise RuntimeError("fileversion must be in range [0, 63]!")

    # Encode version in 3 bases (6 bits = 64 possible values)
    version_bases = byte2QUATS(fileversion)[1:]

    # Magic string marker
    magic_string = "GAGCCAGTGAGTCGTA"

    return version_bases + magic_string


def get_current_file_version(
    semi_automatic_solver: SemiAutomaticReconstructionToolkit,
    magic_string: str = "GAGCCAGTGAGTCGTA",
) -> int:
    """
    Return the largest version number available in the pool.

    Scans all DNA sequences in the pool and extracts version numbers from
    sequences containing the base_dna_version_string marker.

    Args:
        magic_string: The magic DNA string marking version sequences
            (e.g., "GAGCCAGTGAGTCGTA").

    Returns:
        The highest version number found. Returns 0 if no versions are found
        (base version only).

    Note:
        Versions are indexed starting from 0, where version 0 is the base version
        and version 1 is the FIRST version after the base version.

    Example:
        >>> mv_decoder = MultiVersionDecoder(decoder)
        >>> max_version = mv_decoder.get_versions_in_pool("GAGCCAGTGAGTCGTA")
        >>> print(f"Available up to version: {max_version}")
    """
    res = 0
    fasta_entries = load_fasta(semi_automatic_solver.decoder.file)

    for seq in fasta_entries.values():
        idx = seq.find(magic_string)
        if idx != -1 and idx + len(magic_string) < len(seq):
            # Extract 3 bases before the magic string (version encoding)
            version_bases = seq[max(0, idx - 3) : idx]
            # Pad if necessary
            if len(version_bases) < 3:
                version_bases = "A" * (3 - len(version_bases)) + version_bases
            version_num = tranlate_quat_to_byte(f"A{version_bases}")
            try:
                version_value = struct.unpack("B", version_num)[0]
                res = max(res, version_value)
            except struct.error:
                logger.warning(f"Failed to parse version from sequence: {seq[:50]}...")

    return res


def combine_fasta_files(fasta_paths: List[Union[str, Path]], output_path: Union[str, Path]) -> int:
    """
    Combine multiple FASTA files into a single output file.

    This function reads multiple FASTA files and merges all sequences into
    a single output FASTA file. Sequence IDs are preserved but made unique
    by appending a suffix if duplicates are found.

    Args:
        fasta_paths: List of paths to FASTA files to combine
        output_path: Path to the output combined FASTA file

    Returns:
        Total number of sequences written to the output file

    Example:
        >>> total = combine_fasta_files(
        ...     ["existing.fasta", "new_packets.fasta"],
        ...     "combined.fasta"
        ... )
        >>> print(f"Combined {total} sequences")
    """
    output_path = Path(output_path)
    all_sequences: Dict[str, str] = {}
    total_sequences = 0

    for fasta_path in fasta_paths:
        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            logger.warning(f"FASTA file not found: {fasta_path}, skipping...")
            continue

        fasta_entries = load_fasta(str(fasta_path))
        logger.info(f"Loaded {len(fasta_entries)} sequences from {fasta_path.name}")

        for seq_id, seq_data in fasta_entries.items():
            # Ensure unique IDs
            original_id = seq_id
            counter = 1
            while seq_id in all_sequences:
                # Split ID to handle multiple suffixes
                if '_' in original_id:
                    base_id = '_'.join(original_id.rsplit('_', 1)[:-1])
                else:
                    base_id = original_id
                seq_id = f"{base_id}_{counter}"
                counter += 1

            all_sequences[seq_id] = seq_data
            total_sequences += 1

    # Write combined FASTA
    logger.info(f"Writing {total_sequences} sequences to {output_path.name}...")
    with open(output_path, 'w') as f:
        for seq_id, seq_data in all_sequences.items():
            f.write(f">{seq_id}\n")
            f.write(f"{seq_data}\n")

    logger.info(f"✓ Combined FASTA written: {output_path.absolute()}")
    return total_sequences


def insert_dna_version_string(
    packet: RU10Packet,
    dna_version_string: str,
    insertion_position: int = 30,
    diff: typing.Optional[np.ndarray] = None,
) -> None:
    """
    Add a DNA version string at a specific position in an RU10Packet.

    This function inserts a version string into the packet data at the specified
    position, handling padding and seed spacing as needed.

    Args:
        packet: The RU10Packet to modify.
        dna_version_string: The DNA string to insert (e.g., "ACGT").
        insertion_position: Byte position in the data section (not DNA representation).
        diff: Array containing the diff between original and new packet data.
            Used for asserting correct insertion position. If None, skips validation.

    Raises:
        AssertionError: If insertion position overlaps with non-padding region.

    Note:
        The function handles padding automatically if the version string length
        is not a multiple of 4 (since each byte encodes to 4 DNA bases).

    Example:
        >>> insert_dna_version_string(
        ...     packet,
        ...     "ACGTACGT",
        ...     insertion_position=50,
        ...     diff=chunk_diff
        ... )
    """
    # Ensure packet has DNA data
    if packet.dna_data is None or packet.dna_data == "":
        packet.calculate_packed_data()

    # Validate insertion position if diff provided
    if diff is not None:
        insert_len = int((len(dna_version_string) + 1) / 4) + 1
        assert not diff[
            insertion_position : insertion_position + insert_len
        ].any(), "Cannot insert version string into non-padding region!"

    # Calculate padding needed to align to byte boundary
    pre_padding = "A" * ((4 - len(dna_version_string) % 4) % 4)
    binary_dna_version_string = tranlate_quat_to_byte(pre_padding + dna_version_string)
    num_padding_bits = len(pre_padding) * 2

    # Handle first byte if padding was added
    if num_padding_bits > 0:
        original_byte = packet.data[insertion_position]
        mask_keep = (0xFF << (8 - num_padding_bits)) & 0xFF
        mask_new = (0xFF >> num_padding_bits) & 0xFF
        combined_first_byte = (original_byte & mask_keep) | (
            binary_dna_version_string[0] & mask_new
        )
        binary_dna_version_string = bytes([combined_first_byte]) + binary_dna_version_string[1:]

    # Create stub array with version string at insertion position
    stub = np.zeros_like(packet.data)
    end_pos = insertion_position + len(binary_dna_version_string)
    stub[insertion_position:end_pos] = np.frombuffer(binary_dna_version_string, dtype=np.uint8)

    # Apply seed-based XOR if enabled
    stub = xor_with_seed(stub, packet.id)

    # Update packet data
    # FIXME:
    # packet.data[insertion_position:end_pos] = stub[insertion_position:end_pos]
    packet.data[insertion_position:end_pos] = np.frombuffer(
        stub[insertion_position:end_pos], dtype=np.uint8
    )

    # Recalculate DNA structure
    packet.get_dna_struct(True, packet.id_spacing, packet.id_spacing_length, True)

    logger.debug(f"Inserted version string at position {insertion_position} in packet {packet.id}")


def insert_id_string(
    packet: RU10Packet,
    insertion_position: int,
    changed_chunk_id: int,
    semiautomatic_solver: SemiAutomaticReconstructionToolkit,
) -> None:
    """
    Insert the ID of a changed chunk into a packet.

    This function encodes the index of a changed chunk into the packet data,
    allowing the decoder to identify which chunk was modified.

    Args:
        packet: RU10Packet to modify.
        insertion_position: Byte position for ID insertion.
        changed_chunk_id: ID of the changed chunk to encode.
        semiautomatic_solver: SemiAutomaticReconstructionToolkit instance.

    Raises:
        AssertionError: If changed_chunk_id is not in used chunks or >= 256.

    Note:
        The chunk ID is encoded as a single byte (0-255 range). For larger
        chunk indices, this function will need to be extended.

    Example:
        >>> insert_id_string(packet, insertion_position=100, changed_chunk_id=5, solver=solver)
    """
    # Get used chunks list
    used_chunks = from_true_false_list(semiautomatic_solver.decoder.removeAndXorAuxPackets(packet))

    # Validate chunk ID is in used chunks
    assert (
        changed_chunk_id in used_chunks
    ), f"Changed chunk id {changed_chunk_id} must be in used chunks {used_chunks}!"

    # Get index of chunk in used chunks list
    index_of_chunk = used_chunks.index(changed_chunk_id)

    # Validate chunk index fits in single byte
    assert (
        index_of_chunk < 256
    ), f"Cannot encode chunk id index >= 256 in this version! Got {index_of_chunk}"

    # Create XOR mask with chunk ID at insertion position
    to_xor = np.zeros_like(packet.data, dtype=np.uint8)
    to_xor[insertion_position] = index_of_chunk & 0xFF

    # Apply XOR to packet data
    packet.data = xor_numpy(packet.data, to_xor)

    # Recalculate DNA structure
    packet.get_dna_struct(True, packet.id_spacing, packet.id_spacing_length, True)

    logger.debug(
        f"Inserted chunk ID {changed_chunk_id} (index {index_of_chunk}) at position {insertion_position}"
    )


def reduce_packet_to_chunk(
    packet: RU10Packet,
    semiautomatic_solver: SemiAutomaticReconstructionToolkit,
    chunk_to_reduce_to: int = 0,
) -> RU10Packet:
    """
    Reduce a packet to contain only a single target chunk.

    This function removes all chunks from a packet except the specified target
    chunk by XORing with the known chunk data from the decoder state.

    Args:
        packet: RU10Packet to reduce.
        semiautomatic_solver: SemiAutomaticReconstructionToolkit instance.
        chunk_to_reduce_to: Chunk index to keep (default: 0 for header chunk).

    Returns:
        Modified packet containing only the target chunk.

    Raises:
        AssertionError: If packet doesn't contain exactly the target chunk after reduction.

    Note:
        This operation is essential for isolating individual chunks during
        version decoding and repair operations.

    Example:
        >>> reduced = reduce_packet_to_chunk(packet, solver, chunk_to_reduce_to=0)
        >>> assert len(reduced.used_packets) == 1
    """
    # Get normalized used chunks and remove auxiliary packets
    used_chunks = semiautomatic_solver.decoder.removeAndXorAuxPackets(packet)
    packet.used_packets = {i for i, x in enumerate(used_chunks) if x}

    # XOR out all chunks except the target
    for i, must_remove_chunk in enumerate(used_chunks):
        if must_remove_chunk and i != chunk_to_reduce_to:
            packet.data = xor_numpy(packet.data, semiautomatic_solver.decoder.GEPP.b[i])
            packet.used_packets.remove(i)

    # Verify only target chunk remains
    assert len(packet.used_packets) == 1 and chunk_to_reduce_to in packet.used_packets, (
        f"Packet reduction failed: expected only chunk {chunk_to_reduce_to}, "
        f"got {packet.used_packets}"
    )

    return packet


def find_insertion_position(
    version_string_length: int,
    chunk_diff: np.ndarray,
    id_spacing: int,
    id_len: int,
) -> Generator[int, int, int]:
    """
    Find a suitable insertion position for the version string in the chunk diff.

    Scans the chunk diff array for regions of zeros (padding) that are large
    enough to accommodate the version string without overlapping seed positions.

    Args:
        version_string_length: Length of version string in bytes (not quats).
        chunk_diff: Numpy array representing the chunk diff.
        id_spacing: Spacing between chunk ID DNA bases in the packet.
        id_len: Length of the chunk ID in bytes.

    Yields:
        Starting indices of valid insertion positions.

    Returns:
        -1 if no suitable position is found.

    Note:
        The function searches from the end of the array backwards to find
        the last suitable position, which often provides better stability.

    Example:
        >>> positions = list(find_insertion_position(10, chunk_diff, id_spacing=2, id_len=4))
        >>> if positions:
        ...     print(f"Found {len(positions)} insertion positions")
    """
    # Find all zero regions in chunk diff
    zero_regions = np.where(chunk_diff == 0)[0]

    # Search for suitable positions from end backwards
    for start_idx in reversed([x for x in zero_regions]):
        # Check if region is large enough and doesn't overlap with ID spacing
        if (
            start_idx + version_string_length <= len(chunk_diff)
            and id_spacing * id_len * 4 - 1 < start_idx
        ):
            # Verify entire region is zero
            if np.all(chunk_diff[start_idx : start_idx + version_string_length] == 0):
                yield start_idx

    # No suitable position found
    return -1


def find_insertion_position_with_seed(
    version_string_length: int,
    chunk_diff: np.ndarray,
    id_spacing: int,
    id_len: int,
) -> Generator[int, int, int]:
    """
    Find insertion position accounting for seed spacing constraints.

    Similar to find_insertion_position but creates a detailed mask that accounts
    for seed spacing at the quaternary (DNA base) level, ensuring the version
    string doesn't overlap with seed positions.

    Args:
        version_string_length: Length of version string in bytes (not quats).
        chunk_diff: Numpy array representing the chunk diff.
        id_spacing: Spacing between chunk ID DNA bases in the packet.
        id_len: Length of the chunk ID in bytes.

    Yields:
        Starting indices of valid insertion positions that don't overlap with seeds.

    Returns:
        -1 if no suitable position is found.

    Note:
        This function is more precise than find_insertion_position when seed
        spacing is used, as it checks at the DNA base level rather than byte level.

    Example:
        >>> positions = list(
        ...     find_insertion_position_with_seed(10, chunk_diff, id_spacing=2, id_len=4)
        ... )
    """
    # Create mask for valid positions (True = usable, False = blocked by seed)
    num_bytes = len(chunk_diff)
    num_quats = num_bytes * 4  # Each byte = 4 DNA bases (quaternary)

    # Create quaternary-level mask (True = usable)
    quat_mask = np.ones(num_quats, dtype=bool)

    if id_spacing > 0 and id_len > 0:
        # Number of seed bases
        num_seed_bases = id_len * 4
        # Seed bases are placed at positions 0, (id_spacing+1), 2*(id_spacing+1), ...
        seed_step = id_spacing + 1
        for i in range(num_seed_bases):
            seed_quat_pos = i * seed_step
            if seed_quat_pos < num_quats:
                quat_mask[seed_quat_pos] = False

    # Convert quaternary mask to byte mask
    # A byte is only usable if ALL 4 of its quaternary positions are usable
    byte_mask = np.ones(num_bytes, dtype=bool)
    for byte_idx in range(num_bytes):
        quat_start = byte_idx * 4
        quat_end = min(quat_start + 4, num_quats)
        if not np.all(quat_mask[quat_start:quat_end]):
            byte_mask[byte_idx] = False

    # Combine with chunk_diff == 0 condition
    # Valid positions: chunk_diff is zero AND byte is not blocked by seed spacing
    valid_positions = (chunk_diff == 0) & byte_mask

    # Find zero regions in valid positions
    zero_regions = np.where(valid_positions)[0]

    # Search for suitable positions from end backwards
    for start_idx in reversed([x for x in zero_regions]):
        if start_idx + version_string_length <= len(chunk_diff):
            # Verify entire region is valid
            if np.all(valid_positions[start_idx : start_idx + version_string_length]):
                yield start_idx

    # No suitable position found
    return -1


def generate_new_packets(
    semiautomatic_solver: SemiAutomaticReconstructionToolkit,
    encoder: RU10Encoder,
    diff: np.ndarray,
    changed_chunk_ids: np.ndarray,
    new_file_version: int,
) -> Dict[int, Union[List[RU10Packet], List[Tuple[RU10Packet, RU10Packet]]]]:
    """
    Scan ALL seeds and record ALL candidate seeds for each changed chunk.

    This function performs an exhaustive search through all possible seeds to
    find packets that:
    1. Include at least one changed chunk
    2. Include the header chunk (0) or last chunk
    3. Have suitable insertion positions for version strings

    Args:
        semiautomatic_solver: SemiAutomaticReconstructionToolkit instance.
        encoder: RU10Encoder instance for packet generation.
        diff: Diff array from find_affected_chunks.
        changed_chunk_ids: Array of changed chunk indices.
        new_file_version: New version number to encode.

    Returns:
        Dictionary mapping chunk IDs to lists of generated packets (or packet pairs).

    Raises:
        RuntimeError: If no valid seeds found for any changed chunk.

    Optimizations:
        - Precompute changed set for O(1) membership checks
        - Quick-skip seeds that don't mention any changed chunk
        - Quick-skip seeds that don't include header or last chunk

    Example:
        >>> packets = generate_new_packets(solver, encoder, diff, changed_chunks, version)
        >>> for chunk_id, chunk_packets in packets.items():
        ...     print(f"Chunk {chunk_id}: {len(chunk_packets)} packets")
    """
    res: Dict[int, Union[List[RU10Packet], List[Tuple[RU10Packet, RU10Packet]]]] = {}

    max_num = min(encoder.calc_max_size(struct.calcsize("<" + encoder.id_len_format)), int31)

    last_chunk_idx = encoder.number_of_chunks - 1
    changed_set = {int(x) for x in changed_chunk_ids}

    # Initialize mapping for results - one entry per changed chunk
    packet_to_seed_mapping: Dict[int, Set[int]] = {cid: set() for cid in changed_set}

    logger.info(f"Scanning {max_num:,} seeds for {len(changed_set)} changed chunks...")
    logger.info(f"  Changed chunk IDs: {changed_set}")
    logger.info(f"  Last chunk index: {last_chunk_idx}")

    seeds_scanned = 0
    seeds_matched = 0
    last_progress = 0

    # Scan all seeds
    for seed in range(max_num):
        packet_numbers = choose_packet_numbers(
            encoder.number_of_chunks, seed, encoder.dist, systematic=False
        )

        # remove and xor aux packets to get true used chunks for this seed:

        # Create packet and get used chunks
        # used_chunks = semiautomatic_solver.decoder.removeAndXorAuxPackets(packet)
        bool_used_chunks = semiautomatic_solver.decoder.removeAndXorAuxPackets_from_indices(
            packet_numbers
        )
        used_chunks = from_true_false_list(bool_used_chunks)
        """
        try:
            assert np.all(np.equal(used_chunks,  used_chunks_new)), f"{used_chunks} --- {used_chunks_new}"
        except AssertionError:
            print("error")
        """
        # OPTIMIZATION 1: Quick filter using set intersection
        pn_set = set(used_chunks)
        if not (pn_set & changed_set) or not (
            pn_set & {0}
        ):  # TODO: the second comparison is not really needed!
            continue

        # OPTIMIZATION 2: Skip if doesn't include header or last chunk
        # this is not technically needed as we could also use regions without changes to insert version string,
        # but it reduces the search-space and simplifies the POC logic.
        # if 0 not in pn_set or last_chunk_idx not in pn_set:
        #    continue

        # Verify header or last chunk is still used after aux removal
        try:
            if not (bool(bool_used_chunks[0]) or bool(bool_used_chunks[-1])):
                continue
        except Exception:
            continue

        # Find intersection with changed chunks
        touched_changed = pn_set & changed_set
        if touched_changed:
            seeds_matched += 1
            for cid in touched_changed:
                packet_to_seed_mapping[cid].add(seed)

        seeds_scanned += 1

        # Log progress every 10% or every 10000 seeds (whichever is more frequent)
        progress_interval = max(10000, max_num // 10)
        if seeds_scanned % progress_interval == 0 and seeds_scanned != last_progress:
            progress_pct = (seeds_scanned / max_num) * 100
            total_candidates = sum(len(v) for v in packet_to_seed_mapping.values())
            logger.info(
                f"  Progress: {seeds_scanned:,}/{max_num:,} seeds ({progress_pct:.1f}%) - "
                f"Found {total_candidates} candidate seeds across {len([cid for cid, seeds in packet_to_seed_mapping.items() if seeds])} chunks"
            )
            last_progress = seeds_scanned

        # packet = RU10Packet(b"", packet_numbers, encoder.number_of_chunks, seed, encoder.dist, read_only=True)
    """
    logger.error(f"packet.used_packets: {packet.used_packets}")
    logger.error(f"get_bool_array_used_packets: {[i for i, v in enumerate(packet.get_bool_array_used_packets()) if v]}")
    logger.error(
        f"get_bool_array_used_and_ldpc_packets: {[i for i, v in enumerate(packet.get_bool_array_used_and_ldpc_packets()) if v]}")

    r1 = semiautomatic_solver.decoder.removeAndXorAuxPackets(packet)
    r2 = semiautomatic_solver.decoder.removeAndXorAuxPackets_from_indices(packet.used_packets)

    logger.error(f"Original True count: {np.sum(r1)}")
    logger.error(f"New True count: {np.sum(r2)}")
    logger.error(f"Original True indices: {[i for i, v in enumerate(r1) if v]}")
    logger.error(f"New True indices: {[i for i, v in enumerate(r2) if v]}")

    logger.warning(f"=== PACKET BOUNDARIES ===")
    logger.warning(f"packet.total_number_of_chunks: {packet.total_number_of_chunks}")
    logger.warning(f"packet.s: {packet.get_number_of_ldpc_blocks()}")
    logger.warning(f"u_bound: {packet.total_number_of_chunks + packet.get_number_of_ldpc_blocks()}")
    logger.warning(f"135 < u_bound: {135 < packet.total_number_of_chunks + packet.get_number_of_ldpc_blocks()}")

    logger.warning(f"\n=== DECODER BOUNDARIES ===")
    logger.warning(f"decoder.number_of_chunks: {semiautomatic_solver.decoder.number_of_chunks}")
    logger.warning(f"decoder.s: {semiautomatic_solver.decoder.s}")
    logger.warning(f"decoder u_bound: {semiautomatic_solver.decoder.number_of_chunks + semiautomatic_solver.decoder.s}")

    logger.warning(f"\n=== CHECK 135 ===")
    logger.warning(f"135 in packet.used_packets: {135 in packet.used_packets}")
    logger.warning(f"135 < packet u_bound: {135 < packet.total_number_of_chunks + packet.get_number_of_ldpc_blocks()}")

    used_ldpc = packet.get_bool_array_used_and_ldpc_packets()
    logger.warning(
        f"135 in get_bool_array_used_and_ldpc_packets: {used_ldpc[135] if len(used_ldpc) > 135 else 'INDEX OUT OF RANGE'}")
    logger.warning(f"len(used_ldpc): {len(used_ldpc)}")
    """

    # Verify we found at least one seed for every changed chunk
    missing = [cid for cid, seeds in packet_to_seed_mapping.items() if not seeds]
    if missing:
        logger.error(f"✗ Could not find ANY seed for chunks {missing}")
        logger.error(
            "  Each changed chunk must have at least one packet containing header or last chunk"
        )
        raise RuntimeError(
            f"Could not find ANY seed for chunks {missing} that contains either header or last chunk"
        )

    total_candidates = sum(len(v) for v in packet_to_seed_mapping.values())
    logger.info("✓ Scan complete!")
    logger.info(f"  Total seeds scanned: {seeds_scanned:,}")
    logger.info(f"  Seeds matching criteria: {seeds_matched:,}")
    logger.info(f"  Total candidate seeds found: {total_candidates:,}")
    logger.info(
        f"  Candidates per chunk: {dict((k, len(v)) for k, v in packet_to_seed_mapping.items())}"
    )

    # Select best seeds for each chunk
    chunk_to_potential_seed_mapping = select_numbers(packet_to_seed_mapping, n=50)

    # Generate packets for selected seeds
    generated_packets: Dict[int, List[RU10Packet]] = {}
    for chunk_id, seed_set in chunk_to_potential_seed_mapping:
        for seed in seed_set:
            packet = encoder.create_new_packet(False, seed)
            should_drop_packet(encoder.rules, packet, 1.0)  # Calculate error probability

            if chunk_id not in generated_packets:
                generated_packets[chunk_id] = []
            generated_packets[chunk_id].append(packet)

        # Sort by error probability (best first)
        generated_packets[chunk_id] = sorted(
            generated_packets[chunk_id], key=lambda x: x.error_prob
        )

    # Populate header chunk for version string insertion
    semiautomatic_solver.decoder.populate_header_chunk(
        last_chunk_len_str=semiautomatic_solver.decoder.config_map.get("last_chunk_len_str", "I")
    )

    # Process each changed chunk
    changed_chunk_to_new_packets: Dict[int, List[RU10Packet]] = {}
    changed_chunk_to_packet_pair_list: Dict[int, List[Tuple[RU10Packet, RU10Packet]]] = {}

    version_string = generate_dna_version_string(new_file_version)

    for changed_chunk, potential_packets in generated_packets.items():
        packet_added = False

        # Try each potential packet
        for potential_packet in potential_packets:
            modified_packet = potential_packet.copy()
            plain_used_chunks = semiautomatic_solver.decoder.removeAndXorAuxPackets(modified_packet)

            if plain_used_chunks[0]:
                # Header chunk case - find insertion position
                try:
                    insertion_position = next(
                        find_insertion_position(
                            int((len(version_string) + 1) / 4 + 1),
                            diff[changed_chunk],
                            modified_packet.id_spacing,
                            struct.calcsize(modified_packet.id_len_format),
                        )
                    )
                except StopIteration:
                    # No suitable position found, try next packet
                    continue

                # Insert version string
                insert_dna_version_string(
                    modified_packet, version_string, insertion_position, diff[changed_chunk]
                )

                # Insert chunk ID
                insert_id_string(
                    modified_packet,
                    int(insertion_position + (len(version_string) + 1) / 4),
                    changed_chunk,
                    semiautomatic_solver,
                )

            elif plain_used_chunks[-1]:
                # Last chunk case - not yet implemented
                continue
            else:
                # Neither header nor last chunk - not yet implemented
                # TODO: as we only consider the difference between the original and new version to encode the
                #  offset of the changed chunk (based on the first chunk of the used chunk list
                #  _AFTER_ removing aux-packets, from the generated packet, we would be able to work with ANY packet
                #  containing the affected chunk - even if the changed chunk is the first one!
                continue

            # Apply diff for changed chunk
            modified_packet.data = xor_numpy(modified_packet.data, diff[changed_chunk])

            # Recalculate DNA structure
            modified_packet.get_dna_struct(
                True, modified_packet.id_spacing, modified_packet.id_spacing_length, True
            )

            # Calculate error probability
            should_drop_packet(encoder.rules, modified_packet)

            # Store packet
            if changed_chunk not in changed_chunk_to_new_packets:
                changed_chunk_to_new_packets[changed_chunk] = []
            changed_chunk_to_new_packets[changed_chunk].append(modified_packet)
            packet_added = True

        # Handle case where no single packet worked - try packet pairs
        if not packet_added or (
            changed_chunk_to_new_packets.get(changed_chunk)
            and sorted(changed_chunk_to_new_packets[changed_chunk], key=lambda x: x.error_prob)[
                0
            ].error_prob
            >= 1.0
        ):
            logger.warning(
                f"Could not find suitable packet for chunk {changed_chunk} to insert "
                f"version string. Generating two packets with same seed with split diff (50/50)!"
            )

            for potential_packet in potential_packets:
                modified_packet_first = potential_packet.copy()
                modified_packet_second = potential_packet.copy()

                # Split diff in half
                diff_mask = np.zeros_like(diff[changed_chunk], dtype=bool)
                half_point = len(diff[changed_chunk]) // 2
                diff_mask[:half_point] = True
                first_diff = np.where(diff_mask, diff[changed_chunk], 0)
                second_diff = np.where(~diff_mask, diff[changed_chunk], 0)

                # Try to find insertion positions for both halves
                if plain_used_chunks[0]:
                    try:
                        insertion_position_first = next(
                            find_insertion_position(
                                int((len(version_string) + 1) / 4 + 1),
                                diff[changed_chunk],
                                modified_packet_first.id_spacing,
                                struct.calcsize(modified_packet_first.id_len_format),
                            )
                        )
                        insertion_position_second = next(
                            find_insertion_position(
                                int((len(version_string) + 1) / 4 + 1),
                                diff[changed_chunk],
                                modified_packet_second.id_spacing,
                                struct.calcsize(modified_packet_second.id_len_format),
                            )
                        )
                    except StopIteration:
                        continue

                    # Insert version string and chunk ID in both packets
                    insert_dna_version_string(
                        modified_packet_first,
                        version_string,
                        insertion_position_first,
                        first_diff,
                    )
                    insert_id_string(
                        modified_packet_second,
                        int(insertion_position_second + (len(version_string) + 1) / 4),
                        changed_chunk,
                        semiautomatic_solver,
                    )

                    insert_dna_version_string(
                        modified_packet_first,
                        version_string,
                        insertion_position_first,
                        second_diff,
                    )
                    insert_id_string(
                        modified_packet_second,
                        int((insertion_position_second + (len(version_string) + 1)) / 4),
                        changed_chunk,
                        semiautomatic_solver,
                    )

                elif plain_used_chunks[-1]:
                    # Last chunk case - not yet implemented
                    continue

                # Apply split diffs
                modified_packet_first.data = xor_numpy(modified_packet_first.data, first_diff)
                modified_packet_first.get_dna_struct(
                    True,
                    modified_packet_first.id_spacing,
                    modified_packet_first.id_spacing_length,
                    True,
                )

                modified_packet_second.data = xor_numpy(modified_packet_second.data, second_diff)
                modified_packet_second.get_dna_struct(
                    True,
                    modified_packet_second.id_spacing,
                    modified_packet_second.id_spacing_length,
                    True,
                )

                # Calculate error probabilities
                should_drop_packet(encoder.rules, modified_packet_first)
                should_drop_packet(encoder.rules, modified_packet_second)

                # Store packet pair
                changed_chunk_to_new_packets[changed_chunk].append(modified_packet_first)
                changed_chunk_to_new_packets[changed_chunk].append(modified_packet_second)

                if changed_chunk not in changed_chunk_to_packet_pair_list:
                    changed_chunk_to_packet_pair_list[changed_chunk] = []
                changed_chunk_to_packet_pair_list[changed_chunk].append(
                    (modified_packet_first, modified_packet_second)
                )
                packet_added = True

        if not packet_added:
            logger.error(f"Could not create any packet for changed chunk {changed_chunk}!")

    # Select best packets for each changed chunk
    for changed_chunk in changed_chunk_to_new_packets.keys():
        if changed_chunk in changed_chunk_to_new_packets:
            # Try using best single packet first
            changed_chunk_to_new_packets[changed_chunk] = sorted(
                changed_chunk_to_new_packets[changed_chunk], key=lambda x: x.error_prob
            )

            for new_pack in changed_chunk_to_new_packets[changed_chunk]:
                if new_pack.error_prob < 1.0:
                    if changed_chunk not in res:
                        res[changed_chunk] = []
                    res[changed_chunk].append(new_pack)
                    logger.debug(
                        f"Generated packet with error probability {new_pack.error_prob} "
                        f"for chunk {changed_chunk}."
                    )
                else:
                    logger.debug(
                        f"Skipping packet with error probability {new_pack.error_prob} "
                        f"for chunk {changed_chunk}!"
                    )

        elif changed_chunk in changed_chunk_to_packet_pair_list:
            # Use best packet pair
            changed_chunk_to_packet_pair_list[changed_chunk] = sorted(
                changed_chunk_to_packet_pair_list[changed_chunk],
                key=lambda x: max(x[0].error_prob, x[1].error_prob),
            )

            for new_pack_pair in changed_chunk_to_packet_pair_list[changed_chunk]:
                if new_pack_pair[0].error_prob < 1.0 and new_pack_pair[1].error_prob < 1.0:
                    if changed_chunk not in res:
                        res[changed_chunk] = []
                    res[changed_chunk].append(new_pack_pair)
                    logger.debug(
                        f"Generated packet pair with error probability "
                        f"{(new_pack_pair[0].error_prob, new_pack_pair[1].error_prob)} "
                        f"for chunk {changed_chunk}!"
                    )
                else:
                    logger.debug(
                        f"Skipping packet pair with error probability "
                        f"{(new_pack_pair[0].error_prob, new_pack_pair[1].error_prob)} "
                        f"for chunk {changed_chunk}!"
                    )

    return res


def create_perceptual_hash(image: Image.Image) -> str:
    """
    Create a perceptual hash for an image.

    This function generates a perceptual hash (phash) that can be used to
    compare images for similarity. The hash is returned as a string representation.

    Args:
        image: PIL Image object to hash.

    Returns:
        String representation of the perceptual hash.

    Note:
        Perceptual hashes are useful for detecting similar images even after
        minor modifications like compression or resizing.

    Example:
        >>> from PIL import Image
        >>> img = Image.open("image.png")
        >>> hash_value = create_perceptual_hash(img)
        >>> print(f"Image hash: {hash_value}")
    """
    if not HAS_IMAGEHASH:
        raise ImportError("imagehash library not available. Install with: pip install imagehash")

    hash_value = imagehash.phash(image)
    return str(hash_value)


def add_packets(
    encoder: RU10Encoder, new_packets: Union[Set[RU10Packet], List[RU10Packet]]
) -> None:
    """
    Add new packets to an encoder's packet set.

    This is a convenience function for adding generated packets to an encoder.

    Args:
        encoder: RU10Encoder instance to add packets to.
        new_packets: Set or list of RU10Packet objects to add.

    Example:
        >>> add_packets(encoder, generated_packets)
    """
    if isinstance(new_packets, list):
        encoder.encodedPackets.update(new_packets)
    else:
        encoder.encodedPackets |= new_packets


def decode_versions(
    semiautomatic_solver: SemiAutomaticReconstructionToolkit,
    dna_version_string_prefix: str = "",
) -> typing.Dict[int, str]:
    """
    Decode all versions of an encoded file from the packets known to the solver.

    This function iterates through all available versions in the DNA pool and
    decodes each one, saving them with version-prefixed filenames.

    Args:
        semiautomatic_solver: The SemiAutomaticReconstructionToolkit with the
            decoder containing the packets.
        dna_version_string_prefix: The prefix used for version strings in the
            DNA sequences (default: "").

    Returns:
        A dictionary mapping file version numbers to filenames for all versions found.

    Note:
        The decoding process:
        1. Decodes base version using only packets WITHOUT version strings
        2. For each version: decodes changed chunks using packets WITH that version's string
        3. Saves each version as v<version>_<base_filename>.<extension>
        4. Uses each new version as the base for the next version

    Example:
        >>> versions = decode_versions(solver)
        >>> for version, filename in versions.items():
        ...     print(f"Version {version}: {filename}")
    """
    res: typing.Dict[int, str] = {}

    # Get base version string
    if not dna_version_string_prefix:
        dna_version_string_prefix = "GAGCCAGTGAGTCGTA"

    # Get maximum version in pool
    from MultiVersionDecoder import MultiVersionDecoder

    mv_decoder = MultiVersionDecoder(semiautomatic_solver.decoder)
    max_version = mv_decoder.get_versions_in_pool(dna_version_string_prefix)

    logger.info(f"Found versions 0 to {max_version} in pool")

    # Decode base version (version 0)
    logger.info("Decoding base version (v0)...")
    mv_decoder.decode_base_version(dna_version_string_prefix)
    res[0] = f"v0_{semiautomatic_solver.decoder.headerChunk.file_name.decode('utf-8')}"

    # Decode each subsequent version
    for version in range(1, max_version + 1):
        logger.info(f"Decoding version {version}...")
        try:
            decoded = mv_decoder.decode_to_version(dna_version_string_prefix, version)
            if version in decoded:
                res[version] = decoded[version]
                logger.info(f"Version {version} decoded successfully")
        except Exception as e:
            logger.error(f"Failed to decode version {version}: {e}")
            # Continue with next version

    return res


# ============================================================================
# MultiVersionCoder Class (Object-oriented API)
# ============================================================================


class MultiVersionCoder:
    """
    Multi-version coder for NOREC4DNA encoded files.

    This class handles encoding of new file versions into existing DNA pools,
    properly managing version numbers and creating diffs from the latest version.

    Attributes:
        config: ConfigReadAndExecute instance for the current pool
        solver: SemiAutomaticReconstructionToolkit instance
        encoder: RU10Encoder instance for packet generation
        magic_string: DNA magic string marker for versions

    Example:
        >>> from MultiVersionCoder import MultiVersionCoder
        >>> coder = MultiVersionCoder("existing_pool.ini")
        >>> max_version = coder.get_max_version_in_pool()
        >>> print(f"Pool contains versions 0 to {max_version}")
    """

    def __init__(
        self,
        config: Union[ConfigReadAndExecute, str],
        magic_string: str = "GAGCCAGTGAGTCGTA",
    ) -> None:
        """
        Initialize the MultiVersionCoder with configuration.

        Args:
            config: ConfigReadAndExecute instance or path to INI configuration file.
            magic_string: DNA magic string marker for version sequences
                (default: "GAGCCAGTGAGTCGTA").

        Raises:
            ValueError: If configuration cannot be loaded.
            FileNotFoundError: If config file path doesn't exist.
        """
        # Handle both ConfigReadAndExecute instance and file path
        if isinstance(config, str):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config}")
            self.config = ConfigReadAndExecute(config)
        else:
            self.config = config

        # Initialize solver and encoder
        self._initialize_solver()
        self.magic_string = magic_string
        self.encoder: Optional[RU10Encoder] = None

    def _initialize_solver(self) -> None:
        """Initialize the solver and encoder from configuration."""
        try:
            decoder = self.config.execute(return_decoder=True, skip_solve=True)[0]
            self.solver = SemiAutomaticReconstructionToolkit(decoder)
        except Exception as e:
            raise ValueError(f"Failed to initialize solver from config: {e}")

    def get_max_version_in_pool(self) -> int:
        """
        Get the maximum version number currently in the DNA pool.

        Scans all DNA sequences in the pool and extracts the highest version
        number from sequences containing the magic string marker.

        Returns:
            The highest version number found. Returns 0 if no version strings
            are present (base version only).

        Note:
            Version 0 is the base version, version 1 is the first update, etc.

        Example:
            >>> coder = MultiVersionCoder("pool.ini")
            >>> max_ver = coder.get_max_version_in_pool()
            >>> print(f"Latest version in pool: {max_ver}")
        """
        return get_current_file_version(self.solver, self.magic_string)

    def get_next_version_number(self) -> int:
        """
        Get the version number for the next version to be encoded.

        Returns:
            The next version number (max_existing + 1).

        Example:
            >>> next_ver = coder.get_next_version_number()
            >>> print(f"Next version will be: {next_ver}")
        """
        return self.get_max_version_in_pool() + 1

    def get_sequences_for_version(self, version: int) -> List[str]:
        """
        Get all DNA sequences corresponding to a specific version.

        Args:
            version: Version number to retrieve sequences for.

        Returns:
            List of DNA sequences for the specified version.

        Example:
            >>> v1_seqs = coder.get_sequences_for_version(1)
            >>> print(f"Version 1 has {len(v1_seqs)} sequences")
        """
        from MultiVersionDecoder import MultiVersionDecoder

        mv_decoder = MultiVersionDecoder(self.solver.decoder)
        return mv_decoder.get_sequences_for_version(self.magic_string, version)

    def get_all_version_numbers(self) -> List[int]:
        """
        Get a list of all version numbers present in the pool.

        Returns:
            Sorted list of version numbers present in the pool.

        Example:
            >>> versions = coder.get_all_version_numbers()
            >>> print(f"Pool contains versions: {versions}")
        """
        max_version = self.get_max_version_in_pool()
        return list(range(max_version + 1))

    def analyze_existing_pool(self) -> Dict:
        """
        Analyze the existing DNA pool and return version information.

        Returns:
            Dictionary containing:
            - max_version: Highest version number in pool
            - all_versions: List of all version numbers
            - total_sequences: Total number of DNA sequences
            - sequences_per_version: Dict mapping version to sequence count

        Example:
            >>> info = coder.analyze_existing_pool()
            >>> print(f"Pool analysis: {info}")
        """
        fasta_entries = load_fasta(self.solver.decoder.file)
        total_sequences = len(fasta_entries)

        sequences_per_version: Dict[int, int] = {}
        for seq in fasta_entries.values():
            idx = seq.find(self.magic_string)
            if idx != -1 and idx >= 3:
                version_bases = seq[idx - 3 : idx]
                if len(version_bases) < 3:
                    version_bases = "A" * (3 - len(version_bases)) + version_bases
                try:
                    version_num = struct.unpack("B", tranlate_quat_to_byte(f"A{version_bases}"))[0]
                    sequences_per_version[version_num] = (
                        sequences_per_version.get(version_num, 0) + 1
                    )
                except struct.error:
                    logger.warning(f"Failed to parse version from sequence")

        # Count sequences without version marker as base version (0)
        if 0 not in sequences_per_version:
            sequences_per_version[0] = 0

        max_version = self.get_max_version_in_pool()

        return {
            "max_version": max_version,
            "all_versions": list(range(max_version + 1)),
            "total_sequences": total_sequences,
            "sequences_per_version": sequences_per_version,
        }

    def decode_to_latest_version(self) -> SemiAutomaticReconstructionToolkit:
        """
        Decode all versions in the pool up to the latest one.

        This method reconstructs the latest version by iteratively decoding
        from base version (v0) through all updates. This is essential for
        calculating the correct diff when adding a new version.

        Returns:
            SemiAutomaticReconstructionToolkit with the latest version decoded.

        Raises:
            RuntimeError: If decoding fails for any version.

        Example:
            >>> coder = MultiVersionCoder("pool_v2.ini")
            >>> solver = coder.decode_to_latest_version()
            >>> # Now solver contains the reconstructed v2 data
            >>> latest_data = solver.get_file_as_bytes()
        """
        from MultiVersionDecoder import MultiVersionDecoder

        max_version = self.get_max_version_in_pool()

        if max_version == 0:
            logger.info("Pool contains only base version (v0), no decoding needed")
            return self.solver

        logger.info(f"Decoding pool with {max_version + 1} versions (v0 to v{max_version})...")
        logger.info("This is necessary to reconstruct the latest version for diff calculation")

        # Initialize MultiVersionDecoder
        mv_decoder = MultiVersionDecoder(self.solver.decoder)

        # Decode base version (v0)
        logger.info("Decoding base version (v0)...")
        mv_decoder.decode_base_version(self.magic_string)
        logger.info("✓ Base version decoded")

        # Decode each subsequent version
        for version in range(1, max_version + 1):
            logger.info(f"Decoding version {version}/{max_version}...")
            try:
                mv_decoder.decode_to_version(self.magic_string, version)
                logger.info(f"✓ Version {version} decoded successfully")
            except Exception as e:
                logger.error(f"✗ Failed to decode version {version}: {e}")
                raise RuntimeError(f"Failed to decode version {version}: {e}")

        logger.info(
            f"✓ All versions decoded successfully - latest version (v{max_version}) reconstructed"
        )

        # Update solver to use the decoded state
        self.solver = SemiAutomaticReconstructionToolkit(mv_decoder.decoder)
        return self.solver

    def encode_new_version(
        self,
        new_file_data: bytes,
        packets_per_chunk: int = 5,
        validate_diff: bool = True,
    ) -> Tuple[int, List[RU10Packet]]:
        """
        Encode a new file version into the DNA pool.

        This is the main method for adding new versions. It:
        1. Determines the next version number
        2. Decodes all existing versions to reconstruct the latest version
        3. Calculates diff from latest version to new version
        4. Generates DNA packets for changed content
        5. Adds packets to encoder

        Args:
            new_file_data: Bytes of the new file version.
            packets_per_chunk: Number of packets to generate per changed chunk.
            validate_diff: If True, validates that diff calculation succeeds.

        Returns:
            Tuple of (version_number, list_of_generated_packets).

        Raises:
            ValueError: If file size is larger than current version (growth not supported).
            RuntimeError: If packet generation fails or decoding fails.

        Example:
            >>> with open("updated_file.bin", "rb") as f:
            ...     new_data = f.read()
            >>> version, packets = coder.encode_new_version(new_data)
            >>> print(f"Created version {version} with {len(packets)} packets")
        """
        logger.info("=" * 80)
        logger.info("Starting MultiVersionCoder.encode_new_version()")
        logger.info("=" * 80)

        # Get next version number
        current_version = self.get_max_version_in_pool()
        new_version = self.get_next_version_number()
        logger.info(f"Current version in pool: {current_version}")
        logger.info(f"Creating new version: {new_version}")
        logger.info(f"New file size: {len(new_file_data):,} bytes")
        logger.info(f"Packets per chunk: {packets_per_chunk}")

        # CRITICAL: Decode all versions up to the latest one before calculating diff
        if current_version > 0:
            logger.info("=" * 80)
            logger.info(f"Pool contains {current_version + 1} versions (v0 to v{current_version})")
            logger.info(
                f"Decoding up to latest version (v{current_version}) to use as diff base..."
            )
            logger.info("=" * 80)
            self.decode_to_latest_version()
            logger.info(
                f"✓ Latest version (v{current_version}) reconstructed and ready for diff calculation"
            )
        else:
            logger.info("Pool contains only base version (v0) - no decoding needed")

        # Calculate diff from latest version to new version
        logger.info("Calculating diff from latest version to new file...")
        diff, changed_chunks = find_affected_chunks(self.solver, new_file_data)
        logger.info(
            f"✓ Found {len(changed_chunks)} changed chunk(s): {changed_chunks.tolist() if len(changed_chunks) > 0 else 'none'}"
        )

        if not changed_chunks:
            logger.warning("⚠ No changes detected between latest version and new file!")
            logger.warning("⚠ Version will be created but no new packets will be added.")
            return new_version, []

        logger.info(
            f"Changed chunks represent {len(changed_chunks) * diff.shape[1]:,} bytes of data"
        )

        # Create encoder if not already created
        if self.encoder is None:
            logger.info("Initializing encoder from decoder configuration...")
            self.encoder = encoder_from_decoder(self.solver, self.config, rules=FastDNARules())
            logger.info("✓ Encoder initialized")

        # Generate packets for changed content
        logger.info(f"Generating packets for {len(changed_chunks)} changed chunk(s)...")
        logger.info("This may take a while depending on the number of seeds to scan...")
        packet_candidates = generate_new_packets(
            self.solver, self.encoder, diff, changed_chunks, new_version
        )
        logger.info(
            f"✓ Found candidate packets: {dict((k, len(v)) for k, v in packet_candidates.items())}"
        )

        # Select best packets
        generated_packets: List[RU10Packet] = []
        packets_added = 0
        packets_skipped = 0

        logger.info(f"Selecting best packets (max {packets_per_chunk} per chunk)...")
        for chunk_id, packets in packet_candidates.items():
            added_for_chunk = 0
            for packet in packets:
                if packet.error_prob >= 1.0:
                    logger.debug(
                        f"  Skipping packet for chunk {chunk_id} with high error prob {packet.error_prob:.4f}"
                    )
                    packets_skipped += 1
                    continue

                if added_for_chunk >= packets_per_chunk:
                    break

                self.encoder.encodedPackets.add(packet)
                generated_packets.append(packet)
                added_for_chunk += 1
                packets_added += 1
                logger.debug(
                    f"  ✓ Added packet {packet.id} for chunk {chunk_id} (error prob: {packet.error_prob:.4f})"
                )

            if added_for_chunk > 0:
                logger.info(f"✓ Chunk {chunk_id}: Added {added_for_chunk} packet(s)")
            else:
                logger.warning(f"⚠ Chunk {chunk_id}: No suitable packets found!")

        logger.info("-" * 80)
        logger.info("Encoding Summary:")
        logger.info(f"  Version created: {new_version}")
        logger.info(f"  Changed chunks: {len(changed_chunks)}")
        logger.info(f"  Packets added: {packets_added}")
        logger.info(f"  Packets skipped (high error): {packets_skipped}")
        logger.info(f"  Total packets in encoder: {len(self.encoder.encodedPackets)}")
        logger.info("-" * 80)

        return new_version, generated_packets

    def save_updated_pool(self, output_path: Union[str, Path]) -> None:
        """
        Save the updated DNA pool with all versions.

        Args:
            output_path: Path to save the updated FASTA file.

        Raises:
            ValueError: If encoder is not initialized.

        Example:
            >>> coder.save_updated_pool("updated_pool.fasta")
        """
        logger.info("=" * 80)
        logger.info("Saving updated DNA pool...")
        logger.info("=" * 80)

        if self.encoder is None:
            raise ValueError("Encoder not initialized. Call encode_new_version first.")

        output_path = Path(output_path)
        logger.info(f"Output FASTA file: {output_path.absolute()}")

        # Get total packet count before saving
        total_packets = len(self.encoder.encodedPackets)
        logger.info(f"Total packets to save: {total_packets}")

        self.encoder.file = str(output_path)
        logger.info("Writing FASTA file...")
        self.encoder.save_packets_fasta(None, str(output_path), False)
        logger.info(f"✓ FASTA file written: {output_path.absolute()}")

        logger.info("Saving configuration file...")
        config_path = self.encoder.save_config_file(add_dot_fasta=True)
        logger.info(f"✓ Configuration file written: {config_path}")

        # Get file sizes
        try:
            fasta_size = output_path.stat().st_size
            config_size = Path(config_path).stat().st_size
            logger.info(
                f"FASTA file size: {fasta_size:,} bytes ({fasta_size / 1024 / 1024:.2f} MB)"
            )
            logger.info(f"Config file size: {config_size:,} bytes")
        except Exception as e:
            logger.debug(f"Could not get file sizes: {e}")

        logger.info("-" * 80)
        logger.info("✓ Updated pool saved successfully!")
        logger.info(f"  FASTA: {output_path.absolute()}")
        logger.info(f"  Config: {config_path}")
        logger.info("=" * 80)

    def add_packets_to_pool(self, packets: Union[Set[RU10Packet], List[RU10Packet]]) -> None:
        """
        Add packets to the encoder's pool.

        Args:
            packets: Set or list of RU10Packet objects to add.

        Example:
            >>> coder.add_packets_to_pool(generated_packets)
        """
        if self.encoder is None:
            self.encoder = encoder_from_decoder(self.solver, self.config, rules=FastDNARules())

        if isinstance(packets, list):
            self.encoder.encodedPackets.update(packets)
        else:
            self.encoder.encodedPackets |= packets

        logger.info(f"Added {len(packets)} packets to pool")

    def combine_existing_fasta_with_packets(
        self,
        existing_fasta_path: Union[str, Path],
        packets: Union[Set[RU10Packet], List[RU10Packet]],
        output_fasta_path: Union[str, Path],
        output_config_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[int, int]:
        """
        Combine an existing FASTA file with newly generated packets.

        This method reads an existing FASTA file (e.g., from a previous version),
        combines it with newly generated packets, and saves the result to a new
        output file. This is useful when you want to preserve all previously
        generated sequences while adding new ones for updated content.

        Args:
            existing_fasta_path: Path to the existing FASTA file to use as base
            packets: Set or list of new RU10Packet objects to add
            output_fasta_path: Path to save the combined FASTA file
            output_config_path: Optional path to save the config file.
                If None, config is saved next to output_fasta with .ini extension.

        Returns:
            Tuple of (existing_sequence_count, new_packet_count, config_path)

        Raises:
            FileNotFoundError: If existing_fasta_path doesn't exist
            ValueError: If packets list is empty

        Example:
            >>> existing_count, new_count, config_path = coder.combine_existing_fasta_with_packets(
            ...     "pool_v1.fasta",
            ...     new_packets,
            ...     "pool_v2_combined.fasta"
            ... )
            >>> print(f"Combined {existing_count} existing + {new_count} new sequences")
            >>> print(f"Config saved at: {config_path}")
        """
        existing_fasta_path = Path(existing_fasta_path)
        output_fasta_path = Path(output_fasta_path)

        if not existing_fasta_path.exists():
            raise FileNotFoundError(f"Existing FASTA file not found: {existing_fasta_path}")

        if not packets:
            raise ValueError("No packets provided to add")

        # Convert packets to list if it's a set
        if isinstance(packets, set):
            packets = list(packets)

        logger.info("=" * 80)
        logger.info("Combining existing FASTA with new packets...")
        logger.info("=" * 80)
        logger.info(f"Existing FASTA: {existing_fasta_path.absolute()}")
        logger.info(f"New packets to add: {len(packets)}")
        logger.info(f"Output FASTA: {output_fasta_path.absolute()}")

        # Create temporary FASTA file with new packets
        temp_new_fasta = output_fasta_path.parent / f"temp_new_packets_{output_fasta_path.name}"
        logger.info(f"Writing {len(packets)} new packets to temporary file...")
        with open(temp_new_fasta, 'w') as f:
            for packet in packets:
                f.write(f">{packet.id}\n")
                f.write(f"{packet.dna_data}\n")

        try:
            # Combine existing and new FASTA files
            total_sequences = combine_fasta_files(
                [existing_fasta_path, temp_new_fasta],
                output_fasta_path
            )

            # Save config file
            if output_config_path is None:
                output_config_path = output_fasta_path.with_suffix(".ini")

            # Create a temporary encoder to save config if needed
            if self.encoder is None:
                self.encoder = encoder_from_decoder(self.solver, self.config, rules=FastDNARules())

            # Update encoder file path for config
            file_bkp = self.encoder.file
            self.encoder.file = str(output_fasta_path.with_suffix(""))
            self.encoder.out_file = output_fasta_path.with_suffix("")
            config_path = self.encoder.save_config_file(add_dot_fasta=True)
            self.encoder.file = file_bkp

            logger.info(f"✓ Configuration file written: {config_path}")

            # Get file sizes
            try:
                output_size = output_fasta_path.stat().st_size
                config_size = Path(config_path).stat().st_size
                logger.info("-" * 80)
                logger.info("File Sizes:")
                logger.info(
                    f"  Combined FASTA: {output_size:,} bytes ({output_size / 1024 / 1024:.2f} MB)"
                )
                logger.info(f"  Config file: {config_size:,} bytes")
            except Exception as e:
                logger.debug(f"Could not get file sizes: {e}")

            logger.info("=" * 80)
            logger.info("✓ FASTA combination completed successfully!")
            logger.info(f"  Total sequences: {total_sequences}")
            logger.info(f"  Existing sequences: {total_sequences - len(packets)}")
            logger.info(f"  New sequences: {len(packets)}")
            logger.info("=" * 80)

            return total_sequences - len(packets), len(packets), config_path

        finally:
            # Clean up temporary file
            if temp_new_fasta.exists():
                temp_new_fasta.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_new_fasta}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def init_args() -> argparse.Namespace:
    """
    Parse command-line arguments for MultiVersionCoder.

    Returns:
        Parsed arguments namespace.

    Example:
        >>> args = init_args()
        >>> print(f"Processing {args.new_file}")
    """
    parser = argparse.ArgumentParser(
        description="Multi-Version Coder for NOREC4DNA - Encode file updates into DNA"
    )
    parser.add_argument(
        "--ini",
        metavar="ini",
        type=str,
        help="Configuration file (INI format)",
        default="/home/michael/Code/DR4DNA/eval/sleeping_beauty_Mon_Feb_16_13_45_57_2026.ini",
    )
    parser.add_argument(
        "--new_file",
        metavar="new_file",
        type=str,
        help="Updated file path",
        required=True,
    )
    parser.add_argument(
        "--packet_add_limit",
        metavar="packet_add_limit",
        type=int,
        default=5,
        help="Maximum number of packets to add for each changed chunk",
    )
    parser.add_argument(
        "--output",
        metavar="output",
        type=str,
        help="Optional: Output base path (without extension). "
             "If not provided, output will be saved in the same directory as the INI file.",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for MultiVersionCoder CLI.

    This function parses command-line arguments and performs file update encoding.
    It properly handles input INI files that already contain multiple versions,
    creating diffs from the latest version and using the correct version number.

    Example:
        python -m MultiVersionCoder --ini pool.ini --new_file updated.bin
    """
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parsed_args = init_args()

    ini_file = parsed_args.ini
    new_file_path = parsed_args.new_file
    packet_add_limit = parsed_args.packet_add_limit
    output_base_path = parsed_args.output

    logger.info("=" * 80)
    logger.info("MultiVersionCoder CLI - DNA File Update Encoder")
    logger.info("=" * 80)
    logger.info(f"Configuration file: {ini_file}")
    logger.info(f"New file: {new_file_path}")
    logger.info(f"Packet limit per chunk: {packet_add_limit}")
    if output_base_path:
        logger.info(f"Output base path: {output_base_path}")
    logger.info("-" * 80)

    # Load configuration
    logger.info("Loading configuration...")
    cfg_worker = ConfigReadAndExecute(ini_file)
    x = cfg_worker.execute(return_decoder=True, skip_solve=True)[0]
    semiautomatic_solver = SemiAutomaticReconstructionToolkit(x)
    logger.info(f"✓ Configuration loaded from: {ini_file}")
    logger.info(f"  Decoder file: {semiautomatic_solver.decoder.file}")
    logger.info(f"  Number of chunks: {semiautomatic_solver.decoder.number_of_chunks}")

    # Read new file content
    if not Path(new_file_path).exists():
        logger.error(f"✗ New file not found: {new_file_path}")
        logger.error("Please check the file path and try again.")
        return

    with open(new_file_path, "rb") as f:
        new_file_content = f.read()

    logger.info(f"✓ New file loaded: {new_file_path}")
    logger.info(f"  File size: {len(new_file_content):,} bytes")

    # Get current (max) version from pool - handles existing multi-version pools
    current_version = get_current_file_version(semiautomatic_solver)
    new_file_version = current_version + 1

    logger.info("-" * 80)
    logger.info("Version Information:")
    logger.info(f"  Current pool version: {current_version}")
    logger.info(f"  Creating new version: {new_file_version}")

    if current_version > 0:
        logger.info(
            f"  Input pool already contains {current_version + 1} versions (0 to {current_version})"
        )
        logger.info(
            f"  Diff will be calculated from version {current_version} to version {new_file_version}"
        )
    else:
        logger.info(f"  Input pool contains only base version (version 0)")
        logger.info(f"  Creating first update (version 1)")
    logger.info("-" * 80)

    # CRITICAL: Decode all versions up to the latest one before calculating diff
    if current_version > 0:
        logger.info("=" * 80)
        logger.info(f"Pool contains {current_version + 1} versions (v0 to v{current_version})")
        logger.info(f"Decoding up to latest version (v{current_version}) to use as diff base...")
        logger.info("=" * 80)

        # Use MultiVersionDecoder to reconstruct the latest version
        from MultiVersionDecoder import MultiVersionDecoder

        mv_decoder = MultiVersionDecoder(semiautomatic_solver.decoder)

        # Decode base version (v0)
        logger.info("Decoding base version (v0)...")
        mv_decoder.decode_base_version("GAGCCAGTGAGTCGTA")
        #logger.info("✓ Base version decoded")

        # Decode each subsequent version
        for version in range(1, current_version + 1):
            #logger.info(f"Decoding version {version}/{current_version}...")
            try:
                decoded_version = mv_decoder.decode_to_version("GAGCCAGTGAGTCGTA", version)
                logger.info(f"✓ Version {version} decoded successfully")
            except Exception as e:
                logger.error(f"✗ Failed to decode version {version}: {e}")
                logger.error("Cannot proceed without reconstructing the latest version")
                raise e
                # return

        logger.info(f"✓ All versions decoded - latest version (v{current_version}) reconstructed")
        logger.info("=" * 80)

        # Update solver to use the decoded state
        semiautomatic_solver = SemiAutomaticReconstructionToolkit(mv_decoder.decoder)

    # Calculate diff from latest version to new version
    logger.info("Calculating diff from latest version to new file...")
    diff, changed_chunks = find_affected_chunks(semiautomatic_solver, new_file_content)
    logger.info(f"✓ Found {len(changed_chunks)} changed chunk(s)")

    if len(changed_chunks) == 0:
        logger.warning("⚠ No changes detected! The new file is identical to the latest version.")
        logger.warning("⚠ No packets will be generated.")
        return

    # Create encoder
    logger.info("Creating encoder...")
    encoder = encoder_from_decoder(semiautomatic_solver, cfg_worker, rules=FastDNARules())
    logger.info("✓ Encoder created")

    # Generate packets for changed content
    logger.info(f"Generating packets for {len(changed_chunks)} changed chunk(s)...")
    logger.info("Scanning seeds - this may take a while...")

    packet_candidates = generate_new_packets(
        semiautomatic_solver, encoder, diff, changed_chunks, new_file_version
    )

    logger.info(
        f"✓ Found candidate seeds per chunk: {dict((k, len(v)) for k, v in packet_candidates.items())}"
    )

    # Add best packets to encoder
    packets_added = 0
    added_packets = []
    packets_skipped = 0

    logger.info(f"Selecting best packets (max {packet_add_limit} per chunk)...")
    for changed_chunk_packet_group, packets in packet_candidates.items():
        added_packets_for_chunk = 0
        for packet in packets:
            if packet.error_prob >= 1.0:
                packets_skipped += 1
                continue
            if added_packets_for_chunk >= packet_add_limit:
                break

            encoder.encodedPackets.add(packet)
            added_packets.append(packet)
            added_packets_for_chunk += 1
            packets_added += 1

        if added_packets_for_chunk > 0:
            logger.info(
                f"  ✓ Chunk {changed_chunk_packet_group}: Added {added_packets_for_chunk} packet(s)"
            )
        else:
            logger.warning(f"  ⚠ Chunk {changed_chunk_packet_group}: No suitable packets found")

    logger.info("-" * 80)
    logger.info("Packet Generation Summary:")
    logger.info(f"  Packets added: {packets_added}")
    logger.info(f"  Packets skipped (high error): {packets_skipped}")
    logger.info(f"  Total packets in encoder: {len(encoder.encodedPackets)}")
    logger.info("-" * 80)

    # Determine output path
    if output_base_path:
        out_file = Path(output_base_path).absolute()
    else:
        # Get the directory of the INI file to use as output directory
        ini_dir = Path(ini_file).parent.absolute()
        base_filename = Path(semiautomatic_solver.decoder.file).stem  # filename without .fasta
        out_file = ini_dir / f"{base_filename}_v{new_file_version}"
        out_file = Path(out_file).absolute()

    logger.info(f"Output directory: {out_file.parent}")
    logger.info(f"Output base path: {out_file}*.fasta / *.ini")

    logger.info("=" * 80)
    logger.info("Saving output files...")
    logger.info("=" * 80)

    # Use the combine functionality
    logger.info("Combining existing FASTA with new packets...")
    coder = MultiVersionCoder(cfg_worker)
    coder.encoder = encoder

    existing_count, new_count, config_output = coder.combine_existing_fasta_with_packets(
        semiautomatic_solver.decoder.file,
        added_packets,
        out_file.with_suffix(".fasta"),
        out_file.with_suffix(".ini")
    )

    fasta_output = out_file.with_suffix(".fasta")

    logger.info("✓ Output files saved (combined):")
    logger.info(f"  FASTA file (combined): {fasta_output}")
    logger.info(f"  Config file: {config_output}")
    logger.info(f"  Existing sequences: {existing_count}")
    logger.info(f"  New sequences: {new_count}")
    #else:
    """
        # Save all packets (standard mode)
        file_bkp = encoder.file
        encoder.file = str(out_file)
        encoder.save_packets_fasta(str(out_file), "", False)
        encoder.out_file = out_file
        config_output = encoder.save_config_file(add_dot_fasta=True)
        encoder.file = file_bkp

        fasta_output = out_file.with_suffix(".fasta")

        logger.info("✓ Output files saved:")
        logger.info(f"  FASTA file (all versions): {fasta_output}")
        logger.info(f"  Config file: {config_output}")
    """
    # Save added packets to debug file - in same directory as output files
    debug_outfile = out_file.with_suffix(".added_packets.fasta")
    with open(debug_outfile, "w") as f:
        for packet in added_packets:
            f.write(f">{packet.id}\n")
            f.write(f"{packet.dna_data}\n")

    logger.info(f"  Sequences generated for this version: {debug_outfile}")

    # Get file sizes
    try:
        fasta_size = fasta_output.stat().st_size
        debug_size = debug_outfile.stat().st_size
        logger.info("-" * 80)
        logger.info("File Sizes:")
        logger.info(
            f"  FASTA (all versions): {fasta_size:,} bytes ({fasta_size / 1024 / 1024:.2f} MB)"
        )
        logger.info(f"  Added packets only: {debug_size:,} bytes")
    except Exception as e:
        logger.debug(f"Could not get file sizes: {e}")

    logger.info("=" * 80)
    logger.info(f"✓ Version {new_file_version} completed successfully!")
    logger.info("=" * 80)
    logger.info("Next steps:")
    logger.info(f"  1. Use the FASTA file for sequencing: {fasta_output}")
    logger.info(f"  2. To decode, use: python -m MultiVersionDecoder --ini {config_output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
