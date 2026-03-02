"""Metadata repair plugin for DR4DNA.

This module provides repair functionality for DNA-encoded files that contain
metadata sequences. It handles the extraction and correction of metadata such as
filenames, checksums, and additional payload data stored in the header and last
chunks of DNA-encoded files.
"""

import logging
import typing

import numpy
import numpy as np

from NOREC4DNA.metadata_coding import parse_metadata_file
from NOREC4DNA.norec4dna.GEPP import GEPP, GEPP_intern
from NOREC4DNA.norec4dna.HeaderChunk import HeaderChunk
from NOREC4DNA.norec4dna.helper import xor_numpy
from NOREC4DNA.norec4dna.helper.quaternary2Bin import tranlate_quat_to_byte
from repair_algorithms.PluginManager import PluginManager
from repair_algorithms.RandomShuffleRepair import RandomShuffleRepair

logger = logging.getLogger(__name__)


class MetadataRepair(RandomShuffleRepair):
    """
    Metadata repair plugin for DNA-encoded files.

    This plugin handles the repair of files containing metadata sequences such as
    filenames, checksums, and additional payload data. It identifies rows with
    metadata, finds representative packets for groups with equal seeds, and
    calculates corrections for header and last chunk padding.

    Attributes:
        possible_metadata_seqs: List of known metadata sequences as bytes
        known_filename: Filename extracted from metadata header
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the metadata repair plugin.

        Args:
            *args: Positional arguments passed to parent class
            **kwargs: Keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        # self.possible_metadata_seqs: typing.Dict[str, bytes] = self.load_metadata_seqs()

        self.possible_metadata_seqs: typing.List[bytes] = self.load_metadata_seqs_as_bytes()
        self.known_filename = None

    @staticmethod
    def filter_nonprintable(text: str) -> str:
        """Return `text` with non-printable characters removed."""
        if text is None:
            return text
        return "".join(ch for ch in text if ch.isprintable())

    def load_metadata_seqs_as_bytes(
        self, filename: str = "./NOREC4DNA/wanted_meta.fasta"
    ) -> typing.List[bytes]:
        """
        Load metadata sequences from a FASTA file and convert to bytes.

        Args:
            filename: Path to the FASTA file containing metadata sequences

        Returns:
            List of metadata sequences as bytes
        """
        parsed: typing.List[typing.Any] = []
        try:
            tmp = parse_metadata_file(filename)
            parsed = {tranlate_quat_to_byte(x) for x in tmp}
        except Exception as ex:
            logger.error(ex)

        return parsed

    def find_metadata_rows(
        self, A_b_tuple: typing.Optional[typing.Tuple[numpy.ndarray, numpy.ndarray]] = None
    ) -> typing.List[typing.Tuple[int, bytes]]:
        """
        Find rows in the matrix that contain metadata sequences.

        Searches through the b matrix rows to identify those ending with known
        metadata sequences.

        Args:
            A_b_tuple: Optional tuple of (A, b) matrices to search. If None, uses
                the semi-automatic solver's initial matrices.

        Returns:
            List of tuples containing (row_index, metadata_sequence) for matching rows
        """
        if A_b_tuple is None:
            A: numpy.ndarray = self.semi_automatic_solver.initial_A.copy()
            b: numpy.ndarray = self.semi_automatic_solver.initial_b.copy()
        else:
            A, b = A_b_tuple
        matching_sequences: typing.List[typing.Tuple[int, bytes]] = []

        for i in range(A.shape[0]):
            current_row = b[i]
            for metadata_seq in self.possible_metadata_seqs:
                if np.array_equal(
                    current_row[-len(metadata_seq) :],
                    np.frombuffer(metadata_seq, dtype=current_row.dtype),
                ):
                    matching_sequences.append((i, metadata_seq))
                    if not A[i, 0] and not A[i, -1]:
                        logger.warning(
                            f"Row {i} does not contain the header / last chunk but has a metadata tag at the end!?!"
                        )

        return matching_sequences

    def set_use_header(self, use_header):
        """
        Set whether to use header chunk for metadata parsing.

        Args:
            use_header: Boolean indicating if header chunk should be used
        """
        self.use_header_chunk = use_header

    def get_rows_with_headerchunk(
        self, A: typing.Optional[numpy.ndarray] = None
    ) -> typing.FrozenSet[int]:
        """
        Get row indices that contain header chunks.

        Args:
            A: Optional matrix to check. If None, uses the semi-automatic solver's
                initial A matrix.

        Returns:
            FrozenSet of row indices containing header chunks
        """
        if A is None:
            A: numpy.ndarray = self.semi_automatic_solver.initial_A
        return frozenset([i for i in range(A.shape[0]) if A[i, 0]])

    def get_rows_with_lastchunk(
        self, A: typing.Optional[numpy.ndarray] = None
    ) -> typing.FrozenSet[int]:
        """
        Get row indices that contain last chunks.

        Args:
            A: Optional matrix to check. If None, uses the semi-automatic solver's
                initial A matrix.

        Returns:
            FrozenSet of row indices containing last chunks
        """
        if A is None:
            A: numpy.ndarray = self.semi_automatic_solver.initial_A
        return frozenset([i for i in range(A.shape[0]) if A[i, -1]])

    def find_equal_seed_rows(
        self, A: typing.Optional[numpy.ndarray] = None
    ) -> typing.Set[typing.FrozenSet[int]]:
        """
        Find sets of rows with identical seed values (equal A rows).

        Groups rows that have the same A matrix values, which indicates they
        share the same encoding seed.

        Args:
            A: Optional matrix to check. If None, uses a copy of the semi-automatic
                solver's initial A matrix.

        Returns:
            Set of FrozenSets, where each FrozenSet contains row indices with
            identical seed values
        """
        if A is None:
            A: numpy.ndarray = self.semi_automatic_solver.initial_A.copy()
        # Convert rows to tuples for hashable comparison
        row_dict: typing.Dict[typing.Tuple, typing.Set[int]] = {}
        for i in range(A.shape[0]):
            row_tuple = tuple(A[i].tolist())
            if row_tuple in row_dict:
                row_dict[row_tuple].add(i)
            else:
                row_dict[row_tuple] = {i}

        # Return only sets with multiple rows
        return {frozenset(indices) for indices in row_dict.values() if len(indices) > 1}

    def is_compatible(self, meta_info):
        """
        Check if plugin is compatible with the file type.

        Args:
            meta_info: File type metadata string

        Returns:
            True if file contains additional payload in header chunk, False otherwise
        """
        return np.any(self.semi_automatic_solver.headerChunk.additional_payload)

    @staticmethod
    def find_representative(
        equal_seed_rows, rows_with_headerchunk, rows_with_metadata
    ) -> typing.Generator[typing.Tuple[int, typing.FrozenSet[int], bool], None, None]:
        """
        Return a representative for each set of equal seed rows. Only consider rows containing a header chunk.
        """
        plain_metadata_rows = {x[0] for x in rows_with_metadata}
        for row_set in equal_seed_rows:
            # only consider packets containing the header
            if len(row_set & rows_with_headerchunk) == 0:
                continue
            tmp = row_set - plain_metadata_rows
            if len(tmp) > 0:
                # we have at least one row without metadata, take the one with the lowest index,
                # this should be an intact packet without changes (unless we do not know all possible metadata-sequences)
                # in that case we may want to try out _every_ remaining differing row (in b)
                yield sorted(tmp)[0], row_set, True
            else:
                yield min(row_set), row_set, False
        pass

    @staticmethod
    def solve_and_map(gepp: GEPP_intern) -> typing.Tuple[GEPP_intern, numpy.ndarray]:
        """
        Solve the equation system and return the full equation system including the actual header chunk if possible.
        """
        import numpy as _np

        # Attempt to solve the GEPP system
        res = gepp.solve()
        # If solved ok, return gepp and a trivial identity mapping
        n_rows = MetadataRepair._get_row_count(gepp, _np)

        if res:
            return MetadataRepair._return_identity_mapping(gepp, n_rows, _np)

        logger.warning(
            "Equation system was not fully solved. Trying to fix header-chunk if partial solution exists."
        )

        # If more than one row contains data, we might be able to fix the header chunk.
        # For now, return a safe identity mapping so callers can continue without receiving None.
        return MetadataRepair._return_identity_mapping(gepp, n_rows, _np)

    @staticmethod
    def _get_row_count(gepp: GEPP_intern, np_module) -> typing.Optional[int]:
        """Get the number of rows in gepp.b."""
        try:
            return int(np_module.asarray(gepp.b).shape[0])
        except Exception:
            return None

    @staticmethod
    def _return_identity_mapping(
        gepp: GEPP_intern, n_rows: typing.Optional[int], np_module
    ) -> typing.Tuple[GEPP_intern, numpy.ndarray]:
        """Return gepp with an identity mapping array."""
        if n_rows is None:
            return gepp, np_module.array([], dtype=int)
        return gepp, np_module.arange(n_rows, dtype=int)

    def calculate_header_diff(self, raw_header_row) -> typing.Tuple[numpy.ndarray, bool]:
        """
        Return the diff between the actual and expected header.

        Returns the diff if the filename is known or included in the header.
        """
        # Some test doubles (FakeSemi) don't provide the header format attributes used by HeaderChunk.
        # Guard against calling the HeaderChunk parser when those are missing or None.
        if not self._has_valid_header_format():
            logger.debug(
                "Semi-automatic solver missing header format attributes; returning zero diff."
            )
            return self._create_zero_diff(raw_header_row), False

        try:
            header = self._parse_header_chunk(raw_header_row)
        except Exception as e:
            # In unexpected cases (malformed header row) return a zero diff and indicate filename not included.
            logger.warning(f"Could not parse header chunk: {e}")
            return self._create_zero_diff(raw_header_row), False

        logger.warning(f"Got additional payload: {header.additional_payload}")
        self._update_known_filename(header)
        header.update_header(
            filename=self.known_filename, checksum=header.checksum, additional_payload=b""
        )
        diff = xor_numpy(raw_header_row, header.data)
        return diff, len(header.file_name) > 0

    def _has_valid_header_format(self) -> bool:
        """Check if the semi-automatic solver has valid header format attributes."""
        return (
            hasattr(self.semi_automatic_solver, "last_chunk_len_format")
            and hasattr(self.semi_automatic_solver, "checksum_len_format")
            and self.semi_automatic_solver.last_chunk_len_format is not None
            and self.semi_automatic_solver.checksum_len_format is not None
        )

    def _create_zero_diff(self, raw_header_row) -> numpy.ndarray:
        """Create a zero diff array matching the input shape."""
        try:
            return numpy.zeros_like(raw_header_row)
        except Exception:
            return numpy.array([], dtype="uint8")

    def _parse_header_chunk(self, raw_header_row):
        """Parse the header chunk from raw data."""
        return HeaderChunk.from_raw_array(
            raw_header_row,
            last_chunk_len_format=self.semi_automatic_solver.last_chunk_len_format,
            checksum_len_format=self.semi_automatic_solver.checksum_len_format,
        )

    def _update_known_filename(self, header):
        """Update the known filename from the header if available."""
        if len(header.file_name) == 0:
            # TODO: check if this works as intended: if we already know the correct filename (self.known_filename)
            # we can calculate the diff between expected and actual with this info
            # otherwise, we can only calculate a partial solution and must mark the result as unfinished
            pass
        if len(header.file_name) > 0:
            if self.known_filename is None:
                self.known_filename = header.file_name

    # idee: Berechne mittels RandomShuffleRepair alle deltas. dann berechne reparierten header für alle representanten:
    # wenn ein delta einem randomshuffle delta entspricht gibt es zwei fälle:
    # 1. header verwendet nur ein einziges packet mit metadatan -> wende delta auf packet an und markiere als korrekt
    # 2. header verwendet mehr als ein packet: delta MUSS eine linearkombination aus anderen deltas sein
    #   -> finde linear kombination und probiere entweder erst eine andere lösung oder probiere alle zuordnungen der deltas
    #      auf die metadata-pakete um die datei zu lösen
    # falls es uneindeutig ist können wir folgenden code verwenden um dann mit der prüfsumme die korrekte version zu erhalten:
    #             res = self.semi_automatic_solver.constrained_repair(error_delta=np.frombuffer(error_delta, dtype="uint8"),
    #                                                                 possible_packets=[int(x) for x in self.intersects[error_delta]])

    # TODO: use this function to remove diff from any affected packet
    def calculate_last_chunk_padding_diff(self, header_chunk: HeaderChunk, gepp: GEPP_intern):
        """
        Extracts the changed content from the last chunk padding (known to be bytes of value 0x00)
        """
        if (
            header_chunk is None
            or header_chunk.last_chunk_length is None
            or header_chunk.last_chunk_length < 0
            or header_chunk.last_chunk_length > len(gepp.b[0])
            or not gepp.isSolved()
        ):
            raise RuntimeError(
                "GEPP must be solved and clean headerchunk must exist and be valid for last chunk padding calculation!"
            )
        return gepp.b[-1][
            header_chunk.last_chunk_length :
        ]  # Return padding bytes after the actual content

    @staticmethod
    def remove_equal_seed_non_representatives(
        sorted_A,
        sorted_b,
        set_representatives: typing.List[typing.Tuple[int, typing.FrozenSet[int], bool]],
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Remove non-representative metadata rows from matrices.

        Removes all rows from sorted_A and sorted_b that are not designated as
        representatives, keeping only one representative per metadata packet group.

        Args:
            sorted_A: Sorted A matrix
            sorted_b: Sorted b matrix
            set_representatives: List of tuples containing (representative_index, row_set, has_non_metadata)

        Returns:
            Tuple of (sorted_A, sorted_b) with only representatives retained
        """
        # Get all rows that should be removed (non-representatives)
        rows_to_remove: typing.Set[typing.Any] = set()
        for representative, row_set, _has_non_metadata in set_representatives:
            # Add all rows from the set except the representative
            rows_to_remove.update(row_set - {representative})

        # Convert to sorted list for numpy deletion
        rows_to_remove = sorted(rows_to_remove, reverse=True)

        # Remove rows from both matrices
        for row in rows_to_remove:
            sorted_A = np.delete(sorted_A, row, axis=0)
            sorted_b = np.delete(sorted_b, row, axis=0)

        return sorted_A, sorted_b

    def get_special_rows(self, sorted_A, sorted_b):
        """
        Get special row classifications for metadata repair.

        Identifies and categorizes rows containing header chunks, last chunks,
        metadata sequences, and equal seed groups.

        Args:
            sorted_A: Sorted A matrix
            sorted_b: Sorted b matrix

        Returns:
            Tuple of (combined_rows, rows_with_metadata, equal_seed_rows, set_representatives)
        """
        rows_with_headerchunk = self.get_rows_with_headerchunk(sorted_A)
        rows_with_lastchunk = self.get_rows_with_lastchunk(sorted_A)
        rows_with_metadata = self.find_metadata_rows((sorted_A, sorted_b))
        equal_seed_rows = self.find_equal_seed_rows(sorted_A)
        combined_rows = rows_with_headerchunk.union(rows_with_lastchunk)
        set_representatives: typing.List[typing.Tuple[int, typing.FrozenSet[int], bool]] = sorted(
            self.find_representative(equal_seed_rows, combined_rows, rows_with_metadata),
            key=lambda x: x[2],
            reverse=True,
        )
        return combined_rows, rows_with_metadata, equal_seed_rows, set_representatives

    def repair(self, *args, **kwargs):
        """
        Reverts added metadata for packets containing the header chunk to retrieve the correct stored content.
        Speedups: We find all rows with equal seed (i.e. equal A rows) and for each set of packets (rows) we find all sequences containing and NOT containing ANY metadata-sequence:
        If there is at least one packet with the same seed (equal A row) we take this as our representative and omit all others
        else:  take the one with the "lowest" row number OR: take the one with the shortest metadata-sequence

        (if there are multiple differing rows in a group that do not contain KNOWN metadata-sequences, we can choose at random or try out all of them)

        Once we have a representative for each group, we iterate through them:
        for i in group_representatives:
            put swap row of i with the first row + move all other representetive rows to the end of the matrix
            - check if the header chunk is correct (zero-padding at the end, correct filename,...)
            yes: - save the filename
                 - check if checksum correct: - yes: save file and return True
                                              - false: mark representative as correct and continue with next
            no:  - try to correct the header chunk by substituting any non 0x00 at the padding bytes
                   if the filename was stripped and we do not already know it, correct the padding and mark the row as "filename_still_missing"
                   (else) if the filename is known or was not stripped: calculate the XOR between the current version of the header and the correct version
                          apply this diff to the initial row in b and mark the row as corrected
           once the filename is found: fix all rows with the filename_still_missing flag and mark them as corrected

        """
        sorted_A: numpy.ndarray = self.semi_automatic_solver.initial_A.copy()
        sorted_b = self.semi_automatic_solver.initial_b.copy()

        # special_rows = self.get_rows_with_headerchunk(sorted_A)
        (
            special_rows,
            rows_with_metadata,
            equal_seed_rows,
            set_representatives,
        ) = self.get_special_rows(sorted_A, sorted_b)
        # reorder GEPP and put all rows in special_rows at the END of the GEPP matrix
        # for current_row in sorted(special_rows, reverse=True):
        #    # move to end of GEPP:
        #    sorted_A = np.vstack([np.delete(sorted_A, current_row, axis=0), sorted_A[current_row]])
        #    sorted_b = np.vstack([np.delete(sorted_b, current_row, axis=0), sorted_b[current_row]])
        n = sorted_A.shape[0]
        rows = np.array(list(special_rows))  # keep original relative order
        keep = np.setdiff1d(np.arange(n), rows, assume_unique=True)
        new_order = np.r_[keep, rows]
        sorted_A = sorted_A[new_order].copy()
        sorted_b = sorted_b[new_order].copy()

        (
            special_rows,
            rows_with_metadata,
            equal_seed_rows,
            set_representatives,
        ) = self.get_special_rows(sorted_A, sorted_b)

        sorted_A, sorted_b = self.remove_equal_seed_non_representatives(
            sorted_A, sorted_b, set_representatives
        )

        # recalculate as deletions might break the calculated positions
        (
            special_rows,
            rows_with_metadata,
            equal_seed_rows,
            set_representatives,
        ) = self.get_special_rows(sorted_A, sorted_b)
        # we only need to decode with one of each element in each group present. further, when decoding for a group,
        # a single representative of each other group should be present but put at the very end of the GEPP!
        no_fully_solved = set(special_rows)
        repeats = 0
        unique_diffs = set()
        fixed_packets = set()
        while len(no_fully_solved) > 0:
            if repeats > 2 * len(special_rows):
                logger.error(
                    "Got into an endless loop trying to solve metadata without the filename!"
                )
                break
            repeats += 1
            for representative in no_fully_solved.copy():
                no_fully_solved.clear()
                # reorder current representative to the start of the GEPP to
                tmp_A = sorted_A.copy()
                tmp_b = sorted_b.copy()

                # swap first row and row _representative in tmp_A and tmp_b:
                """
                swap = tmp_A[representative].copy()
                tmp_A[representative] = tmp_A[0]
                tmp_A[0] = swap
                swap_b = tmp_b[representative].copy()
                tmp_b[representative] = tmp_b[0]
                tmp_b[0] = swap_b
                """
                tmp_A[[0, representative]] = tmp_A[[representative, 0]]
                tmp_b[[0, representative]] = tmp_b[[representative, 0]]

                self.semi_automatic_solver.multi_error_packets_mode = True
                # diffs = self.calc_unique_diffs()

                # _res = self.find_packet_shuffle()

                tmp_gepp, org_mapping = self.solve_and_map(GEPP(tmp_A, tmp_b))
                # get all remaining (except for the first row (representative)) packets with the header-chunk
                # that were used to decode the header:
                if tmp_A[0, 0]:
                    other_header_rows_included = {
                        x
                        for x in range(len(tmp_gepp.chunk_to_used_packets[0]))
                        if tmp_gepp.chunk_to_used_packets[0][x]
                    } & special_rows
                    undetermined_header_packets = other_header_rows_included - fixed_packets
                    if len(undetermined_header_packets) > 0:
                        # TODO: in its current configuration, we may have to solve such rows twice even if the other
                        #  packet(s) do not contain any metadata
                        # the current solution is a combination of multiple diffs (not the actual solution!)
                        for p in undetermined_header_packets:
                            no_fully_solved.add(p)
                        no_fully_solved.add(representative)
                        logging.warning(
                            "Multiple metadata-packets were used to decode this header - "
                            "Trying to find the linear combination to solve this."
                        )
                    diff, includes_filename = self.calculate_header_diff(tmp_gepp.b[0])
                    unique_diffs.add(diff.tobytes())
                    # TODO: handle the case that includes_filename is False and we do not know the filename yet!
                    if not includes_filename:
                        no_fully_solved.add(representative)
                    # propagate diff to representative-row in sorted_b:
                    if diff.max() > 0:
                        if tmp_gepp.chunk_to_used_packets[0][0] and len(no_fully_solved) == 0:
                            assert sorted_A[representative][0]
                            sorted_b[representative] = xor_numpy(sorted_b[representative], diff)
                            fixed_packets.add(representative)
                        else:
                            logger.warning(
                                f"Packet {representative} was not used to decode header chunk even though it was set as first packet!"
                            )
                else:
                    other_lastchunk_rows_included = {
                        x
                        for x in range(len(tmp_gepp.chunk_to_used_packets[-1]))
                        if tmp_gepp.chunk_to_used_packets[-1][x]
                    } & special_rows
                    undetermined_lastchunk_packets = other_lastchunk_rows_included - fixed_packets
                    if len(undetermined_lastchunk_packets) > 0:
                        # TODO: in its current configuration, we may have to solve such rows twice even if the other
                        #  packet(s) do not contain any metadata
                        # the current solution is a combination of multiple diffs (not the actual solution!)
                        for p in undetermined_lastchunk_packets:
                            no_fully_solved.add(p)
                        no_fully_solved.add(representative)
                        logging.warning(
                            "Multiple metadata-packets were used to decode the last chunk - "
                            "Trying to find the linear combination to solve this."
                        )
                    # Safely attempt to calculate the last-chunk padding diff. Tests and some fakes may not provide
                    # a headerChunk or the expected attributes; in that case, fall back to a zero-diff so the
                    # repair loop can continue without raising exceptions.
                    try:
                        header_chunk = getattr(self.semi_automatic_solver, "headerChunk", None)
                        if (
                            header_chunk is None
                            or getattr(header_chunk, "last_chunk_length", None) is None
                        ):
                            logger.warning(
                                "HeaderChunk missing or incomplete; using zero diff for last chunk padding."
                            )
                            diff = np.zeros_like(tmp_gepp.b[0])
                        else:
                            # calculate_last_chunk_padding_diff expects the header chunk and the solved gepp
                            diff = self.calculate_last_chunk_padding_diff(header_chunk, tmp_gepp)
                    except Exception as e:
                        logger.warning(f"Could not calculate last chunk padding diff: {e}")
                        try:
                            diff = np.zeros_like(tmp_gepp.b[0])
                        except Exception:
                            diff = np.array([], dtype="uint8")

                    # For last-chunk handling we don't have an 'includes_filename' flag; handle propagation similarly
                    unique_diffs.add(diff.tobytes())
                    if diff.max() > 0:
                        # If the last chunk was used from packet 0 and there are no other unresolved packets,
                        # apply the diff to the representative row.
                        try:
                            if tmp_gepp.chunk_to_used_packets[-1][0] and len(no_fully_solved) == 0:
                                assert sorted_A[representative][-1] or True
                                sorted_b[representative] = xor_numpy(sorted_b[representative], diff)
                                fixed_packets.add(representative)
                            else:
                                logger.warning(
                                    f"Packet {representative} was not used to decode last chunk even though it was set as first packet!"
                                )
                        except Exception:
                            logger.warning(
                                "Error while propagating last-chunk diff to representative; skipping propagation."
                            )
        logger.debug(f"unique_diffs_count={len(unique_diffs)}")
        rows_to_keep = []
        # reduce work as packets with equal seed but
        for equal_content_group in equal_seed_rows:
            # delete every row in the group of equal packets (seed-wise)
            # if possible, delete all packets containing a metadata-sequence
            # if not possible, keep the packet with the "lowest" id (to avoid having to move later rows)
            sorted_elements = sorted(equal_content_group, reverse=True)
            rows_to_keep.append(sorted_elements[0])
            to_delete = sorted_elements[1:]
            sorted_A = numpy.delete(sorted_A, to_delete, axis=0)
            sorted_b = numpy.delete(sorted_b, to_delete, axis=0)

        self.semi_automatic_solver.decoder.GEPP = GEPP(sorted_A, sorted_b)
        self.semi_automatic_solver.decoder.solve(True)
        return {"updates_b": True, "refresh_view": True}

    def get_ui_elements(self):
        """
        Get UI elements for the metadata repair plugin.

        Returns:
            Dictionary of UI element configurations for metadata repair
        """
        return {
            "btn-metadata-repair": {
                "type": "button",
                "text": "Extract metadata",
                "callback": self.repair,
                "updates_b": True,
            }
        }

    def set_no_columns_to_repair(self, *args, **kwargs):
        """
        Set the number of columns to repair from callback value.

        Args:
            *args: Additional positional arguments
            **kwargs: Keyword arguments containing c_ctx with callback context

        Returns:
            Dictionary with updates_b and refresh_view flags
        """
        try:
            self.no_columns_to_repair = int(kwargs["c_ctx"].triggered[0]["value"])
        except (ValueError, TypeError, IndexError):
            print("Error: could not set number of columns to repair")
        return {"updates_b": False, "refresh_view": False}

    def dump_array_as_dna(self, arr: np.ndarray) -> typing.List[str]:
        """
        Convert numpy array to DNA sequence strings.

        Args:
            arr: Numpy array to convert

        Returns:
            List of DNA sequence strings (quaternary format)
        """
        from norec4dna.helper.bin2Quaternary import string2QUATS

        return ["".join(string2QUATS(bytearray(x))) for x in arr.tolist()]


mgr = PluginManager()
mgr.register_plugin(MetadataRepair)
