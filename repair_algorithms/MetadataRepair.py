import functools
import itertools
import logging
import string
import typing

import numpy
import numpy as np

from NOREC4DNA.norec4dna.HeaderChunk import HeaderChunk
from NOREC4DNA.norec4dna.helper import xor_numpy

from NOREC4DNA.metadata_coding import parse_metadata_file
from NOREC4DNA.norec4dna.GEPP import GEPP_intern, GEPP
from NOREC4DNA.norec4dna.helper.quaternary2Bin import tranlate_quat_to_byte
from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from googletrans import Translator
import language_tool_python
from collections import Counter
from Levenshtein import distance as levenshtein_distance
from repair_algorithms.PluginManager import PluginManager
from repair_algorithms.RandomShuffleRepair import RandomShuffleRepair

logger = logging.Logger(__name__)


class MetadataRepair(RandomShuffleRepair):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.possible_metadata_seqs: typing.Dict[str, bytes] = self.load_metadata_seqs()

        self.possible_metadata_seqs: typing.List[bytes] = self.load_metadata_seqs_as_bytes()
        self.known_filename = None
        print("tmp")

    def load_metadata_seqs_as_bytes(self, filename: str = "/home/michael/Code/DR4DNA/NOREC4DNA/wanted_meta.fasta") -> \
            typing.List[bytes]:
        parsed = []
        try:
            tmp = parse_metadata_file(filename)
            parsed = set([tranlate_quat_to_byte(x) for x in tmp])
        except Exception as ex:
            logger.error(ex)

        return parsed

    def find_metadata_rows(self, A_b_tuple: typing.Optional[typing.Tuple[numpy.ndarray, numpy.ndarray]] = None) -> \
            typing.List[typing.Tuple[int, bytes]]:
        if A_b_tuple is None:
            A: numpy.ndarray = self.semi_automatic_solver.initial_A.copy()
            b: numpy.ndarray = self.semi_automatic_solver.initial_b.copy()
        else:
            A, b = A_b_tuple
        matching_sequences: typing.List[typing.Tuple[int, bytes]] = []

        for i in range(A.shape[1]):
            if A[i, 0]:
                current_row = b[i]
                for metadata_seq in self.possible_metadata_seqs:
                    if np.array_equal(current_row[-len(metadata_seq):],
                                      np.frombuffer(metadata_seq, dtype=current_row.dtype)):
                        matching_sequences.append((i, metadata_seq))

        return matching_sequences

    def set_use_header(self, use_header):
        self.use_header_chunk = use_header

    def set_no_inspect_chunks(self, *args, **kwargs):
        try:
            self.no_inspect_chunks = int(kwargs["c_ctx"].triggered[0]["value"])
        except:
            print("Error: could not set number of chunks to inspect")
        return {"updates_b": False, "refresh_view": False}

    @staticmethod
    def filter_nonprintable(text):
        import itertools
        # Use characters of control category
        nonprintable = itertools.chain(range(0x00, 0x20), range(0x7f, 0xa0))
        # Use translate to remove all non - printable characters
        return text.translate({character: None for character in nonprintable})

    def find_error_region(self, language=None, *args, **kwargs):
        if language is None:
            language = self.lang
        return self.find_error_region_by_words(language, *args, **kwargs)

    def find_incorrect_rows(self, *args, **kwargs):
        if kwargs is None or kwargs.get("chunk_tag") is None:
            self.chunk_tag = np.zeros(self.gepp.b.shape[0], dtype=np.int32)
        else:
            self.chunk_tag = kwargs.get("chunk_tag")
        """
        max_count, max_col = 0, 0
        max_counter = None
        for i, counter in enumerate(counters):
            most_non_zero_column = np.argmin([x.get(0.0) for x in self.get_column_counter()])
            for diff, count in counter.most_common(4):  # 0.0, 0.1, 0.5 and the most common real error...
                if diff < 1.0:
                    continue
                else:
                    if count > max_count:
                        max_col = i
                        max_count = count
                        max_counter = counter
        """
        incorrect_columns = [x for x in self.find_incorrect_columns()]
        tmp = sorted(incorrect_columns, key=lambda x: x[2], reverse=True)
        column_tags = [x[2] for x in incorrect_columns]
        for i in range(len(self.error_matrix)):
            if tmp[0][1] != 0.0 and self.error_matrix[i, tmp[0][0]] == tmp[0][1]:
                self.chunk_tag[i] = 1
        return {"chunk_tag": self.chunk_tag, "updates_b": False, "refresh_view": True, "column_tag": column_tags}

    def find_correct_rows(self, *args, **kwargs):
        # all rows with no errors according to the spellchecker are treated as correct!
        if kwargs is None or kwargs.get("chunk_tag") is None:
            self.chunk_tag = np.zeros(self.gepp.b.shape[0], dtype=np.int32)
        else:
            self.chunk_tag = kwargs.get("chunk_tag")

        res = []
        if self.error_matrix is None:
            self.error_matrix = self.find_error_region()
        # for each row: count all entrys != 0
        for i in range(len(self.error_matrix)):
            res.append(np.sum(self.error_matrix[i, :]) == 0)
        for i in range(1 if self.use_header_chunk else 0, min(len(self.chunk_tag), len(res))):
            if res[i]:
                self.chunk_tag[i] = 2
        return {"chunk_tag": self.chunk_tag, "updates_b": False, "refresh_view": True}

    def find_incorrect_columns(self, *args, **kwargs):
        column_counters = self.get_column_counter()
        for i, counter in enumerate(column_counters):
            exists_gr_zero = False
            for diff, count in counter.most_common(4):
                if diff < 1.0:
                    continue
                else:
                    exists_gr_zero = True
                    yield i, diff, count, counter
                    break
            if not exists_gr_zero:
                yield i, 0.0, 0, counter

    def get_incorrect_columns(self, *args, **kwargs):
        incorrect_columns = self.find_incorrect_columns()
        column_tags = [x[2] for x in incorrect_columns]
        return {"column_tag": column_tags, "updates_b": False, "refresh_view": True}

    def get_column_counter(self, *args, **kwargs):
        if self.error_matrix is None or self.analyzed_row != self.no_inspect_chunks:
            self.error_matrix = self.find_error_region(*args, **kwargs)
            self.analyzed_row = self.no_inspect_chunks
        avg_errors = []
        row_counters = []
        for i in range(self.gepp.b.shape[1]):
            avg_errors.append(np.mean(self.error_matrix[:, i]))
        for i in range(self.gepp.b.shape[1]):
            # avg_error = np.mean(self.error_matrix[:, i])
            ctr = Counter(self.error_matrix[:, i])
            row_counters.append(ctr)
        #    for key, value in ctr.items():
        #        pass
        return row_counters

    def get_rows_with_headerchunk(self, A: typing.Optional[numpy.ndarray] = None) -> typing.FrozenSet[int]:
        if A is None:
            A: numpy.ndarray = self.semi_automatic_solver.initial_A
        return frozenset([i for i in range(A.shape[0]) if A[i, 0]])

    def find_equal_seed_rows(self, A: typing.Optional[numpy.ndarray] = None) -> typing.Set[typing.FrozenSet[int]]:
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

    def repair(self, *args, **kwargs):
        if self.chunk_tag is None or sum(self.chunk_tag) == 0:
            self.find_error_region(*args, **kwargs)
            self.find_incorrect_rows()
        # np.bitwise_xor(self.gepp.b[i,j], most_common_difference[i])
        tmp = [x for x in self.find_incorrect_columns()]
        if self.no_columns_to_repair is None or self.no_columns_to_repair == 0:
            incorrect_columns = sorted(tmp, key=lambda x: x[2], reverse=True)
        else:
            incorrect_columns = sorted(tmp, key=lambda x: x[2], reverse=True)[:min(len(tmp), self.no_columns_to_repair)]
        for i in range(len(self.error_matrix)):
            if self.error_matrix[i, incorrect_columns[0][0]] == incorrect_columns[0][1]:
                tmp_b_i = self.gepp.b[i].copy()
                tmp_b_i[incorrect_columns[0][0]] = np.bitwise_xor(tmp_b_i[incorrect_columns[0][0]],
                                                                  int(incorrect_columns[0][1]))
                return {"updates_b": True, "repair": {"corrected_row": i, "corrected_value": tmp_b_i},
                        "refresh_view": True, "chunk_tag": self.chunk_tag}

    def is_compatible(self, meta_info):
        # parse magic info string:
        return True
        return meta_info == "data" or "Unicode text" in meta_info

    def find_representative(self, equal_seed_rows, rows_with_headerchunk, rows_with_metadata) -> typing.Generator[
        typing.Tuple[int, typing.FrozenSet[int], bool], None, None]:
        """
        returns a representative for each set of equal seed rows. Only considers rows containing a header chunk.
        @returns:
        """
        plain_metadata_rows = set([x[0] for x in rows_with_metadata])
        for row_set in equal_seed_rows:
            row_set_contains_header = ()
            if len(row_set & rows_with_headerchunk) == 0:
                continue
            tmp = row_set - plain_metadata_rows
            if len(tmp) > 0:
                # we have at least one row without metadata, take the one with the lowest index,
                # this should be an intact packet without changes (unless we do not know all possible metadata-sequences)
                # TODO: in that case we may want to try out _every_ remaining differing row (in b)
                for row in sorted(tmp):
                    yield row, row_set, True
            else:
                yield min(row_set), row_set, False
        pass

    @staticmethod
    def solve_and_map(gepp: GEPP_intern) -> GEPP_intern:
        """
        solves the equation system and returns the full equation system including the actual header chunk if possible
        """
        res = gepp.solve()
        if not res:
            # TODO: we must check if the header chunk is solved (using the FIRST input row) and continue from there!
            logger.warning(
                "Equation system was not fully solved. Please make sure the header chunk is solved before using it to revert metadata-embedding")
            pass
        # ensure that the order is correct / comparable
        tmp_A = np.squeeze(gepp.A[gepp.result_mapping])
        tmp_b = np.squeeze(gepp.b[gepp.result_mapping])
        gepp.A = np.vstack((tmp_A, gepp.A[tmp_A.shape[0]:]))
        gepp.b = np.vstack((tmp_b, gepp.b[tmp_b.shape[0]:]))
        gepp.result_mapping = np.arange(gepp.b.shape[0])
        # store the solution
        return gepp

    def calculate_header_diff(self, raw_header_row) -> typing.Tuple[numpy.ndarray, bool]:
        """
        returns the diff between the actual and expected header (if the filename is known or included in the
        """
        header = HeaderChunk.from_raw_array(raw_header_row,
                                            last_chunk_len_format=self.semi_automatic_solver.last_chunk_len_format,
                                            checksum_len_format=self.semi_automatic_solver.checksum_len_format)
        logger.error(f"Got additional payload: {header.additional_payload}")
        if len(header.file_name) == 0:
            # TODO: fix the
            pass
        if len(header.file_name) > 0:
            if self.known_filename is None:
                self.known_filename = header.file_name
        len(header.additional_payload)
        header.update_header(filename=self.known_filename, checksum=header.checksum, additional_payload=b"")
        diff = xor_numpy(raw_header_row, header.data)
        return diff, len(header.file_name) > 0

        # return header.additional_payload

    def revert_metadata(self, *args, **kwargs):
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
        # TODO: implement!
        rows_with_headerchunk = self.get_rows_with_headerchunk(sorted_A)
        # reorder GEPP and put all rows in rows_with_headerchunk at the END of the GEPP matrix
        for current_row in sorted(rows_with_headerchunk, reverse=True):
            # move to end of GEPP:
            sorted_A = np.vstack([np.delete(sorted_A, current_row, axis=0), sorted_A[current_row]])
            sorted_b = np.vstack([np.delete(sorted_b, current_row, axis=0), sorted_b[current_row]])

        rows_with_headerchunk = self.get_rows_with_headerchunk(sorted_A)
        rows_with_metadata = self.find_metadata_rows((sorted_A, sorted_b))
        equal_seed_rows = self.find_equal_seed_rows(sorted_A)
        set_representatives = sorted(
            [x for x in self.find_representative(equal_seed_rows, rows_with_headerchunk, rows_with_metadata)],
            key=lambda x: x[2], reverse=True)

        # we only need to decode with one of each element in each group present. further, when decoding for a group,
        # a single representative of each other group should be present but put at the very end of the GEPP!
        no_fully_solved = set([x[0] for x in set_representatives])
        repeats = 0
        while len(no_fully_solved) > 0:
            if repeats > 10:
                logger.error("Got into an endless loop trying to solve metadata without the filename!")
                break
            repeats += 1
            for representative in no_fully_solved.copy():
                no_fully_solved.clear()
                # reorder current representative to the start of the GEPP to
                tmp_A = sorted_A.copy()
                tmp_b = sorted_b.copy()

                # swap first row and row _representative in tmp_A and tmp_b:
                tmp_A[[0, representative]] = tmp_A[[representative, 0]]
                tmp_b[[0, representative]] = tmp_b[[representative, 0]]

                tmp_gepp = self.solve_and_map(GEPP(tmp_A, tmp_b))
                print(tmp_gepp.b[0])
                diff, includes_filename = self.calculate_header_diff(tmp_gepp.b[0])
                # TODO: handle the case that includes_filename is False and we do not know the filename yet!
                if not includes_filename:
                    no_fully_solved.add(representative)
                # propagate diff to representative-row in sorted_b:
                if self.semi_automatic_solver.decoder.GEPP.chunk_to_used_packets[0][0]:
                    sorted_b[representative] = xor_numpy(sorted_b[representative], diff)
                else:
                    logger.warning(f"Packet {representative} was not used to decode header chunk even though it was set as first packet!")


        rows_to_keep = []
        # reduce work as packets with equal seed but
        for equal_content_group in equal_seed_rows:
            # delete every row in the group of equal packets (seed-wise)
            # if possible, delete all packets containing a metadata-sequence
            # if not possible, keep the packet with the "lowest" id (to avoid having to move later rows)
            # TODO: fix all code from this function below
            sorted_elements = sorted(equal_content_group, reverse=True)
            rows_to_keep.append(sorted_elements[0])
            to_delete = sorted_elements[1:]
            numpy.delete(sorted_A, to_delete, axis=0)
            numpy.delete(sorted_b, to_delete, axis=0)
            # for row in sorted(equal_content_group, reverse=True)[1:]:
            #    sorted_A.
        rows_with_metadata = self.find_metadata_rows((sorted_A, sorted_b))
        equal_seed_rows = self.find_equal_seed_rows()

        # self.semi_automatic_solver.decoder.GEPP
        # TODO: put all
        # TODO: shuffle all
        self.semi_automatic_solver.decoder.GEPP = GEPP(sorted_A, sorted_b)
        self.semi_automatic_solver.decoder.solve(True)
        return {"updates_b": True, "refresh_view": True}

    def get_ui_elements(self):
        return {"btn-metadata-repair": {"type": "button", "text": "Extract metadata", "callback": self.revert_metadata,
                                        "updates_b": True}}

    def set_no_columns_to_repair(self, *args, **kwargs):
        try:
            self.no_columns_to_repair = int(kwargs["c_ctx"].triggered[0]["value"])
        except:
            print("Error: could not set number of columns to repair")
        return {"updates_b": False, "refresh_view": False}

    def update_chunk_tag(self, chunk_tag):
        super().update_chunk_tag(chunk_tag)
        self.error_matrix = None  # this could be speed-up?!

    def update_gepp(self, gepp):
        # invalidate error matrix:
        self.error_matrix = None
        self.gepp = self.semi_automatic_solver.decoder.GEPP
        # trigger recalculating the error matrix:
        # self.find_error_region()


mgr = PluginManager()
mgr.register_plugin(MetadataRepair)
