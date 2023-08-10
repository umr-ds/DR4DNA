import base64
import io
import itertools
import json
import math
import typing
from collections import Counter

import numpy as np
from PIL import Image
from kaitaistruct import ValidationFailedError

import Kaitai2Html
from repair_algorithms.FileSpecificRepair import FileSpecificRepair

from repair_algorithms.PluginManager import PluginManager
from repair_algorithms.bmp import Bmp

nonprintable = itertools.chain(range(0x00, 0x20), range(0x7f, 0xa0))


class UploadRepair(FileSpecificRepair):
    # TODO: we might want to create and save __all__ possible results for a modified chunk
    # example: we change a byte in a (or multiple) chunk(s) and we want to decode assuming the error happening in all possible packets.
    # to further limit the number of packets we might aswell use the chunktags to pinpoint the corrupt packet!
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_repair_bytes = 2
        self.error_matrix = None
        self.file_bytes = None
        self.reconstructed_file_bytes = None
        self.load()

    def load(self):
        start = 1 if self.use_header_chunk else 0
        start_offset = start * self.semi_automatic_solver.decoder.GEPP.b.shape[1]
        self.semi_automatic_solver.parse_header("I")
        if self.semi_automatic_solver.headerChunk is not None:
            last_chunk_garbage = self.gepp.b.shape[1] - self.semi_automatic_solver.headerChunk.last_chunk_length
        else:
            last_chunk_garbage = 0
        # try to parse the zipfile:
        if last_chunk_garbage > 0:
            self.file_bytes = self.gepp.b[start:self.semi_automatic_solver.decoder.number_of_chunks].reshape(-1)[
                              :-last_chunk_garbage].tobytes()
        else:
            self.file_bytes = self.gepp.b[start:self.semi_automatic_solver.decoder.number_of_chunks].reshape(
                -1).tobytes()
        if self.reconstructed_file_bytes is None:
            self.reconstructed_file_bytes = bytearray(self.file_bytes)
        self.error_matrix = np.zeros((self.gepp.b.shape[0], self.gepp.b.shape[1]), dtype=np.float32)

    def set_use_header(self, use_header):
        self.use_header_chunk = use_header

    def repair(self, *args, **kwargs):
        # user has to tag error regions
        # and a single position (maybe multiple pixel within a chunk) with the corrected color.
        # sort the columns by the number of entries with the same value (use only the rows from the corrupt packet)::
        error_cols = sorted([x for x in self.find_incorrect_columns()], key=lambda x: x[2], reverse=True)
        # find the row that that contains the first _no_inspect_chunks_ errors
        repair_row = -1
        diff_lst = []
        # we could iterate only over the chunk_tag values since we know that they are the only one with known errors
        for row_num, row in enumerate(self.error_matrix):
            for col_no, diff, num, counter in error_cols[:self.num_repair_bytes]:
                if diff < 1.0:
                    # those are either unknown errors (0.5) or correct columns (0.0) or columns of unknown status (-1.0)
                    break
                if row[col_no] != diff:
                    repair_row = -1
                    diff_lst = []
                    break
                else:
                    repair_row = row_num
                    diff_lst.append((col_no, diff))
            if repair_row != -1:
                # we found a row that contains the first _no_inspect_chunks_ errors
                break
        if repair_row == -1:
            return {"info": f"Could not find a row that contains {self.num_repair_bytes} matching errors."}
        # XOR repair the repair_row with all _no_inspect_chunks_ diffs
        new_row_content = bytearray(self.gepp.b[repair_row])
        for col_no, diff in diff_lst:
            new_row_content[col_no] = np.bitwise_xor(new_row_content[col_no], int(diff))

        return {"update_b": True, "repair": {"corrected_row": repair_row, "corrected_value": new_row_content},
                "refresh_view": True, "chunk_tag": self.chunk_tag}

    def repair_multi(self, *args, **kwargs):
        # calculate which chunks to use for which packet in common_packets to repair,
        # then return this mapping to the caller
        row_to_repaired_content: typing.Dict[int, bytes] = {}
        # we could iterate only over the chunk_tag values since we know that they are the only one with known errors
        # error_cols = sorted([x for x in self.find_incorrect_columns()], key=lambda x: x[2], reverse=True)
        for row_num, row in enumerate(self.error_matrix):
            if sum(row) != 0:
                # there was a modification in this row, we should add the row + the changed content to our result.
                row_to_repaired_content[row_num] = bytearray([np.bitwise_xor(i,int(j)) for i,j in zip(self.gepp.b[row_num], self.error_matrix[row_num])])

        return {"update_b": False, "repair_variations": {"variations": row_to_repaired_content, "generate_all": False}, "refresh_view": False,
                "chunk_tag": self.chunk_tag}

    def is_compatible(self, meta_info):
        # upload (offline repair) is always possible...
        return True

    def get_ui_elements(self):
        return {"btn-file-download": {"type": "download", "text": "Download (parsable) original file", "callback": self.download},
                "upload-file": {"type": "upload", "text": "Upload file", "callback": self.upload_file},
                "btn-upload-file-find-incorrect-pos": {"type": "button", "text": "Find incorrect positions",
                                                       "callback": self.find_errors_tags},
                "btn-upload-file-find-columns": {"type": "button", "text": "Tag (in)correct columns",
                                                 "callback": self.get_incorrect_columns, "updates_b": False},
                "btn-upload-file-auto-repair": {"type": "button", "text": "Automatic Repair",
                                                "callback": self.repair, "updates_b": False},
                "btn-upload-file-multifile-repair": {"type": "button", "text": "Automatic Repair (multi-file)",
                                                     "callback": self.repair_multi, "updates_b": False},
                "txt-upload-file-num-repair-bytes": {"type": "int",
                                                     "text": "Number of bytes to repair (should be <= incorrect columns)",
                                                     "default": 2, "callback": self.update_num_repair,
                                                     "updates_b": False}
                }

    def update_num_repair(self, *args, **kwargs):
        num_repair_bytes = kwargs["c_ctx"].triggered[0]["value"]
        # we could check if kwargs["c_ctx"].triggered[X] has a prop_io equal to the textbox's id
        if num_repair_bytes < 1:
            self.num_repair_bytes = self.gepp.b[0]
        else:
            self.num_repair_bytes = num_repair_bytes

        return {"refresh_view": False, "update_b": False}

    def get_incorrect_columns(self, *args, **kwargs):
        incorrect_columns = self.find_incorrect_columns()
        column_tags = [x[2] for x in incorrect_columns]
        return {"column_tag": column_tags, "updates_b": False, "refresh_view": True}

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

    def get_column_counter(self, *args, **kwargs):
        if self.error_matrix is None:
            self.error_matrix = self.find_error_regions(*args, **kwargs)
        avg_errors = []
        row_counters = []
        for i in range(self.gepp.b.shape[1]):
            avg_errors.append(np.mean(self.error_matrix[:, i]))
        for i in range(self.gepp.b.shape[1]):
            ctr = Counter(self.error_matrix[:, i])
            row_counters.append(ctr)
        return row_counters

    def update_chunk_tag(self, chunk_tag):
        super().update_chunk_tag(chunk_tag)
        self.error_matrix = None  # this could be speed-up?!

    def find_error_regions(self, *args, **kwargs):
        # calculate error_matrix by looking at the difference between the original and the reconstructed image
        start_pos = (1 if self.use_header_chunk else 0) * self.gepp.b.shape[1]
        pos_correct = np.zeros(self.gepp.b.shape[0] * self.gepp.b.shape[1], dtype=np.float32)
        for i in range(0, len(self.file_bytes) - start_pos):
            diff = self.file_bytes[i] ^ self.reconstructed_file_bytes[i]
            if diff != 0:
                pos_correct[i + start_pos] = diff
        return pos_correct.reshape(-1, self.gepp.b.shape[1])

    def find_errors_tags(self, *args, **kwargs):
        if kwargs is None or kwargs.get("chunk_tag") is None:
            self.chunk_tag = np.zeros(self.gepp.b.shape[0], dtype=np.int32)
        else:
            self.chunk_tag = kwargs.get("chunk_tag")
        if self.error_matrix is None:
            self.error_matrix = self.find_error_regions(*args, **kwargs)
        # for each row: count all entries != 0
        for i in range(len(self.error_matrix)):
            chunk_res = np.amax(self.error_matrix[i, :])
            if chunk_res == 0:
                # 0 indicates that a single position is correct, however, if there are still locations with "-1",
                # then the row is still undecidable
                chunk_res = np.amin(self.error_matrix[i, :])
                if chunk_res == 0:
                    # if all positions are 0, then the row would be correct:
                    # chunk_res = 2
                    # BUT we do not want to tag positions as valid, since a user might want to perform a partial repair:
                    chunk_res = -1
            elif chunk_res > 0:
                # there is at least ONE error in the row
                chunk_res = 1
            else:
                # there is no information in the current row
                chunk_res = -1
            self.chunk_tag[i] = chunk_res
        return {"chunk_tag": self.chunk_tag, "update_b": False, "refresh_view": True}

    def download(self, *args, **kwargs):
        return {"update_b": False, "refresh_view": False, "download": bytes(self.reconstructed_file_bytes),
                "filename": "raw.bmp"}

    def update_gepp(self, gepp):
        # invalidate error matrix:
        self.gepp = gepp
        self.error_matrix = None
        self.load()
        # self.file_bytes = None
        # user has to refresh the canvas!

    def upload_file(self, content=None, *args, **kwargs):
        start_pos = (1 if self.use_header_chunk else 0) * self.gepp.b.shape[1]
        content = kwargs.get("c_ctx").triggered[0]["value"]
        if isinstance(content, list):
            content = content[0]
        if self.error_matrix is None:
            res = self.find_errors_tags(args, kwargs)
        if content is not None:
            try:
                content_type, content_string = content.split(',')
                new_error_part = np.array([a ^ b for a, b in
                                           zip(base64.b64decode(
                                               content_string),
                                               self.reconstructed_file_bytes)], dtype=self.error_matrix.dtype)
                self.error_matrix = self.error_matrix.reshape(-1)

                for i in range(0, new_error_part.shape[0]):
                    # only update positions with changes!
                    if new_error_part[i] != 0:
                        self.error_matrix[start_pos + i] = new_error_part[i]
                self.error_matrix = self.error_matrix.reshape(-1, self.gepp.b.shape[1])
                self.reconstructed_file_bytes = base64.b64decode(content_string)
                res = self.find_errors_tags()
                res["updates_canvas"] = False
                return res
            except Exception as ex:
                raise ex
        return {"chunk_tag": self.chunk_tag, "update_b": False, "refresh_view": True, "updates_canvas": False}


mgr = PluginManager()
mgr.register_plugin(UploadRepair)
