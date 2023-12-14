# https://ide.kaitai.io/
# https://kaitai.io/
import copy
# pylint: disable=invalid-name,missing-docstring,too-many-public-methods
# pylint: disable=useless-object-inheritance,super-with-arguments,consider-using-f-string
# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
import io
import lzma
import re
import traceback
import typing
import zlib
from typing import Union
import struct

import numpy as np
from kaitaistruct import ValidationNotEqualError, ValidationNotAnyOfError, ValidationFailedError, KaitaiStream

import Kaitai2Html
from CustomExceptions import CustomOutOfBoundsException, InvalidDataException
from repair_algorithms.PluginManager import PluginManager
from repair_algorithms.FileSpecificRepair import FileSpecificRepair

import norec4dna.GEPP

from repair_algorithms.zip import Zip


def inflate(data):
    decompress = zlib.decompressobj(
        -zlib.MAX_WBITS  # see above
    )
    inflated = decompress.decompress(data)
    inflated += decompress.flush()
    return inflated


class ZipFileRepair(FileSpecificRepair):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_structure = None
        self.error_matrix = None
        self.reconstructed_bmp_bytes: typing.Optional[bytearray] = None
        self.parser_error_matrix = None
        self.reconstructed_zip_bytes = None

    def bitwise_hamming_distance(self, a, b):
        r = (1 << np.arange(8))[:, None]
        return np.count_nonzero((np.bitwise_xor(a, b) & r) != 0)

    def parse_zipfile(self, iterations=50):
        start = 1 if self.use_header_chunk else 0
        error_pos = [-1 for _ in
                     range(len(self.gepp.b[start:].reshape(-1)))]  # -1 <= unknown, 0 == correct, >=1 = incorrect
        last_chunk_garbage = self.gepp.b.shape[1] - self.semi_automatic_solver.headerChunk.last_chunk_length
        # try to parse the zipfile:
        if last_chunk_garbage > 0:
            zip_bytes = self.gepp.b[start:self.semi_automatic_solver.decoder.number_of_chunks].reshape(-1)[
                        :-last_chunk_garbage].tobytes()
        else:
            zip_bytes = self.gepp.b[start:self.semi_automatic_solver.decoder.number_of_chunks].reshape(-1).tostring()
        iters = 0
        while iters < iterations:
            iters += 1
            try:
                res = Zip.from_bytes(zip_bytes)
                if self.reconstructed_zip_bytes is not None:
                    self.reconstructed_zip_bytes = bytearray(copy.copy(zip_bytes))
                return res, error_pos  # np.array(error_pos).reshape(-1, self.gepp.b.shape[1])
            except ValidationFailedError as err:
                tb = traceback.format_exc()
                # if magic bytes are wrong:
                if err.src_path == "/types/pk_section/seq/0":
                    # replace magic bytes with correct ones:
                    zip_bytes = zip_bytes[:err.io.pos() - len(err.expected)] + err.expected + zip_bytes[err.io.pos():]
                    for i, diff in enumerate([a - b for a, b in zip(err.expected, err.actual)]):
                        error_pos[
                            (start * self.semi_automatic_solver.decoder.GEPP.b.shape[1]) + err.io.pos() - len(
                                err.expected) + i] = 1 if diff != 0 else 0
                elif err.src_path == "/types/pk_section/seq/1":
                    distance = {}
                    for expected in [513, 1027, 1541, 2055]:
                        distance[expected] = self.bitwise_hamming_distance(expected, err.actual)
                    expected = struct.pack("<H", sorted([(x) for x in distance.items()], key=lambda x: x[1])[0][0])
                    # replace src bytes with correct ones:
                    zip_bytes = zip_bytes[:err.io.pos() - len(expected)] + expected + zip_bytes[err.io.pos():]
                    for i, diff in enumerate([a - b for a, b in zip(expected, struct.pack("<H", err.actual))]):
                        error_pos[
                            (start * self.semi_automatic_solver.decoder.GEPP.b.shape[0]) + err.io.pos() - len(
                                expected) + i] = 1 if diff != 0 else 0
                    # a different approach would be to try to parse the part with all possible solutions and check for
                    # consistency...
                elif err.src_path == "/types/time/seq/1" or err.src_path == "/types/time/seq/2":
                    expected = struct.pack("<B", 0)
                    zip_bytes = zip_bytes[:err.io.pos()] + expected + zip_bytes[err.io.pos():]
                    for i in range(1):
                        error_pos[
                            (start * self.semi_automatic_solver.decoder.GEPP.b.shape[0]) + err.io.pos() - len(
                                expected) + i] = 1
                elif err.src_path == "/types/date/seq/0":
                    expected = struct.pack("<B", 1)
                    zip_bytes = zip_bytes[:err.io.pos() - len(expected)] + expected + zip_bytes[err.io.pos():]
                    for i in range(1):
                        error_pos[
                            (start * self.semi_automatic_solver.decoder.GEPP.b.shape[0]) + err.io.pos() - len(
                                expected) + i] = 1
                elif err.src_path == "/types/date/seq/1":
                    distance = {}
                    for expected in range(1, 13, 1):
                        distance[expected] = self.bitwise_hamming_distance(expected, err.actual)
                    expected = struct.pack("<B", sorted([(x) for x in distance.items()], key=lambda x: x[1])[0][0])
                    # replace src bytes with correct ones:
                    zip_bytes = zip_bytes[:err.io.pos() - len(expected)] + expected + zip_bytes[err.io.pos():]
                    for i, diff in enumerate([a - b for a, b in zip(expected, struct.pack("<B", err.actual))]):
                        error_pos[
                            (start * self.semi_automatic_solver.decoder.GEPP.b.shape[0]) + err.io.pos() - len(
                                expected) + i] = 1 if diff != 0 else 0
                else:
                    raise err
            except CustomOutOfBoundsException as err:
                tb = traceback.format_exc()
                raise err
            except InvalidDataException as err:
                tb = traceback.format_exc()
                if err.args[0] in ["/types/filename/invalid", "/types/comment/invalid", "/types/extra/invalid",
                                   "/types/len_extra/notmultiple4", "/types/raw_extra/invalid",
                                   "/types/raw_body/invalid", "/types/comment_len/invalid", "/types/comment/invalid"]:
                    # update error matrix at position error_pos
                    expected = struct.pack("<H", err.expected)
                    zip_bytes = zip_bytes[:err.error_pos] + expected + zip_bytes[err.error_pos:]
                    for i in range(2):
                        error_pos[
                            (start * self.semi_automatic_solver.decoder.GEPP.b.shape[0]) + err.error_pos - len(
                                expected) + i] = 1
                else:
                    raise err
            except Exception as err:
                tb = traceback.format_exc()
                raise err
        return self.sweep_zip_header()

    def is_compatible(self, meta_info):
        # parse magic info string:
        return "zip" in meta_info.lower()

    def repair(self, *args, **kwargs):
        if self.zip_structure is None or self.parser_error_matrix is None:
            self.zip_structure, self.parser_error_matrix = self.parse_zipfile()
            self.error_matrix = np.array(self.parser_error_matrix).reshape(-1, self.gepp.b.shape[1])
        start = 1 if self.use_header_chunk else 0
        last_chunk_garbage = self.gepp.b.shape[1] - self.semi_automatic_solver.headerChunk.last_chunk_length

        diff = self.semi_automatic_solver.decoder.GEPP.b[start:self.semi_automatic_solver.decoder.number_of_chunks,
               :].reshape(-1)[:-last_chunk_garbage] - self.reconstructed_zip_bytes
        diff_matrix = np.zeros(self.semi_automatic_solver.decoder.GEPP.b.shape, dtype=np.uint8).reshape(-1)
        diff_matrix[start * self.semi_automatic_solver.decoder.GEPP.b.shape[1]:start *
                                                                               self.semi_automatic_solver.decoder.GEPP.b.shape[
                                                                                   1] + len(diff)] = diff
        diff_matrix = diff_matrix.reshape(self.semi_automatic_solver.decoder.GEPP.b.shape)

        invalid_rows = [_i for _i, _x in enumerate(self.chunk_tag) if _x == 1]
        valid_rows = [_i for _i, _x in enumerate(self.chunk_tag) if _x == 2]
        lst = []
        repaired_content = None
        for invalid_row in invalid_rows:
            tmp_invalid_rows = invalid_rows.copy()
            tmp_valid_rows = valid_rows.copy()
            known_false_columns = np.where(diff_matrix[invalid_row] != 0)[0]
            if len(known_false_columns) > 0:
                repaired_content = bytearray(
                    [np.bitwise_xor(i, int(j)) for i, j in zip(self.gepp.b[invalid_row], diff_matrix[invalid_row])])
            for column in known_false_columns:
                # this is more fine-grained since the parser might tag multiple bytes as invalid (e.g. a 2-byte integer)
                for row in range(self.semi_automatic_solver.decoder.GEPP.b.shape[0]):
                    if self.error_matrix[row, column] == 1:
                        tmp_invalid_rows.append(row)
                    elif self.error_matrix[row, column] == 0:
                        tmp_valid_rows.append(row)
                tmp_common_packets = self.semi_automatic_solver.decoder.GEPP.get_common_packets(tmp_invalid_rows,
                                                                                                tmp_valid_rows,
                                                                                                self.semi_automatic_solver.multi_error_packets_mode)
            if repaired_content is not None:
                lst.append((np.where(tmp_common_packets == True)[0], invalid_row, repaired_content))
        return {"update_b": False, "repair_for_each_packet": {"repair_list": lst, "generate_all": False,
                                                              "correctness_function": self.correctness_function},
                "refresh_view": False, "chunk_tag": self.chunk_tag}

    def correctness_function(self, repaired_content, *args, **kwargs):
        # return True if parser found NO errors,
        # return False if parser found errors
        tmpfilerepair = ZipFileRepair(self.semi_automatic_solver, chunk_tag=self.chunk_tag)
        tmpfilerepair.on_load()
        org_b = copy.deepcopy(tmpfilerepair.gepp.b)
        tmpfilerepair.gepp.b = repaired_content
        tmpfilerepair.parse_zipfile()
        if np.any(np.any(tmpfilerepair.error_matrix == 1)):
            res = False
        else:
            res = True
        tmpfilerepair.gepp.b = org_b
        return res

    def get_raw_bytes(self, start, num_bytes):
        header = 1 if self.use_header_chunk else 0
        offset = header * self.gepp.b.shape[1]
        return self.gepp.b.reshape(-1)[start + offset:start + offset + num_bytes]

    def find_error_region(self, *args, **kwargs):
        start = 1 if self.use_header_chunk else 0
        error_pos = [-1 for _ in
                     range(len(self.gepp.b[start:].reshape(-1)))]  # -1 <= unknown, 0 == correct, >=1 = incorrect

        def update_error_pos(_start, _end, new_error_pos=None, corrected_bytes=None, overwrite=False):
            offset = start * self.gepp.b.shape[1]
            _parser_error_pos = self.parser_error_matrix[offset + _start: offset + _end]
            if new_error_pos is None:
                new_error_pos = _parser_error_pos
            error_pos[_start + offset: _end + offset] = [new_error_pos[pos] if (i == -1 or overwrite) else i for pos, i
                                                         in
                                                         enumerate(_parser_error_pos)]
            if corrected_bytes is not None and _end - _start == len(corrected_bytes):
                for i, j in enumerate(range(_start, _end)):
                    self.reconstructed_zip_bytes[j] = corrected_bytes[i]
            return 1 if len(np.array(new_error_pos)[np.where(np.array(new_error_pos) > 0)]) > 0 else 0

        if self.zip_structure is None:
            self.zip_structure, self.parser_error_matrix = self.parse_zipfile()
        for section in self.zip_structure.sections:
            error_counter = 0
            error_pos_bkp = copy.deepcopy(error_pos)
            reconstructed_zip_bytes_bkp = copy.deepcopy(self.reconstructed_zip_bytes)
            if isinstance(section.body, Zip.LocalFile):
                error_counter += update_error_pos(section.body.start - 4, section.body.start,
                                                  [0] * 4)  # PK header will be correct
                if section.body.header.compression_method == Zip.Compression.deflated:
                    try:
                        data = inflate(section.body.body)
                    except zlib.error:
                        data = None
                elif section.body.header.compression_method == Zip.Compression.lzma:
                    try:
                        data = lzma.decompress(section.body.body)
                    except lzma.LZMAError:
                        data = None
                elif section.body.header.compression_method == Zip.Compression.none:
                    data = section.body.body
                else:
                    try:
                        comp = Zip.Compression(section.body.header.compression_method)
                        raise NotImplementedError("Only deflate, lzma and no compression are currently supported")
                    except ValueError:
                        # no valid compression was used, this is an error:
                        error_counter += update_error_pos(section.body.header.start + 8,
                                                          section.body.header.start + 8 + 2,
                                                          [1] * 2)

                if data is not None:
                    actual_crc32 = zlib.crc32(data)
                else:
                    actual_crc32 = None
                if section.body.header.crc32 == actual_crc32:
                    if (
                            len(data) == section.body.header.len_body_uncompressed or section.body.header.len_body_uncompressed == 0xffffffff):
                        # uncompressed size:
                        error_counter += update_error_pos(section.body.header.start + 18,
                                                          section.body.header.start + 18 + 4, [0] * 4)
                    else:
                        error_counter += update_error_pos(section.body.header.start + 18,
                                                          section.body.header.start + 18 + 4, [1] * 4)
                        # we might want to use "struct.pack("<I", len(data)))" but it might be 0xfff...
                    # in chunk_tag: tag content, crc AND header.compression_method as correct
                    # version field:
                    # error_counter += update_error_pos(section.start, section.start + 4, [0] * 4)
                    # compression method:
                    error_counter += update_error_pos(section.body.header.start + 4, section.body.header.start + 4 + 2,
                                                      [0] * 2)
                    # crc32:
                    error_counter += update_error_pos(section.body.header.start + 10,
                                                      section.body.header.start + 10 + 4, [0] * 4)
                    # body:
                    error_counter += update_error_pos(section.body.header.end,
                                                      section.body.header.end + section.body.header.len_body_compressed,
                                                      [0] * section.body.header.len_body_compressed)
                else:
                    if data is not None and len(data) == section.body.header.len_body_uncompressed:
                        # content is corrupt, but uncompressed size is correct
                        error_counter += update_error_pos(section.body.header.start + 10,
                                                          section.body.header.start + 10 + 4, [0] * 4)
                        # body:
                        error_counter += update_error_pos(section.body.header.end,
                                                          section.body.header.end + section.body.header.len_body_compressed,
                                                          [1] * section.body.header.len_body_compressed)
                    else:
                        if data is not None:
                            # content is most likely correct, but uncompressed size is wrong
                            error_counter += update_error_pos(section.body.header.start + 10,
                                                              section.body.header.start + 10 + 4,
                                                              [1] * 4)
                            # body:
                            error_counter += update_error_pos(section.body.header.end,
                                                              section.body.header.end + section.body.header.len_body_compressed,
                                                              [1] * section.body.header.len_body_compressed)
                            # crc is most likely wrong
                            error_counter += update_error_pos(section.body.header.start + 10,
                                                              section.body.header.start + 10 + 4,
                                                              [1] * 4)
                # version needed to extract:
                # error_counter += update_error_pos(section.body.header.start, section.body.header.start + 2, [0] * 2)
                # general purpose bit flag:
                if section.body.header.flags.reserved_1 != 0 or section.body.header.flags.reserved_2 != 0 or \
                        section.body.header.flags.reserved_3 != 0 or section.body.header.flags.reserved_4 != 0:
                    # we might need to check if offset is correct for all reserved fields
                    error_counter += update_error_pos(section.body.header.start + 2 + 1,
                                                      section.body.header.start + 2 + 2, [1])
                # if section.body.header.general_purpose_bit_flag & 0b00000011101011 == 0:
                #    error_counter += update_error_pos(section.body.header.start + 2, section.body.header.start + 4, [0] * 4)
                # else:
                #    error_counter += update_error_pos(section.body.header.start + 2, section.body.header.start + 4, [1] * 4)

                # TODO: make some basic sanity check that these numbers are not too large:
                # these are currently inactive as we have to make sure that all zip implementations correctly use these:
                # file last modification time:
                # error_pos[section.header.start + 10:section.header.start + 10 + 2] = [0] * 2
                # file last modification date:
                # error_pos[section.header.start + 12:section.header.start + 12 + 2] = [0] * 2
                # parser_error_pos = self.parser_error_matrix[
                #                   section.body.header.start + 14: section.body.header.start + 14 + 4]
                # error_pos[section.body.header.start + 14:section.body.header.start + 14 + 4] = [0 for i in
                #                                                                                parser_error_pos
                #                                                                                if i == -1]
                # compressed size:
                if section.body.header.len_body_compressed == len(
                        section.body.body) or section.body.header.len_body_compressed == 0xffffffff:
                    error_counter += update_error_pos(section.body.header.start + 14,
                                                      section.body.header.start + 14 + 4, [0] * 4)
                else:
                    error_counter += update_error_pos(section.body.header.start + 14,
                                                      section.body.header.start + 14 + 4, [1] * 4)
                # filename:
                if len(section.body.header.file_name) < section.body.header.len_file_name:
                    # non unicode characters were removed from filename...
                    error_counter += update_error_pos(section.body.header.start + 26,
                                                      section.body.header.start + 26 + section.body.header.len_file_name,
                                                      [1] * section.body.header.len_file_name)
                else:
                    error_counter += update_error_pos(section.body.header.start + 26,
                                                      section.body.header.start + 26 + section.body.header.len_file_name,
                                                      [0] * section.body.header.len_file_name)
                # filename len
                next_header_signature = self.get_raw_bytes(
                    section.body.header.start + 24 + 2 + section.body.header.len_file_name +
                    section.body.header.len_extra + section.body.header.len_body_compressed, 2)
                if b"".join(next_header_signature) == b'PK':
                    # length of filename and extra field is correct
                    error_counter += update_error_pos(section.body.header.start + 22,
                                                      section.body.header.start + 22 + 2 + 2, [0] * 4,
                                                      overwrite=True)
                else:
                    pass
                    # the "PK" signature might be wrong...
                    # error_counter += update_error_pos(section.body.header.start + 22, section.body.header.start + 22 + 2 + 2, [1] * 4,
                    #                 overwrite=True)
                # extra len field:
                if section.body.header.len_extra == 0:
                    error_counter += update_error_pos(section.body.header.start + 24,
                                                      section.body.header.start + 24 + 2, [0] * 2)

                # if not, the filename or extra field are wrong...

                # parser_error_pos = self.parser_error_matrix[
                #                   section.body.header.start + 18: section.body.header.start + 18 + 4]
                # error_pos[section.body.header.start + 18:section.body.header.start + 18 + 4] = [0 for i in
                #                                                                                parser_error_pos
                #                                                                                if i == -1]
                # parser_error_pos = self.parser_error_matrix[
                #                   section.body.header.end:section.body.header.end + section.body.header.len_body_compressed]
                # error_pos[
                # section.body.header.end:section.body.header.end + section.body.header.len_body_compressed] = [0 for
                #                                                                                              i in
                #                                                                                              parser_error_pos
                #                                                                                              if
                #                                                                                              i == -1]
            elif isinstance(section.body, Zip.CentralDirEntry):
                # check if ofs_local_header points to a local file:
                for local_file in self.zip_structure.sections:
                    if isinstance(local_file.body, Zip.LocalFile) and local_file.start == section.body.ofs_local_header:
                        # in chunk_tag: tag content, crc AND header.compression_method as correct
                        error_pos[section.body.start + 42: section.body.start + 42 + 4] = [0] * 4
                        # general puropse bit flag:

            elif isinstance(section.body, Zip.EndOfCentralDir):
                if section.body.disk_of_end_of_central_dir != 0:
                    error_counter += update_error_pos(section.body.start, section.body.start + 2, [1] * 2, [0] * 2)
                else:
                    error_counter += update_error_pos(section.body.start, section.body.start + 2, [0] * 2)
                if section.body.disk_of_central_dir != 0:
                    error_counter += update_error_pos(section.body.start + 2, section.body.start + 2 + 2, [1] * 2,
                                                      [0] * 2)
                else:
                    error_counter += update_error_pos(section.body.start + 2, section.body.start + 2 + 2, [0] * 2)
                central_sec_count = sum(
                    [1 for i in self.zip_structure.sections if isinstance(i.body, Zip.CentralDirEntry)])
                if section.body.num_central_dir_entries_on_disk != central_sec_count and \
                        section.body.num_central_dir_entries_on_disk != 0xffff and \
                        section.body.num_central_dir_entries_on_disk != len(self.zip_structure.sections) - 1:
                    error_counter += update_error_pos(section.body.start + 4, section.body.start + 4 + 2, [1] * 2,
                                                      struct.pack("<H", central_sec_count))
                else:
                    error_counter += update_error_pos(section.body.start + 4, section.body.start + 4 + 2, [0] * 2)
                if section.body.num_central_dir_entries_total != central_sec_count and \
                        section.body.num_central_dir_entries_total != 0xffff and \
                        section.body.num_central_dir_entries_total != len(self.zip_structure.sections) - 1:
                    error_counter += update_error_pos(section.body.start + 6, section.body.start + 6 + 2, [1] * 2,
                                                      struct.pack("<H", central_sec_count))
                else:
                    error_counter += update_error_pos(section.body.start + 6, section.body.start + 6 + 2, [0] * 2)
                central_sec_len = sum([4 + s.body.end - s.body.start for s in self.zip_structure.sections if
                                       isinstance(s.body, Zip.CentralDirEntry)])
                if section.body.len_central_dir != central_sec_len:
                    # cant be sure that the start and end are correct
                    error_counter += update_error_pos(section.body.start + 8, section.body.start + 8 + 4, [-1] * 4)
                else:
                    error_counter += update_error_pos(section.body.start + 8, section.body.start + 8 + 4, [0] * 4)
                if not any([isinstance(s.body, Zip.CentralDirEntry) and s.body.start - 4 == section.body.ofs_central_dir
                            for s in self.zip_structure.sections]):
                    minium_start_of_a_central_dir = min(
                        [s.body.start for s in self.zip_structure.sections if isinstance(s.body, Zip.CentralDirEntry)])
                    error_counter += update_error_pos(section.body.start + 12, section.body.start + 12 + 4, [1] * 4,
                                                      struct.pack("<I", minium_start_of_a_central_dir))
                else:
                    error_counter += update_error_pos(section.body.start + 12, section.body.start + 12 + 4, [0] * 4)
                if len(section.body.comment) != section.body.len_comment:
                    # the comment length must be correct since foutain codes do not change the length of the data
                    error_counter += update_error_pos(section.body.start + 16, section.body.start + 16 + 2,
                                                      corrected_bytes=struct.pack("<H", len(section.body.comment)))
                else:
                    error_counter += update_error_pos(section.body.start + 16, section.body.start + 16 + 2, [0] * 2)
            if error_counter > 4:
                # TODO: we might want to find best (closest) magic number...
                error_pos = error_pos_bkp
                self.reconstructed_zip_bytes = reconstructed_zip_bytes_bkp
        if start:
            error_pos = [-1] * self.gepp.b.shape[1] + error_pos
            # if we have a header chunk we can check the overhead of the zip file
            if self.gepp.b[self.semi_automatic_solver.decoder.number_of_chunks,
               self.semi_automatic_solver.headerChunk.last_chunk_length:].sum() == 0:
                update_error_pos(self.gepp.b.shape[
                                     1] * (
                                         self.semi_automatic_solver.decoder.number_of_chunks - 1) + self.semi_automatic_solver.headerChunk.last_chunk_length,
                                 self.gepp.b.shape[1] * (self.semi_automatic_solver.decoder.number_of_chunks - 1) +
                                 self.gepp.b.shape[1],
                                 [0] * (self.gepp.b.shape[
                                            1] - self.semi_automatic_solver.headerChunk.last_chunk_length))
            else:
                update_error_pos(self.gepp.b.shape[
                                     1] * (
                                         self.semi_automatic_solver.decoder.number_of_chunks - 1) + self.semi_automatic_solver.headerChunk.last_chunk_length,
                                 self.gepp.b.shape[1] * (self.semi_automatic_solver.decoder.number_of_chunks - 1) +
                                 self.gepp.b.shape[1],
                                 [1] * (self.gepp.b.shape[
                                            1] - self.semi_automatic_solver.headerChunk.last_chunk_length),
                                 [0] * (self.gepp.b.shape[
                                            1] - self.semi_automatic_solver.headerChunk.last_chunk_length))
        # for rows with only some columns == -1 and the rest == 0:
        # check if there are still possible packets if we would tag this row as incorrect together with all row that are already tagged incorrect
        # additionally we could do the same for correct rows and treating the row as correct and check if there is a solution together with all other fully correct or incorrect rows...
        error_pos = self.compare_sections(error_pos, copy.deepcopy(self.zip_structure.sections))
        return np.array(error_pos).reshape(-1, self.gepp.b.shape[1])

    def compare_sections(self, error_pos, sections):
        def update_error_pos(_start, _end, new_error_pos=None, corrected_bytes=None, overwrite=False):
            offset = start * self.gepp.b.shape[1]
            _parser_error_pos = self.parser_error_matrix[offset + _start: offset + _end]
            if new_error_pos is None:
                new_error_pos = _parser_error_pos
            error_pos[_start + offset: _end + offset] = [new_error_pos[pos] if (i == -1 or overwrite) else i for pos, i
                                                         in
                                                         enumerate(_parser_error_pos)]
            if corrected_bytes is not None:
                self.reconstructed_zip_bytes[_start: _end] = corrected_bytes
            return 1 if len(np.array(new_error_pos)[np.where(np.array(new_error_pos) > 0)]) > 0 else 0

        # compare matching PK sections (for each LocalFile section compare with the corresponding CentralDirEntry section)
        start = 1 if self.use_header_chunk else 0
        # error_pos = [-1 for _ in
        #             range(len(self.gepp.b[start:].reshape(-1)))]  # -1 <= unknown, 0 == correct, >=1 = incorrect
        localfile_to_centraldir = {}
        matched_central_dir_entries = set()
        for section in sections:
            error_counter = 0
            error_pos_bkp = copy.deepcopy(error_pos)
            reconstructed_zip_bytes_bkp = copy.deepcopy(self.reconstructed_zip_bytes)
            # if all([i in [-1, 0] for i in self.parser_error_matrix[section.start:section.start+4]]):
            error_counter += update_error_pos(section.body.start - 4, section.body.start,
                                              [0] * 4)  # PK header will be correct
            if isinstance(section.body, Zip.LocalFile):
                # compare local file header and central directory entry header:
                for central_dir_section in sections:
                    if isinstance(central_dir_section.body, Zip.CentralDirEntry):
                        if central_dir_section.body.ofs_local_header == section.start:
                            # we use the ofs_local_header since fountain codes won't change the length of the file!
                            localfile_to_centraldir[section] = central_dir_section
                            matched_central_dir_entries.add(central_dir_section)
                            # ofs_local_header is correct:
                            error_counter += update_error_pos(central_dir_section.start + 42,
                                                              central_dir_section.start + 42 + 4,
                                                              [0] * 4)
                            if central_dir_section.body.version_needed_to_extract == central_dir_section.body.version_made_by:
                                error_counter += update_error_pos(central_dir_section.body.start,
                                                                  central_dir_section.body.start + 2,
                                                                  [0] * 2)
                            # version field(s):
                            if section.body.header.version == central_dir_section.body.version_needed_to_extract:
                                error_counter += update_error_pos(section.body.header.start + 2,
                                                                  section.body.header.start + 2 + 2,
                                                                  [0] * 2)
                                error_counter += update_error_pos(central_dir_section.body.start + 2,
                                                                  central_dir_section.body.start + 2 + 2,
                                                                  [0] * 2)
                            # flags field:
                            # sec_flags_field = self.get_raw_bytes(section.body.start + 8, 2)
                            # central_dir_flags_field = self.get_raw_bytes(central_dir_section.body.start + 10, 2)
                            # if all(np.equal(sec_flags_field, central_dir_flags_field)):
                            #    error_counter += update_error_pos(section.body.start + 8, section.body.start + 10, [0] * 2)
                            #    error_counter += update_error_pos(central_dir_section.body.start + 10, central_dir_section.body.start + 12,
                            #                     [0] * 2)
                            # else:
                            #    error_counter += update_error_pos(section.body.start + 8, section.body.start + 10, [-1] * 2, True)
                            #    error_counter += update_error_pos(central_dir_section.body.start + 10, central_dir_section.body.start + 12,
                            #                     [-1] * 2, True)
                            # comparison of compression field:
                            if section.body.header.compression_method == central_dir_section.body.compression_method:
                                error_counter += update_error_pos(section.body.start + 10, section.body.start + 12,
                                                                  [0] * 2)
                                error_counter += update_error_pos(central_dir_section.body.start + 12,
                                                                  central_dir_section.body.start + 12 + 2,
                                                                  [0] * 2)
                            else:
                                # error_counter += update_error_pos(section.body.header.start, section.body.header.start + 2, [0] * 2)
                                if all([i == 0 for i in
                                        error_pos[section.body.start + 10:section.body.start + 10 + 2]]):
                                    error_counter += update_error_pos(central_dir_section.body.start + 12,
                                                                      central_dir_section.body.start + 12 + 2, [1] * 2)
                            # modification time:
                            if section.body.header.file_mod_time.time.second == central_dir_section.body.file_mod_time.time.second \
                                    and section.body.header.file_mod_time.time.minute == central_dir_section.body.file_mod_time.time.minute \
                                    and section.body.header.file_mod_time.time.hour == central_dir_section.body.file_mod_time.time.hour:
                                error_counter += update_error_pos(section.body.start + 12, section.body.start + 12 + 2,
                                                                  [0] * 2)
                                error_counter += update_error_pos(central_dir_section.body.start + 14,
                                                                  central_dir_section.body.start + 14 + 2,
                                                                  [0] * 2)
                            # modification date:
                            if section.body.header.file_mod_time.date.day == central_dir_section.body.file_mod_time.date.day and \
                                    section.body.header.file_mod_time.date.month == central_dir_section.body.file_mod_time.date.month and \
                                    section.body.header.file_mod_time.date.year == central_dir_section.body.file_mod_time.date.year:
                                error_counter += update_error_pos(section.body.start + 14, section.body.start + 14 + 2,
                                                                  [0] * 2)
                                error_counter += update_error_pos(central_dir_section.body.start + 16,
                                                                  central_dir_section.body.start + 16 + 2,
                                                                  [0] * 2)
                            # crc32:
                            if section.body.header.crc32 == central_dir_section.body.crc32:
                                error_counter += update_error_pos(section.body.start + 16, section.body.start + 16 + 4,
                                                                  [0] * 4)
                                error_counter += update_error_pos(central_dir_section.body.start + 18,
                                                                  central_dir_section.body.start + 18 + 4,
                                                                  [0] * 4)
                            else:
                                if all([i == 0 for i in
                                        error_pos[section.body.start + 16:section.body.start + 16 + 4]]):
                                    # crc of the file is treated as correct, so the error is in the cdf:
                                    error_counter += update_error_pos(central_dir_section.body.start + 18,
                                                                      central_dir_section.body.start + 18 + 4, [1] * 4)
                            # compressed size:
                            if section.body.header.len_body_compressed == central_dir_section.body.len_body_compressed:
                                error_counter += update_error_pos(section.body.start + 20, section.body.start + 20 + 4,
                                                                  [0] * 4)
                                error_counter += update_error_pos(central_dir_section.body.start + 22,
                                                                  central_dir_section.body.start + 22 + 4, [0] * 4)
                            else:
                                if all([i == 0 for i in
                                        error_pos[section.body.start + 20:section.body.start + 20 + 4]]):
                                    # compressed size of the file is treated as correct, so the error is in the cdf:
                                    error_counter += update_error_pos(central_dir_section.body.start + 22,
                                                                      central_dir_section.body.start + 22 + 4, [1] * 4)
                            # uncompressed size:
                            if section.body.header.len_body_uncompressed == central_dir_section.body.len_body_uncompressed:
                                error_counter += update_error_pos(section.body.start + 24, section.body.start + 24 + 4,
                                                                  [0] * 4)
                                error_counter += update_error_pos(central_dir_section.body.start + 26,
                                                                  central_dir_section.body.start + 26 + 4, [0] * 4)
                            else:
                                if all([i == 0 for i in
                                        error_pos[section.body.start + 24:section.body.start + 24 + 4]]):
                                    # uncompressed size of the file is treated as correct, so the error is in the cdf:
                                    error_counter += update_error_pos(central_dir_section.body.start + 26,
                                                                      central_dir_section.body.start + 26 + 4, [1] * 4)
                                # else:
                                #    # unpack the file and check if the uncompressed size is correct
                            # file name len:
                            if section.body.header.len_file_name == central_dir_section.body.len_file_name:
                                error_counter += update_error_pos(section.body.start + 30, section.body.start + 30 + 2,
                                                                  [0] * 2)
                                error_counter += update_error_pos(central_dir_section.body.start + 32,
                                                                  central_dir_section.body.start + 32 + 2, [0] * 2)
                            else:
                                if all([i == 0 for i in
                                        error_pos[section.body.start + 30:section.body.start + 30 + 2]]):
                                    # filename len of the file is treated as correct, so the error is in the cdf:
                                    error_counter += update_error_pos(central_dir_section.body.start + 32,
                                                                      central_dir_section.body.start + 32 + 2, [1] * 2)
                                else:
                                    if central_dir_section.body.len_file_name == len(section.body.header.file_name):
                                        error_counter += update_error_pos(central_dir_section.body.start + 32,
                                                                          central_dir_section.body.start + 32 + 2,
                                                                          [0] * 2)
                            # file name:
                            if section.body.header.file_name == central_dir_section.body.file_name:
                                error_counter += update_error_pos(section.body.start + 46,
                                                                  section.body.start + 46 + section.body.header.len_file_name,
                                                                  [0] * section.body.header.len_file_name)
                                error_counter += update_error_pos(central_dir_section.body.start + 46,
                                                                  central_dir_section.body.start + 46 + central_dir_section.body.len_file_name,
                                                                  [0] * central_dir_section.body.len_file_name)
                            else:
                                # since we strip all non UTF-8 characters, the "correct" version will be the longer one!
                                if len(central_dir_section.body.file_name) == central_dir_section.body.len_file_name:
                                    error_counter += update_error_pos(central_dir_section.body.start + 46,
                                                                      central_dir_section.body.start + 46 + central_dir_section.body.len_file_name,
                                                                      [0] * central_dir_section.body.len_file_name)
                                    error_counter += update_error_pos(section.body.start + 46,
                                                                      section.body.start + 46 + section.body.header.len_file_name,
                                                                      [0 if (central_dir_section.body.file_name[i] ==
                                                                             section.body.header.file_name[
                                                                                 i]) else 1 for i in
                                                                       range(min(central_dir_section.body.len_file_name,
                                                                                 section.body.header.len_file_name))],
                                                                      central_dir_section.body._raw_file_name)
                                    self.reconstructed_zip_bytes[section.body.start + 46:
                                                                 section.body.start + 46 + section.body.header.len_file_name] = \
                                        central_dir_section.body._raw_file_name
                                else:
                                    error_counter += update_error_pos(section.body.start + 46,
                                                                      section.body.start + 46 + section.body.header.len_file_name,
                                                                      [0] * section.body.header.len_file_name)
                                    error_counter += update_error_pos(central_dir_section.body.start + 46,
                                                                      central_dir_section.body.start + 46 + central_dir_section.body.len_file_name,
                                                                      [0 if (central_dir_section[i] == section[
                                                                          i]) else 1 for i in
                                                                       range(min(section.body.header.len_file_name,
                                                                                 central_dir_section.body.len_file_name))],
                                                                      section.body.header._raw_file_name)
                                    self.reconstructed_zip_bytes[central_dir_section.body.start + 46:
                                                                 central_dir_section.body.start + 46 + central_dir_section.body.len_file_name] = \
                                        section.body.header._raw_file_name
                            # extra field len:
                            # if section.body.header.len_extra == central_dir_section.body.len_extra:
                            #    error_counter += update_error_pos(section.body.start + 32, section.body.start + 32 + 2, [0] * 2)
                            #    error_counter += update_error_pos(central_dir_section.body.start + 34,
                            #                     central_dir_section.body.start + 34 + 2, [0] * 2)
                            # else:
                            #    if all([i == 0 for i in
                            #            error_pos[section.body.start + 32:section.body.start + 32 + 2]]):
                            #        # extra field len of the file is treated as correct, so the error is in the cdf:
                            #        error_counter += update_error_pos(central_dir_section.body.start + 34,
                            #                         central_dir_section.body.start + 34 + 2, [1] * 2)
                            # file comment len:
                            next_signature = self.get_raw_bytes(
                                central_dir_section.body.start + 42 + central_dir_section.body.len_file_name
                                + central_dir_section.body.len_extra + central_dir_section.body.len_comment, 2)
                            if b"".join(next_signature) == b'PK':
                                # len file name is correct
                                error_counter += update_error_pos(central_dir_section.body.start + 36,
                                                                  central_dir_section.body.start + 36 + 2, [0] * 2)
                                # len extra field is correct
                                error_counter += update_error_pos(central_dir_section.body.start + 38,
                                                                  central_dir_section.body.start + 38 + 2, [0] * 2)
                                # len file comment is correct
                                error_counter += update_error_pos(central_dir_section.body.start + 40,
                                                                  central_dir_section.body.start + 40 + 2, [0] * 2)
                            # disk number start:
                            if central_dir_section.body.disk_number_start == 0:
                                error_counter += update_error_pos(central_dir_section.body.start + 36,
                                                                  central_dir_section.body.start + 36 + 2, [0] * 2)
                            else:
                                error_counter += update_error_pos(central_dir_section.body.start + 36,
                                                                  central_dir_section.body.start + 36 + 2, [1] * 2)

                            # internal file attributes:
                            # TODO: test if we can check bit 1 and 3-16 (reserved/unused!)
                            # external file attributes:
                            # NO way to check...
                            # error_counter += update_error_pos(section.body.header.start + 18, section.body.header.start + 18 + 4, [0] * 4)
                            break
                        else:
                            if b"".join(self.get_raw_bytes(central_dir_section.body.ofs_local_header, 2)) == b"PK":
                                # the reference is correct, but the local file header is not correct
                                error_counter += update_error_pos(central_dir_section.start + 42,
                                                                  central_dir_section.start + 42 + 4,
                                                                  [0] * 4)
                                # TODO: go trough all local file headers and invalidate the one BEFORE the reference...
                            else:
                                # either the reference is wrong or this was no real central directory entry!
                                error_counter += update_error_pos(central_dir_section.start + 42,
                                                                  central_dir_section.start + 42 + 4,
                                                                  [-1] * 4)
            if error_counter > 4:
                # TODO find best magic number...
                error_pos = error_pos_bkp
                self.reconstructed_zip_bytes = reconstructed_zip_bytes_bkp
        for section in matched_central_dir_entries:
            sections.remove(section)
        for section in sections:
            if section not in localfile_to_centraldir:
                # we have a local file header without a corresponding central directory entry header
                # TODO we might be able to match it with an unmatched central directory entry by comparing
                #  other entries such as the filename
                #  alternatively we choose the central directory entry with the smallest edit distance
                # This might further increase the recovery chance but may increase the complexity.
                pass
        return error_pos

    def get_ui_elements(self):
        return {"kaitai-viewer": {"type": "kaitai_view", "text": "Show KaitaiStruct",
                                  "callback": self.toogle_kaitai_viewer, "updates_b": False},
                "btn-zipfile-find-rows": {"type": "button", "text": "Tag (in)correct rows",
                                          "callback": self.find_correct_rows, "updates_b": True},
                "btn-zipfile-find-columns": {"type": "button", "text": "Tag (in)correct columns",
                                             "callback": self.find_incorrect_columns, "updates_b": True},
                "btn-zipfile-repair": {"type": "button", "text": "Create corrected version(s)", "callback": self.repair,
                                       "updates_b": True}}

    def toogle_kaitai_viewer(self, *args, **kwargs):
        n_clicks = kwargs["c_ctx"].triggered[0]["value"]
        if n_clicks % 2 == 1:
            if self.zip_structure is None:
                self.zip_structure, self.parser_error_matrix = self.parse_zipfile()
            kaitai_html = Kaitai2Html.kaitai2html(self.zip_structure,
                                                  chunk_length=self.semi_automatic_solver.decoder.GEPP.b.shape[1],
                                                  chunk_offset=1 if self.use_header_chunk else 0)
        else:
            kaitai_html = None
        return {"kaitai_content": kaitai_html, "refresh_view": False}

    def find_correct_rows(self, *args, **kwargs):
        if kwargs is None or kwargs.get("chunk_tag") is None:
            self.chunk_tag = np.zeros(self.gepp.b.shape[0], dtype=np.int32)
        else:
            self.chunk_tag = kwargs.get("chunk_tag")

        res = []
        if self.error_matrix is None:
            self.error_matrix = self.find_error_region(*args, **kwargs)
        # for each row: count all entrys != 0
        for i in range(len(self.error_matrix)):
            res.append(np.sum(self.error_matrix[i, :]) == 0)
        for i in range(min(len(self.chunk_tag), len(res))):
            if res[i]:
                self.chunk_tag[i] = 2
        return {"chunk_tag": self.find_incorrect_rows(chunk_tag=self.chunk_tag)["chunk_tag"], "updates_b": False,
                "refresh_view": True}

    # parse each section in the zip file:
    # if the section is a local file header, check the crc32

    def find_incorrect_rows(self, *args, **kwargs):
        if kwargs is None or kwargs.get("chunk_tag") is None:
            self.chunk_tag = np.zeros(self.gepp.b.shape[0], dtype=np.int32)
        else:
            self.chunk_tag = kwargs.get("chunk_tag")

        res = []
        if self.error_matrix is None:
            self.error_matrix = self.find_error_region(*args, **kwargs)
        # for each row: count all entrys != 0
        for i in range(len(self.error_matrix)):
            res.append(1 in self.error_matrix[i, :])
        for i in range(min(len(self.chunk_tag), len(res))):
            if res[i]:
                self.chunk_tag[i] = 1
        return {"chunk_tag": self.chunk_tag, "updates_b": False, "refresh_view": True}

    def find_incorrect_columns(self, *args, **kwargs):
        column_tag = [0] * self.gepp.b.shape[1]
        if self.error_matrix is None:
            self.error_matrix = self.find_error_region(*args, **kwargs)
        # for each row: count all entrys != 0
        for i in range(self.gepp.b.shape[1]):
            for row in range(self.error_matrix.shape[0]):
                if self.error_matrix[row, i] > 0:
                    column_tag[i] += 1
        return {"column_tag": column_tag, "updates_b": False, "refresh_view": True}

    def update_gepp(self, gepp):
        pass
        # update
        # self.zip_structure, error_pos = self.parse_zipfile()
        # self.error_matrix = np.array(error_pos).reshape(-1, self.gepp.b.shape[1])

    def update_chunk_tag(self, chunk_tag):
        pass

    def parse_section(self, zip_bytes, start_offset, error_pos, start):
        iterations = 0
        bkp_error_pos = copy.copy(error_pos)
        while iterations < 100:
            try:
                res = Zip.PkSection(KaitaiStream(io.BytesIO(zip_bytes[start_offset:])), start_offset=start_offset)
                if self.reconstructed_zip_bytes is None:
                    self.reconstructed_zip_bytes = bytearray(copy.copy(zip_bytes))
                return res, error_pos  # np.array(error_pos).reshape(-1, self.gepp.b.shape[1])
            except ValidationFailedError as err:
                iterations += 1
                # if magic bytes are wrong:
                if err.src_path == "/types/pk_section/seq/0":
                    # replace magic bytes with correct ones:
                    zip_bytes = zip_bytes[:start_offset + err.io.pos() - len(err.expected)] + err.expected + zip_bytes[
                                                                                                             start_offset + err.io.pos():]
                    for i, diff in enumerate([a - b for a, b in zip(err.expected, err.actual)]):
                        error_pos[start_offset +
                                  (start * self.semi_automatic_solver.decoder.GEPP.b.shape[1]) + err.io.pos() - len(
                            err.expected) + i] = 1 if diff != 0 else 0
                elif err.src_path == "/types/pk_section/seq/1":
                    distance = {}
                    for expected in [513, 1027, 1541, 2055]:
                        distance[expected] = self.bitwise_hamming_distance(expected, err.actual)
                    expected = struct.pack("<H", sorted([(x) for x in distance.items()], key=lambda x: x[1])[0][0])
                    # replace src bytes with correct ones:
                    zip_bytes = zip_bytes[:start_offset + err.io.pos() - len(expected)] + expected + zip_bytes[
                                                                                                     start_offset + err.io.pos():]
                    for i, diff in enumerate([a - b for a, b in zip(expected, struct.pack("<H", err.actual))]):
                        error_pos[start_offset +
                                  (start * self.semi_automatic_solver.decoder.GEPP.b.shape[0]) + err.io.pos() - len(
                            expected) + i] = 1 if diff != 0 else 0
                    # a different approach would be to try to parse the part with all possible solutions and check for
                    # consistency...
                elif err.src_path == "/types/time/seq/1" or err.src_path == "/types/time/seq/2":
                    expected = struct.pack("<B", 0)
                    zip_bytes = zip_bytes[:start_offset + err.io.pos()] + expected + zip_bytes[
                                                                                     start_offset + err.io.pos():]
                    for i in range(2):
                        error_pos[start_offset +
                                  (start * self.semi_automatic_solver.decoder.GEPP.b.shape[0]) + err.io.pos() - len(
                            expected) + i] = 1
                elif err.src_path == "/types/date/seq/0":
                    expected = struct.pack("<B", 1)
                    zip_bytes = zip_bytes[:start_offset + err.io.pos()] + expected + zip_bytes[
                                                                                     start_offset + err.io.pos():]
                    for i in range(2):
                        error_pos[start_offset +
                                  (start * self.semi_automatic_solver.decoder.GEPP.b.shape[0]) + err.io.pos() - len(
                            expected) + i] = 1
                elif err.src_path == "/types/date/seq/1":
                    distance = {}
                    for expected in range(1, 13, 1):
                        distance[expected] = self.bitwise_hamming_distance(expected, err.actual)
                    expected = struct.pack("<B", sorted([(x) for x in distance.items()], key=lambda x: x[1])[0][0])
                    # replace src bytes with correct ones:
                    zip_bytes = zip_bytes[:err.io.pos() - len(expected)] + expected + zip_bytes[err.io.pos():]
                    for i, diff in enumerate([a - b for a, b in zip(expected, struct.pack("<B", err.actual))]):
                        error_pos[
                            (start * self.semi_automatic_solver.decoder.GEPP.b.shape[0]) + err.io.pos() - len(
                                expected) + i] = 1 if diff != 0 else 0
                else:
                    return None, error_pos
                    raise err
            except CustomOutOfBoundsException as err:
                iterations += 1
                raise err
            except InvalidDataException as err:
                iterations += 1
                if err.args[0] in ["/types/filename/invalid", "/types/file_name/invalid", "/types/comment/invalid",
                                   "/types/extra/invalid", "/types/len_extra/notmultiple4", "/types/raw_extra/invalid",
                                   "/types/raw_body/invalid", "/types/comment_len/invalid", "/types/comment/invalid",
                                   "/types/compressed_data/invalid"]:
                    # update error matrix at position error_pos
                    expected = struct.pack("<H", err.expected)
                    zip_bytes = zip_bytes[:start_offset + err.error_pos] + expected + zip_bytes[
                                                                                      start_offset + err.error_pos:]
                    rnge = 4 if err.args[0] == "/types/compressed_data/invalid" else 2
                    for i in range(rnge):
                        try:
                            error_pos[start_offset + (
                                    start * self.semi_automatic_solver.decoder.GEPP.b.shape[0]) + err.error_pos - len(
                                expected) + i] = 1
                        except:
                            pass
                else:
                    raise err
            except Exception as err:
                iterations += 1
                return None, bkp_error_pos
        error_pos = bkp_error_pos
        return None, error_pos

    def sweep_zip_header(self, error_pos=None):
        start = 1 if self.use_header_chunk else 0
        if error_pos is None:
            error_pos = [-1 for _ in
                         range(len(self.gepp.b[start:].reshape(-1)))]  # -1 <= unknown, 0 == correct, >=1 = incorrect
        last_chunk_garbage = self.gepp.b.shape[1] - self.semi_automatic_solver.headerChunk.last_chunk_length
        if last_chunk_garbage > 0:
            zip_bytes = self.gepp.b[start:self.semi_automatic_solver.decoder.number_of_chunks].reshape(-1)[
                        :-last_chunk_garbage].tobytes()
        else:
            zip_bytes = self.gepp.b[start:self.semi_automatic_solver.decoder.number_of_chunks].reshape(
                -1).tobytes()
        signature_positions = {}
        for signature in [rb'\x50\x4b\x01\x02', rb'\x50\x4b\x03\x04', rb'\x50\x4b\x05\x06', rb'\x50\x4b\x07\x08']:
            try:
                signature_positions[signature] = [x.regs[0][0] for x in re.finditer(signature, zip_bytes)]
            except:
                pass
        # flatten the dict to list of lists:
        flat_signature_positions = sorted([signature_positions[signature] for signature in signature_positions])
        # flatten to single list:
        flat_signature_positions = [item for sublist in flat_signature_positions for item in sublist]
        sections = []
        for start_offset in flat_signature_positions:
            # create a copy of error_pos for each section canididate and merge them at the end (only if the section was "valid")
            # make sure the sections are not overlapping and if they are, choose the one that produces the least errors

            error_pos_bkp = error_pos.copy()
            tmp = self.parse_section(zip_bytes, start_offset, error_pos, start)
            error_pos = error_pos_bkp
            sct = tmp[0]
            if sct is not None:
                sections.append(sct)

        # iterate over all sections to find section at location of ofs_local_header:
        for section in sections:
            if isinstance(section.body, Zip.CentralDirEntry):
                error_pos_bkp = error_pos.copy()
                ofs_local_header = section.body.ofs_local_header
                tmp = self.parse_section(zip_bytes, ofs_local_header, error_pos, start)
                if tmp[0] is not None and tmp[0].start not in [x.start for x in sections]:
                    sections.append(tmp[0])
                else:
                    # we might want to delete the section if there is no valid matchin file section
                    error_pos = error_pos_bkp

        tmp = Zip.from_bytes(b"")
        tmp.sections = sections
        if self.reconstructed_zip_bytes is None:
            self.reconstructed_zip_bytes = bytearray(copy.copy(zip_bytes))
        return tmp, error_pos


mgr = PluginManager()
mgr.register_plugin(ZipFileRepair)
