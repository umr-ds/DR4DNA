import base64
import io
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


class BmpFileRepair(FileSpecificRepair):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_repair_bytes = 2
        self.error_matrix = None
        self.parser_error_matrix = None
        self.no_inspect_chunks = self.gepp.b.shape[0]
        self.width = None
        self.height = None
        self.bmp_structure = None
        self.reconstructed_bmp_bytes: typing.Optional[bytearray] = None
        self.image_matrix = None
        self.bmp_bytes = None

    def parse_bmp(self, *args, **kwargs):
        start = 1 if self.use_header_chunk else 0
        start_offset = start * self.semi_automatic_solver.decoder.GEPP.b.shape[1]
        error_pos = np.array([-1 for _ in
                              range(self.gepp.b.shape[0] * self.gepp.b.shape[1])],
                             dtype=np.float32)  # -1 <= unknown, 0 == correct, >=1 = incorrect
        self.semi_automatic_solver.parse_header("I")
        if self.semi_automatic_solver.headerChunk is not None:
            last_chunk_garbage = self.gepp.b.shape[1] - self.semi_automatic_solver.headerChunk.last_chunk_length
        else:
            last_chunk_garbage = 0
        # try to parse the zipfile:
        if last_chunk_garbage > 0:
            self.bmp_bytes = self.gepp.b[start:self.semi_automatic_solver.decoder.number_of_chunks].reshape(-1)[
                             :-last_chunk_garbage].tobytes()
        else:
            self.bmp_bytes = self.gepp.b[start:self.semi_automatic_solver.decoder.number_of_chunks].reshape(
                -1).tobytes()
        if self.reconstructed_bmp_bytes is None:
            self.reconstructed_bmp_bytes = bytearray(self.bmp_bytes)
        while True:
            try:
                res = Bmp.from_bytes(self.reconstructed_bmp_bytes)
                # check all known parameters + all other parameters for sanity:
                if res.file_hdr.len_file != len(self.reconstructed_bmp_bytes):
                    self.reconstructed_bmp_bytes[2:6] = len(self.reconstructed_bmp_bytes).to_bytes(4, "little")
                    error_pos[start_offset + 2:start_offset + 6] = np.array(
                        [a ^ b for a, b in zip(self.bmp_bytes[2:6], self.reconstructed_bmp_bytes[2:6])],
                        dtype=error_pos.dtype)
                    res = Bmp.from_bytes(self.reconstructed_bmp_bytes)
                allowed_file_types = ["BM", "BA", "CI", "CP", "IC", "PT"]
                if res.file_hdr.file_type not in allowed_file_types:
                    self.reconstructed_bmp_bytes[0:2] = allowed_file_types[0].encode("ascii")
                    error_pos[start_offset:start_offset + 2] = np.array(
                        [a ^ b for a, b in zip(self.bmp_bytes[0:2], self.reconstructed_bmp_bytes[0:2])],
                        dtype=error_pos.dtype)
                    res = Bmp.from_bytes(self.reconstructed_bmp_bytes)

                # TODO: check if correct:
                if res.file_hdr.reserved1 != 0:
                    self.reconstructed_bmp_bytes[6] = 0
                    self.reconstructed_bmp_bytes[7] = 0
                    error_pos[start_offset + 6: start_offset + 8] = np.array(
                        [a ^ b for a, b in zip(self.bmp_bytes[6:8], self.reconstructed_bmp_bytes[6:8])],
                        dtype=error_pos.dtype)
                    res = Bmp.from_bytes(self.reconstructed_bmp_bytes)
                if res.file_hdr.reserved2 != 0:
                    self.reconstructed_bmp_bytes[8] = 0
                    self.reconstructed_bmp_bytes[9] = 0
                    error_pos[start_offset + 8: start_offset + 9] = np.array(
                        [a ^ b for a, b in zip(self.bmp_bytes[8:10], self.reconstructed_bmp_bytes[8:10])],
                        dtype=error_pos.dtype)
                    res = Bmp.from_bytes(self.reconstructed_bmp_bytes)
                if res.file_hdr.ofs_bitmap != res.dib_info.end:
                    self.reconstructed_bmp_bytes[10:14] = res.dib_info.end.to_bytes(4, "little")
                    error_pos[start_offset + 10: start_offset + 14] = np.array(
                        [a ^ b for a, b in zip(self.bmp_bytes[10:14], self.reconstructed_bmp_bytes[10:14])],
                        dtype=error_pos.dtype)
                    res = Bmp.from_bytes(self.reconstructed_bmp_bytes)
                # TODO: depends: right now this is only correct for 16bit
                # check up res.dib_info.header.bits_per_pixel!
                mask_mask = res.dib_info.color_mask_red ^ res.dib_info.color_mask_blue ^ res.dib_info.color_mask_alpha ^ res.dib_info.color_mask_green
                if mask_mask != 2 ** res.dib_info.header.bits_per_pixel - 1 or (
                        res.dib_info.header.bits_per_pixel == 32 and mask_mask | 0b11100000000000000000000000000000 != 2 ** res.dib_info.header.bits_per_pixel - 1):
                    error_pos[start_offset + 54: start_offset + 54 + math.sqrt(
                        res.dib_info.header.bits_per_pixel)] = np.array(
                        [0.5 for _ in range(math.sqrt(res.dib_info.header.bits_per_pixel))], dtype=error_pos.dtype)
                    res = Bmp.from_bytes(self.reconstructed_bmp_bytes)

                return res, error_pos  # np.array(error_pos).reshape(-1, self.gepp.b.shape[1])
            except ValidationFailedError as err:
                if err.src_path == "/types/file_header/seq/0":
                    self.reconstructed_bmp_bytes = self.reconstructed_bmp_bytes[:err.io.pos() - len(
                        err.expected)] + err.expected + self.reconstructed_bmp_bytes[err.io.pos():]
                    for i, diff in enumerate([a - b for a, b in zip(err.expected, err.actual)]):
                        error_pos[
                            (start * self.semi_automatic_solver.decoder.GEPP.b.shape[1]) + err.io.pos() - len(
                                err.expected) + i] = 1 if diff != 0 else 0
                else:
                    self.reconstructed_bmp_bytes = bytearray(self.bmp_bytes)
                    return None, None

    def set_use_header(self, use_header):
        self.use_header_chunk = use_header

    def repair(self, *args, **kwargs):
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

    def reload_image(self, *args, **kwargs):
        # todo: load canvas_json from args + chunk_tag to find the invalid packet
        self.parser_error_matrix = None
        self.no_inspect_chunks = self.gepp.b.shape[0]
        if self.reconstructed_bmp_bytes is not None:
            self.reconstructed_bmp_bytes = bytearray(self.bmp_bytes)
        self.bmp_structure, self.parser_error_matrix = self.parse_bmp()
        if self.width is None or self.height is None:
            self.width = self.bmp_structure.dib_info.header.image_width
            self.height = self.bmp_structure.dib_info.header.image_height
        # convert sample_arry to image array by using all bitmasks on the elements of the sample_array
        self.image_matrix = np.array(Image.open(io.BytesIO(self.reconstructed_bmp_bytes)).getdata(),
                                     dtype=np.uint8).reshape(self.height, self.width, -1)
        res = self.find_errors_tags()
        return {"update_b": False, "refresh_view": True, "width": self.width, "height": self.height,
                "updates_canvas": True, "canvas_data": self.image_matrix, "chunk_tag": res["chunk_tag"]}

    def is_compatible(self, meta_info, *args, **kwargs):
        # parse magic info string:
        # TODO: add check for filename / extension
        return meta_info == "Bitmap" or "PC bitmap" in meta_info

    def set_image_width(self, width, *args, **kwargs):
        res_str = "Reload the image first!"
        res = {"refresh_view": False}
        self.width = width[0]
        if self.reconstructed_bmp_bytes is not None:
            calculated_height = len(self.bmp_structure._raw_bitmap) / (
                    self.width * self.bmp_structure.dib_info.header.bits_per_pixel)
            if calculated_height.is_integer():
                res_str = f"height must be set to = {calculated_height}"
                if self.height == int(calculated_height):
                    res = self.update_shape()
                    res_str = "Image size changed!"
            else:
                res_str = f"Width can not be {self.width}: Image size is not a multiple of width * bits per pixel!"
        res["update_b"] = False
        res["info"] = res_str
        return res

    def set_image_height(self, height, *args, **kwargs):
        res_str = "Reload the image first!"
        res = {"refresh_view": False}
        self.height = height[1]
        if self.reconstructed_bmp_bytes is not None:
            calculated_width = len(self.bmp_structure._raw_bitmap) / (
                    self.height * self.bmp_structure.dib_info.header.bits_per_pixel)
            if calculated_width.is_integer():
                res_str = f"Width must be set to = {calculated_width}"
                if self.width == int(calculated_width):
                    res = self.update_shape()
                    res_str = "Image size changed!"
            else:
                res_str = f"Height can not be {self.height}: Image size is not a multiple of height * bits per pixel!"
        res["update_b"] = False
        res["info"] = res_str
        return res

    def update_shape(self):
        width_pos = self.bmp_structure.dib_info.header.image_width_pos
        height_pos = self.bmp_structure.dib_info.header.image_height_raw_pos
        if self.bmp_structure.dib_info.header.image_height_raw < 0:
            write_height = -self.height  # we have to store it as negative value
        else:
            write_height = self.height
        length = 2 if self.bmp_structure.dib_info.header.is_core_header else 4
        self.reconstructed_bmp_bytes[width_pos:width_pos + length] = [x for x in self.width.to_bytes(length,
                                                                                                     byteorder='little')]
        self.reconstructed_bmp_bytes[height_pos:height_pos + length] = [x for x in write_height.to_bytes(length,
                                                                                                         byteorder='little')]
        self.error_matrix = None  # invalidate error matrix
        return self.find_errors_tags()

    def get_ui_elements(self):
        return {"btn-bmpfile-reload": {"type": "button", "text": "Reload image",
                                       "callback": self.reload_image},
                "btn-bmpfile-download": {"type": "download", "text": "Download image", "callback": self.download},
                "kaitai-viewer": {"type": "kaitai_view", "text": "Show KaitaiStruct",
                                  "callback": self.toogle_kaitai_viewer, "updates_b": False},
                "txt-bmpfile-width": {"type": "int", "text": "Width of the image",
                                      "callback": self.set_image_width},
                "txt-bmpfile-height": {"type": "int", "text": "Height of the image",
                                       "callback": self.set_image_height},
                "cnvs-bmpfile-repair": {"type": "canvas", "width": self.width, "height": self.height},
                "upload-bmpfile": {"type": "upload", "text": "Upload image", "callback": self.upload_image},
                "btn-bmpfile-find-incorrect-pos": {"type": "button", "text": "Find incorrect positions",
                                                   "callback": self.find_errors_tags},
                "btn-bmpfile-find-columns": {"type": "button", "text": "Tag (in)correct columns",
                                             "callback": self.get_incorrect_columns, "updates_b": False},
                "btn-bmpfile-auto-repair": {"type": "button", "text": "Automatic Repair",
                                            "callback": self.repair, "updates_b": False},
                "txt-num-repair-bytes": {"type": "int",
                                         "text": "Number of bytes to repair (should be <= incorrect columns)",
                                         "default": 2, "callback": self.update_num_repair, "updates_b": False},
                }

    def toogle_kaitai_viewer(self, *args, **kwargs):
        n_clicks = kwargs["c_ctx"].triggered[0]["value"]
        if n_clicks % 2 == 1:
            kaitai_html = Kaitai2Html.kaitai2html(self.bmp_structure,
                                                  chunk_length=self.semi_automatic_solver.decoder.GEPP.b.shape[1],
                                                  chunk_offset=1 if self.use_header_chunk else 0)
        else:
            kaitai_html = None
        return {"kaitai_content": kaitai_html, "refresh_view": False}

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
        if self.parser_error_matrix is not None:
            pos_correct = np.array(self.parser_error_matrix)
        else:
            pos_correct = np.zeros(self.gepp.b.shape[0] * self.gepp.b.shape[1], dtype=np.float32)
        for i in range(0, len(self.bmp_bytes) - start_pos):
            diff = self.bmp_bytes[i] ^ self.reconstructed_bmp_bytes[i]
            if diff != 0:
                pos_correct[i + start_pos] = diff
        return pos_correct.reshape(-1, self.gepp.b.shape[1])

    def find_errors_tags(self, *args, **kwargs):
        if kwargs is None or kwargs.get("chunk_tag") is None:
            if self.chunk_tag is None:
                self.chunk_tag = np.zeros(self.gepp.b.shape[0], dtype=np.int32)
            # else: we already got some values, we do not want to overwrite them...
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
                    # if all positions are 0, then the row is correct ("1")
                    chunk_res = 2
            elif chunk_res > 0:
                # there is at least ONE error in the row
                chunk_res = 1
            else:
                # there is no information in the current row
                chunk_res = -1
            if chunk_res != -1:
                self.chunk_tag[i] = chunk_res
        return {"chunk_tag": self.chunk_tag, "update_b": False, "refresh_view": True}

    def download(self, *args, **kwargs):
        if self.reconstructed_bmp_bytes is None:
            res = self.reload_image()
            res["download"] = bytes(self.reconstructed_bmp_bytes)
            res["filename"] = "raw.bmp"
            return res
        return {"update_b": False, "refresh_view": False, "download": bytes(self.reconstructed_bmp_bytes),
                "filename": "raw.bmp"}

    def update_gepp(self, gepp):
        # invalidate error matrix:
        self.gepp = gepp
        self.error_matrix = None
        self.bmp_bytes = None
        self.reconstructed_bmp_bytes = None
        self.bmp_structure, self.parser_error_matrix = self.parse_bmp()
        # user has to refresh the canvas!

    def upload_image(self, *args, **kwargs):
        start_pos = (1 if self.use_header_chunk else 0) * self.gepp.b.shape[1]
        content = kwargs['c_ctx'].triggered[0]['value']
        if isinstance(content, list):
            content = content[0]
        if content is not None:
            try:
                content_type, content_string = content.split(',')
                new_error_part = np.array([a ^ b for a, b in
                                           zip(base64.b64decode(
                                               content_string),
                                               self.reconstructed_bmp_bytes)], dtype=self.error_matrix.dtype)
                self.error_matrix = self.error_matrix.reshape(-1)

                for i in range(0, new_error_part.shape[0]):
                    # only update positions with changes!
                    if new_error_part[i] != 0:
                        self.error_matrix[start_pos + i] = new_error_part[i]
                self.error_matrix = self.error_matrix.reshape(-1, self.gepp.b.shape[1])
                self.reconstructed_bmp_bytes = base64.b64decode(content_string)
                res = self.find_errors_tags()
                res["image_content"] = content
                res["updates_canvas"] = True
                return res
            except Exception as ex:
                raise ex
        return {"updates_canvas": True, "image_content": content}

    def update_canvas(self, canvas_json, *args, **kwargs):
        # if the base image is different from what we already have, the user uploaded a new (repaired) image, act accordingly
        # if the canvas_json is different from what we already have, the user tagged the image, act accordingly
        mask = parse_jsonstring(canvas_json, shape=(self.height, self.width))
        # reshape the mask to -1
        mask = mask.reshape(-1)
        # set the error matrix to the mask (with respect to the header)
        start_pos = (1 if self.use_header_chunk else 0) * self.gepp.b.shape[1]
        # offset of the image data in the bmp file:
        self.error_matrix = self.error_matrix.reshape(-1)
        image_offset = self.bmp_structure.file_hdr.ofs_bitmap
        for i, elem in enumerate(mask):
            if elem:
                self.error_matrix[i + start_pos + image_offset] = 0.5
        self.error_matrix = self.error_matrix.reshape(-1, self.gepp.b.shape[1])
        res = self.find_errors_tags()
        res["updates_canvas"] = True
        res["image_content"] = None  # todo: add line to each chunk that is incorrect?
        return {"updates_canvas": True, "image_content": None}


def parse_jsonstring(json_string, shape=None, scale=1):
    if shape is None:
        shape = (500, 500)
    mask = np.zeros(shape, dtype=np.bool)
    try:
        data = json.loads(json_string)
    except:
        return mask
    scale = 1
    for obj in data['objects']:
        if obj['type'] == 'image':
            scale = obj['scaleX']
        elif obj['type'] == 'path':
            pass  # not supported (yet?)
        elif obj['type'] == 'line':
            # calculate the middle of the line
            scale_obj = obj['scaleX']
            x1 = round((obj['left'] + obj['x1']) / scale * scale_obj)
            x2 = round((obj['left'] + obj['x2']) / scale * scale_obj)
            y1 = round((obj['top'] + obj['y1']) / scale * scale_obj)
            y2 = round((obj['top'] + obj['y2']) / scale * scale_obj)
            # calculate the middle of the line
            x = int(np.floor((x1 + x2) / 2))
            y = int(np.floor((y1 + y2) / 2))
            mask[y, x] = 1
        elif obj['type'] == 'rect':
            # calculate the middle of the rect
            scale_obj = obj['scaleX']
            x1 = round((obj['left']) / scale * scale_obj)
            x2 = round((obj['left'] + obj['width']) / scale * scale_obj)
            y1 = round((obj['top']) / scale * scale_obj)
            y2 = round((obj['top'] + obj['height']) / scale * scale_obj)
            # calculate the middle of the rect
            x = int(np.floor((x1 + x2) / 2))
            y = int(np.floor((y1 + y2) / 2))
            mask[y, x] = 1
    return mask


mgr = PluginManager()
mgr.register_plugin(BmpFileRepair)
