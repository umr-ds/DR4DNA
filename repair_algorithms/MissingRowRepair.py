import numpy as np
from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from repair_algorithms.PluginManager import PluginManager


class MissingRowRepair(FileSpecificRepair):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if rank of matrix is smaller than the number of chunks (columns of A), then there is a missing row
        # and thus not all chunks can be reconstructed -> find the chunks by inspecting which rows in A have more than
        # one "1" in them -> these rows are the missing rows

        # automatic repair very limited, for images, one could use in-painting to fill in the missing pixels and for
        # text, models such a ChatGPT could be used to fill the missing parts, but for unstructured or compressed data,
        # this is not possible. Thus, the user has to manually inspect the missing chunks and decide what to do with them.
        # Depending  on the length and the scenario, a bruteforce approach could be used to find the correct data for
        # a chunk which would then be used propagate to the equation system to solve all other missing chunks.
        self.error_matrix = None
        self.no_inspect_chunks = self.semi_automatic_solver.decoder.GEPP.b.shape[0]
        self.missing_rows = None
        self.added_rows = []
        self.added_row_content = []
        self.fill_row_content = None
        self.fill_row_num = 0

    def parse(self, *args, **kwargs):
        self.missing_rows = (self.semi_automatic_solver.decoder.GEPP.result_mapping == -1).transpose()[0]
        for i in range(len(self.missing_rows)):
            if self.missing_rows[i]:
                self.chunk_tag[i] = 3
            elif self.chunk_tag[i] == 3:
                self.chunk_tag[i] = 0
        return {"update_b": False, "refresh_view": True, "chunk_tag": self.chunk_tag,
                "info": f"Found {sum(self.missing_rows)} missing rows!"}

    def set_use_header(self, use_header):
        self.use_header_chunk = use_header

    def repair(self, *args, **kwargs):
        if len(self.fill_row_content) != self.semi_automatic_solver.decoder.GEPP.b.shape[1]:
            return {"refresh_view": False, "update_b": False,
                    "info": f"Invalid length of the row content! Content must be exactly {self.semi_automatic_solver.decoder.GEPP.b.shape[1]} bytes long!"}
        if self.fill_row_num < 0 or self.fill_row_num >= self.semi_automatic_solver.decoder.GEPP.A.shape[0]:
            return {"refresh_view": False, "update_b": False,
                    "info": f"Invalid row number! Row number must be between 0 and {self.semi_automatic_solver.decoder.GEPP.A.shape[0] - 1}!"}
        if self.chunk_tag[self.fill_row_num] != 3 and self.fill_row_num not in self.added_rows:
            return {"refresh_view": False, "update_b": False,
                    "info": "Invalid row number! Row number must be a missing row! (Try 'Analyze' button?!)"}
        if self.fill_row_num not in self.added_rows:
            a_row = np.zeros(self.semi_automatic_solver.decoder.GEPP.A.shape[1], dtype=np.bool)
            a_row[self.fill_row_num] = True
            self.added_rows.append(self.fill_row_num)  # the position in this list represents the position in A and b
            self.added_row_content.append(self.fill_row_content)
            self.semi_automatic_solver.decoder.GEPP.A = np.vstack((self.semi_automatic_solver.decoder.GEPP.A, a_row))
            self.semi_automatic_solver.decoder.GEPP.b = np.vstack(
                (self.semi_automatic_solver.decoder.GEPP.b, self.fill_row_content))
            self.semi_automatic_solver.decoder.GEPP.addRow(a_row, self.fill_row_content)
        else:
            self.semi_automatic_solver.decoder.GEPP.b[self.added_rows.index(self.fill_row_num)] = self.fill_row_content
            self.added_row_content[self.added_rows.index(self.fill_row_num)] = np.array(self.fill_row_content,
                                                                                        dtype="uint8")
        # ideally we should overwrite the initial A and be to prevent a reset from a different plugin but this
        # would counter the purpose of the initial A and b variables...
        self.semi_automatic_solver.decoder.solve(partial=True)
        # update chunk_tag to fix wrong tags after new solve
        self.parse()
        info_str = f"There are still {sum(self.missing_rows)} missing rows." if any(
            self.missing_rows) else "All missing rows should be solved now."
        return {"update_b": True, "refresh_view": True, "chunk_tag": self.chunk_tag,
                "info": f"Packet with content for row {self.fill_row_num} was added to the LES. {info_str}"}

    def is_compatible(self, meta_info, *args, **kwargs):
        # only activate this module if the gepp did not fully solve the equation system
        # we could alternatively use:
        # all(self.A.sum(axis=1) == 1)
        return not self.semi_automatic_solver.decoder.GEPP.isSolved()

    def get_ui_elements(self):
        return {"btn-analyze-missing-row": {"type": "button", "text": "Analyze",
                                            "callback": self.parse, "updates_b": False, "refresh_view": True},
                "btn-missing-row-repair": {"type": "button", "text": "Automatic Repair",
                                           "callback": self.repair, "updates_b": False},
                "btn-commit-added-rows": {"type": "button", "text": "Commit added rows to initial GEPP",
                                          "callback": self.commit_rows, "updates_b": False},
                "txt-missing-row-row_num": {"type": "int",
                                            "text": "Row to manually update",
                                            "default": 0, "callback": self.update_num_repair, "updates_b": False},
                "txt-missing-row-row": {"type": "text",
                                        "text": "Row content to manually update (as HEX)",
                                        "default": "", "callback": self.update_repair_content, "updates_b": False},
                }

    def commit_rows(self, *args, **kwargs):
        for i, row in enumerate(self.added_rows):
            added_row_a = np.zeros(self.semi_automatic_solver.decoder.GEPP.A.shape[1], dtype=np.bool)
            added_row_a[self.added_rows[row]] = True
            self.semi_automatic_solver.initial_A = np.vstack((self.semi_automatic_solver.initial_A, added_row_a))
            self.semi_automatic_solver.initial_b = np.vstack(
                (self.semi_automatic_solver.initial_b, np.array(self.added_row_content[row], dtype="uint8")))
            self.semi_automatic_solver.decoder.GEPP.A = np.vstack((self.semi_automatic_solver.decoder.GEPP.A, added_row_a))
            self.semi_automatic_solver.decoder.GEPP.b = np.vstack(
                (self.semi_automatic_solver.decoder.GEPP.b, np.array(self.added_row_content[row], dtype="uint8")))
        self.added_rows.clear()
        return {"refresh_view": True, "update_b": True, "info": "Rows commited to the initial GEPP"}

    def update_repair_content(self, *args, **kwargs):
        try:
            self.fill_row_content = bytearray.fromhex(kwargs["c_ctx"].triggered[0]["value"].replace(" ", ""))
        except ValueError:
            return {"refresh_view": False, "update_b": False,
                    "info": "Invalid row content! Content must be a hex string!"}
        if len(self.fill_row_content) != self.semi_automatic_solver.decoder.GEPP.b.shape[1]:
            return {"refresh_view": False, "update_b": False,
                    "info": f"Invalid length of the row content! Content must be exactly {self.semi_automatic_solver.decoder.GEPP.b.shape[1]} bytes long!"}
        return {"refresh_view": False, "update_b": False, "info": "Row content updated"}

    def update_num_repair(self, *args, **kwargs):
        fill_row_num = kwargs["c_ctx"].triggered[0]["value"]
        if fill_row_num == "" or fill_row_num is None:
            return {"refresh_view": False, "update_b": False,
                    "info": f"Invalid row number! row number must be in [0, ..., {self.semi_automatic_solver.decoder.number_of_chunks}]"}
        if fill_row_num < 0:
            self.fill_row_num = 0
        else:
            self.fill_row_num = fill_row_num
        return {"refresh_view": False, "update_b": False, "info": f"Row number set to {self.fill_row_num}"}

    def update_chunk_tag(self, chunk_tag):
        super().update_chunk_tag(chunk_tag)
        self.error_matrix = None  # this could be speed-up?!

    def update_gepp(self, gepp):
        # invalidate error matrix:
        self.gepp = gepp
        self.error_matrix = None
        return self.parse()


mgr = PluginManager()
mgr.register_plugin(MissingRowRepair)
