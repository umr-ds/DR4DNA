"""Missing row repair plugin for DR4DNA.

This module provides repair functionality for DNA-encoded files with missing rows
(packets) in the linear equation system. When the rank of the matrix is smaller
than the number of chunks, not all chunks can be reconstructed. This plugin helps
identify and manually fill in missing rows.
"""

import numpy as np

from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from repair_algorithms.PluginManager import PluginManager


class MissingRowRepair(FileSpecificRepair):
    """
    Missing row repair plugin for DNA-encoded files.

    This plugin handles cases where the linear equation system cannot be fully
    solved due to missing rows (packets). It identifies missing rows and allows
    users to manually provide content for those rows to complete the system.

    Attributes:
        error_matrix: Matrix tracking error positions
        no_inspect_chunks: Number of chunks to inspect
        missing_rows: Boolean array indicating missing rows
        added_rows: List of row indices that were added
        added_row_content: List of content for added rows
        fill_row_content: Content to fill in a row
        fill_row_num: Row number to fill
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the missing row repair plugin.

        Args:
            *args: Positional arguments passed to parent class
            **kwargs: Keyword arguments passed to parent class
        """
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
        """
        Parse and identify missing rows in the equation system.

        Analyzes the GEPP result mapping to identify rows that couldn't be
        reconstructed (marked as -1) and tags them accordingly.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with chunk_tag updates and count of missing rows
        """
        self.missing_rows = (
            self.semi_automatic_solver.decoder.GEPP.result_mapping == -1
        ).transpose()[0]
        for i in range(len(self.missing_rows)):
            if self.missing_rows[i]:
                self.chunk_tag[i] = 3
            elif self.chunk_tag[i] == 3:
                self.chunk_tag[i] = 0
        return {
            "update_b": False,
            "refresh_view": True,
            "chunk_tag": self.chunk_tag,
            "info": f"Found {sum(self.missing_rows)} missing rows!",
        }

    def set_use_header(self, use_header):
        """
        Set whether to use header chunk for parsing.

        Args:
            use_header: Boolean indicating if header chunk should be used
        """
        self.use_header_chunk = use_header

    def repair(self, *args, **kwargs):
        """
        Repair missing rows by adding user-provided content.

        Validates the provided row content and row number, then adds the row
        to the equation system to help solve missing chunks.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments containing fill_row_content
                and fill_row_num

        Returns:
            Dictionary with repair results and status information
        """
        if len(self.fill_row_content) != self.semi_automatic_solver.decoder.GEPP.b.shape[1]:
            return {
                "refresh_view": False,
                "update_b": False,
                "info": f"Invalid length of the row content! Content must be exactly {self.semi_automatic_solver.decoder.GEPP.b.shape[1]} bytes long!",
            }
        if (
            self.fill_row_num < 0
            or self.fill_row_num >= self.semi_automatic_solver.decoder.GEPP.A.shape[0]
        ):
            return {
                "refresh_view": False,
                "update_b": False,
                "info": f"Invalid row number! Row number must be between 0 and {self.semi_automatic_solver.decoder.GEPP.A.shape[0] - 1}!",
            }
        if self.chunk_tag[self.fill_row_num] != 3 and self.fill_row_num not in self.added_rows:
            return {
                "refresh_view": False,
                "update_b": False,
                "info": "Invalid row number! Row number must be a missing row! (Try 'Analyze' button?!)",
            }
        if self.fill_row_num not in self.added_rows:
            a_row = np.zeros(self.semi_automatic_solver.decoder.GEPP.A.shape[1], dtype=np.bool)
            a_row[self.fill_row_num] = True
            self.added_rows.append(
                self.fill_row_num
            )  # the position in this list represents the position in A and b
            self.added_row_content.append(self.fill_row_content)
            self.semi_automatic_solver.decoder.GEPP.A = np.vstack(
                (self.semi_automatic_solver.decoder.GEPP.A, a_row)
            )
            self.semi_automatic_solver.decoder.GEPP.b = np.vstack(
                (self.semi_automatic_solver.decoder.GEPP.b, self.fill_row_content)
            )
            self.semi_automatic_solver.decoder.GEPP.addRow(a_row, self.fill_row_content)
        else:
            self.semi_automatic_solver.decoder.GEPP.b[
                self.added_rows.index(self.fill_row_num)
            ] = self.fill_row_content
            self.added_row_content[self.added_rows.index(self.fill_row_num)] = np.array(
                self.fill_row_content, dtype="uint8"
            )
        # ideally we should overwrite the initial A and be to prevent a reset from a different plugin but this
        # would counter the purpose of the initial A and b variables...
        self.semi_automatic_solver.decoder.solve(partial=True)
        # update chunk_tag to fix wrong tags after new solve
        self.parse()
        info_str = (
            f"There are still {sum(self.missing_rows)} missing rows."
            if any(self.missing_rows)
            else "All missing rows should be solved now."
        )
        return {
            "update_b": True,
            "refresh_view": True,
            "chunk_tag": self.chunk_tag,
            "info": f"Packet with content for row {self.fill_row_num} was added to the LES. {info_str}",
        }

    def is_compatible(self, meta_info, *args, **kwargs):
        """
        Check if plugin is compatible with the current file state.

        Args:
            meta_info: File type metadata string
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            True if the equation system is not fully solved, False otherwise
        """
        # only activate this module if the gepp did not fully solve the equation system
        # we could alternatively use:
        # all(self.A.sum(axis=1) == 1)
        return not self.semi_automatic_solver.decoder.GEPP.isSolved()

    def get_ui_elements(self):
        """
        Get UI elements for the missing row repair plugin.

        Returns:
            Dictionary of UI element configurations for missing row repair
        """
        return {
            "btn-analyze-missing-row": {
                "type": "button",
                "text": "Analyze",
                "callback": self.parse,
                "updates_b": False,
                "refresh_view": True,
            },
            "btn-missing-row-repair": {
                "type": "button",
                "text": "Automatic Repair",
                "callback": self.repair,
                "updates_b": False,
            },
            "btn-commit-added-rows": {
                "type": "button",
                "text": "Commit added rows to initial GEPP",
                "callback": self.commit_rows,
                "updates_b": False,
            },
            "txt-missing-row-row_num": {
                "type": "int",
                "text": "Row to manually update",
                "default": 0,
                "callback": self.update_num_repair,
                "updates_b": False,
            },
            "txt-missing-row-row": {
                "type": "text",
                "text": "Row content to manually update (as HEX)",
                "default": "",
                "callback": self.update_repair_content,
                "updates_b": False,
            },
        }

    def commit_rows(self, *args, **kwargs):
        """
        Commit added rows to the initial GEPP matrix.

        Permanently adds all manually added rows to the initial A and b matrices
        of the semi-automatic solver.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with refresh flags and confirmation message
        """
        for _i, row in enumerate(self.added_rows):
            added_row_a = np.zeros(
                self.semi_automatic_solver.decoder.GEPP.A.shape[1], dtype=np.bool
            )
            added_row_a[self.added_rows[row]] = True
            self.semi_automatic_solver.initial_A = np.vstack(
                (self.semi_automatic_solver.initial_A, added_row_a)
            )
            self.semi_automatic_solver.initial_b = np.vstack(
                (
                    self.semi_automatic_solver.initial_b,
                    np.array(self.added_row_content[row], dtype="uint8"),
                )
            )
            self.semi_automatic_solver.decoder.GEPP.A = np.vstack(
                (self.semi_automatic_solver.decoder.GEPP.A, added_row_a)
            )
            self.semi_automatic_solver.decoder.GEPP.b = np.vstack(
                (
                    self.semi_automatic_solver.decoder.GEPP.b,
                    np.array(self.added_row_content[row], dtype="uint8"),
                )
            )
        self.added_rows.clear()
        return {"refresh_view": True, "update_b": True, "info": "Rows commited to the initial GEPP"}

    def update_repair_content(self, *args, **kwargs):
        """
        Update the fill row content from hex string input.

        Args:
            *args: Additional positional arguments
            **kwargs: Keyword arguments containing c_ctx with callback context

        Returns:
            Dictionary with status information about the update
        """
        try:
            self.fill_row_content = bytearray.fromhex(
                kwargs["c_ctx"].triggered[0]["value"].replace(" ", "")
            )
        except ValueError:
            return {
                "refresh_view": False,
                "update_b": False,
                "info": "Invalid row content! Content must be a hex string!",
            }
        if len(self.fill_row_content) != self.semi_automatic_solver.decoder.GEPP.b.shape[1]:
            return {
                "refresh_view": False,
                "update_b": False,
                "info": f"Invalid length of the row content! Content must be exactly {self.semi_automatic_solver.decoder.GEPP.b.shape[1]} bytes long!",
            }
        return {"refresh_view": False, "update_b": False, "info": "Row content updated"}

    def update_num_repair(self, *args, **kwargs):
        """
        Update the row number to repair from input value.

        Args:
            *args: Additional positional arguments
            **kwargs: Keyword arguments containing c_ctx with callback context

        Returns:
            Dictionary with status information about the update
        """
        fill_row_num = kwargs["c_ctx"].triggered[0]["value"]
        if fill_row_num == "" or fill_row_num is None:
            return {
                "refresh_view": False,
                "update_b": False,
                "info": f"Invalid row number! row number must be in [0, ..., {self.semi_automatic_solver.decoder.number_of_chunks}]",
            }
        if fill_row_num < 0:
            self.fill_row_num = 0
        else:
            self.fill_row_num = fill_row_num
        return {
            "refresh_view": False,
            "update_b": False,
            "info": f"Row number set to {self.fill_row_num}",
        }

    def update_chunk_tag(self, chunk_tag):
        """
        Update chunk tags and invalidate cached error matrix.

        Args:
            chunk_tag: New chunk tag list
        """
        super().update_chunk_tag(chunk_tag)
        self.error_matrix = None  # this could be speed-up?!

    def update_gepp(self, gepp):
        """
        Update GEPP matrix and re-parse for missing rows.

        Args:
            gepp: New GEPP instance

        Returns:
            Result from parse method
        """
        # invalidate error matrix:
        self.gepp = gepp
        self.error_matrix = None
        return self.parse()


mgr = PluginManager()
mgr.register_plugin(MissingRowRepair)
