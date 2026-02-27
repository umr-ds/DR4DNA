"""
Using the currently loaded file, this plugin analyzes the required tags (valid and invalid) for a given packet and
stores the results in a json file.

Running this script on its own will parse the selected json file and produce a csv in the analysis folder called
"output_count.csv" containing the average number of possible packets for each combination of valid and invalid tags.
This csv can be used in the script "analysis/CountRequiredAnalysis.py" for further visualization.

The separation of json and csv files was done to allow for easier and faster analysis of multiple files at once while
preventing loss of raw data.
"""

import csv
import json
import typing

import numpy as np

from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from repair_algorithms.PluginManager import PluginManager


def bool_array_to_index(arr):
    """Return a list of indices where arr is True."""
    return [i for i, x in enumerate(arr) if x]


class CountRequiredTags(FileSpecificRepair):
    """
    Plugin to analyze required tags (valid and invalid) for packets.

    Analyzes the currently loaded file and stores results in a JSON file.
    Can produce CSV output for visualization in CountRequiredAnalysis.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inspect_packet_num = 0
        self.no_permutations = 100
        self.no_inspect_packets = self.gepp.b.shape[0]

    def set_use_header(self, use_header):
        """
        Set whether to use header chunk.

        Args:
            use_header: Boolean indicating if header chunk should be used
        """
        self.use_header_chunk = use_header

    def set_no_inspect_packet(self, *args, **kwargs):
        """
        Set the number of packets to inspect from callback value.

        Args:
            *args: Additional positional arguments
            **kwargs: Keyword arguments containing c_ctx with callback context

        Returns:
            Dictionary with updates_b and refresh_view flags
        """
        try:
            self.inspect_packet_num = int(kwargs["c_ctx"].triggered[0]["value"])
        except (ValueError, TypeError, IndexError):
            print("Error: could not set number of packets to inspect")
        return {"updates_b": False, "refresh_view": False}

    def set_no_permutations(self, *args, **kwargs):
        """
        Set the number of permutations to try from callback value.

        Args:
            *args: Additional positional arguments
            **kwargs: Keyword arguments containing c_ctx with callback context

        Returns:
            Dictionary with updates_b and refresh_view flags
        """
        try:
            self.no_permutations = int(kwargs["c_ctx"].triggered[0]["value"])
        except (ValueError, TypeError, IndexError):
            print("Error: could not set number of permutations to perform")
        return {"updates_b": False, "refresh_view": False}

    def is_compatible(self, meta_info):
        """
        Check if plugin is compatible with file type.

        Args:
            meta_info: File type metadata

        Returns:
            False (this plugin is not file-type specific)
        """
        return False

    def get_ui_elements(self):
        """
        Get UI elements for the plugin.

        Returns:
            Dictionary of UI element configurations
        """
        return {
            "txt-packet-num": {
                "type": "int",
                "text": "Packet to analyze",
                "default": 0,
                "callback": self.set_no_inspect_packet,
            },
            "txt-packet-permutations": {
                "type": "int",
                "text": "# of permutations to try",
                "default": 100,
                "callback": self.set_no_permutations,
            },
            "btn-analyze-for-packet": {
                "type": "button",
                "text": "Analyze required tags for chosen packet",
                "callback": self.analyze_selected_packet,
            },
            "btn-analyze-all-packet": {
                "type": "button",
                "text": "Analyze for all packets",
                "callback": self.analyze_all_packets,
            },
            # "btn-textfile-lt-find-columns": {"type": "button", "text": "Tag (in)correct columns",
            #                                 "callback": self.get_incorrect_columns, "updates_b": False},
            # "btn-textfile-lt-repair": {"type": "button", "text": "Repair", "callback": self.repair,
            #                           "updates_b": True}
        }

    def analyze_selected_packet(self, inspect_num=None, *args, **kwargs):
        """
        Analyze required tags for a selected packet.

        Tests combinations of valid and invalid row tags to determine
        how many corrupt packets are identified for each combination.

        Args:
            inspect_num: Packet number to inspect (default: self.inspect_packet_num)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary mapping (valid_count, invalid_count) tuples to lists of
            corrupt packet indices found for each combination
        """
        if inspect_num is None:
            inspect_num = self.inspect_packet_num
        invalid = self.semi_automatic_solver.get_corrupt_chunks_by_packets(
            [self.inspect_packet_num]
        )  # == invalid_rows
        # calculate valid rows:
        valid = bool_array_to_index(invalid * -1 + 1)
        invalid = bool_array_to_index(invalid)
        res: typing.Dict[str, typing.Any] = {}
        for v in np.arange(0, 20):
            for i in np.arange(0, 20):
                if v == 0 and i == 0:
                    continue
                res[f"({int(v)}, {int(i)})"] = []
                for _ in np.arange(self.no_permutations):
                    valid_rows = np.random.choice(valid, v)
                    invalid_rows = np.random.choice(invalid, i)
                    res[f"({int(v)}, {int(i)})"].append(
                        bool_array_to_index(
                            self.semi_automatic_solver.decoder.GEPP.get_common_packets(
                                invalid_rows,
                                valid_rows,
                                self.semi_automatic_solver.multi_error_packets_mode,
                            )
                        )
                    )
        with open(f"count_{inspect_num}.json", "w") as fp:
            json.dump(res, fp)
        return res

    def analyze_all_packets(self, *args, **kwargs):
        """
        Analyze required tags for all packets.

        Runs analyze_selected_packet for each packet and saves results.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary mapping packet numbers to analysis results
        """
        res: typing.Dict[str, typing.Any] = {}
        for i in np.arange(0, self.semi_automatic_solver.decoder.number_of_chunks):
            res[int(i)] = self.analyze_selected_packet(int(i), as_json=False)
        with open(f"count_all.json", "w") as fp:
            json.dump(res, fp)
        return res


mgr = PluginManager()
mgr.register_plugin(CountRequiredTags)

if __name__ == "__main__":
    with open("../count_all.json", "r") as fp:
        js = json.load(fp)
    res: typing.Dict[str, typing.Any] = {}
    for chosen_packet in js.keys():
        res[chosen_packet] = {}
        for tpl in js[chosen_packet].keys():
            res[chosen_packet][tpl] = 0
            tmp = 0
            for it in js[chosen_packet][tpl]:
                tmp += len(it)
            res[chosen_packet][tpl] = 1.0 * tmp / len(js[chosen_packet][tpl])

    with open("../analysis/output_count.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["selected_packet", "valid", "invalid", "avg_degree"]
        writer.writerow(header)

        # Write the data rows
        for row_key, row_data in res.items():
            row = [row_key] + list(row_data.values())
            for key, val in row_data.items():
                valid, invalid = key.replace("(", "").replace(")", "").split(",")
                row = [int(row_key), int(valid), int(invalid), val]
                writer.writerow(row)
