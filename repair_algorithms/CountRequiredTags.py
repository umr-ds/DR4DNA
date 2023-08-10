import json

import numpy as np
import csv
from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from repair_algorithms.PluginManager import PluginManager


def bool_array_to_index(arr):
    """ returns a list of indices where arr is True """
    return [i for i, x in enumerate(arr) if x]


class CountRequiredTags(FileSpecificRepair):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inspect_packet_num = 0
        self.no_permutations = 100
        self.no_inspect_packets = self.gepp.b.shape[0]

    def set_use_header(self, use_header):
        self.use_header_chunk = use_header

    def set_no_inspect_packet(self, *args, **kwargs):
        try:
            self.inspect_packet_num = int(kwargs["c_ctx"].triggered[0]["value"])
        except:
            print("Error: could not set number of packets to inspect")
        return {"updates_b": False, "refresh_view": False}

    def set_no_permutations(self, *args, **kwargs):
        try:
            self.no_permutations = int(kwargs["c_ctx"].triggered[0]["value"])
        except:
            print("Error: could not set number of permutations to perform")
        return {"updates_b": False, "refresh_view": False}

    def is_compatible(self, meta_info):
        return True

    def get_ui_elements(self):
        return {"txt-packet-num": {"type": "int", "text": "Packet to analyze", "default": 0,
                                   "callback": self.set_no_inspect_packet},
                "txt-packet-permutations": {"type": "int", "text": "# of permutations to try", "default": 100,
                                            "callback": self.set_no_permutations},
                "btn-analyze-for-packet": {"type": "button", "text": "Analyze required tags for chosen packet",
                                           "callback": self.analyze_selected_packet},
                "btn-analyze-all-packet": {"type": "button", "text": "Analyze for all packets",
                                           "callback": self.analyze_all_packets},
                # "btn-textfile-lt-find-columns": {"type": "button", "text": "Tag (in)correct columns",
                #                                 "callback": self.get_incorrect_columns, "updates_b": False},
                # "btn-textfile-lt-repair": {"type": "button", "text": "Repair", "callback": self.repair,
                #                           "updates_b": True}
                }

    def analyze_selected_packet(self, inspect_num=None, as_json=True, *args, **kwargs):
        if inspect_num is None:
            inspect_num = self.inspect_packet_num
        invalid = self.semi_automatic_solver.get_corrupt_chunks_by_packets([self.inspect_packet_num])  # == invalid_rows
        # calculate valid rows:
        valid = bool_array_to_index(invalid * -1 + 1)
        invalid = bool_array_to_index(invalid)
        res = {}
        for v in np.arange(0, 20):
            for i in np.arange(0, 20):
                if v == 0 and i == 0:
                    continue
                res[f"({int(v)}, {int(i)})"] = []
                for perm in np.arange(self.no_permutations):
                    valid_rows = np.random.choice(valid, v)
                    invalid_rows = np.random.choice(invalid, i)
                    res[f"({int(v)}, {int(i)})"].append(
                        bool_array_to_index(self.semi_automatic_solver.decoder.GEPP.get_common_packets(invalid_rows,
                                                                                                       valid_rows,
                                                                                                       self.semi_automatic_solver.multi_error_packets_mode)))
        if as_json:
            with open(f"count_{inspect_num}.json", "w") as fp:
                json.dump(res, fp)
        return res

    def analyze_all_packets(self, *args, **kwargs):
        res = {}
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
    res = {}
    for chosen_packet in js.keys():
        res[chosen_packet] = {}
        for tpl in js[chosen_packet].keys():
            res[chosen_packet][tpl] = 0
            tmp = 0
            for it in js[chosen_packet][tpl]:
                tmp += len(it)
            res[chosen_packet][tpl] = 1.0 * tmp / len(js[chosen_packet][tpl])

    with open('../analysis/output_331.csv', 'w', newline='') as csvfile:
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
