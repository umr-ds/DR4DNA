import itertools
import typing
from collections import Counter
from functools import reduce

import norec4dna
import numpy
import numpy as np

from NOREC4DNA.norec4dna import helper
from NOREC4DNA.norec4dna.GEPP import GEPP
from repair_algorithms.FileSpecificRepair import FileSpecificRepair

from repair_algorithms.PluginManager import PluginManager


class RandomShuffleRepair(FileSpecificRepair):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_shuffles = 5
        self.error_matrix = None
        self.file_bytes = None
        self.reconstructed_file_bytes = None
        self.load()
        self.solutions: typing.List[norec4dna.GEPP] = []
        self.perms = []  # permutation used for solutions
        self.calculated_diff_set = None
        self.intersects = None

    def load(self):
        start = 1 if self.use_header_chunk else 0
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

    def repair(self, *args, **kwargs):
        if self.solutions is None or self.solutions == [] or self.chunk_tag is None or self.intersects is None or self.error_matrix is None:
            return {"info": f"Calculate the corrupt packet using 'Find corrupt packet by shuffling' first.",
                    "update_b": False, "refresh_view": True}
        # find a corrupt chunk from chunk_tag and repair it using the diff calculated earlier.
        repaired_rows = 0
        for packet_diff, corrupt_packet in self.intersects.items():
            if len(corrupt_packet) > 1:
                return {
                    "info": f"Found multiple possible corrupt packets for the same diff: {corrupt_packet}. "
                            f"This indicates that the errors are linearly dependent!",
                    "update_b": False, "refresh_view": True}
            corrupt_packet = corrupt_packet[0]
            for i, row in enumerate(self.chunk_tag):
                if row == 1 and self.semi_automatic_solver.decoder.GEPP.chunk_to_used_packets[i, int(corrupt_packet)]:
                    # ensure, this row was reduced using the corrupt_packet
                    new_row_content = helper.xor_numpy(self.semi_automatic_solver.decoder.GEPP.b[i],
                                                       np.frombuffer(packet_diff, dtype="uint8")).astype("uint8")
                    self.semi_automatic_solver.manual_repair(i, int(corrupt_packet), new_row_content)
                    repaired_rows += 1
                    break
        if repaired_rows == 0:
            return {
                "info": f"Could not find a chunk matching the corrupt packet. This usually only happens if the corrupt packet was not used for any row.",
                "update_b": False, "refresh_view": True}
        return {"update_b": True, "refresh_view": True}

    def generate_permutations(self, num_shuffles, input_order, include_original=True):
        rng = np.random.default_rng()
        offset = (1 if include_original else 0)
        res = np.zeros((num_shuffles + offset, self.gepp.b.shape[0]), dtype=np.int32)
        j = 0
        i = offset
        if include_original:
            res[0] = [x for x in input_order]
        while i < num_shuffles + offset:
            tmp = rng.permutation(input_order)
            if not any(np.equal(res, tmp).all(1)):
                res[i] = tmp
                i += 1
            j += 1
            if j > 100 * num_shuffles:
                # make sure we don't get stuck in an infinite loop
                break
        return res

    def find_packet_shuffle(self, *args, **kwargs):
        self.calculated_diff_set = None
        row_to_lin_comb = {}
        # add initial GEPP solution:
        if self.num_shuffles - len(self.solutions) > 0:
            self.perms.extend(
                self.generate_permutations(self.num_shuffles - len(self.solutions), range(0, self.gepp.b.shape[0]),
                                           len(self.perms) == 0))

            for permutation in self.perms[len(self.solutions):]:
                # permutate A and b & solve:
                tmp_gepp = GEPP(self.semi_automatic_solver.initial_A.copy()[permutation],
                                self.semi_automatic_solver.initial_b.copy()[permutation])
                tmp_gepp.solve()

                # ensure that the order is correct / comparable
                tmp_gepp.A = np.squeeze(tmp_gepp.A[tmp_gepp.result_mapping])
                tmp_gepp.b = np.squeeze(tmp_gepp.b[tmp_gepp.result_mapping])
                tmp_gepp.result_mapping = np.arange(tmp_gepp.b.shape[0])
                # store the solution
                self.solutions.append(tmp_gepp)

        # compare the results with the different solutions:
        # for each differing chunk, calculate the symmetric difference of the common packets for that chunk
        # for the total of all generated solutions take the intersection of all the symmetric differences.
        # the resulting packet(s) should contain the corrupt packet.
        correct_incorrect_diff_lst = []
        possible_packets = [[i for i in range(self.solutions[0].b.shape[0])]]
        # remove all packets that were not part of the first solution: (since we would not try to repair them...)
        for packet_num, row in enumerate(self.semi_automatic_solver.decoder.GEPP.chunk_to_used_packets[
                                         :self.semi_automatic_solver.decoder.number_of_chunks].T):
            # chunk i was
            if not any(row) and packet_num in possible_packets[0]:
                possible_packets[0].remove(packet_num)
        self.intersects = {b'base': np.array(possible_packets.copy(), dtype="uint64")}
        correct_packets = set()
        info_str = ""
        # calculate the unique diffs between the solutions:
        # (we only need to compare the first solution with the others!) - since the user only sees the first solution!

        unique_diffs = np.zeros((1, self.semi_automatic_solver.decoder.GEPP.b.shape[1]), dtype="uint8")
        for sol_i, solution in enumerate(self.solutions[1:]):
            diff: numpy.array = np.array([helper.xor_numpy(self.solutions[0].b[i], solution.b[i]) for i in
                                          range(0, self.semi_automatic_solver.decoder.number_of_chunks)], dtype="uint8")
            if not np.any(diff):
                # the solutions are identical
                continue
            unique_diffs = np.unique(np.vstack((unique_diffs, np.unique(diff, axis=0))), axis=0)
        # remove all zero rows:
        unique_diffs = unique_diffs[~np.all(unique_diffs == 0, axis=1)]

        if len(unique_diffs) > 1 and not self.semi_automatic_solver.multi_error_packets_mode:
            return {"info": f"Found multiple diffs between solutions. This indicates multiple corrupt packets. "
                            f"Turn on Multi-Error Mode to find them.", "update_b": False, "refresh_view": True}
        # remove all rows that are linear combinations of others,
        # for this we assume that a row with a low number of diffs is more likely to be a real error and
        # a higher number of diffs is more likely to be a linear combination of other rows.
        # sort rows in unique_diffs by number of columns equal to 0
        num_diff_bytes = np.count_nonzero(unique_diffs, axis=1)
        unique_diffs = unique_diffs[np.argsort(num_diff_bytes * -1)]
        row_mask = np.ones(unique_diffs.shape[0], dtype=bool)
        for i in range(unique_diffs.shape[0]):
            row_mask[i] = False
            res = self.semi_automatic_solver.solve_lin_dep(unique_diffs[row_mask], unique_diffs[i])
            if res is None:
                # the row is not a linear combination of other rows, restore the row and continue
                row_mask[i] = True
                self.intersects[unique_diffs[i].tobytes()] = self.intersects[b'base'].copy()
            else:
                row_to_lin_comb[unique_diffs[i].tobytes()] = res  # save the mapping to speed up later calculations

        for sol_i, solution in enumerate(self.solutions):
            for cmp_sol_i, cmp_solution in enumerate(self.solutions):  # self.solution[sol_i:] instead of all?
                if cmp_sol_i <= sol_i or (
                        np.array_equal(solution.b, cmp_solution.b) and
                        np.array_equal(solution.chunk_to_used_packets[self.perms[sol_i]],
                                       cmp_solution.chunk_to_used_packets[self.perms[cmp_sol_i]])):
                    continue
                for row_i, row in enumerate(solution.b[:self.semi_automatic_solver.decoder.number_of_chunks]):
                    possible_packets_sol = [self.perms[sol_i][i] for i, x in
                                            enumerate(solution.chunk_to_used_packets[row_i]) if x]
                    possible_packets_sol_cmp = [self.perms[cmp_sol_i][i] for i, x in
                                                enumerate(cmp_solution.chunk_to_used_packets[row_i]) if x]
                    if not np.array_equal(row, cmp_solution.b[row_i]):
                        # we found a difference
                        # calculate the symmetric difference:
                        to_append = helper.xor_numpy(row, cmp_solution.b[row_i]).astype('uint8')
                        if len(correct_incorrect_diff_lst) == 0:
                            correct_incorrect_diff_lst = np.array(to_append)
                        else:
                            correct_incorrect_diff_lst = np.unique(np.vstack((np.array(correct_incorrect_diff_lst),
                                                                              to_append)), axis=0)
                        # check if all numpy arrays in correct_incorrect_diff_lst are equal:

                        if len(correct_incorrect_diff_lst.shape) > 1:
                            diff_bytes = to_append
                            if not all([np.array_equal(correct_incorrect_diff_lst[0], x) for x in
                                        correct_incorrect_diff_lst]):
                                info_str = "Found multiple different solutions, this might indicate that there are multiple incorrect packets. "
                        else:
                            if self.calculated_diff_set is None:
                                self.calculated_diff_set = np.empty((0, correct_incorrect_diff_lst.shape[0]),
                                                                    dtype=np.uint8)
                            diff_bytes = correct_incorrect_diff_lst
                        # check if diff_bytes can be described as a linear combination of the other
                        # diffs in self.calculated_diff_set
                        # if so, we have to apply the process for all corrupt packets that are described by the linear combination:
                        diff_bytes_lst = row_to_lin_comb.get(diff_bytes.tobytes(), [diff_bytes.tobytes()])
                        possible_packets = []
                        # calculate the symmetric difference:
                        possible_packets_intersect = np.setxor1d(possible_packets_sol, possible_packets_sol_cmp,
                                                                 assume_unique=True)
                        packet_intersect_for_row = np.intersect1d(possible_packets_sol, possible_packets_sol_cmp,
                                                                  assume_unique=True)
                        possible_packets.append(possible_packets_intersect)
                        for diff_bytes in diff_bytes_lst:
                            if not isinstance(diff_bytes, bytes):
                                diff_bytes = diff_bytes.tobytes()
                            if len(possible_packets) > 1:
                                possible_packets = [reduce(np.intersect1d, possible_packets)]
                            if diff_bytes not in self.intersects.keys():
                                print("WARNING: diff_bytes not in self.intersects.keys()!")
                                self.intersects[diff_bytes] = np.array(possible_packets, dtype=np.uint64)
                            self.intersects[diff_bytes] = np.intersect1d(self.intersects[diff_bytes],
                                                                         possible_packets)
                            self.intersects[diff_bytes] = np.setdiff1d(self.intersects[diff_bytes],
                                                                       np.array(packet_intersect_for_row,
                                                                                dtype=np.uint64),
                                                                       assume_unique=True).tolist()
                    else:  # rows are equal, check if there are packets in possible_packets that are in only ONE of the solutions
                        # if so, remove them from possible_packets
                        # iterate over all rows:
                        # check if the row is equal in both solutions:
                        # check if there are packets in possible_packets that are in only ONE of the solutions
                        # if so, remove them from possible_packets
                        possible_packets_intersect = np.setxor1d(possible_packets_sol, possible_packets_sol_cmp)
                        correct_packets = correct_packets.union(possible_packets_intersect)
                        for diff_bytes in self.intersects.keys():
                            self.intersects[diff_bytes] = np.setdiff1d(self.intersects[diff_bytes],
                                                                       list(correct_packets)).tolist()
                base = self.intersects.pop(b'base')
                if all([len(x) == 0 for x in self.intersects.values()]):
                    return {"info": "Found no viable solution, try multi error mode!", "update_b": False,
                            "refresh_view": True}
                if all([len(intersect) == 1 for intersect in self.intersects.values()]):
                    # calculate error_matrix by iterating over all corrupt packets with their diffs
                    for packet_diff_bytes, incorrect_packet in self.intersects.items():
                        # iterate over all chunks and check if the packet was used:
                        for i, is_affected in enumerate(
                                self.semi_automatic_solver.get_corrupt_chunks_by_packets(incorrect_packet)):
                            if is_affected == 1:
                                self.error_matrix[i] = helper.xor_numpy(self.error_matrix[i].astype("uint8"),
                                                                        np.frombuffer(packet_diff_bytes, dtype="uint8"))
                    # tag all known good packets in the chunk tag:
                    self.chunk_tag = self.semi_automatic_solver.get_corrupt_chunks_by_packets(correct_packets,
                                                                                              self.chunk_tag, tag_num=2)
                    # tag all known bad packets in the chunk_tag:
                    self.chunk_tag = self.semi_automatic_solver.get_corrupt_chunks_by_packets(
                        [y for x in self.intersects.values() for y in x],
                        self.chunk_tag)
                    # in multiple error mode, the main UI will now correctly calculate the possible corrupt packets
                    # by using all packets tagged as correct!
                    # this plugin will additionally know the incorrect packets and their diff to the correct ones
                    return {
                        "info": f"{info_str}Found {len(self.intersects)} corrupt packets: #{[self.intersects[i][0] for i in self.intersects]}",
                        "chunk_tag": self.chunk_tag, "update_b": False, "refresh_view": True}
                else:
                    info_str = f" Only partial solutions found, try increasing the number of permutations: {self.intersects} "
                    self.intersects[b'base'] = base
        if len(possible_packets) > 1 and len(self.intersects) > 1:
            if self.semi_automatic_solver.multi_error_packets_mode:
                pass
            else:
                self.intersects.pop(b'base')
                return {
                    "info": f"{info_str}Found multiple corrupt packets: {self.intersects}. You might want to increase the number of permutations.",
                    "update_b": False, "refresh_view": True}
        if info_str == "":
            info_str = "Found no corrupt packet: The LES seems to be correct (or the corrupt packet cannot be described by a linear combination of other packets)."
            # we could use this information to remove all packets that do HAVE a linear combination of other packets.
        return {"info": info_str, "update_b": False, "refresh_view": True}

    def is_compatible(self, meta_info):
        # upload (offline repair) is always possible...
        rank_augmented_matrix = self.semi_automatic_solver.calculate_rank_augmented_matrix()
        return rank_augmented_matrix > self.semi_automatic_solver.decoder.number_of_chunks

    def get_ui_elements(self):
        return {"btn-shuffle-find-packet": {"type": "button", "text": "Find corrupt packet by shuffling",
                                            "callback": self.find_packet_shuffle, "updates_b": False},
                "btn-shuffle-repair": {"type": "button", "text": "Attempt automatic repair",
                                       "callback": self.repair, "updates_b": False},
                "txt-number-of-shuffles": {"type": "int",
                                           "text": "Number of random permutations to perform",
                                           "default": 5, "callback": self.update_num_shuffle,
                                           "updates_b": False}
                }

    def update_num_shuffle(self, *args, **kwargs):
        num_shuffle = kwargs["c_ctx"].triggered[0]["value"]
        # we could check if kwargs["c_ctx"].triggered[X] has a prop_io equal to the textbox's id
        if num_shuffle < 1:
            self.num_shuffles = self.gepp.b[0]
        else:
            self.num_shuffles = num_shuffle

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

    def update_gepp(self, gepp):
        # invalidate error matrix:
        self.gepp = gepp
        self.error_matrix = None
        self.load()


mgr = PluginManager()
mgr.register_plugin(RandomShuffleRepair)
