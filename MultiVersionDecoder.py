"""
This tool should allow a user to:
1) Decode a file encoded with NOREC4DNA
2) if there are not enough packets to decode the file, the user should get:
    - a list of missing chunks
    - a partial result with \x00 for missing chunks
    - ideally a ranking of the missing chunks based on how many additional chunks could be retrived if it was present
3) view the file (either as hex, image or as a text) and manually select corrupt chunks
    - based on the selected chunks the tool will then suggest which packet(s) might have caused the corruption
    - the used can then request a new decoding with the detected packet removed

Automatic mode:
1) if there are multiple packets with the same packet-id (or very close hamming distance in total):
    - the tool should try each combination of these packets
    - if there are (multiple) checksums in the header chunks, the tool could automatically find the corrupt packets and either:
        - remove them from the decoding because there are still enough packets left to decode the file
        - bruteforce the corrupt chunks until the checksums match (this can be done in parallel and using believe propagation)
2) if there is only a single packet with this id:
    - the tool can only try to bruteforce the corrupt chunks / packets:
        IF WE BRUTEFORCE THE CHUNK WE MIGHT HAVE A PROBLEM IF THE PACKET HAD A MUTATION AT THE START (wrong ID!)
            we can avoid this pitfall by NOT using the chunk-mapping of the corrupt packet!
        IF WE BRUTEFORCE THE PACKET WE CANT DIRECTLY USE THE CRC (we must always perform a belief propagation / gauss elimination) - this is slower
"""
import argparse
import os
import shutil
import struct
import typing
from functools import reduce
from importlib.metadata import metadata
from io import BytesIO
from itertools import combinations
from pathlib import Path
from time import sleep
import numpy as np
import magic
import crcmod

from NOREC4DNA.file_update_coding import reduce_packet_to_chunk
from NOREC4DNA.metadata_coding import parse_metadata_file
from NOREC4DNA.norec4dna.GEPP import GEPP
from NOREC4DNA.norec4dna.RU10Packet import RU10Packet
from NOREC4DNA.norec4dna.helper.RU10Helper import from_true_false_list
from NOREC4DNA.norec4dna.helper.helper_cpu_single_core import xor_numpy

import NOREC4DNA.norec4dna.helper as helper
from NOREC4DNA.ConfigWorker import ConfigReadAndExecute
from NOREC4DNA.norec4dna.HeaderChunk import HeaderChunk
from NOREC4DNA.norec4dna.Packet import Packet
from NOREC4DNA.norec4dna.RU10Decoder import RU10Decoder
from NOREC4DNA.norec4dna.OnlineDecoder import OnlineDecoder
from NOREC4DNA.norec4dna.LTDecoder import LTDecoder
from numpy.linalg import matrix_rank
from NOREC4DNA.invivo_window_decoder import load_fasta
from NOREC4DNA.norec4dna.helper.quaternary2Bin import tranlate_quat_to_byte
from semi_automatic_reconstruction_toolkit import SemiAutomaticReconstructionToolkit


class MultiVersionDecoder(SemiAutomaticReconstructionToolkit):
    def __init__(self, decoder: typing.Union[RU10Decoder, LTDecoder, OnlineDecoder], metadata_list=None):

        super().__init__(decoder)
        self.last_chunk_len_format = "I"
        self.checksum_len_format = None
        self.decoder: typing.Union[RU10Decoder, LTDecoder, OnlineDecoder] = decoder
        decoder.read_all_before_decode = True
        self.headerChunk: typing.Optional[HeaderChunk] = None
        self.decoder.GEPP.insert_tmp()
        self.initial_A = self.decoder.GEPP.A.copy()
        self.initial_b = self.decoder.GEPP.b.copy()
        self.initial_packet_mapping = None  # self.decoder.GEPP.packet_mapping.copy()
        self.multi_error_packets_mode = False
        self.get_versions_in_pool("GAGCCAGTGAGTCGTA")
        if metadata_list is None:
            self.metadata_list = []
        else:
            self.metadata_list = metadata_list

    def get_versions_in_pool(self, base_dna_version_string) -> int:
        """
        Returns the largest version number that is available in the pool for the given base_dna_version_string.
        If no version is available, it should return 0 (base-version only)
        Versions are indexed starting from 0, where version 1 is the FIRST version after the base version.
        """
        res = 0
        fasta_entries = load_fasta(self.decoder.file)
        for seq in fasta_entries.values():
            idx = seq.find(base_dna_version_string)
            if idx != -1 and idx + len(base_dna_version_string) < len(seq):
                version_num = tranlate_quat_to_byte(f"A{seq[idx - 3:idx]}")
                res = max(res, struct.unpack("B", version_num)[0])
        return res

    def get_sequences_for_version(self, base_dna_version_string, version) -> typing.List[str]:
        """
        Returns a list of all sequences in the file that correspond to the given version.
        If no version is available, it should return an empty list.
        """
        res = []
        fasta_entries = load_fasta(self.decoder.file)
        for seq in fasta_entries.values():
            idx = seq.find(base_dna_version_string)
            if idx != -1 and idx + len(base_dna_version_string) < len(seq):
                version_num = tranlate_quat_to_byte(f"A{seq[idx - 3:idx]}")
                if struct.unpack("B", version_num)[0] == version:
                    res.append(seq)
        return res

    def contains_metadata(self, seq, metadata_list=None):
        """
        returns True
        """
        if metadata_list is None:
            metadata_list = self.metadata_list
        return any([metadata in seq for metadata in metadata_list])

    def decode_base_version(self, base_dna_version_string, known_base_file=None):
        """
        Decode the base version (version 0) and store the result on disk. If the base version is already decoded, it should load the result from disk instead of decoding it again.
        """
        """
        if known_base_file is not None and Path(known_base_file).exists():
            print(f"Base version already decoded, loading from {known_base_file}", flush=True)
            # TODO: manipulate decode state to reflect the loaded file (e.g. by loading the header chunk and updating the GEPP state accordingly)
            with open(known_base_file, "rb") as f:
                data = f.read()
            chunk_size = self.decoder.GEPP.chunk_size
            res = [np.frombuffer(data[i: i + chunk_size], dtype=np.uint8) for i in range(0, len(data), chunk_size)]
            iden = np.identity(self.decoder.number_of_chunks)
            self.decoder.GEPP = GEPP(iden, np.frombuffer(res, dtype=np.uint8))
            self.decoder.input_new_packet()
            return "TODO" # TODO
        """
        # a = self.decoder.solve()
        # file_name = self.decoder.saveDecodedFile(return_file_name=True)
        #tmp_packets = self.decoder.packets.copy()
        #self.decoder_bkp = self.decoder
        self.decoder = type(self.decoder).from_config_map(self.decoder.config_map)
        # if we dont have a known base file, we have to decode the base version first
        print("Decoding base version...", flush=True)
        # we MUST filter out packets that contain the base_dna_version_string:
        fasta_entries = load_fasta(self.decoder.file)
        # fasta_seqs = [seq for seq in fasta_entries.values() if base_dna_version_string not in seq]
        # store the fasta_seqs WITH metadata in a temporary list:
        version_seqs = [seq for seq in fasta_entries.values() if self.contains_metadata(seq, [base_dna_version_string])]
        metadata_seqs = [seq for seq in fasta_entries.values() if self.contains_metadata(seq, self.metadata_list)]
        # we must filter out any sequences containing metadata information (otherwise we would have to fallback to DR4DNA to revert the changed content due to the metadata insertion)
        fasta_seqs = [seq for seq in fasta_entries.values() if not self.contains_metadata(seq, self.metadata_list)]
        fasta_seqs = [seq for seq in fasta_seqs if not self.contains_metadata(seq, [base_dna_version_string])]

        # FIX ME: [x for x in fasta_seqs if x in ground_fasta],[x for x in ground_fasta if x not in fasta_seqs ]
        #  it seems like we omit some sequences during creation of the new version!?!? (org. version has 12 sequences not present in the new version!
        #  eg: CATCATCTCTGAAGGGCTTTCGGTTGTATCACGCAATACATCAGTACGATCTGTCTGCACGACACCGACTATAGTGCGGAAGTAAGGACCGATTTGTAGTTCCTCACGGTAATCCTGCTCAGCCCATCCGACCGCTATGAATTTCCGAATCTACAGCGTATTTGTAAT
        #  this may be caused by generation sequences with the same seed (as they are stored as a set they may overwrite the original seq!)
        #  but then again: why does the decoder solve to True?
        #  also: we must enforce that we do not use the "version"-pattern in the base version.
        #
        # FIX ME: ok, it seems like the decoding fails with the "new" ini-file pointing to the old fasta file (which works with the old ini file)!
        #  so it seems like it is a problem with the RU10Decoder instance getting wrong / incorrect values due to the ini file mismatch!=

        # FIX ME: after inserting all known good packets we may try to repair all metadata and version packets using the reduction to chunk 0 (headerchunk) and inserting them after repair

        id_len_format = self.decoder.config_map.get("id_len_format", "")
        crc_len_format = self.decoder.config_map.get("crc_len_format", "")
        packet_len_format = self.decoder.config_map.get("packet_len_format", "")
        for seq in fasta_seqs:
            # revert seed spacing as it is a DNA-based method and thus not part of parse_raw_packet
            seq = self.decoder.revert_seed_spacing(seq, id_len_format)
            packet = self.decoder.parse_raw_packet(BytesIO(tranlate_quat_to_byte(seq)).read(),
                                                   crc_len_format=crc_len_format,
                                                   number_of_chunks_len_format="",
                                                   packet_len_format=packet_len_format,
                                                   id_len_format=id_len_format)
            self.decoder.input_new_packet(packet)
            self.decoder.packets.append(packet)
            if len(self.decoder.packets) >= self.decoder.static_number_of_chunks:
                if res := self.decoder.solve():
                    break
        # self.decoder.GEPP()
        if self.decoder.use_headerchunk:
            self.decoder.populate_header_chunk()
        if self.decoder.headerChunk is not None and self.decoder.headerChunk.file_name is not None:
            try:
                Path(self.decoder.headerChunk.file_name.decode("utf-8")).rename(
                    "v0_" + self.decoder.headerChunk.file_name.decode("utf-8"))
            except FileNotFoundError as e:
                # if the file does not exist, we can safely ignore the error!
                pass
        file_name = self.decoder.saveDecodedFile(
            last_chunk_len_format=self.decoder.config_map.get("last_chunk_len_str", "I"), return_file_name=True)
        Path(file_name).rename("v0_" + file_name)
        return self.decoder

    def decode_to_version(self, base_dna_version_string, version):
        """
        Decodes the file up to the given version. As each version is based on the previous version, this function iteratively decodes each version up to the selected version and stores all intermediate versions on disk.
        Existing versions are loaded from disk and do not need to be decoded again.
        If the version is not in the pool, it should return an error message.

        @param version: the version to decode to
        """
        # versions = []
        # versions.append(self.decode_base_version(base_dna_version_string))
        id_len_format = self.decoder.config_map.get("id_len_format", "")
        for i in range(1, version + 1):
            version_seqs = self.get_sequences_for_version(base_dna_version_string, i)
            solved_chunks: typing.Dict[int, typing.List[RU10Packet]] = {}
            for seq in version_seqs:
                # create packet from seq while ignoring any broken checksum /error correction!
                # FIXME: only revert seed spcaing if seed spacing was used during encoding!
                reverted_dna_str = self.decoder.revert_seed_spacing(seq, id_len_format)
                # TODO: we must correctly handle reed-solomon / crc calculation:
                #  either: 1) recalculate crc / rs for modified packet during encoding such that we do not have to change the code here or
                #  2) keep the encoded packet as is and handle broken crc / rs during decoding (ignore, or check with unchanged version)
                packet = self.decoder.parse_raw_packet(BytesIO(tranlate_quat_to_byte(reverted_dna_str)).read(),
                                                       crc_len_format=self.decoder.config_map.get("crc_len_format", ""),
                                                       number_of_chunks_len_format="",
                                                       packet_len_format=self.decoder.config_map.get(
                                                           "packet_len_format", ""),
                                                       id_len_format=id_len_format)
                used_chunks_list = from_true_false_list(self.decoder.removeAndXorAuxPackets(packet))
                # self.decoder.input_new_packet(packet)
                bin_dna_version_str = tranlate_quat_to_byte(base_dna_version_string)
                find_result = packet.packed_used_packets.find(bin_dna_version_str)
                header_size = packet.get_packet_header_size()

                offset_pos = find_result - header_size + len(bin_dna_version_str)
                reduced = reduce_packet_to_chunk(packet.copy(), self,
                                                 used_chunks_list[0])  # always the first (usually the header chunk!)
                # get the offset of the changed chunk (index / position from the used_chunks_list!) from the reduced packet:
                target_chunk, = struct.unpack("<B", reduced.data[offset_pos: offset_pos + 1])
                zeros_mask = np.zeros(len(packet.data), dtype=np.uint8)
                zeros_mask[offset_pos - len(bin_dna_version_str) - 1:offset_pos + 1] = np.frombuffer(
                    reduced.data[offset_pos - len(bin_dna_version_str) - 1:offset_pos + 1], dtype=np.uint8)
                # xor the packet data with the mask AND the target_chunk to revert the insertion of the version information:
                repaired_data = xor_numpy(packet.data, zeros_mask)
                # TODO: set content (data) of the packet to repaired_data and update decoder accordingly
                res = RU10Packet(repaired_data, packet.used_packets, packet.total_number_of_chunks, packet.id,
                                 read_only=True,
                                 packet_len_format=packet.packet_len_format, crc_len_format=packet.crc_len_format,
                                 number_of_chunks_len_format=packet.number_of_chunks_len_format,
                                 id_len_format=id_len_format,
                                 save_number_of_chunks_in_packet=packet.total_number_of_chunks is None)
                if used_chunks_list[target_chunk] not in solved_chunks:
                    solved_chunks[used_chunks_list[target_chunk]] = []
                # solve to target_chunk and store the result of later parsing
                reduced_packet = reduce_packet_to_chunk(res.copy(), self, used_chunks_list[target_chunk])
                # we must defer packet insertion as we might have split packets due to missing unchanged space for version-string insertion
                solved_chunks[used_chunks_list[target_chunk]].append(reduced_packet)
            # after parsing all version packets, we can combine the data if more than one differing solution for a chunk exists
            # and add the combined packets to the decoder to solve the new file version. For this we must ensure that the chunks for the new version are used instead of the old version!
            # TODO: for this we may replace the affected rows of GEPP.b
            for key, values in solved_chunks.items():
                unique_data_parts = {bytes(v.data) for v in values}
                #if len(unique_data_parts) > 1:
                tmp = np.zeros_like(self.decoder.GEPP.b[key], dtype=np.uint8)
                # we must combine the parts: xor all parts with the original version, then xor them together and add the original version via xor:
                for part in unique_data_parts:
                    tmp = xor_numpy(tmp, xor_numpy(part, self.decoder.GEPP.b[key]))
                insertion_packet = values[0].copy()
                insertion_packet.data = xor_numpy(tmp, self.decoder.GEPP.b[key])
                self.decoder.packets.append(insertion_packet)
                self.decoder.GEPP.b[key] = insertion_packet.data
            # save the new version:
            # TODO: add logic for crc calculation. for now: just ignore the faulty crc in the header!
            file_name = self.decoder.saveDecodedFile(
                last_chunk_len_format=self.decoder.config_map.get("last_chunk_len_str", "I"),
                return_file_name=True, ignore_crc = True, print_to_output=False)
            Path(file_name).rename(f"v{i}_" + file_name)

    @staticmethod
    def solve_lin_dep(a, b):
        """
        Calculates which rows in vector a can be used to create the target b
        @param a: a matrix , where each row is either used to create b or not
        @param b: the target vector
        @return: a list of rows in a that can be used to create b or None if no solution exists
        """
        combs = [[x for x in combinations(a, i)] for i in range(1, min(4, len(a) + 1))]
        for comb in combs:
            for elem in comb:
                if len(elem) > 1:
                    r = reduce(lambda x, y: xor_numpy(x.astype("uint8"), y.astype("uint8")), elem)
                else:
                    r = elem[0]
                if np.array_equal(r.astype('uint8'), b):
                    return [x.astype("uint8") for x in elem]
        return None

    def repair_and_store_by_packet(self, chunk_id, packet_id, hex_value, clear_working_dir=False,
                                   correctness_function=None):
        # this function will be used if we have multiple invalid packets (and corrected chunks) to save multiple version,
        # where each saved version used a different possible packet to repair the chunk.
        bkp_A = self.decoder.GEPP.A.copy()
        bkp_b = self.decoder.GEPP.b.copy()
        self.manual_repair(chunk_id, packet_id, hex_value)
        working_dir = "multi_file_repair"
        if clear_working_dir:
            # delete the folder working_dir if it exists:
            if Path(working_dir).exists():
                shutil.rmtree(working_dir)
            # create the folder working_dir:
            Path(working_dir).mkdir(parents=True, exist_ok=True)
        # we might have to check if header chunk is used!
        self.parse_header("I")
        if self.headerChunk is not None and self.headerChunk.checksum_len_format is not None:
            is_correct = self.is_checksum_correct()
        else:
            if correctness_function is not None:
                is_correct = correctness_function(self.decoder.GEPP.b)
            else:
                is_correct = False
        try:
            filename = self.decoder.saveDecodedFile(return_file_name=True, print_to_output=False)
        except ValueError as ve:
            filename = ve.args[1]
        _file = Path(filename)
        stem = ("CORRECT_" if is_correct else "") + _file.stem + f"_{chunk_id}_{packet_id}"
        _new_file = _file.rename(Path(working_dir + "/" + stem + _file.suffix))
        self.decoder.GEPP.A = bkp_A
        self.decoder.GEPP.b = bkp_b
        return f"{_new_file.name}"


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ini",
        metavar="ini",
        type=str,
        help="config file (ini)",
        default="/home/michael/Code/DR4DNA/eval/sleeping_beauty_no_error.ini",
    )
    # metadata files (comma separated list of files containing metadata sequences that should be expected when decoding):
    # --unwanted_metadata_file /home/michael/Code/DR4DNA/NOREC4DNA/unwanted_meta.fasta
    metadata_arg_group = parser.add_mutually_exclusive_group(required=False)
    metadata_arg_group.add_argument("--metadata_file", metavar="metafile", type=str,
                                    help="file containing metadata in the fasta format")
    metadata_arg_group.add_argument("--metadata", metavar="dmeta", type=str,
                                    help="comma-separated list of metadata DNA sequences")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = init_args()
    # file = "eval/sleeping_beauty_no_error_v1_Thu_Feb__5_13_49_48_2026.ini"
    # file = "eval/sleeping_beauty_no_error.ini"

    file = parsed_args.ini
    if parsed_args.metadata_file is not None:
        # split the arg at "," and parse each file as fasta file, then extract the sequences and store them in a list:
        metadata = []
        for metadata_file in parsed_args.metadata_file.split(","):
            fasta_entries = load_fasta(metadata_file)
            metadata.extend(fasta_entries.values())
    else:
        metadata = parsed_args.metadata.split(",")

    x = ConfigReadAndExecute(file).execute(return_decoder=True)[0]
    semi_automatic_solver = SemiAutomaticReconstructionToolkit(x)
    print(semi_automatic_solver.view_file_with_chunkborders(False, False, "I"), flush=True)
    mv_decoder = MultiVersionDecoder(x, metadata)
    mv_decoder.decode_base_version("GAGCCAGTGAGTCGTA")

    mv_decoder.decode_to_version("GAGCCAGTGAGTCGTA", 1)

    """
    sleep(1)
    print("Enter the rows that are INVALID (as hex; separated by space): ")
    invalid_rows = input().split(" ")
    invalid_rows = [int(i, 16) for i in invalid_rows]

    print("Enter the rows that are VALID (as hex; separated by space): ")
    valid_rows = input().split(" ")
    valid_rows = [int(i, 16) for i in valid_rows]

    common_packets = semi_automatic_solver.decoder.GEPP.get_common_packets(invalid_rows, valid_rows)
    print("potentially invalid Packets:")
    print(" ".join(map(lambda x: "1" if x else "0", common_packets)), flush=True)
    while np.count_nonzero(common_packets == True) > 1:
        rem_possible_chunks = semi_automatic_solver.get_possible_invalid_chunks_from_common_packets(common_packets)
        print("possible invalid chunks:")
        print(" ".join(map(lambda _x: f"{_x[0]:08x}" if _x[1] else "_", enumerate(rem_possible_chunks))), flush=True)

        print(
            "Result unambiguous, enter additional rows that are INVALID (as hex; separated by space), if there are none, just hit [ENTER]: ",
            flush=True)
        tmp_invalid_rows = input()
        if len(tmp_invalid_rows) != 0:
            for new_invalid_line in tmp_invalid_rows.split(" "):
                invalid_rows.append(int(new_invalid_line, 16))
        print(
            "Result unambiguous, enter additional rows that are VALID (as hex; separated by space), if there are none, just hit [ENTER]: ",
            flush=True)
        tmp_valid_rows = input()
        if len(tmp_valid_rows) != 0:
            for new_valid_line in tmp_valid_rows.split(" "):
                valid_rows.append(int(new_valid_line, 16))
        common_packets = semi_automatic_solver.decoder.GEPP.get_common_packets(invalid_rows, valid_rows)
        print(" ".join(map(lambda _X: "1" if _X else "0", common_packets)), flush=True)
        if len(tmp_valid_rows) == 0 and len(tmp_invalid_rows) == 0:
            break
    print("Missing chunks:")
    print(" ".join(map(lambda _x: "1" if _x else "0", semi_automatic_solver.decoder.GEPP.find_missing_chunks())),
          flush=True)
    """
