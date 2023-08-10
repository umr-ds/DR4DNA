import copy
import csv
import json
import multiprocessing
import os
import contextlib
import datetime
import shutil
from functools import partial
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from matplotlib.lines import Line2D

from norec4dna import nocode, get_error_correction_encode
from NOREC4DNA.ConfigWorker import ConfigReadAndExecute

from NOREC4DNA.demo_raptor_encode import demo_raptor_encode

packet_count_already_encoded = set()

__chunk_size = 73
__insert_header = True
__file = "box.bmp"
__error_correction = "nocode"
__no_repair_symbols = 0


def create_en_de_coders(file, chunk_size=__chunk_size, error_correction_str="nocode", no_repair_symbols=0,
                        overhead=0.05):
    decoder_res = {}
    while len(decoder_res) < 1:
        error_correction = get_error_correction_encode(error_correction_str, no_repair_symbols)
        encoder_data = demo_raptor_encode.encode(file, asdna=True, chunk_size=chunk_size,
                                                 error_correction=error_correction,
                                                 insert_header=__insert_header, save_number_of_chunks_in_packet=False,
                                                 mode_1_bmp=False, prepend="", append="", upper_bound=2.0,
                                                 save_as_fasta=True, save_as_zip=False, overhead=overhead,
                                                 checksum_len_str="I")

        conf = {'error_correction': error_correction_str, 'repair_symbols': no_repair_symbols, 'asdna': True,
                'number_of_splits': 0, 'read_all': True}
        config_filename = encoder_data.save_config_file(conf, add_dot_fasta=True)
        print("Saved config file: %s" % config_filename)

        decoder_data = ConfigReadAndExecute(config_filename)
        decoders = decoder_data.execute(return_decoder=True, skip_solve=True)
        decoder_res = {}
        for decoder in decoders:
            # create a copy of the GEPP:
            inital_decoder_gepp = copy.deepcopy(decoder.GEPP)
            if not decoder.solve():
                os.remove(decoder.file)
                os.remove(config_filename)
                continue  # skip if not decodable at all...
            res = []
            for i in range(0, len(decoder.GEPP.b)):
                decoder.GEPP = copy.deepcopy(inital_decoder_gepp)
                # remove the i-th row of the GEPP:
                decoder.GEPP.remove_row(i)
                res.append(decoder.solve())
            decoder_res[decoder.file] = (res, decoder.number_of_chunks, decoder.GEPP.b.shape[0])
            # delete the file decoder.file:
            os.remove(decoder.file)
            os.remove(config_filename)
    return decoder_res


if __name__ == "__main__":
    repeats = 100
    res = {}
    num_processors = multiprocessing.cpu_count() - 4
    pool = multiprocessing.Pool(processes=num_processors)

    for overhead in [0.00042176297, 0.00084352594]:
        # [0.00042194093, 0.00084388186, 0.00126582278, 0.00168776371, 0.00210970464, 0.00253164557,
        #             0.0029535865, 0.00337552743, 0.00379746835, 0.00421940928]:  # np.arange(0.001, 0.01, 0.001):
        # [0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3]:
        file_size = os.stat(__file).st_size
        number_of_chunks = ceil(1.0 * file_size / __chunk_size) + (1 if __insert_header else 0)
        packets_to_generate = ceil(1.0 * number_of_chunks * (1 + overhead))
        if packets_to_generate in packet_count_already_encoded:
            continue
        else:
            packet_count_already_encoded.add(packets_to_generate)
        # create a copy of __file for each num_processors:
        to_encode = []
        for i in range(repeats):
            shutil.copy(__file, __file + str(i))
            to_encode.append(__file + str(i))
        single_result_list = pool.map(partial(create_en_de_coders, chunk_size=__chunk_size, overhead=overhead,
                                              error_correction_str=__error_correction, no_repair_symbols=0), to_encode)

        for single_result in single_result_list:
            if len(single_result) == 0:
                continue
            for key in single_result.keys():
                if key in res:
                    res[key].append(single_result[key])
                else:
                    res[key] = [single_result[key]]
    date_time_str = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    with open(f"exp_{date_time_str}_raw.json", "w") as fp:
        json.dump(res, fp)
    parsed_data = {}  # {filename -> { number_of_chunks -> {number_of_rows -> [list of results]}}}
    for key in res.keys():
        parsed_data_filename = {}
        print("File: %s" % key)
        for run in res[key]:
            # print("Run: %s" % str(run))
            lst_results, number_of_chunks, number_of_rows = run
            # run is a tuple of (list of results, number of chunks, number of rows in GEPP)
            if number_of_chunks not in parsed_data_filename:
                parsed_data_filename[number_of_chunks] = {}
            if number_of_rows not in parsed_data_filename[number_of_chunks]:
                parsed_data_filename[number_of_chunks][number_of_rows] = []
            parsed_data_filename[number_of_chunks][number_of_rows].extend(lst_results)
        parsed_data[key] = parsed_data_filename
    # print(parsed_data)
    with open(f"exp_{date_time_str}.json", "w") as fp:
        json.dump(parsed_data, fp)
    for key in parsed_data.keys():
        for number_of_chunks in parsed_data[key].keys():
            for number_of_rows in parsed_data[key][number_of_chunks].keys():
                print(
                    f"{key}, {number_of_chunks}, {number_of_rows}: {sum(parsed_data[key][number_of_chunks][number_of_rows])} _ {len(parsed_data[key][number_of_chunks][number_of_rows])}")

    with open(f"exp_{date_time_str}.csv", 'w', newline='') as csvfile:
        fieldnames = ['filename', 'num_chunks', 'num_rows', 'num_success', 'run']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key in res.keys():
            i = 0
            old_number_of_rows = -1
            for run in res[key]:
                lst_results, number_of_chunks, number_of_rows = run
                if number_of_rows != old_number_of_rows:
                    i = 0
                    old_number_of_rows = number_of_rows
                else:
                    i += 1
                writer.writerow({'filename': key, 'num_chunks': number_of_chunks, 'num_rows': number_of_rows,
                                 'num_success': sum(lst_results), 'run': i})

date_time_str = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
df = pandas.read_csv(f"exp_{date_time_str}.csv")
ax = sns.violinplot(x='num_rows', y='num_success', data=df, scale='count')

# add a horizontal line at the maximum value for each num_rows group
min_num_rows = df['num_rows'].min()
for num_rows, group_data in df.groupby('num_rows'):
    max_value = max(group_data['num_rows'].max(), 0)
    group_violin = ax.collections[num_rows - min_num_rows]
    group_center = ax.get_xticks()[num_rows - min_num_rows]
    group_width = 1.0
    ax.hlines(max_value, group_center - group_width / 2, group_center + group_width / 2, linewidth=1, colors='red')

# add a custom legend
custom_legend = [Line2D([0], [0], color='red', lw=1, label='encoded packets')]
ax.legend(handles=custom_legend)

plt.grid(True)
plt.xlabel("Overhead")
plt.ylabel("#non-critical packets")
plt.gcf().savefig('exp_04_04_2023_15_31_33.pdf', bbox_inches='tight')
plt.gcf().savefig('exp_04_04_2023_15_31_33.svg', format='svg', bbox_inches='tight')
plt.show()
