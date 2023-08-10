import random
from NOREC4DNA.invivo_window_decoder import load_fasta
from Levenshtein import distance as levenshtein_distance

from NOREC4DNA.norec4dna import reed_solomon_decode
from NOREC4DNA.norec4dna.helper.bin2Quaternary import string2QUATS
from NOREC4DNA.norec4dna.helper.quaternary2Bin import tranlate_quat_to_byte

"""
to_shrink = load_fasta("/home/michael/PycharmProjects/NOREC4DNA_SAR/005Dorn247_A_assembled_constraint_repaired.fasta")
# randomly choose 71% of the sequences:
to_shrink = random.sample(list(to_shrink.items()), k=int(0.71*len(to_shrink)))
to_shrink = dict(to_shrink)
# save to file:
with open("0071Dorn247_A_assembled_constraint_repaired.fasta", "w") as fp:
    for key, seq in to_shrink.items():
        fp.write(f">{key}\n{seq}\n")

exit(0)
"""


def count_diffs_per_row(fasta1: str, fasta2: str):
    fasta1 = load_fasta(fasta1)
    fasta2 = load_fasta(fasta2)
    """
    # draw 600 entries from fasta2:
    import random
    itms = random.sample(list(fasta2.items()), k=660)
    with open("sair_test.zip_RU10_mutated_660.fasta", "w") as fp:
        for key, seq in itms:
            fp.write(f">{key}\n{seq}\n")
    """
    row2distance, row2min_row = {}, {}
    i_s = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    for title, row in fasta1.items():
        # find the row with the smallest levenstein distance in fasta2:
        min_dist = 10
        min_row = None
        for title2, row2 in fasta2.items():
            dist = levenshtein_distance(row, row2)
            if dist < min_dist:
                min_dist = dist
                min_row = row2
        i_s[min_dist] += 1
        # if min_dist > 0:
        #    print("Title: %s, min_dist: %d, min_row: %s, c_row: %s" % (title, min_dist, min_row, row))
        row2distance[row] = min_dist
        row2min_row[row] = min_row
        if min_dist > 0:
            # the row does have a difference:
            org = check_rs(row, 3)
            org_min_row = min_row
            for i in range(0):
                corrected = check_rs(min_row, 3)
                if corrected is not None and corrected != org:
                    print(
                        f"levenshtein_distance = {levenshtein_distance(row, min_row)} | Found a good error: {row} -> {org_min_row} -> {min_row}")
                    break
                else:
                    min_row = random_mutate(min_row, 1)
    print(i_s)
    return row2distance, row2min_row


def random_mutate(seq: str, n_mutations: int):
    import random
    for i in range(n_mutations):
        pos = random.randint(0, len(seq) - 1)
        seq = seq[:pos] + random.choice("ACGT") + seq[pos + 1:]
    return seq


def check_rs(seq: str, repair_symbols: int = 2):
    try:
        res = reed_solomon_decode(tranlate_quat_to_byte(seq), repair_symbols)
        # string2QUATS(res)
    except:
        res = None
    return res


if __name__ == "__main__":
    # count_diffs_per_row("sb_RU10.fasta", "sb_RU10_mutated.fasta")

    count_diffs_per_row("sb_RU10.fasta", "sb_RU10mutated.fasta")
    #
    # count_diffs_per_row("NOREC4DNA_17.zip_RU10.fasta", "NOREC4DNA_17.zip_RU10_mutated.fasta")
