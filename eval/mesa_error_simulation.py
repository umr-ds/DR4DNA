import argparse
import json
import requests

from NOREC4DNA.invivo_window_decoder import load_fasta

# MESA_URL = 'http://pc12291.mathematik.uni-marburg.de:5000/api/all'


MESA_URL = 'http://192.168.0.48:5000/api/all'


def mutate_sequence(seq: str, payload: str) -> str:
    header = {'content-type': 'application/json;charset=UTF-8'}
    payload['sequence'] = seq
    try:
        res = requests.post(MESA_URL, data=json.dumps(payload), headers=header)
        return [x for x in res.json().values()][0]["res"]["modified_sequence"].replace(" ", "")
    except Exception as ex:
        print("Error occuered during API-Call:", ex)


def main(filename: str, apikey: str, mesa_config_filename: str) -> None:
    title_to_seq = {}

    with open(mesa_config_filename, "r") as fp:
        payload = json.load(fp)
    payload['asHTML'] = False
    payload['key'] = apikey
    payload['retention_time'] = 0
    payload["do_max_expect"] = False

    with open(filename, "r") as fp:
        # parse fasta file:
        for line in fp.readlines():
            if line.startswith(">"):
                title = line[1:].strip()
                title_to_seq[title] = ""
            else:
                title_to_seq[title] += line.strip()
    res = {}
    for key, seq in title_to_seq.items():
        res[key] = mutate_sequence(seq, payload)
    with open(f"{filename.replace('.fasta', '_mutated')}.fasta", "w") as fp:
        for key, seq in res.items():
            fp.write(f">{key}\n{seq}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", metavar="apikey", type=str, help="MESA api key")
    parser.add_argument("--fasta", metavar="fasta", type=str, help="fasta file")
    parser.add_argument("--mesa_config", metavar="mesa_config", type=str, help="mesa config file")
    args = parser.parse_args()
    apikey = args.apikey
    fasta_filename = args.fasta
    mesa_config_filename = args.mesa_config
    main(fasta_filename, apikey, mesa_config_filename)

    org_fasta = load_fasta(r"NOREC4DNA_17.zip_RU10.fasta")
    mut_fasta = load_fasta(r"NOREC4DNA_17.zip_RU10_mutated.fasta")

    for val in mut_fasta.values():
        if val not in org_fasta.values():
            print(f"{val} not in org_fasta")
