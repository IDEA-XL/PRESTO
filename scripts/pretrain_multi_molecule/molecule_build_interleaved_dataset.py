import os
import argparse
import json

from datasets import Dataset

MOLECULE_TOKEN = "<molecule_2d>"
PATENT_MOL_TOKEN = "<-#MOL#->"

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_dataset(path):
    assert os.path.exists(path), f"{path} not exists"
    rxn_data = read_json(path)
    dataset = []
    for pid in rxn_data:
        text = rxn_data[pid]["text"].replace(PATENT_MOL_TOKEN, MOLECULE_TOKEN)
        title = "" if rxn_data[pid]["SMILES_title"] is None or str(rxn_data[pid]["SMILES_title"])=='nan' else rxn_data[pid]["SMILES_title"]
        text = " ".join([title, text])
        dataset.append({
            "pid": pid,
            "text": text,
            "smiles": rxn_data[pid]["SMILES_list"],
        })
    return dataset

def main(args):
    rows = load_dataset(args.path)
    print(f"Loaded {len(rows)} rows")

    def gen(rows):
        for row in rows:
            ret = {
                "pid": row["pid"],
                "molecules": {"smiles": row["smiles"]},
                "text": row["text"],
            }
            yield ret

    dataset = Dataset.from_generator(gen, gen_kwargs={"rows": rows}, num_proc=args.num_proc)
    dataset.save_to_disk(args.out_dir)
    print(f"Saved to {args.out_dir}, {len(dataset)} rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()
    main(args)

# python molecule_build_interleaved_dataset.py --path /cto_labs/AIDD/DATA/React/USPTO/uspto_rxn_data_filtered.json --out_dir /cto_labs/AIDD/DATA/React/USPTO/Interleaved