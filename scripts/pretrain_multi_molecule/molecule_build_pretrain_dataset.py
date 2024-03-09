import os
import random
import argparse
import json

from datasets import Dataset
from rdkit import Chem
import selfies as sf

from bioagent.constants import ROLE_ASSISTANT, ROLE_USER, ROLE_SYSTEM

MOLECULE_TOKEN = "<molecule_2d>"

SYSTEM_PROMPT = """You are a chemist. Now you are given a representation of a molecule. Please help me to understand the molecule.
The molecule representation <REPRESENTATION> is extracted from a strong molecule encoder."""

SELFIES_PHRASES = [
    f"Could you give me the SELFIES representation of {MOLECULE_TOKEN}?",
    f"What is the SELFIES representation of {MOLECULE_TOKEN}?",
    f"Please provide the SELFIES representation of {MOLECULE_TOKEN}.",
    f"Give me the SELFIES representation of {MOLECULE_TOKEN}.",
    f"SELFIES representation of {MOLECULE_TOKEN}?",
]

SELFIES_ANSWER_PHRASES = "The SELFIES is {selfies}."


def load_dataset(qa_path):
    assert os.path.exists(qa_path), f"{qa_path} not exists"
    with open(os.path.join(qa_path, "CID2text.json"), "rt") as f:
        cid2text = json.load(f)
    dataset = []
    with open(os.path.join(qa_path, "CID2SMILES.csv"), "rt") as f:
        f.readline()
        for line in f.readlines():
            _, cid, smiles = line.strip().split(",")
            if cid not in cid2text or Chem.MolFromSmiles(smiles) is None:
                continue
            try:
                selfies = sf.encoder(smiles)
            except Exception as e:
                print(f"Failed to encode {smiles} with error {e}")
                continue
            dataset.append({
                "cid": cid,
                "smiles": smiles,
                "selfies": selfies,
                "text": cid2text[cid]
            })
    return dataset

def main(args):
    rows = load_dataset(args.qa_path)
    print(f"Loaded {len(rows)} rows")

    def gen(rows):
        for row in rows:
            qrows = random.sample(rows, args.num_choices - 1) + [row]
            ret = {
                "id": int(row["cid"]),
                "molecules": {"smiles": [qrow["smiles"] for qrow in qrows], "selfies": [qrow["selfies"] for qrow in qrows]},
                "messages": [
                    {
                        "role": ROLE_SYSTEM,
                        "content": SYSTEM_PROMPT
                    },
                ]
            }
            for qrow in qrows:
                ret["messages"].append({
                    "role": ROLE_USER,
                    "content": random.choice(SELFIES_PHRASES)
                })
                ret["messages"].append({
                    "role": ROLE_ASSISTANT,
                    "content": SELFIES_ANSWER_PHRASES.format(selfies=qrow["selfies"])
                })
            yield ret

    dataset = Dataset.from_generator(gen, gen_kwargs={"rows": rows}, num_proc=args.num_proc)
    dataset.save_to_disk(args.out_dir)
    print(f"Saved to {args.out_dir}, {len(dataset)} rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--num_choices", type=int, default=5)
    args = parser.parse_args()
    main(args)

# python molecule_build_pretrain_dataset.py --qa_path /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/PubChemSTM_data/raw --out_dir /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/pretrain_multi_molecule --num_choices 5