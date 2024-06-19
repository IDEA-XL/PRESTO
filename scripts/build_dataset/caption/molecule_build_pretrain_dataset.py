"""
Prepare pretrain dataset for stage-1 (caption dataset).
Convert to huggingface dataset format.
"""

import os
import random
import argparse
import json
from tqdm import tqdm

from datasets import Dataset
from rdkit import Chem
import selfies as sf

from presto.constants import ROLE_ASSISTANT, ROLE_USER, ROLE_SYSTEM
from presto.chemistry_tools.smiles import convert_to_canonical_smiles

MOLECULE_TOKEN = "<molecule_2d>"

SYSTEM_PROMPT = """You are a chemist. Now you are given a representation of a molecule. Please help me to understand the molecule."""

PRETRAIN_PHRASES = [
   f'Could you give me a brief overview of this molecule {MOLECULE_TOKEN} ?',
   f'Could you provide a description of this molecule {MOLECULE_TOKEN} ?',
   f'Describe this molecule: {MOLECULE_TOKEN} .',
   f'Please give me some details about this molecule {MOLECULE_TOKEN} .',
   f'Provide a brief overview of this molecule {MOLECULE_TOKEN} .',
   f'Provide a description of this molecule {MOLECULE_TOKEN} .',
   f'Given the molecule {MOLECULE_TOKEN}, what can you tell me about it?',
   f'Tell me something about this molecule: {MOLECULE_TOKEN}',
   f'{MOLECULE_TOKEN} What do you know about the molecule?',
   f"I'd like a short overview about this molecule {MOLECULE_TOKEN}. Can you do that?",
   f"{MOLECULE_TOKEN} The above is a compound. Could you please tell me something about it?"
]


def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_dataset(qa_path):
    assert os.path.exists(qa_path), f"{qa_path} not exists"
    qa = read_json(qa_path)

    dataset = []
    
    smiles_list = list(qa.keys())
    for smiles in tqdm(smiles_list):
        canonical_smiles = convert_to_canonical_smiles(smiles)
        if canonical_smiles is None:
            continue
        # encode to selfies
        try:
            selfies = sf.encoder(canonical_smiles)
        except Exception as e:
            print(f"Failed to encode {canonical_smiles} with error {e}, skip it.")
            continue
        # flatten the history
        history = qa[smiles]
        dataset.append({
            "smiles": canonical_smiles,
            "selfies": selfies,
            "text": [description for _, description in history],
        })
    print(f"filtered out {len(smiles_list) - len(dataset)} invalid smiles")
    return dataset

def main(args):
    rows = load_dataset(args.qa_path)
    print(f"Loaded {len(rows)} rows")

    def gen(rows):
        for row in rows:
            for text in row["text"]:
                yield {
                    "molecules": {"smiles": [row["smiles"]]},
                    "messages": [
                        {
                            "role": ROLE_SYSTEM,
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": ROLE_USER,
                            "content": random.choice(PRETRAIN_PHRASES),
                        },
                        {
                            "role": ROLE_ASSISTANT,
                            "content": text,
                        }
                    ]
                }

    dataset = Dataset.from_generator(gen, gen_kwargs={"rows": rows}, num_proc=args.num_proc)
    dataset.save_to_disk(args.out_dir)
    print(f"Saved to {args.out_dir}, {len(dataset)} rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()
    main(args)

# python scripts/pretrain_molecule/molecule_build_pretrain_dataset.py --qa_path /cto_labs/AIDD/DATA/MolFM/pubchemsft_desc/all_clean.json --out_dir /cto_labs/AIDD/DATA/MolFM/pubchemsft_desc