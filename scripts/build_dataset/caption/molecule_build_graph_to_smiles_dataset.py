"""
Prepare pretrain dataset for stage-1 (graph to smiles).
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
   f'Could you give me the SMILES of this molecule {MOLECULE_TOKEN} ?',
   f'Could you provide the SMILES of this molecule {MOLECULE_TOKEN} ?',
   f'Please give me the SMILES of this molecule {MOLECULE_TOKEN} .',
   f'Provide the SMILES of this molecule {MOLECULE_TOKEN} .',
   f'Given the molecule {MOLECULE_TOKEN}, write its SMILES notation.',
   f'Tell me the SMILES of this molecule: {MOLECULE_TOKEN}',
   f'{MOLECULE_TOKEN} What is the SMILES of this molecule?',
   f"I'd like the SMILES of this molecule {MOLECULE_TOKEN}. Can you do that?",
   f"{MOLECULE_TOKEN} The above is a compound. Could you please tell me the SMILES of it?",
   f"{MOLECULE_TOKEN} The above is a compound. What is the corresponding SMILES representation?",
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
                        "content": row["smiles"]
                    }
                ]
            }

    dataset = Dataset.from_generator(gen, gen_kwargs={"rows": rows}, num_proc=args.num_proc)
    print(f"Generated {len(dataset)} rows\nExample: {dataset[0]}")
    dataset.push_to_hub(args.repo_id, private=args.private)
    if args.output_dir:
        dataset.save_to_disk(args.output_dir)
        print(f"Saved to {args.output_dir}, {len(dataset)} rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", type=str, required=True, help="Path to the QA json file")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID on the Hugging Face Hub")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save the dataset")
    parser.add_argument("--private", action="store_true", help="Set to make the dataset private on the Hugging Face Hub")
    args = parser.parse_args()
    main(args)

# python molecule_build_graph_to_smiles_dataset.py --qa_path /home/ys792/data/open-mol/PubChemSFT/all_clean.json --repo_id Open-Mol/PubChem_SMILES-MMPretrain --output_dir /home/ys792/data/open-mol/SMolInst-NC/g2s_mmchat_smiles