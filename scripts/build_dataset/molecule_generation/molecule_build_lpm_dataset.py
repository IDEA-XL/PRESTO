import os
import random
import argparse
import json
import pprint
from tqdm import tqdm

from datasets import Dataset, load_dataset, DatasetDict
from rdkit import Chem
import selfies as sf

from presto.constants import ROLE_ASSISTANT, ROLE_USER, ROLE_SYSTEM
from presto.chemistry_tools.smiles import convert_to_canonical_smiles

SYSTEM_PROMPT = """You are a chemist. Now you are given a description of a molecule. Please generate a molecule that meets the description."""

GENERATION_PHRASES = [
    'Based on the given information, generate a molecule that meets the desired specifications: <CAPTION>',
    'Give me a molecule that satisfies the conditions outlined in the description: <CAPTION>',
    'Generate a molecule based on this description: <CAPTION>',
    'Can you create a molecule that matches the given characteristics? <CAPTION>',
    'I need a molecule that meets the following conditions: <CAPTION> Please represent the molecule in SMILES.',
    'Suppose there is a molecule that meets the following description: <CAPTION> Please write the SMILES representation of it.',
    '<CAPTION> Use the above information to create a molecule.',
    'Build a molecule that meets the requirement: <CAPTION>',
    'Generate a molecule that fulfills the requirement: <CAPTION>',
    'Conceptualize a molecule that meets the specified attribute(s): <CAPTION>',
    'Come up with a molecule based on the description: <CAPTION>',
    'Could you please return a molecule that adheres to this description? <CAPTION>',
    'I give you a description of a molecule, and you need to return one molecule in SMILES that meets the description. The description: <CAPTION>', 
]


def main(args):
    train_full = load_dataset("language-plus-molecules/LPM-24_train", split="train")
    train_extra = load_dataset("language-plus-molecules/LPM-24_train-extra", split="train")
    eval_full = load_dataset("language-plus-molecules/LPM-24_eval-molgen", split="train")

    def gen(rows, split):
        for row in rows:
            if split == "train":
                smiles = row["molecule"]
                canonical_smiles = convert_to_canonical_smiles(smiles)
                if canonical_smiles is None:
                    if split == "train":
                        continue
                    raise ValueError(f"Failed to convert {smiles} to canonical smiles")
            yield {
                    "messages": [
                        {
                            "role": ROLE_SYSTEM,
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": ROLE_USER,
                            "content": random.choice(GENERATION_PHRASES).replace("<CAPTION>", row["caption"])
                        }
                    ] + ([
                        {
                            "role": ROLE_ASSISTANT,
                            "content": canonical_smiles
                        }
                    ] if split == "train" else [])
                }

    train_full = Dataset.from_generator(gen, gen_kwargs={"rows": train_full, "split": "train"}, num_proc=args.num_proc)
    train_extra = Dataset.from_generator(gen, gen_kwargs={"rows": train_extra, "split": "train"}, num_proc=args.num_proc)
    eval_full = Dataset.from_generator(gen, gen_kwargs={"rows": eval_full, "split": "test"}, num_proc=args.num_proc)

    print(f"train_full: {len(train_full)}, train_extra: {len(train_extra)}, eval_full: {len(eval_full)}")
    pprint.pprint(train_full[0])
    pprint.pprint(train_extra[0])
    pprint.pprint(eval_full[0])

    train_full = DatasetDict({"train": train_full, 'test': eval_full})
    train_extra = DatasetDict({"train": train_extra, 'test': eval_full})

    train_full.push_to_hub("OpenMol/LPM-24_molgen-MMChat", private=args.private)
    train_extra.push_to_hub("OpenMol/LPM-24_molgen_extra-MMChat", private=args.private)

    if args.output_dir:
        train_full.save_to_disk(os.path.join(args.output_dir, "train"))
        train_extra.save_to_disk(os.path.join(args.output_dir, "train-extra"))
        eval_full.save_to_disk(os.path.join(args.output_dir, "eval"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save the dataset")
    parser.add_argument("--private", action="store_true", help="Set to make the dataset private on the Hugging Face Hub")
    args = parser.parse_args()
    main(args)

# python scripts/build_dataset/molecule_generation/molecule_build_lpm_dataset.py --private