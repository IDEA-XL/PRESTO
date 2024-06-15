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

MOLECULE_TOKEN = "<molecule_2d>"

SYSTEM_PROMPT = """You are a chemist. Now you are given a representation of a molecule. Please help me to understand the molecule."""

CAPTION_PHRASES = [
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


def main(args):
    train_full = load_dataset("language-plus-molecules/LPM-24_train", split="train")
    train_extra = load_dataset("language-plus-molecules/LPM-24_train-extra", split="train")
    eval_full = load_dataset("language-plus-molecules/LPM-24_eval-caption", split="train")

    def gen(rows, split):
        for row in rows:
            smiles = row["molecule"]
            canonical_smiles = convert_to_canonical_smiles(smiles)
            if canonical_smiles is None:
                if split == "train":
                    continue
                raise ValueError(f"Failed to convert {smiles} to canonical smiles")
            yield {
                    "molecules": {"smiles": [canonical_smiles]},
                    "messages": [
                        {
                            "role": ROLE_SYSTEM,
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": ROLE_USER,
                            "content": random.choice(CAPTION_PHRASES)
                        }
                    ] + ([
                        {
                            "role": ROLE_ASSISTANT,
                            "content": row["caption"]
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

    train_full.push_to_hub("OpenMol/LPM-24_caption", private=args.private)
    train_extra.push_to_hub("OpenMol/LPM-24_caption_extra", private=args.private)

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

# python scripts/pretrain_molecule/molecule_build_lpm_dataset.py