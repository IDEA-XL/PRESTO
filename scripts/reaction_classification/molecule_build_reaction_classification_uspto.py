import json
import os
import argparse
import random
import pandas as pd

import selfies as sf
from datasets import load_dataset, DatasetDict, Dataset

from bioagent.constants import ROLE_ASSISTANT, ROLE_USER, ROLE_SYSTEM
from bioagent.chemistry_tools.reaction import multicomponent_smiles_to_list, list_to_multicomponent_smiles
from bioagent.chemistry_tools.smiles import convert_to_canonical_smiles

MOLECULE_TOKEN = "<molecule_2d>"

SYSTEM_PROMPT = """You are a chemist. Now you are given a reaction equation. Please predict the class of the reaction.
The reaction equation has the following format:
```
reactant1.reactant2. ... .reactantN>>product
```
Your task is to predict the class number. We provide the <REP_1> of the reactants."""

FEW_SHOT_PROMPT = """Here are some examples of reaction equations."""


def process_reaction_equation(reaction, format = "smiles", token=True):
    smiles = multicomponent_smiles_to_list(reaction)
    smiles = [convert_to_canonical_smiles(smi) for smi in smiles]
    assert all(smiles), f"Invalid SMILES: {reaction}"
    try:
        selfies = [sf.encoder(smi) for smi in smiles]
    except:
        selfies = []
    if token:
        molecules = list_to_multicomponent_smiles([MOLECULE_TOKEN for _ in range(len(smiles))])
    elif format == "smiles":
        molecules = list_to_multicomponent_smiles(smiles)
    elif format == "selfies":
        molecules = list_to_multicomponent_smiles(selfies)
    else:
        raise ValueError(f"Unsupported molecule format: {format}")

    return selfies, smiles, molecules

def conversation_train(id, instruction, input, output, format = "smiles", token=True):
    selfies, smiles, molecules = [], [], []
    for part in input.split(">>"):
        part = process_reaction_equation(part, format, token)
        selfies.extend(part[0])
        smiles.extend(part[1])
        molecules.append(part[2])
    molecules = ">>".join(molecules)
    instruction = instruction + "\n" + molecules
    system_prompt = SYSTEM_PROMPT.replace("<REP_1>", format.upper())
    
    return {
        "id": id,
        "molecules": {"selfies": selfies, "smiles": smiles},
        "messages": [
            {
                "role": ROLE_SYSTEM,
                "content": system_prompt
            },
            {
                "role": ROLE_USER,
                "content": instruction
            },
            {
                "role": ROLE_ASSISTANT,
                "content": output
            }
        ],
    }

def conversation_test(id, instruction, input, output, few_shots: list = None, format = "smiles", token=True):
    selfies, smiles, molecules = [], [], []
    for part in input.split(">>"):
        part = process_reaction_equation(part, format, token)
        selfies.extend(part[0])
        smiles.extend(part[1])
        molecules.extend(part[2])
    molecules = ">>".join(molecules)
    instruction = instruction + "\n" + molecules
    system_prompt = SYSTEM_PROMPT.replace("<REP_1>", format.upper())
    
    if not few_shots:
        content = instruction
    else:
        few_shot_examples = "\n".join(
            f"Few-shot example {i+1}: {example['input']} -> {example['output']}" for i, example in enumerate(few_shots)
        )
        content = FEW_SHOT_PROMPT + "\n" + few_shot_examples + "\n" + instruction
        
    return {
        "id": id,
        "molecules": {"selfies": selfies, "smiles": smiles},
        "ground_truth": output,
        "messages": [
            {
                "role": ROLE_SYSTEM,
                "content": system_prompt
            },
            {
                "role": ROLE_USER,
                "content": content
            }
        ],
    }

def generate_few_shot_examples(rows, num_examples=5):
    if not num_examples:
        return None
    return random.sample(sorted(rows, key=lambda x: random.random()), num_examples)

def main(args):
    dataset = load_dataset(args.data_dir)
    
    def gen(split):
        for id, item in enumerate(dataset[split]):
            try:
                if split == "train":
                    result = conversation_train(id, item['instruction'], item['input'], item['output'], format=args.format, token=args.token)
                elif split == "validation":
                    result = conversation_train(id, item['instruction'], item['input'], item['output'], format=args.format, token=args.token)
                elif split == "test":
                    result = conversation_test(id, item['instruction'], item['input'], item['output'], generate_few_shot_examples(dataset[split], num_examples=0), format=args.format, token=args.token)
                yield result
            except Exception as e:
                print(f"invalid example: {e}, id: {id}")
                continue

    # Create dataset info dictionary
    dataset_info = {
        "description": "Forward synthesis dataset for SMolInstruct",
        "version": "1.0.0",
        "license": "Apache-2.0",
        "splits": {
            "train": {"num_examples": len(dataset["train"])},
            "dev": {"num_examples": len(dataset["validation"])},
            "test": {"num_examples": len(dataset["test"])}
        }
    }

    dataset_dict = {}
    for split in ["train", "validation", "test"]:
        dataset_split = Dataset.from_generator(gen, gen_kwargs={"split": split}, num_proc=args.num_proc)
        if split == "validation":
            split = "dev"
        dataset_dict[split] = dataset_split
        print(f"{split} size: {len(dataset_dict[split])}\n{split} example: {dataset_dict[split][0]}")

    dataset_info["features"] = dataset_dict["test"].features

    dataset_dict = DatasetDict(dataset_dict, info=dataset_info)
    dataset_dict.save_to_disk(args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--token", type=bool, default=True)
    parser.add_argument("--format", type=str, default="smiles", choices=["smiles", "selfies"])
    args = parser.parse_args()
    main(args)

# python molecule_build_reaction_classification_uspto.py --data_dir OpenMol/USPTO_1k_TPL-SFT --out_dir /home/ys792/data/mol-llama/SMolInst-Reactions/uspto_tpl_mmchat_smiles