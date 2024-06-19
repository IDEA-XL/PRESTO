import json
import os
import argparse
import random
import pandas as pd

import selfies as sf
from datasets import load_dataset, DatasetDict, Dataset

from presto.constants import ROLE_ASSISTANT, ROLE_USER, ROLE_SYSTEM
from presto.chemistry_tools.reaction import multicomponent_smiles_to_list, list_to_multicomponent_smiles
from presto.chemistry_tools.smiles import convert_to_canonical_smiles

MOLECULE_TOKEN = "<molecule_2d>"

SYSTEM_PROMPT = """You are a chemist. Please follow the instructions to convert the structure to the corresponding name."""

FEW_SHOT_PROMPT = """Here are some examples of name conversion."""

PROMPT_TEMPLATES = [
    {
        "input": "<INPUT> is the SMILES representation of a molecule. What is its IUPAC name?",
        "output": "<OUTPUT>"
    },
    {
        "input": "Convert the SMILES representation of a molecule <INPUT> into IUPAC name.",
        "output": "<OUTPUT>"
    },
    {
        "input": "What is the IUPAC name of the molecule <INPUT> ?",
        "output": "<OUTPUT>"
    },
    {
        "input": "Can you give the IUPAC name of the molecule <INPUT> ?",
        "output": "<OUTPUT>"
    },
    {
        "input": "Please write the IUPAC name of the molecule <INPUT> .",
        "output": "<OUTPUT>"
    },
    {
        "input": "<INPUT> The above is a SMILES representation. Write the IUPAC name of the corresponding molecule.",
        "output": "<OUTPUT>"
    },
    {
        "input": "Determine the IUPAC name for the molecule represented by the following SMILES representation: <INPUT> .",
        "output": "<OUTPUT>"
    },
    {
        "input": "What is the IUPAC name for the molecule whose SMILES representation is <INPUT> ?",
        "output": "<OUTPUT>"
    },
    {
        "input": "Determine the IUPAC name for the molecule denoted by <INPUT> .",
        "output": "<OUTPUT>"
    },
    {
        "input": "Translate the given SMILES formula of a molecule <INPUT> into its IUPAC name.",
        "output": "<OUTPUT>"
    },
    {
        "input": "Provide the IUPAC name for the molecule represented as <INPUT> .",
        "output": "<OUTPUT>"
    },
    {
        "input": "Convert the following SMILES notation <INPUT> into its IUPAC nomenclature.",
        "output": "<OUTPUT>"
    },
    {
        "input": "Turn the given SMILES symbol of a molecule <INPUT> into its respective IUPAC name.",
        "output": "<OUTPUT>"
    }
]

def process_input(input, format = "smiles", token=True):
    smiles = convert_to_canonical_smiles(input)
    selfies = sf.encoder(smiles)
    if token:
        molecule = MOLECULE_TOKEN
    elif format == "smiles":
        molecule = smiles
    elif format == "selfies":
        molecule = selfies
    else:
        raise ValueError(f"Unsupported molecule format: {format}")
    
    return selfies, smiles, molecule

def conversation_train(id, input, output, format = "smiles", token=True):
    selfies, smiles, molecule = process_input(input, format, token)
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<INPUT>", molecule)
    output_template = prompt_template["output"].replace("<OUTPUT>", output)
    
    return {
        "id": id,
        "molecules": {"selfies": selfies, "smiles": smiles},
        "ground_truth": output,
        "messages": [
            {
                "role": ROLE_SYSTEM,
                "content": SYSTEM_PROMPT
            },
            {
                "role": ROLE_USER,
                "content": input_template
            },
            {
                "role": ROLE_ASSISTANT,
                "content": output_template
            }
        ],
    }

def conversation_test(id, input, output, few_shots: list = None, format = "smiles", token=True):
    selfies, smiles, molecule = process_input(input, format, token)
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<INPUT>", molecule)
    
    if not few_shots:
        content = input_template
    else:
        few_shot_examples = "\n".join(
            f"Few-shot example {i+1}: {example['input']} -> {example['output']}" for i, example in enumerate(few_shots)
        )
        content = FEW_SHOT_PROMPT + "\n" + few_shot_examples + "\n" + input_template
        
    return {
        "id": id,
        "molecules": {"selfies": selfies, "smiles": smiles},
        "ground_truth": output,
        "messages": [
            {
                "role": ROLE_SYSTEM,
                "content": SYSTEM_PROMPT
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
    data_files = {
        "train": os.path.join(args.data_dir, "train/name_conversion-s2i.jsonl"),
        "dev": os.path.join(args.data_dir, "dev/name_conversion-s2i.jsonl"),
        "test": os.path.join(args.data_dir, "test/name_conversion-s2i.jsonl")
    }
    dataset = {
        "train": Dataset.from_json(data_files["train"]),
        "dev": Dataset.from_json(data_files["dev"]),
        "test": Dataset.from_json(data_files["test"])
    }
    
    def gen(split):
        for id, item in enumerate(dataset[split]):
            try:
                if split == "train":
                    result = conversation_train(id, item['input'], item['output'], format=args.format, token=args.token)
                elif split == "dev":
                    result = conversation_train(id, item['input'], item['output'], format=args.format, token=args.token)
                elif split == "test":
                    result = conversation_test(id, item['input'], item['output'], generate_few_shot_examples(dataset[split], num_examples=0), format=args.format, token=args.token)
                yield result
            except Exception as e:
                pass

    dataset_dict = {}
    for split in ["train", "dev", "test"]:
        dataset_split = Dataset.from_generator(gen, gen_kwargs={"split": split}, num_proc=args.num_proc)
        dataset_dict[split] = dataset_split
        print(f"{split} size: {len(dataset_dict[split])}\n{split} example: {dataset_dict[split][0]}")

    dataset_dict = DatasetDict(dataset_dict)
    dataset_dict.push_to_hub(args.repo_id, private=args.private)
    if args.output_dir:
        dataset_dict.save_to_disk(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--token", type=bool, default=True)
    parser.add_argument("--format", type=str, default="smiles", choices=["smiles", "selfies"])
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID on the Hugging Face Hub")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for the dataset")
    parser.add_argument("--private", action="store_true", help="Set to make the dataset private on the Hugging Face Hub")
    args = parser.parse_args()
    main(args)
