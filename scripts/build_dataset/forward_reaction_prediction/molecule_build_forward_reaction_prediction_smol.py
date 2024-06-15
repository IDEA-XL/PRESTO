import json
import os
import argparse
import random
import pandas as pd
from typing import List
import selfies as sf
from datasets import load_dataset, DatasetDict, Dataset

from presto.constants import ROLE_ASSISTANT, ROLE_USER, ROLE_SYSTEM
from presto.chemistry_tools.reaction import multicomponent_smiles_to_list, list_to_multicomponent_smiles
from presto.chemistry_tools.smiles import convert_to_canonical_smiles

MOLECULE_TOKEN = "<molecule_2d>"

SYSTEM_PROMPT = """You are a chemist. Now you are given a reaction equation. Please predict the product of the reaction. The reaction equation has the following format:
```
reactant1.reactant2. ... .reactantN>>product
```
Your task is to predict the <REP_1> representation of the product molecule. We provide the <REP_2> of the reactants."""

FEW_SHOT_PROMPT = """Here are some examples of reaction equations."""

PROMPT_TEMPLATES = [
    {
        "input": "<MOLECULE> Based on the reactants and reagents given above, suggest a possible product.",
        "output": "<OUTPUT>",
    },
    {
        "input": "Based on the given reactants and reagents: <MOLECULE>, what product could potentially be produced?",
        "output": "<OUTPUT>",
    },
    {
        "input": "Given the following reactants and reagents, please provide a possible product. <MOLECULE>",
        "output": "<OUTPUT>",
    },
    {
        "input": "<MOLECULE> Given the above reactants and reagents, what could be a probable product of their reaction?",
        "output": "<OUTPUT>",
    },
    {
        "input": "Please provide a feasible product that could be formed using these reactants and reagents: <MOLECULE> .",
        "output": "<OUTPUT>",
    },
    {
        "input": "Consider that for a chemical reaction, if <MOLECULE> is/are the reactants and reagents, what can be the product?",
        "output": "<OUTPUT>",
    },
    {
        "input": "Propose a potential product given these reactants and reagents. <MOLECULE>",
        "output": "<OUTPUT>",
    },
    {
        "input": "Predict the product of a chemical reaction with <MOLECULE> as the reactants and reagents.",
        "output": "<OUTPUT>",
    },
    {
        "input": "Can you tell me the potential product of a chemical reaction that uses <MOLECULE> as the reactants and reagents?",
        "output": "<OUTPUT>",
    },
    {
        "input": "Using <MOLECULE> as the reactants and reagents, tell me the potential product.",
        "output": "<OUTPUT>",
    },
    {
        "input": "Predict a possible product from the listed reactants and reagents. <MOLECULE>",
        "output": "<OUTPUT>",
    },
    {
        "input": "<MOLECULE> Considering the given starting materials, what might be the resulting product in a chemical reaction?",
        "output": "<OUTPUT>",
    },
    {
        "input": "A chemical reaction has started with the substance(s) <MOLECULE> as the reactants and reagents, what could be a probable product?",
        "output": "<OUTPUT>",
    }
]

def process_reaction_equation(reaction, format = "smiles", token=True) -> List[str]:
    smiles = multicomponent_smiles_to_list(reaction)
    smiles = [convert_to_canonical_smiles(smi) for smi in smiles]
    selfies = [sf.encoder(smi) for smi in smiles]
    if token:
        molecules = ".".join([MOLECULE_TOKEN for _ in range(len(smiles))])
    elif format == "smiles":
        molecules = ".".join(smiles)
    elif format == "selfies":
        molecules = ".".join(selfies)
    else:
        raise ValueError(f"Unsupported molecule format: {format}")
    
    return selfies, smiles, molecules

def conversation_train(id, input, output, format = "smiles", token=True):
    selfies, smiles, molecules = process_reaction_equation(input, format, token)
    _, _, output = process_reaction_equation(output, format, False)
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<MOLECULE>", molecules)
    output_template = prompt_template["output"].replace("<OUTPUT>", output)
    system_prompt = SYSTEM_PROMPT.replace("<REP_1>", "structure" if token else format.upper()).replace("<REP_2>", format.upper())
    
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
                "content": input_template
            },
            {
                "role": ROLE_ASSISTANT,
                "content": output_template
            }
        ],
    }

def conversation_test(id, input, output, few_shots: list = None, format = "smiles", token=True):
    selfies, smiles, molecules = process_reaction_equation(input, format, token)
    _, _, output = process_reaction_equation(output, format, False)
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<MOLECULE>", molecules)
    system_prompt = SYSTEM_PROMPT.replace("<REP_1>", "structure" if token else format.upper()).replace("<REP_2>", format.upper())
    
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
    data_files = {
        "train": os.path.join(args.data_dir, "train/forward_synthesis.jsonl"),
        "dev": os.path.join(args.data_dir, "dev/forward_synthesis.jsonl"),
        "test": os.path.join(args.data_dir, "test/forward_synthesis.jsonl")
    }
    dataset = {
        "train": Dataset.from_json(data_files["train"]),
        "dev": Dataset.from_json(data_files["dev"]),
        "test": Dataset.from_json(data_files["test"])
    }
    
    def gen(split):
        for id, item in enumerate(dataset[split]):
            if split == "train":
                result = conversation_train(id, item['input'], item['output'], format=args.format, token=args.token)
            elif split == "dev":
                result = conversation_train(id, item['input'], item['output'], format=args.format, token=args.token)
            elif split == "test":
                result = conversation_test(id, item['input'], item['output'], generate_few_shot_examples(dataset[split], num_examples=0), format=args.format, token=args.token)
            yield result

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
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save the dataset")
    parser.add_argument("--private", action="store_true", help="Set to make the dataset private on the Hugging Face Hub")
    args = parser.parse_args()
    main(args)