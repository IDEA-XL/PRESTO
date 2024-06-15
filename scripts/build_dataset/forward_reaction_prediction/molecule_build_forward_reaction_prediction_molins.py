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
        "output": "A possible product can be <OUTPUT> .",
    },
    {
        "input": "Based on the given reactants and reagents: <MOLECULE>, what product could potentially be produced?",
        "output": "The product can be <OUTPUT> .",
    },
    {
        "input": "Given the following reactants and reagents, please provide a possible product. <MOLECULE>",
        "output": "<OUTPUT> .",
    },
    {
        "input": "<MOLECULE> Given the above reactants and reagents, what could be a probable product of their reaction?",
        "output": "A probable product could be <OUTPUT> .",
    },
    {
        "input": "Please provide a feasible product that could be formed using these reactants and reagents: <MOLECULE> .",
        "output": "<OUTPUT> .",
    },
    {
        "input": "Consider that for a chemical reaction, if <MOLECULE> is/are the reactants and reagents, what can be the product?",
        "output": "<OUTPUT> .",
    },
    {
        "input": "Propose a potential product given these reactants and reagents. <MOLECULE>",
        "output": "<OUTPUT> .",
    },
    {
        "input": "Predict the product of a chemical reaction with <MOLECULE> as the reactants and reagents.",
        "output": "<OUTPUT> .",
    },
    {
        "input": "Can you tell me the potential product of a chemical reaction that uses <MOLECULE> as the reactants and reagents?",
        "output": "Sure. A potential product: <OUTPUT> .",
    },
    {
        "input": "Using <MOLECULE> as the reactants and reagents, tell me the potential product.",
        "output": "<OUTPUT> .",
    },
    {
        "input": "Predict a possible product from the listed reactants and reagents. <MOLECULE>",
        "output": "<OUTPUT> .",
    },
    {
        "input": "<MOLECULE> Considering the given starting materials, what might be the resulting product in a chemical reaction?",
        "output": "<OUTPUT> .",
    },
    {
        "input": "A chemical reaction has started with the substance(s) <MOLECULE> as the reactants and reagents, what could be a probable product?",
        "output": "A probable product: <OUTPUT> .",
    }
]

def process_reaction_equation(reaction, format = "smiles", token=True)->List[str]:
    smiles_list = multicomponent_smiles_to_list(reaction)
    smiles_list = [convert_to_canonical_smiles(smi) for smi in smiles_list]
    selfies_list = [sf.encoder(smi) for smi in smiles_list]
    if token:
        molecules = ".".join([MOLECULE_TOKEN for _ in range(len(smiles_list))])
    elif format == "smiles":
        molecules = ".".join(smiles_list)
    elif format == "selfies":
        molecules = ".".join(selfies_list)
    else:
        raise ValueError(f"Unsupported molecule format: {format}")
    
    return selfies_list, smiles_list, molecules

def conversation_train(id, input, output, format = "smiles", token=True):
    selfies_list, smiles_list, molecules = process_reaction_equation(input, format, token)
    _, _, output = process_reaction_equation(output, format, False)
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<MOLECULE>", molecules)
    output_template = prompt_template["output"].replace("<OUTPUT>", output)
    system_prompt = SYSTEM_PROMPT.replace("<REP_1>", "structure" if token else format.upper()).replace("<REP_2>", format.upper())
    
    return {
        "id": id,
        "molecules": {"selfies": selfies_list, "smiles": smiles_list},
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
        "train": os.path.join(args.data_dir, f"forward_reaction_prediction_train.json"),
        "test": os.path.join(args.data_dir, "forward_reaction_prediction_test.json"),
    }
    dataset = {
        "train": Dataset.from_json(data_files["train"]),
        "test": Dataset.from_json(data_files["test"])
    }
    
    def gen(split):
        for id, item in enumerate(dataset[split]):
            if split == "train":
                result = conversation_train(id, sf.decoder(item['input']), sf.decoder(item['output']), format=args.format, token=args.token)
            elif split == "test":
                # set num_examples to 0 to disable fs-test
                result = conversation_test(id, sf.decoder(item['input']), sf.decoder(item['output']), generate_few_shot_examples(dataset[split], num_examples=args.few_shot), format=args.format, token=args.token)
            yield result

    # Create dataset info dictionary
    dataset_info = {
        "description": "Forward synthesis dataset for MolInstruct",
        "version": "1.0.0",
        "license": "Apache-2.0",
        "splits": {
            "train": {"num_examples": len(dataset["train"])},
            "test": {"num_examples": len(dataset["test"])}
        }
    }

    dataset_dict = {}
    for split in ["train", "test"]:
        dataset_split = Dataset.from_generator(gen, gen_kwargs={"split": split}, num_proc=args.num_proc)
        dataset_dict[split] = dataset_split
        print(f"{split} size: {len(dataset_dict[split])}\n{split} example: {dataset_dict[split][0]}")

    dataset_info["features"] = dataset_dict["test"].features

    dataset_dict = DatasetDict(dataset_dict, info=dataset_info)
    dataset_dict.save_to_disk(args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--token", type=bool, default=True)
    parser.add_argument("--format", type=str, default="smiles", choices=["smiles", "selfies"])
    parser.add_argument("--few_shot", type=int, default=0, help="Number of few-shot examples, set to 0 to disable fs-test")
    args = parser.parse_args()
    main(args)

# python molecule_build_forward_reaction_prediction_molins.py --data_dir /cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions --out_dir /cto_labs/AIDD/DATA/React/MolInstruct/forward_mmchat_smiles --num_proc 4