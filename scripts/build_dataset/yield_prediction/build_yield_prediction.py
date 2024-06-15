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

SYSTEM_PROMPT = """You are a chemist. Now you are given a reaction equation. Please predict the possible reagents of the reaction. The reaction equation has the following format:
```
reactant1.reactant2. ... .reactantN>>product
```
The return value should be in range of 0-1. The higher the value, the more likely the reaction is to occur. 
We provide the <REP_1> of the reactions."""

FEW_SHOT_PROMPT = """Here are some examples of reaction equations."""


PROMPT_TEMPLATES = [
    {
        "input": "<MOLECULE> Based on the given chemical reaction, what is the yield ratio of the reaction?",
        "output": "<OUTPUT>",
    },
    {
        "input": "According to the provided chemical equation <MOLECULE>, what is the yield ratio of the reaction?",
        "output": "<OUTPUT>",
    },
    {
        "input": "Given the chemical reaction <MOLECULE>, what is the yield ratio of the reaction?",
        "output": "<OUTPUT>",
    },
    {
        "input": "What is the yield ratio of the reaction <MOLECULE>?",
        "output": "<OUTPUT>",
    },
    {
        "input": "<MOLECULE> Using the chemical reaction information, what is the ratio of the reaction's yield?",
        "output": "<OUTPUT>",
    },
    {
        "input": "<MOLECULE> Referring to the chemical equation, what is the yield ratio of the reaction?",
        "output": "<OUTPUT>",
    },
    {
        "input": "Examining the chemical equation <MOLECULE>, what is the yield ratio of the reaction?",
        "output": "<OUTPUT>",
    },
    {
        "input": "Analyzing the chemical reaction <MOLECULE>, what is the yield ratio?",
        "output": "<OUTPUT>",
    },
    {
        "input": "Please provide the yield ratio of the reaction <MOLECULE>.",
        "output": "<OUTPUT>",
    },
    {
        "input": "Predict the yield ratio of the reaction <MOLECULE>.",
        "output": "<OUTPUT>",
    },
]

def process_reaction_equation(reaction, format = "smiles", token=True)->List[str]:
    smiles_list = multicomponent_smiles_to_list(reaction)
    smiles_list = [convert_to_canonical_smiles(smi) for smi in smiles_list]
    selfies_list = []
    for smi in smiles_list:
        try:
            selfies_list.append(sf.encoder(smi))
        except:
            selfies_list.append(smi)
    if token:
        molecules = ".".join([MOLECULE_TOKEN for _ in range(len(smiles_list))])
    elif format == "smiles":
        molecules = ".".join(smiles_list)
    elif format == "selfies":
        molecules = ".".join(selfies_list)
    else:
        raise ValueError(f"Unsupported molecule format: {format}")
    
    return selfies_list, smiles_list, molecules

def conversation_train(id, reactants, products, output, format = "smiles", token=True):
    react_selfies_list, react_smiles_list, react_molecules = process_reaction_equation(reactants, format, token)
    prod_selfies_list, prod_smiles_list, prod_molecules = process_reaction_equation(products, format, token)
    selfies_list = react_selfies_list + prod_selfies_list
    smiles_list = react_smiles_list + prod_smiles_list
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<MOLECULE>", react_molecules+">>"+prod_molecules)
    output_template = prompt_template["output"].replace("<OUTPUT>", output)
    system_prompt = SYSTEM_PROMPT.replace("<REP_1>", "structure" if token else format.upper())
    
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
                "content": str(output_template)
            }
        ],
    }

def conversation_test(id, reactants, products, output, few_shots: list = None, format = "smiles", token=True):
    react_selfies_list, react_smiles_list, react_molecules = process_reaction_equation(reactants, format, token)
    prod_selfies_list, prod_smiles_list, prod_molecules = process_reaction_equation(products, format, token)
    selfies_list = react_selfies_list + prod_selfies_list
    smiles_list = react_smiles_list + prod_smiles_list
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<MOLECULE>", react_molecules+">>"+prod_molecules)
    output_template = prompt_template["output"].replace("<OUTPUT>", output)
    system_prompt = SYSTEM_PROMPT.replace("<REP_1>", "structure" if token else format.upper())
    
    if not few_shots:
        content = input_template
    else:
        few_shot_examples = "\n".join(
            f"Few-shot example {i+1}: reaction:{example['input']}, reagents:{example['output']}" for i, example in enumerate(few_shots)
        )
        content = FEW_SHOT_PROMPT + "\n" + few_shot_examples + "\n" + input_template
        
    return {
        "id": id,
        "molecules": {"selfies": selfies_list, "smiles": smiles_list},
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
        "train": [
            os.path.join(args.data_dir, "Buchwald-Hartwig/train.json"),
            os.path.join(args.data_dir, "Suzuki-Miyaura/train.json"),
        ],
        "test": [
            os.path.join(args.data_dir, "Buchwald-Hartwig/test_100.json"),
            os.path.join(args.data_dir, "Suzuki-Miyaura/test_100.json"),
        ],
    }
    dataset = {
        "train": Dataset.from_json(data_files["train"]),
        "test": Dataset.from_json(data_files["test"])
    }
    
    def gen(split):
        for id, item in enumerate(dataset[split]):
            rxn, y = item["rxn"], item["y"]
            y = str(round(y, 4))
            reactants, products = rxn.split(">>")
            if split == "train":
                result = conversation_train(id, reactants, products, y, format=args.format, token=args.token)
            elif split == "test":
                # set num_examples to 0 to disable fs-test
                result = conversation_test(id, reactants, products, y, generate_few_shot_examples(dataset[split], num_examples=args.few_shot), format=args.format, token=args.token)
            yield result

    # Create dataset info dictionary
    dataset_info = {
        "description": "Forward synthesis dataset for SMolInstruct",
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

# python build_yield_prediction.py --data_dir /cto_labs/AIDD/DATA/yields --out_dir /cto_labs/AIDD/DATA/yields/mmchat_smiles --num_proc 4