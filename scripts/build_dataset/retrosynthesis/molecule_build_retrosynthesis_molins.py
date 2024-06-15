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

SYSTEM_PROMPT = """You are a chemist. Now you are given a product molecule. Please predict the the reactant molecules of the reaction.
The reaction equation has the following format:
```
reactant1.reactant2. ... .reactantN>>product
```
Your task is to predict the <REP_1> representation of the reactant molecule. We provide the <REP_2> of the reactants."""

FEW_SHOT_PROMPT = """Here are some examples of reaction equations."""

PROMPT_TEMPLATES = [
    {
        "input": "Based on the given product, provide some plausible reactants that might have been utilized to prepare it. <INPUT>",
        "output": "<OUTPUT>"
    },
    {
        "input": "Can you identify the reactant(s) that might result in the given product <INPUT> ?",
        "output": "<OUTPUT>"
    },
    {
        "input": "Given the following product, please provide possible reactants. <INPUT>",
        "output": "Possible reactant(s): <OUTPUT> ."
    },
    {
        "input": "Do retrosynthesis with the product <INPUT> .",
        "output": "OK. The reactants may be <OUTPUT> ."
    },
    {
        "input": "<INPUT> Given the product provided, propose some possible reactants that could have been employed in its formation.",
        "output": "Here are possible reactants: <OUTPUT> ."
    },
    {
        "input": "To synthesis <INPUT>, what are the possible reactants? Write in the SMILES representation.",
        "output": "<OUTPUT>"
    },
    {
        "input": "Provide the potential reactants that may be used to produce the product <INPUT> .",
        "output": "The potential reactants: <OUTPUT> ."
    },
    {
        "input": "What reactants could lead to the production of the following product? <INPUT>",
        "output": "<OUTPUT>"
    },
    {
        "input": "With the given product <INPUT>, suggest some likely reactants that were used in its synthesis.",
        "output": "<OUTPUT>"
    },
    {
        "input": "Identify possible reactants that could have been used to create the specified product. <INPUT>",
        "output": "<OUTPUT>"
    },
    {
        "input": "Could you tell which reactants might have been used to generate the following product? <INPUT>",
        "output": "<OUTPUT>"
    },
    {
        "input": "Suggest possible substances that may have been involved in the synthesis of the presented compound. <INPUT>",
        "output": "<OUTPUT>"
    },
    {
        "input": "Can you list the reactants that might result in the chemical product <INPUT> ?",
        "output": "<OUTPUT>"
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
    input_template = prompt_template["input"].replace("<INPUT>", molecules)
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
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<INPUT>", molecules)
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
        "train": os.path.join(args.data_dir, f"retrosynthesis_train.json"),
        "test": os.path.join(args.data_dir, "retrosynthesis_test.json")
    }
    dataset = {
        "train": Dataset.from_json(data_files["train"]),
        "test": Dataset.from_json(data_files["test"])
    }
    
    def gen(split):
        for id, item in enumerate(dataset[split]):
            item["input"], item["output"] = sf.decoder(item["input"]), sf.decoder(item["output"])
            if split == "train":
                result = conversation_train(id, item['input'], item['output'], format=args.format, token=args.token)
            elif split == "test":
                result = conversation_test(id, item['input'], item['output'], generate_few_shot_examples(dataset[split], num_examples=args.few_shot), format=args.format, token=args.token)
            yield result

    # Create dataset info dictionary
    dataset_info = {
        "description": "Retrosynthesis dataset for MolInstruct",
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

# python molecule_build_retrosynthesis_molins.py --data_dir /cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions --out_dir /cto_labs/AIDD/DATA/React/MolInstruct/retro_mmchat_smiles --num_proc 4