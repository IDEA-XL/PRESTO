import json
import os
import argparse
import random

from datasets import Dataset
import selfies as sf

from bioagent.chemistry_tools.reaction import multicomponent_smiles_to_list, list_to_multicomponent_smiles
from bioagent.constants import ROLE_ASSISTANT, ROLE_USER, ROLE_SYSTEM

MOLECULE_TOKEN = "<molecule_2d>"

SYSTEM_PROMPT = """You are a reagent prediction expert. Now you are given a product molecule and a reaction equation. Please help me to predict the SELFIES representation of the reagent molecule.
The reaction equation has the following format. We also have a special representation of a molecule. It is extracted from a strong molecule encoder.
```
reactant1.reactant2. ... .reactantN>>product1.product2. ... .productM
```
Your task is to predict the SELFIES representation of all the reactant molecules. If there are multiple reactant molecules, please separate them with a period. If there are no reactant molecules, please predict an empty string."""

FEW_SHOT_PROMPT = """Here are some examples of reaction equations."""

FEW_SHOT_TEMPLATE = """Instruction: {instruction}

Input: {input}

Molecule representation of the reaction: {molecules_input}

Output: {output}

Molecule representation of the reagent: {molecules_output}"""

PROMPT_TEMPLATE = """Instruction: {instruction}

Input: {input}

Molecule representation of the reaction: {molecules}"""

OUTPUT_TEMPLATE = """{output}"""


def load_dataset(reaction_data_path):
    assert os.path.exists(reaction_data_path), f"{reaction_data_path} does not exist"

    with open(reaction_data_path, "r") as file:
        rows = json.load(file)
    
    return rows


def process_reaction_equation(reaction):
    reactants, products = reaction.split(">>")
    selfies = multicomponent_smiles_to_list(reactants) + multicomponent_smiles_to_list(products)
    smiles = [sf.decoder(selfie) for selfie in selfies]
    molecules = ".".join([MOLECULE_TOKEN for _ in range(len(reactants.split(".")))]) + ">>" + ".".join([MOLECULE_TOKEN for _ in range(len(products.split(".")))])
    return selfies, smiles, molecules


def conversation_train(id, instruction, input, output):
    selfies, smiles, molecules = process_reaction_equation(input)
    return {
        "id": id,
        "molecules": {"selfies": selfies, "smiles": smiles},
        "messages": [
            {
                "role": ROLE_SYSTEM,
                "content": SYSTEM_PROMPT
            },
            {
                "role": ROLE_USER,
                "content": PROMPT_TEMPLATE.format(instruction=instruction, input=input, molecules=molecules)
            },
            {
                "role": ROLE_ASSISTANT,
                "content": OUTPUT_TEMPLATE.format(output=output)
            }
        ],
    }


def conversation_test(id, instruction, input, output, few_shots: list):
    selfies, smiles = [], []
    for i, row in enumerate(few_shots):
        selfies_, smiles_, molecules_input = process_reaction_equation(row["input"])
        selfies.extend(selfies_)
        smiles.extend(smiles_)
        selfies_, smiles_, molecules_output = process_reaction_equation(row["output"])
        selfies.extend(selfies_)
        smiles.extend(smiles_)
        few_shots[i] = {
            "input": row["input"],
            "output": row["output"],
            "molecules_input": molecules_input,
            "molecules_output": molecules_output
        }
        
    selfies_, smiles_, molecules = process_reaction_equation(input)
    selfies.extend(selfies_)
    smiles.extend(smiles_)

    if not few_shots:
        content = PROMPT_TEMPLATE.format(instruction=instruction, input=input, molecules=molecules)
    else:
        content = FEW_SHOT_PROMPT + "\n" + "\n".join(
            FEW_SHOT_TEMPLATE.format(
                instruction=instruction,
                input=item["input"],
                output=item["output"],
                molecules_input=item["molecules_input"],
                molecules_output=item["molecules_output"]
            ) for item in few_shots
        ) + "\n" + PROMPT_TEMPLATE.format(instruction=instruction, input=input, molecules=molecules)
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
    return random.sample(rows, num_examples)

def main(args):
    rows = load_dataset(args.reaction_data_path)
    print(f"Loaded {len(rows)} rows")

    def gen(rows, split="train"):
        for id, item in enumerate(rows):
            if item['metadata']['split'] != split:
                continue

            if split == "train":
                yield conversation_train(
                    id,
                    item['instruction'],
                    item['input'],
                    item['output']
                )
            else:
                yield conversation_test(
                    id,
                    item['instruction'],
                    item['input'],
                    item['output'],
                    generate_few_shot_examples(rows, num_examples=0),
                )

    # train test split based on item['metadata']['split']
    for split in ["train", "test"]:
        dataset = Dataset.from_generator(gen, gen_kwargs={"rows": rows, "split": split}, num_proc=args.num_proc)
        dataset.save_to_disk(os.path.join(args.out_dir, split))

        # display the first 5 examples
        print(dataset[:5])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reaction_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()
    main(args)

# python molecule_build_reagent_prediction_dataset.py --reaction_data_path /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/reagent_prediction.json --out_dir /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/reagent_prediction