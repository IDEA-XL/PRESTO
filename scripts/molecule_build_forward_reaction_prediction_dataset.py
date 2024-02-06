import json
import os
import argparse
import random

from datasets import Dataset
import selfies as sf

from bioagent.chemistry_tools.reaction import multicomponent_smiles_to_list, list_to_multicomponent_smiles
from bioagent.constants import ROLE_ASSISTANT, ROLE_USER, ROLE_SYSTEM

MOLECULE_TOKEN = "<molecule_2d>"

SYSTEM_PROMPT = """You are a chemist. Now you are given a reaction equation. Please predict the product of the reaction.
The reaction equation has the following format (<REPRESENTATION> is extracted from a strong molecule encoder):
```
reactant1<REPRESENTATION>.reactant2<REPRESENTATION>. ... .reactantN<REPRESENTATION>>product
```
Your task is to predict the SELFIES representation of the product molecule."""

FEW_SHOT_PROMPT = """Here are some examples of reaction equations."""

FEW_SHOT_TEMPLATE = """Instruction: {instruction}

Input: {input}

Output: {output}"""

PROMPT_TEMPLATE = """Instruction: {instruction}

Input: {input}"""

OUTPUT_TEMPLATE = """Output: {output}"""


def load_dataset(reaction_data_path):
    assert os.path.exists(reaction_data_path), f"{reaction_data_path} does not exist"

    with open(reaction_data_path, "r") as file:
        rows = json.load(file)
    
    return rows


def process_reaction_equation(reaction):
    selfies = multicomponent_smiles_to_list(reaction)
    input = list_to_multicomponent_smiles((sf + f"<{MOLECULE_TOKEN}>" for sf in selfies))
    smiles = [sf.decoder(selfie) for selfie in selfies]
    return input, selfies, smiles


def conversation_train(id, instruction, input, output):
    input, selfies, smiles = process_reaction_equation(input)
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
                "content": PROMPT_TEMPLATE.format(instruction=instruction, input=input)
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
        input_, selfies_, smiles_ = process_reaction_equation(row["input"])
        selfies.extend(selfies_)
        smiles.extend(smiles_)
        output_, selfies_, smiles_ = process_reaction_equation(row["output"])
        selfies.extend(selfies_)
        smiles.extend(smiles_)
        few_shots[i] = {
            "input": input_,
            "output": output_
        }
        
    input, selfies_, smiles_ = process_reaction_equation(input)
    selfies.extend(selfies_)
    smiles.extend(smiles_)

    if not few_shots:
        content = PROMPT_TEMPLATE.format(instruction=instruction, input=input)
    else:
        content = FEW_SHOT_PROMPT + "\n" + "\n".join(
            FEW_SHOT_TEMPLATE.format(
                instruction=instruction,
                input=item["input"],
                output=item["output"]
            ) for item in few_shots
        ) + "\n" + PROMPT_TEMPLATE.format(instruction=instruction, input=input)
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
                    generate_few_shot_examples(rows, num_examples=3)
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


# python molecule_build_forward_reaction_prediction_dataset.py --reaction_data_path /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/forward_reaction_prediction.json --out_dir /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/forward_reaction_prediction_few_shot