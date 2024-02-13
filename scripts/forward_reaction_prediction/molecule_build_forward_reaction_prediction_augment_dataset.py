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
Your task is to predict the SELFIES representation of the product molecule.
Note that you can ask for help from a ChemT5 model, but it is not always correct."""

PROMPT_TEMPLATE = """Instruction: {instruction}

Input: {input}

ChemT5 Output: {output}"""

OUTPUT_TEMPLATE = """Thoughts:
{thoughts}

Output:
{output}"""

CORRECT_THOUGHTS = "ChemT5 model predicts the correct product. I will use it as an answer."
INCORRECT_THOUGHTS = "ChemT5 model predicts the incorrect product. I will not use it as a reference. Instead, I will generate the product by myself."

def load_dataset(reaction_data_path):
    assert os.path.exists(reaction_data_path), f"{reaction_data_path} does not exist"

    with open(reaction_data_path, "r") as file:
        rows = json.load(file)
    
    return rows

def reaction_equation_to_conversation(id, instruction, input, chemT5_output, chemT5_thoughts, output=None, split="train"):
    """
    Convert a reaction equation to a list of messages.
    """
    selfies = multicomponent_smiles_to_list(input)
    input = list_to_multicomponent_smiles((sf + f"<{MOLECULE_TOKEN}>" for sf in selfies))
    try:
        [sf.decoder(selfie) for selfie in multicomponent_smiles_to_list(chemT5_output)]
    except:
        chemT5_output = None
    if chemT5_output is None:
        chemT5_output = "Unable to generate a valid answer."
    else:
        selfies += multicomponent_smiles_to_list(chemT5_output)
        chemT5_output = chemT5_output + f"<{MOLECULE_TOKEN}>"
    smiles = [sf.decoder(selfie) for selfie in selfies]
    ret = {
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
                "content": PROMPT_TEMPLATE.format(instruction=instruction, input=input, output=chemT5_output)
            },
        ],
    }
    if split == "train":
        ret["messages"].append(
            {
                "role": ROLE_ASSISTANT,
                "content": OUTPUT_TEMPLATE.format(thoughts=chemT5_thoughts, output=output)
            }
        )
    return ret

def main(args):
    rows = load_dataset(args.reaction_data_path)
    print(f"Loaded {len(rows)} rows")

    def gen(rows, split="train"):
        for id, item in enumerate(rows):
            if item['metadata']['split'] != split:
                continue

            yield reaction_equation_to_conversation(
                id,
                item['instruction'],
                item['input'],
                item['chemT5_output'],
                CORRECT_THOUGHTS if item['metadata']['chemT5_match'] else INCORRECT_THOUGHTS,
                item['output'],
                item['metadata']['split']
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

# python molecule_build_forward_reaction_prediction_augment_dataset.py --reaction_data_path /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/forward_reaction_prediction_chemT5.json --out_dir /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/forward_reaction_prediction_augmented
