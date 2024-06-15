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

SYSTEM_PROMPT_MAP = {
    "reagent_selection": 
        """You are an expert chemist. Given selected one reactant, two reagents and solvent of a Suzuki reaction, predict the optimal reactant that maximize the yield with the rest of reaction components by using your experienced chemical reactant selection knowledge. No explanations and other information. Only return the reactant smiles from the given list. Please strictly follow the format, the template are provided as follows:\nGiven the rest of reaction components:\nreactant: reactant 1\n ligand: ligand 1\n base: base 1\n solvent: solvent 1 \nReactant list: reactant1,reactant2,reactant3,.etc\nOptimal reactant: reactant2 \n\nGiven the rest of reaction components:\nreactant: <REACTANT_1>\nligand: <LIGAND_1>\nsolvent: <SOLVENT_1>\nbase: <BASE_1>  \nReactants list for selection:\n<CANDIDATES_LIST>\nOptimal reactant:\n""",
    "ligand_selection":
        """You are an expert chemist. Given selected two reactants, one reagent and solvent of a Suzuki reaction, predict the optimal ligand that maximize the yield with the rest of reaction components by using your experienced chemical ligand selection knowledge. No explanations and other information. Only return the ligand smiles from the given list. Please strictly follow the format, the template are provided as follows:\nGiven the rest of reaction components:\nreactant: reactant 1\nreactant: reactant 2\nbase: base 1\nsolvent: solvent 1 \nLigand list: ligand1,ligand2,ligand3,.etc\nOptimal reactant: ligand2 \n\nGiven the rest of reaction components:\nreactant 1: <REACTANT_1>\nreactant 2: <REACTANT_2>\nbase: <BASE_1>\nsolvent: <SOLVENT_1>  \nLigand list for selection:\n<CANDIDATES_LIST>\nOptimal ligand:\n""",
    "solvent_selection":
        """You are an expert chemist. Given selected two reactants, two reagents of a Suzuki reaction, predict the optimal solvent that maximize the yield with the rest of reaction components by using your experienced chemical solvent selection knowledge. No explanations and other information. Only return the solvent smiles from the given list. Please strictly follow the format, the template are provided as follows:\nGiven the rest of reaction components:\nreactant: reactant 1\nreactant: reactant 2\nligand: ligand 1\nbase: base 1 \nsolvent list: solvent 1,solvent 2,solvent 3,.etc\nOptimal solvent: solvent 2 \n\nGiven the rest of reaction components:\nreactant 1: <REACTANT_1>\nreactant 2: <REACTANT_2>\nligand: <LIGAND_1>\nbase: <BASE_1>  \nSolvent list for selection:\n<CANDIDATES_LIST>\nOptimal solvent:\n""",
}

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def encode_smiles_to_selfies(smiles_list: List[str]):
    selfies_list = []
    for smiles in smiles_list:
        try:
            selfies = sf.encoder(smiles)
            selfies_list.append(selfies)
        except Exception as e:
            print(f"Failed to encode {smiles} with error {e}, back to smiles.")
            selfies_list.append(smiles)
    return selfies_list

def load_dataset(mapping_dict, split, selection_target:str):
    system_prompt, user_query_template = SYSTEM_PROMPT_MAP[selection_target].split("\n\n")
    rows = mapping_dict[split][selection_target]
    dataset = []
    for i,row in enumerate(rows):
        smiles_list = []
        if selection_target == "reagent_selection":
            user_query = user_query_template.replace("<REACTANT_1>", MOLECULE_TOKEN)
            smiles_list.append(row["reactant"])
            user_query = user_query.replace("<LIGAND_1>", MOLECULE_TOKEN)
            smiles_list.append(row["ligand"])
            user_query = user_query.replace("<SOLVENT_1>", MOLECULE_TOKEN)
            smiles_list.append(row["solvent"])
            user_query = user_query.replace("<BASE_1>", MOLECULE_TOKEN)
            smiles_list.append(row["base"])
        elif selection_target == "ligand_selection":
            user_query = user_query_template.replace("<REACTANT_1>", MOLECULE_TOKEN)
            smiles_list.append(row["reactant"][0])
            user_query = user_query.replace("<REACTANT_2>", MOLECULE_TOKEN)
            smiles_list.append(row["reactant"][1])
            user_query = user_query.replace("<BASE_1>", MOLECULE_TOKEN)
            smiles_list.append(row["base"])
            user_query = user_query.replace("<SOLVENT_1>", MOLECULE_TOKEN)
            smiles_list.append(row["solvent"])
        elif selection_target == "solvent_selection":
            user_query = user_query_template.replace("<REACTANT_1>", MOLECULE_TOKEN)
            smiles_list.append(row["reactant"][0])
            user_query = user_query.replace("<REACTANT_2>", MOLECULE_TOKEN)
            smiles_list.append(row["reactant"][1])
            user_query = user_query.replace("<LIGAND_1>", MOLECULE_TOKEN)
            smiles_list.append(row["ligand"])
            user_query = user_query.replace("<BASE_1>", MOLECULE_TOKEN)
            smiles_list.append(row["base"])
        user_query = user_query.replace("<CANDIDATES_LIST>", ",".join([MOLECULE_TOKEN] * len(row["candidates"])))
        for candidate in row["candidates"]:
            smiles_list.append(candidate)
        assert user_query.count(MOLECULE_TOKEN) == len(smiles_list)
        
        messages = [
            {
                "role": ROLE_SYSTEM,
                "content": system_prompt
            },
            {
                "role": ROLE_USER,
                "content": user_query,
            },
        ]
        if split == "train":
            messages.append(
                {
                    "role": ROLE_ASSISTANT,
                    "content": str(row["optimal_index"])
                }
            )
        dataset.append({
            "task": selection_target,
            "ground_truth": str(row["optimal_index"]),
            "molecules": {"smiles": smiles_list, "selfies": encode_smiles_to_selfies(smiles_list)},
            "messages": messages
        })
    return dataset
    


def main(args):
    mapping_dict = read_json(os.path.join(args.data_dir, "extracted_smiles.json"))
    dataset = {
        "train": [],
        "test": []
    }
    for split in ["train", "test"]:
        for selection_target in ["reagent_selection", "ligand_selection", "solvent_selection"]:
            dataset[split].extend(load_dataset(mapping_dict, split, selection_target))
    
    def gen(split):
        for id, item in enumerate(dataset[split]):
            item["id"] = id
            yield item

    # Create dataset info dictionary
    dataset_info = {
        "description": "Reagent selection instruction dataset",
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
    args = parser.parse_args()
    main(args)

# python build_reagent_selection.py --data_dir /cto_labs/AIDD/DATA/ChemLLMBench/reagent_selection --out_dir /cto_labs/AIDD/DATA/ChemLLMBench/reagent_selection/mmchat_smiles --num_proc 1