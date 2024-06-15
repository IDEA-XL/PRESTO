import json
import os
import argparse
import pandas as pd
from typing import Union, List, Tuple

import selfies as sf
from datasets import load_dataset, DatasetDict, Dataset

import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
import warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import csv
import difflib

from presto.constants import ROLE_USER, ROLE_SYSTEM

MOLECULE_TOKEN = "<molecule_2d>"

SYSTEM_PROMPT = """You are an expert chemist. Given the molecular requirements description, your task is to design a new molecule using your experienced chemical Molecular Design knowledge. \nPlease strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the designed molecule. The SMILES must be valid and chemically reasonable. \n"""

FEW_SHOT_PROMPT = "A few examples are provided in the beginning."

def molToCanonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    canonical_smiles = Chem.MolToSmiles(mol)
    return canonical_smiles

def read_data(filename):
    # Open the file for reading
    with open(filename, 'r') as f:
        # Create a CSV reader with tab delimiter
        reader = csv.reader(f, delimiter='\t')
        # Read the data into a list of tuples
        data = [tuple(row) for row in reader]
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

def similarity_ratio(s1, s2):
    # Calculate the similarity ratio between the two strings
    ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
    
    # Return the similarity ratio
    return ratio

def top_n_similar_strings(query:str, candidates:List[str], n:int=5):
    # Calculate the Levenshtein distance between the query and each candidate
    distances = []
    for c in tqdm(candidates):
        distances.append((c, similarity_ratio(query, c)))
    
    # Sort the candidates by their Levenshtein distance to the query
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)
    
    # Get the top n candidates with the smallest Levenshtein distance
    top_candidates = [d[0] for d in sorted_distances[:n]]
    
    # Return the top n candidates
    return top_candidates


def conversation_test(id, input:str, output, examples:List = None):
    system_prompt = SYSTEM_PROMPT
    # no molecule examples
    smiles = []
    selfies = []
    # build few shot examples
    content = ""
    for description, smiles in examples:
        content += f"Molecular requirements description: {description}\nMolecular SMILES: {smiles}\n"
    content += f"Molecular requirements description: {input}\nMolecular SMILES:"
        
    return {
        "id": id,
        "molecules": {"selfies": selfies, "smiles": smiles},
        "ground_truth": str(output),
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

def main(args):
    # read raw data
    folder = args.data_dir
    train = read_data(folder + "train.txt")
    valid = read_data(folder + "validation.txt")
    test = pd.read_csv("/cto_labs/AIDD/DATA/ChemLLMBench/molecule_design/molecule_design_test.csv")
    
    def gen(n_shot:int):
        for index, row in test.iterrows():
            try:
                input_col = 'description'
                output_col = 'SMILES'
                sims = top_n_similar_strings(
                    test.loc[index, input_col], list(train[input_col]), n=n_shot,
                )
                chunk = train[train[input_col].isin(sims)]
                examples = list(zip(chunk[input_col], chunk[output_col]))
                result = conversation_test(index, input=row[input_col], output=row[output_col], examples=examples)
                yield result
            except Exception as e:
                print(f"invalid example: {e}, id: {id}")
                continue

    # Create dataset info dictionary
    dataset_info = {
        "description": "Property Prediction for ICL test",
        "version": "1.0.0",
        "license": "Apache-2.0",
        "splits": {
            "test": {"num_examples": len(test)}
        }
    }

    dataset_dict = {}
    dataset_split = Dataset.from_generator(gen, gen_kwargs={"n_shot": args.n_shot}, num_proc=args.num_proc)
    dataset_dict["test"] = dataset_split
    print(f"{args.n_shot} size: {len(dataset_split)}\n example: {dataset_split[0]}")
    dataset_info["features"] = dataset_dict["test"].features
    dataset_dict = DatasetDict(dataset_dict, info=dataset_info)
    dataset_dict.save_to_disk(args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # set random seed
    random.seed(args.seed)
    main(args)

# python build_molecule_design.py --data_dir /cto_labs/AIDD/DATA/MolT5/ChEBI-20_data/ --out_dir /cto_labs/AIDD/DATA/ChemLLMBench/ICL/molecule_design/3shot --num_proc 1 --n_shot 3 --seed 42