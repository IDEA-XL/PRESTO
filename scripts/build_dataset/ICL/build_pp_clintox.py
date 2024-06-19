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
RDLogger.DisableLog('rdApp.warning')
import datetime

from bioagent.constants import ROLE_USER, ROLE_SYSTEM

MOLECULE_TOKEN = "<molecule_2d>"

SYSTEM_PROMPT = """You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically whether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule. The FDA-approved status will specify if the drug is approved by the FDA for clinical trials(Yes) or Not approved by the FDA for clinical trials(No). You will be provided with task template. The task is to predict the binary label for a given molecule, please answer with only Yes(1) or No(0).\n"""

FEW_SHOT_PROMPT = "A few examples are provided in the beginning."

# random sampling
def random_sample_examples(data, k:int, label_column:str, smiles_column:str, extra_column:str):
    positive_examples = data[data[label_column] == 1].sample(int(k/2))
    negative_examples = data[data[label_column] == 0].sample(int(k/2))
    smiles = positive_examples[smiles_column].tolist() + negative_examples[smiles_column].tolist()
    
    extra_infos = positive_examples[extra_column].tolist() + negative_examples[extra_column].tolist()
    
    class_label = positive_examples[label_column].tolist() + negative_examples[label_column].tolist()
    #convert 1 to "Yes" and 0 to "No"" in class_label
    class_label = ["1" if i == 1 else "0" for i in class_label]
    examples = list(zip(smiles, extra_infos, class_label))
    return examples

# scaffold sampling
def scaffold_sample_examples(target_smiles:str, data, k:int, label_column:str, smiles_column:str, extra_column:str):
    #drop the target_smiles from the dataset
    data = data[data[smiles_column] != target_smiles]
    molecule_smiles_list = data[smiles_column].tolist()
    extra_infos = data[extra_column].tolist()
    label_list = data[label_column].tolist()
    label_list = ["1" if i == 1 else "0" for i in label_list]

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is not None:
        target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
    else:
        print("Error: Unable to create a molecule from the provided SMILES string.")
        #drop the target_smiles from the dataset
        return None

    target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
    target_fp = rdMolDescriptors.GetMorganFingerprint(target_scaffold, 2)
    RDLogger.DisableLog('rdApp.warning')
    warnings.filterwarnings("ignore", category=UserWarning)
    similarities = []
    
    for i,smiles in enumerate(molecule_smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_fp = rdMolDescriptors.GetMorganFingerprint(scaffold, 2)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(target_fp, scaffold_fp)
            # print(tanimoto_similarity)
            similarities.append((smiles, tanimoto_similarity, extra_infos[i], label_list[i]))
        except:
            continue
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_similar_molecules = similarities[:k]
    # drop out the similarity score
    top_k_similar_molecules = [(smiles, extra, label) for smiles, _, extra, label in top_k_similar_molecules]
    return top_k_similar_molecules


def conversation_test(
    id, 
    input:str, 
    output:Union[str, int], 
    pp_examples:List[Tuple[str, int]],
    extra_info:Union[str, int]
):
    system_prompt = SYSTEM_PROMPT
    # insert few shot examples
    smiles = []
    selfies = []
    for smi, *_ in pp_examples:
        smiles.append(smi)
        try:
            selfies.append(sf.encoder(smi))
        except:
            selfies.append(smi)
    smiles.append(input)
    try:
        selfies.append(sf.encoder(input))
    except:
        selfies.append(input)
    # build few shot examples
    content = ""
    for example in pp_examples:
        content += f"Mol: {MOLECULE_TOKEN}\nFDA-approved: {example[-2]}\nClinically-trail-toxic: {example[-1]}\n"
    content += f"Mol: {MOLECULE_TOKEN}\nFDA-approved: {extra_info}\nClinically-trail-toxic:\n"
        
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
    test_dataset = pd.read_csv(os.path.join(args.test_csv))
    dataset = pd.read_csv(os.path.join(args.train_csv))
    smiles_column = "smiles"
    label_column = "CT_TOX"
    extra_column = "FDA_APPROVED"

    def gen(n_shot:int, sample_strategy:str):
        for index, row in test_dataset.iterrows():
            try:
                target_smiles = row[smiles_column]
                if sample_strategy == "random":
                    pp_examples=random_sample_examples(dataset, n_shot, label_column, smiles_column, extra_column)
                else:
                    pp_examples=scaffold_sample_examples(target_smiles, dataset, n_shot, label_column, smiles_column,extra_column)
                result = conversation_test(
                    index, 
                    input=row[smiles_column], 
                    output=row[label_column], 
                    pp_examples=pp_examples,
                    extra_info=row[extra_column]
                )
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
            "test": {"num_examples": len(test_dataset)}
        }
    }

    dataset_dict = {}
    dataset_split = Dataset.from_generator(gen, gen_kwargs={"n_shot": args.n_shot, "sample_strategy": args.sample_strategy}, num_proc=args.num_proc)
    dataset_dict["test"] = dataset_split
    print(f"{args.n_shot} size: {len(dataset_split)}\n example: {dataset_split[0]}")
    dataset_info["features"] = dataset_dict["test"].features
    dataset_dict = DatasetDict(dataset_dict, info=dataset_info)
    dataset_dict.save_to_disk(args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--n_shot", type=int, default=4)
    parser.add_argument("--sample_strategy", type=str, default="random", choices=["random", "scaffold"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # set random seed
    random.seed(args.seed)
    main(args)

"""
sample_strategy=scaffold
n_shot=4
python build_pp_clintox.py --test_csv /cto_labs/AIDD/DATA/ChemLLMBench/property_prediction/ClinTox_test.csv --train_csv /cto_labs/AIDD/DATA/ChemLLMBench/property_prediction/ClinTox.csv --out_dir /cto_labs/AIDD/DATA/ChemLLMBench/ICL/pp/clintox_${sample_strategy}_${n_shot}shot_int --n_shot ${n_shot} --sample_strategy ${sample_strategy}
"""