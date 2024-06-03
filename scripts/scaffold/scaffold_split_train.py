import random
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import argparse
import tqdm
import pickle

def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    if not scaffold:
        return None
    return scaffold

def read_files(file_names):
    inputs = []
    outputs = []

    for file_name in file_names:
        with open(f"{file_name}_input.txt", "r") as f:
            inputs += [line.strip() for line in f]

        with open(f"{file_name}_output.txt", "r") as f:
            outputs += [line.strip() for line in f]

    return inputs, outputs

def write_files(file_name, inputs, outputs):
    with open(f"{file_name}_input.txt", "w") as f:
        f.write('\n'.join(inputs))
    
    with open(f"{file_name}_output.txt", "w") as f:
        f.write('\n'.join(outputs))

def save_scaffolds(scaffolds, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(scaffolds, f)

def load_scaffolds(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def split_train_set(train_input, train_output, split_ratio=0.5, scaffold_file='scaffolds.pkl'):
    if args.fs:
        scaffold_file = 'mol_fs_' + scaffold_file
    elif args.rs:
        scaffold_file = 'mol_rs_' + scaffold_file

    scaffold_to_indices = load_scaffolds(scaffold_file)
    if scaffold_to_indices is None:
        scaffold_to_indices = defaultdict(list)
        for i, smiles in tqdm.tqdm(enumerate(train_input), total=len(train_input), desc="Generating scaffolds"):
            scaffold = generate_scaffold(smiles)
            scaffold_to_indices[scaffold].append(i)

    save_scaffolds(scaffold_to_indices, scaffold_file)

    # merge scaffolds with less than int(1/split_ratio) compounds
    scaffold_to_indices_misc = [idx[0] for idx in scaffold_to_indices.values() if len(idx) < int(1/split_ratio)]
    scaffold_to_indices = {scaffold: indices for scaffold, indices in scaffold_to_indices.items() if len(indices) > int(1/split_ratio)}

    print(f"Number of scaffolds: {len(scaffold_to_indices)}")
    print(f"Number of compounds: {len(train_input)}")
    selected_indices = []
    
    for scaffold, indices in scaffold_to_indices.items():
        num_select = round(len(indices) * split_ratio)
        selected = random.sample(indices, num_select)
        selected_indices.extend(selected)

    if len(selected_indices) < int(len(train_input) * split_ratio):
        selected_indices.extend(random.sample(scaffold_to_indices_misc, int(len(train_input) * split_ratio) - len(selected_indices)))

    selected_input = [train_input[i] for i in selected_indices]
    selected_output = [train_output[i] for i in selected_indices]
    
    print(f"Selected train set size: {len(selected_indices)}")
    
    return selected_input, selected_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", action="store_true")
    parser.add_argument("--rs", action="store_true")
    parser.add_argument("--split_ratios", nargs="+", type=float, default=[0.5, 0.25, 0.125])
    args = parser.parse_args()

    split_fractions = ["1_" + str(int(1/ratio)) for ratio in args.split_ratios]

    if args.fs:
        mol_fs_train_input, mol_fs_train_output = read_files(["mol_fs_train"])
        
        for split_ratio, fraction in zip(args.split_ratios, split_fractions):
            selected_train_input, selected_train_output = split_train_set(
                mol_fs_train_input, mol_fs_train_output, split_ratio=split_ratio, scaffold_file='mol_fs_train_scaffolds.pkl')

            write_files(f"mol_fs_scaffold_train_{fraction}", selected_train_input, selected_train_output)

    elif args.rs:
        mol_rs_train_input, mol_rs_train_output = read_files(["mol_rs_train"])
        
        for split_ratio, fraction in zip(args.split_ratios, split_fractions):
            selected_train_input, selected_train_output = split_train_set(
                mol_rs_train_input, mol_rs_train_output, split_ratio=split_ratio, scaffold_file='mol_rs_train_scaffolds.pkl')

            write_files(f"mol_rs_scaffold_train_{fraction}", selected_train_input, selected_train_output)