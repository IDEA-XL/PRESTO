# %%
from datasets import load_dataset, DatasetDict, load_from_disk
import random
from collections import defaultdict
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
import argparse
import pickle
import tqdm
import multiprocessing

RDLogger.DisableLog('rdApp.*')

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--fs", action="store_true")
parser.add_argument("--rs", action="store_true")
parser.add_argument("--num_proc", type=int, default=10)
args = parser.parse_args()

# %%
def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    if not scaffold:
        return None
    return scaffold

def save_similarities(similarities, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(similarities, f)

def load_similarities(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def calc_train_test_similarities(train_scaffolds, test_scaffolds, threshold=0.5, similarity_file='similarities.pkl', num_proc=1):
    similarities = load_similarities(similarity_file)
    if similarities is None:
        similarities = {}
        with multiprocessing.Pool(processes=num_proc) as pool:
            results = []
            for test_scaffold in test_scaffolds:
                if test_scaffold is None:
                    continue
                result = pool.apply_async(_scaffold_similarities, (test_scaffold, train_scaffolds, threshold, None))
                results.append(result)
            for result, test_scaffold in tqdm.tqdm(list(zip(results, test_scaffolds)), desc="Calculating scaffold similarities"):
                train_scaffold_similarities = result.get()
                for train_scaffold, similarity in train_scaffold_similarities.items():
                    similarities[(test_scaffold, train_scaffold)] = similarity
        save_similarities(similarities, similarity_file)
    return similarities

def _scaffold_similarities(test_scaffold, train_scaffolds, threshold, similarities):
        scaffold_similarities = {}
        for train_scaffold in train_scaffolds:
            similarity = scaffold_similarity(test_scaffold, train_scaffold, threshold, similarities)
            scaffold_similarities[train_scaffold] = similarity
        return scaffold_similarities

@lru_cache(maxsize=None)
def _scaffold_similarity(scaffold1, scaffold2, threshold=0.5):
    try:
        fps1 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(scaffold1), 2, nBits=1024)
        fps2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(scaffold2), 2, nBits=1024)
        similarity = DataStructs.TanimotoSimilarity(fps1, fps2)
    except KeyboardInterrupt:
        raise
    except:
        similarity = threshold
    return similarity


def scaffold_similarity(scaffold1, scaffold2, threshold=0.5, similarities={}):
    if similarities:
        if (scaffold1, scaffold2) in similarities:
            return similarities[(scaffold1, scaffold2)]
        elif (scaffold2, scaffold1) in similarities:
            return similarities[(scaffold2, scaffold1)]
    return _scaffold_similarity(scaffold1, scaffold2, threshold)


def calc_similarity_distribution(test_scaffolds, train_scaffolds, threshold=0.5, bin_size=0.1, similarities={}):
    distribution = defaultdict(int)
    for test_scaffold in tqdm.tqdm(test_scaffolds, desc="Calculating similarity distribution"):
        if test_scaffold is None:
            continue
        max_similarity = max(scaffold_similarity(test_scaffold, train_scaffold, threshold, similarities)
                             for train_scaffold in train_scaffolds)
        bin_index = int(max_similarity / bin_size)
        distribution[bin_index] += 1
    return distribution

# %%
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

mol_fs_test_input, mol_fs_test_output = read_files(["mol_fs_test", "smol_fs_test", "smol_fs_dev"])
mol_fs_train_input, mol_fs_train_output = read_files(["mol_fs_train"])
mol_rs_test_input, mol_rs_test_output = read_files(["mol_rs_test", "smol_rs_test", "smol_rs_dev"])
mol_rs_train_input, mol_rs_train_output = read_files(["mol_rs_train"])

# %%
def select_test_set(train_input, test_input, test_output, num_select=1000, similarity_threshold=0.5, bin_size=0.1, similarity_file='similarities.pkl', num_proc=1):
    train_scaffolds = list({generate_scaffold(smiles) for smiles in tqdm.tqdm(train_input) if generate_scaffold(smiles) is not None})
    test_scaffolds = [generate_scaffold(smiles) for smiles in tqdm.tqdm(test_input)]

    print(f"Number of train scaffolds: {len(train_scaffolds)}, Number of test scaffolds: {len(test_scaffolds)}")
    train_test_similarities = calc_train_test_similarities(train_scaffolds, test_scaffolds, similarity_threshold, similarity_file, num_proc=num_proc)
    similarity_distribution = calc_similarity_distribution(test_scaffolds, train_scaffolds, similarity_threshold, bin_size, similarities=train_test_similarities)
    print("Similarity distribution:")
    for bin_index, count in sorted(similarity_distribution.items()):
        print(f"{bin_index * bin_size:.1f} - {(bin_index + 1) * bin_size:.1f}: {count}")

    scaffold_to_indices = {}
    for i, scaffold in enumerate(test_scaffolds):
        if scaffold not in scaffold_to_indices:
            scaffold_to_indices[scaffold] = [i]
        else:
            scaffold_to_indices[scaffold].append(i)
    
    selected_indices = []
    remaining_indices = []

    for scaffold, indices in tqdm.tqdm(scaffold_to_indices.items(), desc="Selecting test set"):
        similar_to_train = any(scaffold_similarity(scaffold, train_scaffold, threshold=similarity_threshold, similarities=train_test_similarities) >= similarity_threshold 
                               for train_scaffold in train_scaffolds)
        
        if not similar_to_train:
            if len(selected_indices) < num_select:
                selected_indices.extend(indices)
            else:
                remaining_indices.extend(indices)
        else:
            remaining_indices.extend(indices)
    
    selected_input = [test_input[i] for i in selected_indices]
    selected_output = [test_output[i] for i in selected_indices]
    remaining_input = [test_input[i] for i in remaining_indices]
    remaining_output = [test_output[i] for i in remaining_indices]
    
    print(f"Selected test set size: {len(selected_indices)}")
    print(f"Remaining test set size: {len(remaining_indices)}")
    
    return selected_input, selected_output, remaining_input, remaining_output

# %%
if args.fs:
    selected_test_input, selected_test_output, remaining_test_input, remaining_test_output = select_test_set(
        mol_fs_train_input, mol_fs_test_input, mol_fs_test_output, num_select=1000, similarity_threshold=0.8, bin_size=0.1, similarity_file='similarities_fs.pkl', num_proc=args.num_proc)

    write_files("mol_fs_scaffold_selected_test", selected_test_input, selected_test_output)
    write_files("mol_fs_scaffold_remaining_test", remaining_test_input, remaining_test_output)

elif args.rs:
    selected_test_input, selected_test_output, remaining_test_input, remaining_test_output = select_test_set(
        mol_rs_train_input, mol_rs_test_input, mol_rs_test_output, num_select=1000, similarity_threshold=0.8, bin_size=0.1, similarity_file='similarities_rs.pkl', num_proc=args.num_proc)

    write_files("mol_rs_scaffold_selected_test", selected_test_input, selected_test_output)
    write_files("mol_rs_scaffold_remaining_test", remaining_test_input, remaining_test_output)

 