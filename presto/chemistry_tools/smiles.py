from typing import Dict

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType, BondDir, ChiralType

RDLogger.DisableLog('rdApp.*')


NUM_ATOM_TYPE = 120     # including the extra mask tokens
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 6       # including aromatic and self-loop edge, and extra masked tokens
NUM_BOND_DIRECTION = 3

# WARN: we just set all UNSPECIFIED bond as SINGLE bond
BOND_TYPE = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3, BondType.UNSPECIFIED:0}
BOND_DIR = {BondDir.NONE: 0, BondDir.ENDUPRIGHT: 1, BondDir.ENDDOWNRIGHT: 2}
CHI = {ChiralType.CHI_UNSPECIFIED: 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: 2, ChiralType.CHI_OTHER: 3}


def _feature_value(feature_dict, feature_method):
    feature = feature_method()
    return feature_dict[feature]


def atom_to_feature(atom):
    num = atom.GetAtomicNum() - 1
    if num == -1:
        num = 118
    chiral = _feature_value(CHI, atom.GetChiralTag)
    return [num, chiral]


def bond_to_feature(bond):
    bond_type = _feature_value(BOND_TYPE, bond.GetBondType)
    bond_dir = _feature_value(BOND_DIR, bond.GetBondDir)
    return [bond_type, bond_dir]


def smiles_to_graph(smiles_string, device="cpu"):
    mol = Chem.MolFromSmiles(smiles_string)
    atom_features_list = [atom_to_feature(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features_list, dtype=torch.int64, device=device)

    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_feature = bond_to_feature(bond)
            edges_list.extend([(i, j), (j, i)])
            edge_features_list.extend([edge_feature, edge_feature])

        edge_index = torch.tensor(edges_list, dtype=torch.int64, device=device).t().contiguous()
        edge_attr = torch.tensor(edge_features_list, dtype=torch.int64, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64, device=device)
        edge_attr = torch.zeros((0, 2), dtype=torch.int64, device=device)

    return x, edge_index, edge_attr

def graph_to_smiles(x, edge_index, edge_attr):
    mol = Chem.RWMol()
    for atom_feature in x:
        atom = Chem.Atom(atom_feature[0])
        atom.SetChiralTag(CHI[atom_feature[1]])
        mol.AddAtom(atom)

    for i, j in edge_index.t().tolist():
        bond_feature = edge_attr[i][j]
        bond = Chem.BondType(BOND_TYPE[bond_feature[0]])
        bond.SetBondDir(BOND_DIR[bond_feature[1]])
        mol.AddBond(i, j, bond)
    return Chem.MolToSmiles(mol)


def smiles_to_coords(smiles, filter_h_only=True, max_attempts=1000, random_seed=42):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    if filter_h_only and all(atom == 'H' for atom in atoms):
        return None

    res = AllChem.EmbedMolecule(mol, maxAttempts=max_attempts, randomSeed=random_seed)
    if res == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
    elif res == -1:
        mol_tmp = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol_tmp, maxAttempts=max_attempts, randomSeed=random_seed)
        mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol_tmp)
        except:
            pass
        mol = mol_tmp

    coordinates = mol.GetConformer().GetPositions()

    assert len(atoms) == len(coordinates), f"Coordinates shape is not aligned with {smiles}"

    return atoms, coordinates


def convert_to_canonical_smiles(smiles, isomeric=False, kekulization=False):
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=isomeric, canonical=True, kekuleSmiles=kekulization)
        return canonical_smiles
    else:
        return None
    

def convert_multiple_smiles_to_canonical(smiles, isomeric=False, kekulization=False, allow_empty_part=False, sort_things=True):
    """
    Args:
        smiles: str, SMILES string
        isomeric: bool, whether to include isomeric information
        kekulization: bool, whether to kekulize the SMILES
        allow_empty_part: bool, whether to allow empty part in the SMILES
        sort_things: bool, whether to sort the SMILES parts
    Returns:
        str, canonical SMILES; None if the input SMILES is invalid
    """
    things = smiles.split('.')
    new_things = []
    for thing in things:
        try:
            if thing == '' and not allow_empty_part:
                raise ValueError('SMILES contains empty part.')
            mol = Chem.MolFromSmiles(thing)
            assert mol is not None, 'Molecule is None.'
            new_thing = Chem.MolToSmiles(mol, isomericSmiles=isomeric, canonical=True, kekuleSmiles=kekulization)
            assert new_thing is not None, f"cannot convert {thing} to canonical smiles"
        except:
            return None
        new_things.append(new_thing)
    if sort_things:
        things = sorted(new_things)
    return '.'.join(new_things)