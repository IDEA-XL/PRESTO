import pytest
import os

SKIP_TESTS = False
try:
    from MoleculeSTM.datasets.utils import mol_to_graph_data_obj_simple
except ImportError:
    SKIP_TESTS = True

import torch
from bioagent.chemistry_tools import smiles_to_graph


def smiles_to_graph_mock(smiles):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(smiles)
    data = mol_to_graph_data_obj_simple(molecule)
    return data.x, data.edge_index, data.edge_attr


@pytest.mark.skipif(SKIP_TESTS, reason="MoleculeSTM not installed")
@pytest.mark.parametrize("smiles", ["COc1cc2c(cc1OC)C(=O)C(CC1CCNCC1)C2", "COc1cc2c(cc1OC)C(=O)C(=CC1CCN(Cc3ccccc3)CC1)C2", "COc1cc2c(cc1O)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", "COc1cc2c(cc1O)CC(CC1CCN(Cc3ccccc3)CC1)C2=O"])
def test_smiles_to_graph(smiles):
    x_1, edge_index_1, edge_attr_1 = smiles_to_graph(smiles, device="cpu")
    x_2, edge_index_2, edge_attr_2 = smiles_to_graph_mock(smiles)
    assert torch.allclose(x_1, x_2), f"Atom features are not equal for {smiles}"
    assert torch.allclose(edge_index_1, edge_index_2), f"Edge index is not equal for {smiles}"
    assert torch.allclose(edge_attr_1, edge_attr_2), f"Edge attributes are not equal for {smiles}"