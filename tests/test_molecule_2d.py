import pytest
import os

SKIP_TESTS = False
try:
    from bioagent.modalities.molecule_2d import GNN
except ImportError:
    SKIP_TESTS = True

import torch

from bioagent.modalities import Molecule2DModality
from bioagent.chemistry_tools import smiles_to_graph

INIT_CHECKPOINT = os.environ.get("MOLECULE_2D_PATH", None)
SKIP_TESTS = SKIP_TESTS or INIT_CHECKPOINT is None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_modality(device="cpu"):
    modality = Molecule2DModality(model_name_or_path=INIT_CHECKPOINT + "/molecule_model.pth").to(torch.float32, device)
    return modality
    
@pytest.mark.skipif(SKIP_TESTS, reason="MoleculeSTM not installed")
@pytest.mark.parametrize("smiles", [["CCO", "C1=CC=CC=C1"]])
def test_molecule_multiple_2d_modality(smiles):
    graph1 = smiles_to_graph(smiles[0], device=DEVICE)
    node_num1 = graph1[0].shape[0]
    
    graph2 = smiles_to_graph(smiles[1], device=DEVICE)
    node_num2 = graph2[0].shape[0]

    modality = load_modality(device=DEVICE)
    modality_output = modality.forward([graph1, graph2])
    assert modality_output[0][0].shape == (node_num1, 300)
    assert modality_output[0][1].shape == (node_num2, 300)