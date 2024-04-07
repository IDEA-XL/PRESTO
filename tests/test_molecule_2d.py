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

# def load_moleculeSTM(device="cpu"):
#     moleculeSTM = GNN_graphpred(
#         num_layer=5,
#         emb_dim=300,
#         num_tasks=1,
#         JK="last",
#         graph_pooling="mean",
#         molecule_node_model=GNN(
#             num_layers=5,
#             emb_dim=300,
#             JK="last",
#             drop_ratio=0.,
#             gnn_type="gin",
#         ),
#     )
#     moleculeSTM = moleculeSTM.from_pretrained(INIT_CHECKPOINT + "/molecule_model.pth").to(device)
#     return moleculeSTM

def load_modality(device="cpu"):
    modality = Molecule2DModality(model_name_or_path=INIT_CHECKPOINT + "/molecule_model.pth").to(torch.float32, device)
    return modality

# @pytest.mark.skipif(SKIP_TESTS, reason="MoleculeSTM not installed")
# @pytest.mark.parametrize("smiles", ["CCO", "C1=CC=CC=C1", "C1=CC=CC=C1C", "C1=CC=CC=C1CC", "C1=CC=CC=C1CCC", "C1=CC=CC=C1CCCC"])
# def test_molecule_2d_modality(smiles):
#     x, edge_index, edge_attr = smiles_to_graph(smiles, device=DEVICE)
#     node_num = x.shape[0]

#     # moleculeSTM = load_moleculeSTM(device=DEVICE)
#     # moleculeSTM.eval()
#     # with torch.no_grad():
#     #     moleculeSTM_output = moleculeSTM(x, edge_index, edge_attr, None)[0]
#     #     assert moleculeSTM_output.shape == (node_num, 300)

#     modality = load_modality(device=DEVICE)
#     modality_output = modality.forward(x, edge_index, edge_attr)
#     assert modality_output.shape == (node_num, 300)

#     # assert torch.allclose(moleculeSTM_output, modality_output, atol=1e-4)
    
    
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