import os
from typing import List, Dict, Tuple, Union

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from torch_geometric.nn import (MessagePassing, global_mean_pool, global_add_pool, global_max_pool)

from presto.chemistry_tools import smiles_to_graph
from presto.modalities.base_modality import Modality
from presto.modalities.projectors import build_mlp_vector_projector

MOLECULE_2D_PATH = os.environ.get("MOLECULE_2D_PATH", "")

NODE_LAYER = -2
GRAPH_LAYER = -1

MOL_NUM_LAYERS = 5
MOL_HIDDEN_SIZE = 300
DROP_RATIO = 0.1

NUM_ATOM_TYPE = 120     # including the extra mask tokens
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 6       # including aromatic and self-loop edge, and extra masked tokens
NUM_BOND_DIRECTION = 3


class Molecule2DModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = os.path.join(MOLECULE_2D_PATH, "molecule_model.pth"),
        num_projector_layers: int = 2,
    ):
        self.model_name_or_path = model_name_or_path
        self.module = MoleculeSTM(
            num_layers = MOL_NUM_LAYERS,
            hidden_size = MOL_HIDDEN_SIZE,
            drop_ratio = DROP_RATIO
        )
        self.module.load_state_dict(torch.load(self.model_name_or_path, map_location="cpu"))
        self.module.eval()

        self.dtype = torch.float32
        self.device = 'cpu'
        self.num_projector_layers = num_projector_layers

    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        return build_mlp_vector_projector(
            input_hidden_size=self.module.output_dim,
            lm_hidden_size=lm_hidden_size,
            num_layers=self.num_projector_layers,
        )

    @property
    def name(self) -> str:
        return "molecule_2d"

    @property
    def token(self) -> str:
        return "<molecule_2d>"

    @property
    def data_key(self) -> str:
        return "molecules"

    def to(self, dtype: torch.dtype, device: torch.device) -> "Molecule2DModality":
        self.dtype = dtype
        self.device = device
        self.module.to(device=device)
        return self

    def preprocess_rows(self, rows: List[Dict]) -> List[Dict]:
        """
        Preprocesses a list of rows into a list of molecule graph representations.
        """
        row_values = []
        for row in rows:
            smiles = row.get(self.data_key, {}).get("smiles", [])
            if not smiles:
                selfies = row.get(self.data_key, {}).get("selfies", [])
                smiles = [sf.decoder(selfie) for selfie in selfies]
            if not isinstance(smiles, list):
                smiles = [smiles]
            row_values.append(tree_map(smiles_to_graph, smiles))
        return row_values

    @torch.no_grad()
    def forward(self, *argv) -> Union[torch.Tensor, List[torch.Tensor]]:
        if len(argv) > 1:
            return self.module(*tree_map(lambda x: x.to(self.device), argv)).to(self.dtype) # take the node feature
        features = []
        for batch in argv[0]:
            batch_features = []
            if len(batch) == 0:
                batch_features.append(None)
            else:
                assert len(batch[0]) == 3, "The input should be a tuple of (x, edge_index, edge_attr)"
                for item in batch:
                    batch_features.append(self.module(*tree_map(lambda x: x.to(self.device), item)).to(self.dtype))
            features.append(batch_features)
        return features
        


class GINConv(MessagePassing):
    def __init__(self, hidden_size, aggr="add"):
        super(GINConv, self).__init__(aggr=aggr)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 2 * hidden_size),
            torch.nn.BatchNorm1d(2 * hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_size, hidden_size)
        )

        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(hidden_size)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.bond_encoder(edge_attr)
        # WARN: some weird thing happend if excute in bfloat16, so we force to cast to float32
        dtype = x.dtype
        inter = (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if dtype == torch.bfloat16:
            inter = inter.float()
            out = self.mlp.float()(inter)
            out = out.to(dtype)
        else:
            out = self.mlp(inter)
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN(nn.Module):
    def __init__(self, num_layers, hidden_size, drop_ratio=0.):
        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layers =num_layers 
        self.output_dim = hidden_size

        self.atom_encoder = AtomEncoder(hidden_size)

        self.gnns = nn.ModuleList(
            [GINConv(hidden_size, aggr="add") for _ in range(num_layers)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.atom_encoder(x)
        for layer in range(self.num_layers):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        return h


# Reference: https://github.com/chao1224/MoleculeSTM/blob/main/MoleculeSTM/models/molecule_gnn_model.py
class MoleculeSTM(nn.Module):
    def __init__(self, num_layers, hidden_size, drop_ratio=0.):
        super(MoleculeSTM, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layers = num_layers
        self.output_dim = hidden_size

        self.molecule_node_model = GNN(num_layers, hidden_size, drop_ratio)
        self.graph_pred_linear = nn.Linear(hidden_size, 1)      # unused
        self.pooler = global_mean_pool

    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
        return self.molecule_node_model(x, edge_index, edge_attr)
