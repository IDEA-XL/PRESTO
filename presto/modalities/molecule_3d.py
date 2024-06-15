# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import List, Dict, Any, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from unicore import utils
from unicore.modules import LayerNorm, TransformerEncoderLayer, init_bert_params
from unicore.data import Dictionary

from presto.modalities.base_modality import Modality
from presto.modalities.projectors import build_mlp_vector_projector

MOLECULE_3D_PATH = os.environ.get("MOLECULE_3D_PATH", "")


class Molecule3DModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = os.path.join(MOLECULE_3D_PATH, "molecule_model.pth"),
        model_dictionary_path: str = os.path.join(MOLECULE_3D_PATH, "dictionary.txt"),
        num_projector_layers: int = 2,
        num_tokens_output: int = 1,
    ):
        self.model_name_or_path = model_name_or_path
        self.module = UniMolModel()
        self.module.load_state_dict(torch.load(self.model_name_or_path, map_location="cpu")['model'], strict=False)

        self.dictionary = Dictionary.load(model_dictionary_path)
        self.dictionary.add_symbol("[MASK]", is_special=True)

        self.dtype = torch.float32
        self.device = 'cpu'
        self.num_projector_layers = num_projector_layers
        self.num_tokens_output = num_tokens_output

    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        return build_mlp_vector_projector(
            input_hidden_size=self.module.output_dim,
            lm_hidden_size=lm_hidden_size,
            num_layers=self.num_projector_layers,
            num_tokens=self.num_tokens_output,
        )

    @property
    def name(self) -> str:
        return "molecule_3d"

    @property
    def token(self) -> str:
        return "<molecule_3d>"

    @property
    def data_key(self) -> str:
        return "molecules"

    @property
    def token_width(self) -> int:
        return self.num_tokens_output

    def to(self, dtype: torch.dtype, device: torch.device) -> "Molecule3DModality":
        self.dtype = dtype
        self.device = device
        self.module.to(device=device)
        return self

    def preprocess_rows(self, rows: List[Dict]) -> List[Any | None]:
        pass 


    @torch.no_grad()
    def forward(self, encoded_values: List[Dict]) -> List[torch.Tensor]:
        mol_features = []
        for encoded_value in encoded_values:
            mol_feature = []
            for mol in encoded_value:
                mol_feature.append(self.module(
                    **tree_map(lambda x: x.to(self.device), mol)
                ))
            mol_features.append(torch.stack(mol_feature).to(self.dtype) if len(mol_feature) > 0 else None)
        return mol_features

# def __getitem__(self, index):
#         data = self.lmdb_dataset[index]
#         smiles = data['smi']
#         ## deal with 3d coordinates
#         atoms_orig = np.array(data['atoms'])
#         atoms = atoms_orig.copy()
#         coordinate_set = data['coordinates_list']
#         coordinates = random.sample(coordinate_set, 1)[0].astype(np.float32)
#         assert len(atoms) == len(coordinates) and len(atoms) > 0
#         assert coordinates.shape[1] == 3

#         ## deal with the hydrogen
#         if self.remove_hydrogen:
#             mask_hydrogen = atoms != "H"
#             if sum(mask_hydrogen) > 0:
#                 atoms = atoms[mask_hydrogen]
#                 coordinates = coordinates[mask_hydrogen]

#         if not self.remove_hydrogen and self.remove_polar_hydrogen:
#             end_idx = 0
#             for i, atom in enumerate(atoms[::-1]):
#                 if atom != "H":
#                     break
#                 else:
#                     end_idx = i + 1
#             if end_idx != 0:
#                 atoms = atoms[:-end_idx]
#                 coordinates = coordinates[:-end_idx]

#         ## deal with cropping
#         if self.max_atoms > 0 and len(atoms) > self.max_atoms:
#             index = np.random.permutation(len(atoms))[:self.max_atoms]
#             atoms = atoms[index]
#             coordinates = coordinates[index]

#         assert 0 < len(atoms) < self.__max_atoms, print(len(atoms), atoms_orig, index)
#         atom_vec = torch.from_numpy(self.dictionary.vec_index(atoms)).long()

#         if self.normalize_coords:
#             coordinates = coordinates - coordinates.mean(axis=0)

#         if self.add_special_token:
#             atom_vec = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
#             coordinates = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)

#         ## obtain edge types; which is defined as the combination of two atom types
#         edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
#         dist = distance_matrix(coordinates, coordinates).astype(np.float32)
#         coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)
#         return atom_vec, coordinates, edge_type, dist, smiles


class UniMolModel(nn.Module):

    def __init__(
        self,
        encoder_layers: int = 15,
        encoder_embed_dim: int = 512,
        encoder_ffn_embed_dim: int = 2048,
        encoder_attention_heads: int = 64,
        activation_fn: str = "gelu",
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 512,
        delta_pair_repr_norm_loss: float = -1.0,
        max_atoms: int = 256,
        dictionary: Optional[Dictionary] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), encoder_embed_dim, self.padding_idx
        )
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=encoder_layers,
            embed_dim=encoder_embed_dim,
            ffn_embed_dim=encoder_ffn_embed_dim,
            attention_heads=encoder_attention_heads,
            emb_dropout=emb_dropout,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            max_seq_len=max_seq_len,
            activation_fn=activation_fn,
            no_final_head_layer_norm=delta_pair_repr_norm_loss < 0,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, encoder_attention_heads, activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.num_features = encoder_embed_dim
        self.apply(init_bert_params)


    def forward(
        self,
        src_tokens,
        src_distance,
        src_edge_type,
    ):
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_rep = encoder_rep * (1 - padding_mask.unsqueeze(-1).type_as(encoder_rep))
        return encoder_rep.mean(dim=1)



class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz = emb.size(0)
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        for i in range(len(self.layers)):
            x, attn_mask, _ = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm



class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)