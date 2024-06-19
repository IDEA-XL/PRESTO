import torch.nn as nn
import torch


def build_patch_mlp_projector(
    input_hidden_size: int, lm_hidden_size: int, num_layers: int
) -> nn.Module:
    modules = [nn.Linear(input_hidden_size, lm_hidden_size)]
    for _ in range(1, num_layers):
        modules.append(nn.GELU())
        modules.append(nn.Linear(lm_hidden_size, lm_hidden_size))
    return nn.Sequential(*modules)


class _MLPVectorProjector(nn.Module):
    def __init__(
        self, input_hidden_size: int, lm_hidden_size: int, num_layers: int, 
    ):
        super(_MLPVectorProjector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_hidden_size, lm_hidden_size),
        )
        for _ in range(1, num_layers):
            self.mlp.append(nn.GELU())
            self.mlp.append(nn.Linear(lm_hidden_size, lm_hidden_size))

    def forward(self, x):
        return self.mlp(x)


def build_mlp_vector_projector(
    input_hidden_size: int, lm_hidden_size: int, num_layers: int,
):
    return _MLPVectorProjector(
        input_hidden_size, lm_hidden_size, num_layers,
    )
