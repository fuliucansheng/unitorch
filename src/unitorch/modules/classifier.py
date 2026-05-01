# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers.activations import quick_gelu


class ResLayer(nn.Module):
    """Residual MLP block: Linear → (BN) → ReLU → Linear, with skip connection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        assert input_dim == output_dim, "input_dim must equal output_dim for residual connection"
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else None

        for fc in (self.fc1, self.fc2):
            fc.weight.data.normal_(mean=0.0, std=0.02)
            fc.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        if self.bn is not None:
            out = self.bn(out)
        out = torch.relu(out)
        return torch.relu(self.fc2(out) + x)


class MLPLayer(nn.Module):
    """MLP block with optional pre/post layer norm and a residual connection.

    Architecture: (pre-LN) → Linear → QuickGELU → Linear + skip → (post-LN).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        add_pre_layer_norm: bool = True,
        add_post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        assert input_dim == output_dim, "input_dim must equal output_dim for residual connection"
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.pre_layer_norm = nn.LayerNorm(input_dim) if add_pre_layer_norm else None
        self.post_layer_norm = nn.LayerNorm(output_dim) if add_post_layer_norm else None

        for fc in (self.fc1, self.fc2):
            fc.weight.data.normal_(mean=0.0, std=0.02)
            fc.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pre_layer_norm(x) if self.pre_layer_norm is not None else x
        out = quick_gelu(self.fc1(out))
        out = self.fc2(out) + x
        if self.post_layer_norm is not None:
            out = self.post_layer_norm(out)
        return out
