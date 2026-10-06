"""Shared four-tap gated residual fusion for dense predictions."""

import torch
from torch import nn


class SpatiotemporalDenseFusion(nn.Module):
    def __init__(self, dim_in=2048, intermediate_layer_idx=(4, 11, 17, 23)):
        super().__init__()
        self.indices = tuple(intermediate_layer_idx)
        if len(self.indices) != 4 or len(set(self.indices)) != 4:
            raise ValueError("Dense fusion requires four distinct cached taps")
        self.norm = nn.LayerNorm(2 * dim_in)
        self.projection = nn.Linear(2 * dim_in, dim_in)
        self.alpha = nn.Parameter(torch.zeros(4))

    def forward(self, spatial, temporal):
        if temporal is None:
            return spatial
        if len(spatial) != len(temporal):
            raise ValueError("Fusion branch depths must match")
        output = list(spatial)
        for gate, index in zip(self.alpha, self.indices):
            s, t = spatial[index], temporal[index]
            if s is None or t is None or s.shape != t.shape:
                raise ValueError(f"Missing or mismatched fusion tap {index}")
            delta = self.projection(self.norm(torch.cat((s, t), dim=-1))).to(s.dtype)
            output[index] = s + gate.to(s.dtype) * delta
        return output
