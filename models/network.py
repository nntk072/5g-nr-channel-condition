"""Neural model definitions, with framework imports delayed until construction."""

from __future__ import annotations


def build_domain_mlp(input_dim: int):
    """Build the shared MLP embedding and classification head."""
    from torch import nn

    class DomainMLP(nn.Module):
        def __init__(self, dimension: int):
            super().__init__()
            self.embed = nn.Sequential(
                nn.Linear(dimension, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

        def forward(self, value):
            embedding = self.embed(value)
            return self.head(embedding), embedding

    return DomainMLP(input_dim)
