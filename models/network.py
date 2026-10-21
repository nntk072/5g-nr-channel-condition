"""Neural model definitions, with framework imports delayed until construction."""

from __future__ import annotations


def build_domain_mlp(input_dim: int, capacity: str = "medium"):
    """Build the shared MLP embedding and classification head."""
    from torch import nn

    from models.capacity import capacity_spec

    width1, width2, head_width = capacity_spec(capacity).mlp_widths

    class DomainMLP(nn.Module):
        def __init__(self, dimension: int):
            super().__init__()
            self.embed = nn.Sequential(
                nn.Linear(dimension, width1),
                nn.LayerNorm(width1),
                nn.ReLU(),
                nn.Linear(width1, width2),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(width2, head_width), nn.ReLU(), nn.Linear(head_width, 2)
            )

        def forward(self, value):
            embedding = self.embed(value)
            return self.head(embedding), embedding

    return DomainMLP(input_dim)
