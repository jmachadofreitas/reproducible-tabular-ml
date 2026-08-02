from __future__ import annotations

from collections.abc import Sequence

from torch import nn


class MLP(nn.Sequential):
    """Small dense block shared by current multi-instance Torch models."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        super().__init__(*layers)
