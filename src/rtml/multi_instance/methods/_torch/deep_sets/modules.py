from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from rtml.multi_instance.methods._torch.modules import MLP

BagPool = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class DeepSets(nn.Module):
    """Permutation-invariant neural model for bag-level tabular prediction.

    Implements the ``rho(sum(phi(x)))`` architecture, with optional masked mean
    pooling, from Zaheer et al., "Deep Sets", NeurIPS 2017.

    References:
        https://arxiv.org/abs/1703.06114
        https://github.com/manzilzaheer/DeepSets
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        encoder_dims: Sequence[int] = (64, 64),
        latent_dim: int = 64,
        predictor_dims: Sequence[int] = (64,),
        pooling: str = "sum",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = MLP(
            input_dim,
            encoder_dims,
            latent_dim,
            dropout=dropout,
        )
        self.predictor = MLP(
            latent_dim,
            predictor_dims,
            output_dim,
            dropout=dropout,
        )
        self.pooling = pooling
        self._pool = _resolve_pooling(pooling)

    def forward(self, instances: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict one output from each padded bag and its valid-instance mask."""
        if instances.ndim != 3:
            raise ValueError("DeepSets expects instances shaped [batch, instances, features]")
        if mask.shape != instances.shape[:2]:
            raise ValueError("DeepSets mask must match the batch and instance dimensions")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("DeepSets does not support empty bags")

        encoded = self.encoder(instances)
        bag_embedding = self._pool(encoded, mask)
        return self.predictor(bag_embedding)


def _resolve_pooling(name: str) -> BagPool:

    def masked_sum(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (encoded * mask.unsqueeze(-1)).sum(dim=1)

    def masked_mean(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        summed = masked_sum(encoded, mask)

        return summed / mask.sum(dim=1, keepdim=True)

    if name == "sum":
        return masked_sum
    if name == "mean":
        return masked_mean

    raise ValueError(f"DeepSets pooling must be 'sum' or 'mean', got {name!r}")
