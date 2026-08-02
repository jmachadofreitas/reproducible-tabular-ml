from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rtml.multi_instance.methods._torch.modules import MLP


class Attention(nn.Module):
    """Plain attention pooling from Attention-based Deep MIL."""

    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
        n_branches: int = 1,
    ) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, n_branches),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(embeddings).transpose(1, 2)
        return _masked_attention(scores, mask)


class GatedAttention(nn.Module):
    """Gated attention pooling from Attention-based Deep MIL."""

    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
        n_branches: int = 1,
    ) -> None:
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_dim, n_branches)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention_w(
            self.attention_v(embeddings) * self.attention_u(embeddings)
        ).transpose(1, 2)
        return _masked_attention(scores, mask)


class AttentionDeepMIL(nn.Module):
    """Attention-based deep MIL for bag-level binary classification.

    This tabular implementation retains the plain and gated attention pooling
    proposed by Ilse, Tomczak, and Welling. It replaces the reference image
    encoder with an MLP and applies masked attention to padded bags.

    References:
        https://proceedings.mlr.press/v80/ilse18a.html
        https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    """

    def __init__(
        self,
        input_dim: int,
        *,
        encoder_dims: Sequence[int] = (128,),
        embedding_dim: int = 128,
        attention_dim: int = 64,
        attention: str = "gated",
        n_attention_branches: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if attention_dim < 1:
            raise ValueError("attention_dim must be positive")
        if n_attention_branches < 1:
            raise ValueError("n_attention_branches must be positive")

        self.feature_extractor = nn.Sequential(
            MLP(
                input_dim,
                encoder_dims,
                embedding_dim,
                dropout=dropout,
            ),
            nn.ReLU(),
        )
        self.attention = _attention_builder(attention)(
            embedding_dim,
            attention_dim,
            n_attention_branches,
        )
        self.classifier = nn.Linear(n_attention_branches * embedding_dim, 1)

    def forward(self, instances: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return one bag logit per padded bag."""
        embeddings = self.feature_extractor(instances)
        weights = self.attention(embeddings, mask)
        bag_embeddings = torch.bmm(weights, embeddings).flatten(start_dim=1)
        return self.classifier(bag_embeddings)

    def attention_weights(
        self,
        instances: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized attention weights for each bag and branch."""
        embeddings = self.feature_extractor(instances)
        return self.attention(embeddings, mask)


def _masked_attention(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 3:
        raise ValueError("attention scores must have shape [batch, branches, instances]")
    if mask.shape != (scores.shape[0], scores.shape[2]):
        raise ValueError("attention mask must match the batch and instance dimensions")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("AttentionDeepMIL does not support empty bags")

    weights = torch.softmax(scores.masked_fill(~mask.unsqueeze(1), float("-inf")), dim=2)
    return weights.masked_fill(~mask.unsqueeze(1), 0.0)


def _attention_builder(name: str) -> type[Attention] | type[GatedAttention]:
    if name == "plain":
        return Attention
    if name == "gated":
        return GatedAttention
    raise ValueError("attention must be 'plain' or 'gated'")
