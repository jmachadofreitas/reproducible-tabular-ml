from typing import List, Callable, Optional

from torch import nn


class MLP(nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_features (int): Number of features of the input
        hidden_dims (List[int]): List of the hidden dimensions

    Reference:
        https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
        last_dropout: bool = True,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_features
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            if activation_layer is not None:
                layers.append(activation_layer(**params))
            if dropout > 0:
                layers.append(nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dims[-1], bias=bias))

        if last_dropout and dropout > 0:
            layers.append(nn.Dropout(dropout, **params))

        super().__init__(*layers)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1, use_layer_norm=False):
        """
        Initializes the ResidualMLPBlock.

        Args:
            input_dim (int): Dimension of the input and output features.
            hidden_dim (int): Dimension of the hidden layer.
            dropout_prob (float): Dropout probability.
            use_layer_norm (bool): Whether to use LayerNorm as the first layer.
        """
        super().__init__()

        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        residual = x  # Store the input for the residual connection

        if self.use_layer_norm:
            x = self.layer_norm(x)

        # MLP block
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        # Residual connection
        return x + residual


def create_residual_mlp_blocks(
    num_blocks,
    input_dim,
    hidden_dim,
    dropout_prob=0.1,
) -> list:
    blocks = []

    # Create the first block without LayerNorm
    blocks.append(ResidualMLP(input_dim, hidden_dim, dropout_prob, use_layer_norm=False))

    # Create the remaining blocks with LayerNorm
    for _ in range(1, num_blocks):
        blocks.append(ResidualMLP(input_dim, hidden_dim, dropout_prob, use_layer_norm=True))

    return blocks
