from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class BagDatasetBundle(NamedTuple):
    """Bag tensor datasets produced from one benchmark split."""

    train: BagTensorDataset
    validation: BagTensorDataset | None
    test: BagTensorDataset
    classes: np.ndarray | None
    input_dim: int


class BagLoaderBundle(NamedTuple):
    """Dataloaders produced for one multi-instance Torch run."""

    train: DataLoader
    validation: DataLoader | None
    test: DataLoader


class BagTensorDataset(Dataset):
    """Contiguous instance tensors indexed as variable-length bags."""

    def __init__(
        self,
        instances: np.ndarray | torch.Tensor,
        bag_offsets: np.ndarray | torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self.instances = torch.as_tensor(instances, dtype=torch.float32)
        self.bag_offsets = torch.as_tensor(bag_offsets, dtype=torch.long)
        self.targets = targets

        if self.instances.ndim != 2:
            raise ValueError("MIL instance tensor must be two-dimensional")
        if self.bag_offsets.ndim != 1:
            raise ValueError("bag offsets must be one-dimensional")
        if len(self.bag_offsets) != len(self.targets) + 1:
            raise ValueError("bag offsets length must equal target count plus one")
        if self.bag_offsets[0].item() != 0:
            raise ValueError("bag offsets must start at zero")
        if self.bag_offsets[-1].item() != len(self.instances):
            raise ValueError("last bag offset must equal the number of instances")
        if torch.any(torch.diff(self.bag_offsets) <= 0):
            raise ValueError("Torch MIL methods do not support empty bags")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.bag_offsets[index])
        stop = int(self.bag_offsets[index + 1])
        return self.instances[start:stop], self.targets[index]


def collate_bags(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length bags and mark the real instances in each bag."""
    bags, targets = zip(*batch, strict=True)
    lengths = torch.as_tensor([len(bag) for bag in bags], dtype=torch.long)
    padded = pad_sequence(bags, batch_first=True)
    mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask, torch.stack(targets)
