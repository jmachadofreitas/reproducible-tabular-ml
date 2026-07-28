"""Multiple-instance learning support."""

from rtml.multi_instance.datasets import MultiInstanceDataset
from rtml.multi_instance.resampling import build_multi_instance_resampling_plan
from rtml.multi_instance.tasks import MultiInstanceTask

__all__ = [
    "MultiInstanceDataset",
    "MultiInstanceTask",
    "build_multi_instance_resampling_plan",
]
