"""Multi-instance method implementations"""

from rtml.multi_instance.methods._torch import (
    default_multi_instance_backends,
    MultiInstanceTorchBackend,
)

__all__ = ["MultiInstanceTorchBackend", "default_multi_instance_backends"]
