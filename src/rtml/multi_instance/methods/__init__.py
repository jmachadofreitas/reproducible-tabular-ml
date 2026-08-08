"""Multi-instance method implementations."""

from rtml.methods.backends.base import MethodBackend
from rtml.multi_instance.methods._sklearn import MultiInstanceSklearnBackend
from rtml.multi_instance.methods._torch import MultiInstanceTorchBackend


def default_multi_instance_backends() -> tuple[MethodBackend, ...]:
    """Return the built-in backends for multi-instance methods."""
    return MultiInstanceSklearnBackend(), MultiInstanceTorchBackend()


__all__ = [
    "MultiInstanceSklearnBackend",
    "MultiInstanceTorchBackend",
    "default_multi_instance_backends",
]
