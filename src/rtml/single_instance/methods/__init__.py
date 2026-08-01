"""Single-instance method implementations."""

from rtml.methods.backends.base import MethodBackend
from rtml.single_instance.methods._sklearn import SklearnBackend
from rtml.single_instance.methods._torch import TorchBackend


def default_single_instance_backends() -> tuple[MethodBackend, ...]:
    """Return the built-in single-instance method backends."""
    return (SklearnBackend(), TorchBackend())


__all__ = [
    "SklearnBackend",
    "TorchBackend",
    "default_single_instance_backends",
]
