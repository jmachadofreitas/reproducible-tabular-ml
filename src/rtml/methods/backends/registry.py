from __future__ import annotations

from rtml.methods.backends.base import MethodBackend
from rtml.methods.backends.sklearn import SklearnBackend


def default_method_backends() -> tuple[MethodBackend, ...]:
    """Return the built-in method backends available without optional integrations."""
    return (SklearnBackend(),)
