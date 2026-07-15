from __future__ import annotations

from rtml.methods.backends.base import MethodBackend
from rtml.methods.base import MethodSpec
from rtml.methods.backends.sklearn import SklearnBackend


def default_method_backends() -> tuple[MethodBackend, ...]:
    """Return the built-in method backends available without optional integrations."""
    return (SklearnBackend(),)


def method_backend_name(method: MethodSpec) -> str:
    """Return the backend selected by a method spec."""
    return method.model.backend
