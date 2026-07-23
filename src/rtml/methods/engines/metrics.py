from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import torch


class Metric(Protocol):
    """Metric interface accepted by the torch engine metric collection."""

    def reset(self) -> None: ...

    def update(self, *values: Any) -> None: ...

    def compute(self) -> Any: ...


class Metrics:
    """Route named step outputs into reset/update/compute metric objects."""

    def __init__(self, metrics: Mapping[str, Metric] | None = None, **kwargs: Metric) -> None:
        self.metrics = dict(metrics or {})
        self.metrics.update(kwargs)
        self.update_called = False

    def __bool__(self) -> bool:
        return bool(self.metrics)

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self.update_called = False

    def update(self, **kwargs: Any) -> None:
        updated = False
        for key, metric in self.metrics.items():
            value = kwargs.get(key)
            if value is None:
                continue
            if _is_metric_sequence(value):
                metric.update(*value)
            else:
                metric.update(value)
            updated = True
        self.update_called = updated or self.update_called

    def compute(self) -> dict[str, float]:
        if not self.update_called:
            return {}
        return {name: _as_float(metric.compute()) for name, metric in self.metrics.items()}


class IgniteMetric:
    """Adapt Ignite's update(output) metrics to RTML's update(*values) protocol."""

    def __init__(self, metric: Any) -> None:
        self.metric = metric

    def reset(self) -> None:
        self.metric.reset()

    def update(self, *values: Any) -> None:
        if len(values) == 1:
            self.metric.update(values[0])
            return
        self.metric.update(tuple(values))

    def compute(self) -> Any:
        return self.metric.compute()


def _is_metric_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)
