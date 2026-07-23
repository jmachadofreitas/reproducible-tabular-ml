"""Checkpoint helpers for the Torch/Ignite engine."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from ignite.handlers import Checkpoint, DiskSaver, ModelCheckpoint


class CheckpointState:
    """Small state object saved alongside Ignite-managed torch objects."""

    def __init__(self) -> None:
        self.epoch = 0
        self.step = 0
        self.score_name: str | None = None
        self.score: float | None = None
        self.best_score: float | None = None

    def update(
        self,
        *,
        epoch: int,
        step: int,
        score_name: str | None,
        score: float | None,
        best_score: float | None,
    ) -> None:
        self.epoch = epoch
        self.step = step
        self.score_name = score_name
        self.score = score
        self.best_score = best_score

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "score_name": self.score_name,
            "score": self.score,
            "best_score": self.best_score,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.step = int(state.get("step", 0))
        self.score_name = state.get("score_name")
        self.score = _optional_float(state.get("score"))
        self.best_score = _optional_float(state.get("best_score"))


class CheckpointManager:
    """Ignite-backed last/best checkpoint manager for one training run."""

    def __init__(
        self,
        *,
        directory: str | Path,
        score_mode: str = "min",
        save_last: bool = True,
        save_best: bool = True,
        every_n_epochs: int = 1,
        delay_n_epochs: int = 0,
        n_saved: int | None = 1,
        atomic: bool = True,
        require_empty: bool = False,
        resume_from: str | Path | None = None,
    ) -> None:
        if score_mode not in {"min", "max"}:
            raise ValueError("checkpoint score_mode must be 'min' or 'max'")
        if every_n_epochs < 1:
            raise ValueError("checkpoint every_n_epochs must be >= 1")
        if delay_n_epochs < 0:
            raise ValueError("checkpoint delay_n_epochs must be >= 0")

        self.directory = Path(directory)
        self.score_mode = score_mode
        self.save_last = save_last
        self.save_best = save_best
        self.every_n_epochs = every_n_epochs
        self.delay_n_epochs = delay_n_epochs
        self.n_saved = n_saved
        self.atomic = atomic
        self.require_empty = require_empty
        self.resume_from = resume_from
        self.state = CheckpointState()
        self._objects: dict[str, Any] = {}
        self._score: float | None = None
        self.best_score = float("inf") if score_mode == "min" else float("-inf")
        self.last_path: Path | None = None
        self.best_path: Path | None = None
        self.saved_paths: list[Path] = []

        self._last_handler = Checkpoint(
            {},
            DiskSaver(
                self.directory,
                atomic=self.atomic,
                create_dir=True,
                require_empty=self.require_empty,
            ),
            filename_prefix="last",
            n_saved=self.n_saved,
            filename_pattern=self._filename_pattern("last"),
        )
        self._best_handler = ModelCheckpoint(
            self.directory,
            filename_prefix="best",
            score_function=self._ignite_score,
            score_name="score",
            n_saved=self.n_saved,
            atomic=self.atomic,
            require_empty=False,
            create_dir=True,
            filename_pattern=self._filename_pattern("best"),
        )

    def set_objects(self, objects: Mapping[str, Any]) -> None:
        self._objects = {**dict(objects), "checkpoint": self.state}
        self._last_handler.to_save = self._objects

    def should_save(self, epoch: int) -> bool:
        return epoch > self.delay_n_epochs and epoch % self.every_n_epochs == 0

    def save(
        self,
        *,
        engine: Any,
        epoch: int,
        step: int,
        score_name: str | None = None,
        score: float | None = None,
    ) -> tuple[Path | None, Path | None]:
        if not self._objects:
            raise ValueError("checkpoint objects must be set before saving")

        self._score = score
        if score is not None and self._is_better(score):
            self.best_score = score
        self.state.update(
            epoch=epoch,
            step=step,
            score_name=score_name,
            score=score,
            best_score=None if _is_inf(self.best_score) else self.best_score,
        )

        last_path = None
        if self.save_last:
            self._last_handler(engine)
            last_path = _path_or_none(self._last_handler.last_checkpoint)
            self.last_path = last_path

        best_path = None
        if self.save_best and score is not None:
            previous_best = self._best_handler.last_checkpoint
            self._best_handler(engine, self._objects)
            candidate_best = _path_or_none(self._best_handler.last_checkpoint)
            if candidate_best is not None and str(candidate_best) != str(previous_best):
                best_path = candidate_best
                self.best_path = candidate_best

        self.saved_paths.extend(path for path in (last_path, best_path) if path is not None)
        return last_path, best_path

    def resume_path(self) -> Path | None:
        if self.resume_from in {None, "", False}:
            return None
        if self.resume_from == "last":
            return self.directory / "last.ckpt"
        if self.resume_from == "best":
            return self.directory / "best.ckpt"
        return Path(self.resume_from)  # type: ignore

    def load_resume_checkpoint(self) -> Path | None:
        path = self.resume_path()
        if path is None:
            return None
        load_checkpoint(path, to_load=self._objects)
        if self.state.best_score is not None:
            self.best_score = self.state.best_score
        return path

    def _ignite_score(self, engine: Any) -> float:
        if self._score is None:
            return float("-inf")
        if self.score_mode == "min":
            return -float(self._score)
        return float(self._score)

    def _is_better(self, score: float) -> bool:
        if self.score_mode == "min":
            return score < self.best_score
        return score > self.best_score

    def _filename_pattern(self, prefix: str) -> str:
        if self.n_saved == 1:
            return "{filename_prefix}.ckpt"
        if prefix == "best":
            return "{filename_prefix}_{score_name}_{score}_{global_step}.ckpt"
        return "{filename_prefix}_{global_step}.ckpt"


def load_checkpoint(
    path: str | Path,
    *,
    to_load: Mapping[str, Any] | None = None,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: Any | None = None,
) -> dict[str, Any]:
    """Load an Ignite checkpoint and optionally restore common training objects."""
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    objects: dict[str, Any] = dict(to_load or {})
    if model is not None:
        objects["model"] = model
    if optimizer is not None:
        objects["optimizer"] = optimizer
    if lr_scheduler is not None:
        objects["lr_scheduler"] = lr_scheduler
    if objects:
        Checkpoint.load_objects(to_load=objects, checkpoint=checkpoint)
    return checkpoint


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _is_inf(value: float) -> bool:
    return value in {float("inf"), float("-inf")}
