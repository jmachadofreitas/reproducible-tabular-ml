from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from ignite.engine import Engine, Events, State
from ignite.handlers import EarlyStopping
from torch.utils.data import DataLoader

from rtml.loggers import Logger
from rtml.methods.engines.checkpointing import CheckpointManager
from rtml.methods.engines.metrics import Metrics

Batch = Any
BatchPreparer = Callable[[Batch, torch.device | str | None], Batch]
StepOutput = Mapping[str, Any] | torch.Tensor | float | int | None
TrainingStep = Callable[[Batch], StepOutput]
EvaluationStep = Callable[[Batch], Mapping[str, Any] | None]


def send_to_device(value: Any, device: torch.device | str | None) -> Any:
    """Recursively move tensors in one batch to the requested device."""
    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: send_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(send_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [send_to_device(item, device) for item in value]
    return value


def default_prepare_batch(batch: Batch, device: torch.device | str | None) -> Batch:
    return send_to_device(batch, device)


class Evaluator(Engine):
    """Ignite evaluator with output collection for later metric computation."""

    def __init__(
        self,
        evaluation_step: EvaluationStep,
        *,
        metrics: Metrics | None = None,
        prepare_batch: BatchPreparer = default_prepare_batch,
        device: torch.device | str | None = None,
        name: str | None = None,
    ) -> None:
        self.evaluation_step = evaluation_step
        self.prepare_batch = prepare_batch
        self.device = device
        self.name = name
        self.metrics = metrics or Metrics()
        self.outputs: list[dict[str, Any]] = []

        def _process_function(engine: Engine, batch: Batch) -> dict[str, Any]:
            prepared_batch = self.prepare_batch(batch, self.device)
            output = self.evaluation_step(prepared_batch)
            return dict(output or {})

        super().__init__(_process_function)

        self.add_event_handler(Events.STARTED, self._reset_outputs)
        self.add_event_handler(Events.ITERATION_COMPLETED, self._store_output)
        self.add_event_handler(Events.ITERATION_COMPLETED, self._update_metrics)
        self.add_event_handler(Events.COMPLETED, self._store_metrics)

    def _reset_outputs(self, engine: Engine) -> None:
        self.outputs = []
        self.metrics.reset()

    def _store_output(self, engine: Engine) -> None:
        output = engine.state.output
        if isinstance(output, Mapping):
            self.outputs.append(dict(output))

    def _update_metrics(self, engine: Engine) -> None:
        output = engine.state.output
        if isinstance(output, Mapping):
            self.metrics.update(**output)

    def _store_metrics(self, engine: Engine) -> None:
        engine.state.metrics = self.metrics.compute()

    def _collect_outputs(self) -> dict[str, list[Any]]:
        collected: dict[str, list[Any]] = {}
        for output in self.outputs:
            for key, value in output.items():
                collected.setdefault(key, []).append(value)
        return collected

    @torch.inference_mode()
    def evaluate(self, dataloader: DataLoader) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        state = self.run(dataloader)
        return self._collect_outputs(), dict(state.metrics)


class Trainer(Engine):
    """Ignite trainer with optional validation, testing, and early stopping."""

    def __init__(
        self,
        training_step: TrainingStep,
        *,
        train_metrics: Metrics | None = None,
        lr_scheduler: Any | None = None,
        hp_scheduler: Any | None = None,
        val_evaluator: Evaluator | None = None,
        test_evaluator: Evaluator | None = None,
        score_name: str | None = None,
        score_mode: str = "min",
        prepare_batch: BatchPreparer = default_prepare_batch,
        device: torch.device | str | None = None,
        max_epochs: int = 1,
        val_every_n_epochs: int = 1,
        early_stopping_patience: int | None = None,
        logger: Logger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self.training_step = training_step
        self.train_metrics = train_metrics or Metrics()
        self.lr_scheduler = lr_scheduler
        self.hp_scheduler = hp_scheduler
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self.score_name = score_name
        self.score_mode = score_mode
        self.prepare_batch = prepare_batch
        self.device = device
        self.default_max_epochs = max_epochs
        self.val_every_n_epochs = val_every_n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.run_logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.train_history: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []
        self.test_history: list[dict[str, Any]] = []
        self.latest_train_metrics: dict[str, Any] = {}
        self.latest_validation_metrics: dict[str, Any] = {}
        self.latest_test_metrics: dict[str, Any] = {}
        self.checkpoint_paths: list[str] = []
        self.best_checkpoint_path: str | None = None
        self.last_checkpoint_path: str | None = None
        self.resume_checkpoint_path: str | None = None
        self._val_dataloader: DataLoader | None = None
        self._test_dataloader: DataLoader | None = None

        def _process_function(engine: Engine, batch: Batch) -> dict[str, Any]:
            prepared_batch = self.prepare_batch(batch, self.device)
            output = self.training_step(prepared_batch)
            return self._normalize_output(output)

        super().__init__(_process_function)

        self.add_event_handler(Events.STARTED, self._reset_train_metrics)
        if self.train_metrics:
            self.add_event_handler(Events.ITERATION_COMPLETED, self._update_train_metrics)
            self.add_event_handler(Events.EPOCH_COMPLETED, self._complete_train_epoch)

        if self.lr_scheduler is not None:
            self.add_event_handler(Events.EPOCH_COMPLETED, self._step_lr_scheduler)

        if self.hp_scheduler is not None:
            self.add_event_handler(Events.EPOCH_COMPLETED, self._step_hp_scheduler)

        if self.val_evaluator is not None:
            self.add_event_handler(
                Events.EPOCH_COMPLETED(every=self.val_every_n_epochs),
                self._run_validation,
            )
        elif self.test_evaluator is not None:
            self.add_event_handler(
                Events.EPOCH_COMPLETED(every=self.val_every_n_epochs),
                self._run_test,
            )

        if self.checkpoint_manager is not None:
            self.add_event_handler(Events.EPOCH_COMPLETED, self._save_checkpoint)

        if self.val_evaluator is not None and self.score_name is not None:
            if self.early_stopping_patience is not None:
                early_stopping = EarlyStopping(
                    patience=self.early_stopping_patience,
                    score_function=self._score_from_validation,
                    trainer=self,
                )
                self.val_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    @staticmethod
    def _normalize_output(output: StepOutput) -> dict[str, Any]:
        """Convert common step return shapes into one mapping for Ignite state.output."""
        if output is None:
            return {}
        if isinstance(output, Mapping):
            return dict(output)
        if isinstance(output, torch.Tensor):
            if output.ndim == 0:
                return {"loss": output.item()}
            return {"output": output}
        return {"loss": output}

    def _step_lr_scheduler(self, engine: Engine) -> None:
        if self.lr_scheduler is None:
            return
        self.lr_scheduler.step()

    def _step_hp_scheduler(self, engine: Engine) -> None:
        if self.hp_scheduler is None:
            return
        self.hp_scheduler.step()

    def _reset_train_metrics(self, engine: Engine) -> None:
        self.train_history = []
        self.latest_train_metrics = {}
        self.train_metrics.reset()

    def _update_train_metrics(self, engine: Engine) -> None:
        output = engine.state.output
        if isinstance(output, Mapping):
            self.train_metrics.update(**output)

    def _complete_train_epoch(self, engine: Engine) -> None:
        metrics = self.train_metrics.compute()
        self.latest_train_metrics = metrics
        self.train_history.append(metrics)
        self._log_metrics("train", metrics, step=engine.state.epoch)
        self.train_metrics.reset()

    def _run_validation(self, engine: Engine) -> None:
        if self._val_dataloader is None or self.val_evaluator is None:
            return
        _, metrics = self.val_evaluator.evaluate(self._val_dataloader)
        self.latest_validation_metrics = metrics
        self.validation_history.append(metrics)
        self._log_metrics("validation", metrics, step=engine.state.epoch)

        self._run_test(engine)

    def _run_test(self, engine: Engine) -> None:
        if self.test_evaluator is None or self._test_dataloader is None:
            return
        _, test_metrics = self.test_evaluator.evaluate(self._test_dataloader)
        self.latest_test_metrics = test_metrics
        self.test_history.append(test_metrics)
        self._log_metrics("test", test_metrics, step=engine.state.epoch)

    def _log_metrics(self, prefix: str, metrics: Mapping[str, Any], *, step: int) -> None:
        if self.run_logger is None or not metrics:
            return
        self.run_logger.log_metrics(
            {f"{prefix}.{name}": float(value) for name, value in metrics.items()},
            step=step,
        )

    def _save_checkpoint(self, engine: Engine) -> None:
        if self.checkpoint_manager is None:
            return
        epoch = int(engine.state.epoch)
        if not self.checkpoint_manager.should_save(epoch):
            return
        score = self._checkpoint_score()
        last_path, best_path = self.checkpoint_manager.save(
            engine=engine,
            epoch=epoch,
            step=int(engine.state.iteration),
            score_name=self.score_name,
            score=score,
        )
        for path in (last_path, best_path):
            if path is not None:
                self.checkpoint_paths.append(str(path))
                self._log_artifact(str(path), artifact_path="checkpoints")
        if last_path is not None:
            self.last_checkpoint_path = str(last_path)
        if best_path is not None:
            self.best_checkpoint_path = str(best_path)

    def _checkpoint_score(self) -> float | None:
        if self.score_name is None:
            return None
        if self.score_name in self.latest_validation_metrics:
            return float(self.latest_validation_metrics[self.score_name])
        return None

    def _log_artifact(self, path: str, *, artifact_path: str | None = None) -> None:
        if self.run_logger is None:
            return
        log_artifact = getattr(self.run_logger, "log_artifact", None)
        if log_artifact is not None:
            log_artifact(path, artifact_path=artifact_path)

    def _score_from_validation(self, engine: Engine) -> float:
        if self.score_name is None:
            raise ValueError("score_name must be set when early stopping is enabled")
        metric_value = engine.state.metrics[self.score_name]
        score = float(metric_value)
        if self.score_mode == "max":
            return score
        if self.score_mode == "min":
            return -score
        raise ValueError("score_mode must be 'min' or 'max'")

    def train(
        self,
        train_dataloader: DataLoader,
        *,
        val_dataloader: DataLoader | None = None,
        test_dataloader: DataLoader | None = None,
        max_epochs: int | None = None,
        epoch_length: int | None = None,
    ) -> State:
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        return self.run(
            train_dataloader,
            max_epochs=max_epochs or self.default_max_epochs,
            epoch_length=epoch_length,
        )
