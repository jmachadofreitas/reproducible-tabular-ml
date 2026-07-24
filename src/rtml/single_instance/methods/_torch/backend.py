from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.methods import MethodSpec
from rtml.core.metrics import compute_metrics
from rtml.core.resampling import Resample
from rtml.core.results import PredictionSet
from rtml.core.runtime import RuntimeSpec
from rtml.core.tasks import TaskType
from rtml.loggers import Logger
from rtml.methods.backends.base import BackendResult, MethodBackend
from rtml.methods.engines import (
    CheckpointManager,
    Evaluator,
    TorchFitConfig,
    Trainer,
    create_lr_scheduler,
    create_optimizer,
)
from rtml.single_instance.methods._torch.common.helpers import (
    as_float32_array,
    require_supervised_target,
    resolve_device,
    seed_torch,
    target_tensors,
)
from rtml.single_instance.methods._torch.common.outputs import build_prediction_set
from rtml.single_instance.methods._torch.common.bundles import (
    DataLoaderBundle,
    TensorDatasetBundle,
    TorchModelBundle,
)
from rtml.single_instance.methods._torch.common.task_adapters import (
    infer_score_mode,
    resolve_score_name,
)
from rtml.single_instance.methods._torch.mlp.factory import build_mlp_bundle
from rtml.single_instance.preprocessing import build_preprocessor
from rtml.core.tasks import TaskSpec


class TorchModelBuilder(Protocol):
    """Build a torch model bundle for one model kind and prepared input shape."""

    def __call__(
        self,
        *,
        task: TaskSpec,
        input_dim: int,
        n_classes: int | None,
        params: Mapping[str, Any],
        fit_config: TorchFitConfig,
        device: torch.device,
    ) -> TorchModelBundle: ...


class TorchBackend(MethodBackend):
    """Single-instance backend for methods implemented with torch."""

    name = "torch"
    DEFAULT_MODEL_BUILDERS: Mapping[str, TorchModelBuilder] = {
        "simple_mlp": build_mlp_bundle,
    }

    def __init__(
        self,
        model_builders: Mapping[str, TorchModelBuilder] | None = None,
    ) -> None:
        self._model_builders = dict(model_builders or self.DEFAULT_MODEL_BUILDERS)
        if not self._model_builders:
            raise ValueError("torch backend requires at least one model builder")

    def validate_method(self, method: MethodSpec) -> None:
        if method.model.kind not in self._model_builders:
            supported = ", ".join(sorted(self._model_builders)) or "<none>"
            raise ValueError(
                f"torch backend does not support model kind {method.model.kind!r}; "
                f"supported model kinds: {supported}"
            )

    def run(
        self,
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        resample_id: str | None = None,
        seed: int = 0,
        runtime: RuntimeSpec | None = None,
        logger: Logger | None = None,
    ) -> BackendResult:
        self.validate_method(method)
        case.task.validate_columns(case.dataset)
        resample = case.resampling.get_resample(resample_id)
        device = resolve_device(runtime)
        generator = seed_torch(
            seed, deterministic=None if runtime is None else runtime.deterministic
        )
        transform_config, policy = self._preprocessing_config(method)

        fit_start = perf_counter()
        data = self._prepare_data(
            case=case,
            resample=resample,
            policy=policy,
            transform_config=transform_config,
        )
        bundle = self._build_model_bundle(
            case=case,
            method=method,
            input_dim=data.input_dim,
            n_classes=None if data.classes is None else len(data.classes),
            device=device,
        )
        self._validate_task(case=case)
        loaders = self._build_loaders(
            case=case,
            data=data,
            bundle=bundle,
            seed=seed,
            generator=generator,
            device=device,
        )
        trainer = self._build_trainer(
            case=case,
            method=method,
            resample=resample,
            bundle=bundle,
            loaders=loaders,
            device=device,
            logger=logger,
            seed=seed,
        )
        trainer.train(
            loaders.train,
            val_dataloader=loaders.validation,
            test_dataloader=loaders.test if loaders.test_evaluator is not None else None,
            max_epochs=bundle.fit_config.max_epochs,
        )
        fit_time = perf_counter() - fit_start

        predict_start = perf_counter()
        predictions = self._evaluate(
            case=case,
            method=method,
            resample=resample,
            bundle=bundle,
            test_loader=loaders.test,
            classes=data.classes,
            device=device,
        )
        predict_time = perf_counter() - predict_start

        return BackendResult(
            predictions=predictions,
            metrics=compute_metrics(case.task.metrics, predictions),
            fit_time=fit_time,
            predict_time=predict_time,
            metadata=self._metadata(bundle=bundle, policy=policy, device=device, trainer=trainer),
        )

    def _preprocessing_config(self, method: MethodSpec) -> tuple[dict[str, Any], str]:
        transform_config = dict(method.transform)
        policy = transform_config.pop("policy", "neural_default")
        return transform_config, policy

    def _model_builder_for(self, method: MethodSpec) -> TorchModelBuilder:
        self.validate_method(method)
        return self._model_builders[method.model.kind]

    def _prepare_data(
        self,
        *,
        case: BenchmarkCase,
        resample: Resample,
        policy: str,
        transform_config: dict[str, Any],
    ) -> TensorDatasetBundle:
        x = case.task.source_frame(case.dataset)
        y = require_supervised_target(case)
        x_train = x.iloc[resample.train_idx]
        y_train = y.iloc[resample.train_idx]
        x_test = x.iloc[resample.test_idx]
        y_test = y.iloc[resample.test_idx]

        preprocessor = build_preprocessor(
            dataset=case.dataset,
            task=case.task,
            policy=policy,
            options=transform_config,
        )
        x_train_array = as_float32_array(preprocessor.fit_transform(x_train, y_train))
        x_test_array = as_float32_array(preprocessor.transform(x_test))
        y_train_tensor, y_test_tensor, classes = target_tensors(
            task=case.task,
            y_train=y_train,
            y_eval=y_test,
        )
        if case.task.task_type == TaskType.BINARY_CLASSIFICATION and (
            classes is None or len(classes) != 2
        ):
            raise ValueError(
                "binary classification requires exactly two classes in the training split"
            )

        return TensorDatasetBundle(
            train_dataset=TensorDataset(torch.as_tensor(x_train_array), y_train_tensor),
            test_dataset=TensorDataset(torch.as_tensor(x_test_array), y_test_tensor),
            y_train_tensor=y_train_tensor,
            classes=classes,
            input_dim=x_train_array.shape[1],
        )

    def _build_model_bundle(
        self,
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        input_dim: int,
        n_classes: int | None,
        device: torch.device,
    ) -> TorchModelBundle:
        return self._model_builder_for(method)(
            task=case.task,
            input_dim=input_dim,
            n_classes=n_classes,
            params=method.model.params,
            fit_config=TorchFitConfig.from_mapping(method.fit),
            device=device,
        )

    def _validate_task(self, *, case: BenchmarkCase) -> None:
        if case.task.task_type == TaskType.UNSUPERVISED:
            raise ValueError("torch backend currently supports supervised tasks only")

    def _build_loaders(
        self,
        *,
        case: BenchmarkCase,
        data: TensorDatasetBundle,
        bundle: TorchModelBundle,
        seed: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> DataLoaderBundle:
        train_dataset: TensorDataset | Subset = data.train_dataset
        validation_loader = None
        validation_evaluator = None

        if bundle.fit_config.validation_fraction > 0:
            train_indices, validation_indices = train_test_split(
                np.arange(len(data.train_dataset)),
                test_size=bundle.fit_config.validation_fraction,
                random_state=seed,
                shuffle=True,
                stratify=self._stratify_values(case=case, data=data),
            )
            train_dataset = Subset(data.train_dataset, train_indices.tolist())
            validation_dataset = Subset(data.train_dataset, validation_indices.tolist())
            validation_loader = DataLoader(
                validation_dataset, batch_size=bundle.fit_config.batch_size
            )
            validation_evaluator = self._build_evaluator(
                bundle=bundle,
                device=device,
                with_metrics=True,
            )

        return DataLoaderBundle(
            train=DataLoader(
                train_dataset,
                batch_size=bundle.fit_config.batch_size,
                shuffle=True,
                generator=generator,
            ),
            validation=validation_loader,
            test=DataLoader(
                data.test_dataset, batch_size=bundle.fit_config.batch_size, shuffle=False
            ),
            validation_evaluator=validation_evaluator,
            test_evaluator=self._build_test_evaluator(bundle=bundle, device=device),
        )

    def _stratify_values(
        self,
        *,
        case: BenchmarkCase,
        data: TensorDatasetBundle,
    ) -> np.ndarray | None:
        """Return stratification values for train/validation split, if applicable."""
        if case.task.task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }:
            return data.y_train_tensor.numpy()
        return None

    def _build_evaluator(
        self,
        *,
        bundle: TorchModelBundle,
        device: torch.device,
        with_metrics: bool,
    ) -> Evaluator:
        metrics = bundle.make_validation_metrics() if with_metrics else None
        return Evaluator(bundle.make_evaluation_step(), metrics=metrics, device=device)

    def _build_test_evaluator(
        self,
        *,
        bundle: TorchModelBundle,
        device: torch.device,
    ) -> Evaluator | None:
        if not bundle.fit_config.tracking.get("log_test_metrics", False):
            return None
        return Evaluator(
            bundle.make_evaluation_step(),
            metrics=bundle.make_test_metrics(),
            device=device,
        )

    def _build_trainer(
        self,
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        resample: Resample,
        bundle: TorchModelBundle,
        loaders: DataLoaderBundle,
        device: torch.device,
        logger: Logger | None,
        seed: int,
    ) -> Trainer:
        score_name = resolve_score_name(case.task)
        optimizer = create_optimizer(bundle.model, **bundle.fit_config.optimizer)
        lr_scheduler = create_lr_scheduler(
            optimizer,
            config=None
            if bundle.fit_config.lr_scheduler is None
            else dict(bundle.fit_config.lr_scheduler),
            max_epochs=bundle.fit_config.max_epochs,
        )
        if bundle.fit_config.hp_scheduler is not None:
            raise NotImplementedError(
                "hp_scheduler requires method-owned hyperparameter state; "
                "only lr_scheduler is wired for now"
            )
        checkpoint_manager = self._build_checkpoint_manager(
            case=case,
            method=method,
            resample=resample,
            bundle=bundle,
            score_mode=infer_score_mode(score_name),
            seed=seed,
        )

        trainer = Trainer(
            bundle.create_training_step(optimizer),
            train_metrics=bundle.make_train_metrics(),
            lr_scheduler=lr_scheduler,
            val_evaluator=loaders.validation_evaluator,
            test_evaluator=loaders.test_evaluator,
            score_name=score_name,
            score_mode=infer_score_mode(score_name),
            device=device,
            max_epochs=bundle.fit_config.max_epochs,
            val_every_n_epochs=bundle.fit_config.every_n_epochs,
            early_stopping_patience=bundle.fit_config.early_stopping_patience,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
        )
        if checkpoint_manager is not None:
            objects: dict[str, Any] = {
                "model": bundle.model,
                "optimizer": optimizer,
                "trainer": trainer,
            }
            if lr_scheduler is not None:
                objects["lr_scheduler"] = lr_scheduler
            checkpoint_manager.set_objects(objects)
            resume_path = checkpoint_manager.load_resume_checkpoint()
            trainer.resume_checkpoint_path = None if resume_path is None else str(resume_path)
        return trainer

    def _build_checkpoint_manager(
        self,
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        resample: Resample,
        bundle: TorchModelBundle,
        score_mode: str,
        seed: int,
    ) -> CheckpointManager | None:
        config = dict(bundle.fit_config.checkpoint)
        enabled = bool(config.pop("enabled", bool(config.get("resume_from"))))
        if not config and not enabled:
            return None
        if not enabled:
            return None
        directory = self._checkpoint_dir(
            case=case,
            method=method,
            resample=resample,
            seed=seed,
            root=config.pop("dir", ".runs/checkpoints"),
        )
        score_mode_name = str(config.pop("score_mode", score_mode))
        save_last = bool(config.pop("save_last", True))
        save_best = bool(config.pop("save_best", True))
        every_n_epochs = int(config.pop("every_n_epochs", 1))
        delay_n_epochs = int(config.pop("delay_n_epochs", 0))
        n_saved = config.pop("n_saved", 1)
        atomic = bool(config.pop("atomic", True))
        require_empty = bool(config.pop("require_empty", False))
        resume_from = config.pop("resume_from", None)
        if config:
            unknown = ", ".join(sorted(config))
            raise ValueError(f"unknown torch checkpoint config: {unknown}")
        return CheckpointManager(
            directory=directory,
            score_mode=score_mode_name,
            save_last=save_last,
            save_best=save_best,
            every_n_epochs=every_n_epochs,
            delay_n_epochs=delay_n_epochs,
            n_saved=None if n_saved is None else int(n_saved),
            atomic=atomic,
            require_empty=require_empty,
            resume_from=resume_from,
        )

    @staticmethod
    def _checkpoint_dir(
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        resample: Resample,
        seed: int,
        root: str | Path,
    ) -> Path:
        root = Path(root)
        return (
            root
            / _safe_path_part(case.name)
            / _safe_path_part(method.name)
            / _safe_path_part(resample.id)
            / f"seed_{seed}"
        )

    def _evaluate(
        self,
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        resample: Resample,
        bundle: TorchModelBundle,
        test_loader: DataLoader,
        classes: np.ndarray | None,
        device: torch.device,
    ) -> PredictionSet:
        evaluator = self._build_evaluator(bundle=bundle, device=device, with_metrics=False)
        outputs, _ = evaluator.evaluate(test_loader)
        return build_prediction_set(
            case=case,
            method=method,
            resample_id=resample.id,
            test_indices=resample.test_idx,
            outputs=outputs,
            classes=classes,
        )

    def _metadata(
        self,
        *,
        bundle: TorchModelBundle,
        policy: str,
        device: torch.device,
        trainer: Trainer,
    ) -> dict[str, Any]:
        metadata = {
            "preprocessing_policy": policy,
            "device": str(device),
            "max_epochs": bundle.fit_config.max_epochs,
            "batch_size": bundle.fit_config.batch_size,
            **dict(bundle.metadata),
        }
        if bundle.fit_config.tracking.get("store_history", False):
            metadata["train_history"] = trainer.train_history
            metadata["validation_history"] = trainer.validation_history
            metadata["test_history"] = trainer.test_history
        if trainer.last_checkpoint_path is not None:
            metadata["last_checkpoint_path"] = trainer.last_checkpoint_path
        if trainer.best_checkpoint_path is not None:
            metadata["best_checkpoint_path"] = trainer.best_checkpoint_path
        if trainer.resume_checkpoint_path is not None:
            metadata["resume_checkpoint_path"] = trainer.resume_checkpoint_path
        return metadata


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "run"
