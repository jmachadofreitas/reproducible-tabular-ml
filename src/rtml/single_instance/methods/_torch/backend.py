from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.datasets import Dataset
from rtml.core.methods import MethodSpec, ModelSpec
from rtml.core.metrics import EvaluationMetrics
from rtml.core.resampling import Resample
from rtml.core.results import PredictionSet
from rtml.core.runtime import RuntimeSpec
from rtml.core.tasks import MetricSpec, TaskSpec, TaskType
from rtml.loggers import Logger
from rtml.methods.backends.base import BackendResult, MethodBackend
from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.checkpointing import (
    CheckpointManager,
    build_checkpoint_manager,
    checkpoint_directory,
)
from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.core import Evaluator, Trainer
from rtml.methods.engines.fitting import fit_model_bundle
from rtml.methods.engines.runtime import resolve_device, seed_torch
from rtml.methods.engines.task_adapters import (
    infer_score_mode,
    require_supervised_target,
    resolve_score_metric,
    target_tensors,
)
from rtml.single_instance.methods._torch.data import (
    DataLoaderBundle,
    TensorDatasetBundle,
    as_float32_array,
)
from rtml.single_instance.methods._torch.fitted import TorchFittedMethod
from rtml.single_instance.methods._torch.mlp.factory import build_mlp_bundle
from rtml.single_instance.methods._torch.outputs import build_prediction_set
from rtml.single_instance.methods._torch.tabm.factory import build_tabm_bundle
from rtml.single_instance.preprocessing import build_preprocessor


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
        "tabm": build_tabm_bundle,
    }

    def __init__(
        self,
        model_builders: Mapping[str, TorchModelBuilder] | None = None,
    ) -> None:
        builders = self.DEFAULT_MODEL_BUILDERS if model_builders is None else model_builders
        self._model_builders = dict(builders)
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
        model_builder = self._model_builder_for(method)
        case.task.validate_columns(case.dataset)
        if case.task.task_type == TaskType.UNSUPERVISED:
            raise ValueError("torch backend currently supports supervised tasks only")
        resample = case.resampling.get_resample(resample_id)
        fit_config = TorchFitConfig.from_mapping(method.fit)
        resample = self._with_validation_split(
            case=case,
            resample=resample,
            fit_config=fit_config,
            seed=seed,
        )
        device = resolve_device(runtime)
        generator = seed_torch(
            seed, deterministic=None if runtime is None else runtime.deterministic
        )
        transform_config = dict(method.transform)
        policy = transform_config.pop("policy", "neural_default")

        fit_start = perf_counter()
        data = self._prepare_data(
            case=case,
            resample=resample,
            policy=policy,
            transform_config=transform_config,
        )
        bundle = model_builder(
            task=case.task,
            input_dim=data.input_dim,
            n_classes=None if data.classes is None else len(data.classes),
            params=method.model.params,
            fit_config=fit_config,
            device=device,
        )
        loaders = self._build_loaders(
            data=data,
            bundle=bundle,
            generator=generator,
        )
        score_metric: MetricSpec = resolve_score_metric(
            case.task.primary_metric,
            case.task.metrics,
        )
        score_mode = infer_score_mode(score_metric)
        trainer = fit_model_bundle(
            bundle,
            loaders.train,
            validation_dataloader=loaders.validation,
            test_dataloader=loaders.test,
            score_name=score_metric.name,
            score_mode=score_mode,
            device=device,
            logger=logger,
            checkpoint_manager=self._build_checkpoint_manager(
                case=case,
                method=method,
                resample=resample,
                bundle=bundle,
                score_mode=score_mode,
                seed=seed,
            ),
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
            metrics=EvaluationMetrics(case.task.metrics).compute(predictions),
            fit_time=fit_time,
            predict_time=predict_time,
            metadata=self._metadata(bundle=bundle, policy=policy, device=device, trainer=trainer),
        )

    def _model_builder_for(self, method: MethodSpec) -> TorchModelBuilder:
        self.validate_method(method)
        return self._model_builders[method.model.kind]

    @staticmethod
    def _with_validation_split(
        *,
        case: BenchmarkCase,
        resample: Resample,
        fit_config: TorchFitConfig,
        seed: int,
    ) -> Resample:
        if fit_config.validation_fraction <= 0:
            return resample
        if resample.valid_idx is not None:
            raise ValueError(
                "validation is defined by both resample.valid_idx and fit.validation_fraction"
            )

        target = require_supervised_target(case)
        stratify = None
        if case.task.task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }:
            stratify = target.iloc[resample.train_idx]
        train_idx, valid_idx = train_test_split(
            resample.train_idx,
            test_size=fit_config.validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        return Resample(
            id=resample.id,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=resample.test_idx,
            metadata={
                **resample.metadata,
                "validation_fraction": fit_config.validation_fraction,
            },
        )

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
        x_validation = None if resample.valid_idx is None else x.iloc[resample.valid_idx]
        y_validation = None if resample.valid_idx is None else y.iloc[resample.valid_idx]
        x_test = x.iloc[resample.test_idx]
        y_test = y.iloc[resample.test_idx]

        preprocessor = build_preprocessor(
            dataset=case.dataset,
            task=case.task,
            policy=policy,
            options=transform_config,
        )
        x_train_array = as_float32_array(preprocessor.fit_transform(x_train, y_train))
        x_validation_array = (
            None if x_validation is None else as_float32_array(preprocessor.transform(x_validation))
        )
        x_test_array = as_float32_array(preprocessor.transform(x_test))
        y_train_tensor, y_test_tensor, classes = target_tensors(
            task_type=case.task.task_type,
            y_train=y_train,
            y_eval=y_test,
        )
        validation_dataset = None
        if x_validation_array is not None and y_validation is not None:
            _, y_validation_tensor, _ = target_tensors(
                task_type=case.task.task_type,
                y_train=y_train,
                y_eval=y_validation,
            )
            validation_dataset = TensorDataset(
                torch.as_tensor(x_validation_array),
                y_validation_tensor,
            )

        return TensorDatasetBundle(
            train=TensorDataset(torch.as_tensor(x_train_array), y_train_tensor),
            validation=validation_dataset,
            test=TensorDataset(torch.as_tensor(x_test_array), y_test_tensor),
            classes=classes,
            input_dim=x_train_array.shape[1],
        )

    def _build_loaders(
        self,
        *,
        data: TensorDatasetBundle,
        bundle: TorchModelBundle,
        generator: torch.Generator,
    ) -> DataLoaderBundle:
        batch_size = bundle.fit_config.batch_size
        return DataLoaderBundle(
            train=DataLoader(
                data.train,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            ),
            validation=None
            if data.validation is None
            else DataLoader(data.validation, batch_size=batch_size, shuffle=False),
            test=DataLoader(
                data.test,
                batch_size=batch_size,
                shuffle=False,
            ),
        )

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
        config.setdefault("save_best", resample.valid_idx is not None)
        directory = checkpoint_directory(
            config.pop("dir", ".runs/checkpoints"),
            case_name=case.name,
            method_name=method.name,
            resample_id=resample.id,
            seed=seed,
        )
        return build_checkpoint_manager(
            config,
            directory=directory,
            default_score_mode=score_mode,
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
        evaluator = Evaluator(bundle.evaluation_step, device=device)
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
