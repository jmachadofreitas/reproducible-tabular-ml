from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import joblib
import numpy as np
import torch
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
from rtml.methods.backends.base import BackendRefitResult, BackendResult, MethodBackend
from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.checkpointing import build_checkpoint_manager
from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.core import (
    Evaluator,
    Trainer,
    as_float32_array,
    resolve_device,
    seed_torch,
)
from rtml.methods.engines.fitting import fit_model_bundle
from rtml.methods.engines.task_adapters import (
    build_supervised_prediction_set,
    infer_score_mode,
    resolve_score_metric,
    target_tensors,
)
from rtml.single_instance.methods._torch.data import (
    DataLoaderBundle,
    TensorDatasetBundle,
)
from rtml.single_instance.methods._torch.fitted import TorchFittedMethod
from rtml.single_instance.methods._torch.mlp.factory import build_mlp_bundle
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
        if case.task.sample_weight is not None:
            raise ValueError("torch backend does not support sample-weighted tasks")
        resample = case.resampling.get_resample(resample_id)
        fit_config = TorchFitConfig.from_mapping(method.fit)
        device = resolve_device(runtime)
        generator = seed_torch(seed)
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
        score_metric = resolve_score_metric(
            case.task.primary_metric,
            case.task.metrics,
        )
        score_name = None if score_metric is None else score_metric.name
        score_mode = "min" if score_metric is None else infer_score_mode(score_metric)
        trainer = fit_model_bundle(
            bundle,
            loaders.train,
            validation_dataloader=loaders.validation,
            test_dataloader=loaders.test,
            score_name=score_name,
            score_mode=score_mode,
            device=device,
            deterministic=None if runtime is None else runtime.deterministic,
            logger=logger,
            checkpoint_manager=build_checkpoint_manager(
                bundle.fit_config.checkpoint,
                case_name=case.name,
                method_name=method.name,
                resample_id=resample.id,
                seed=seed,
                default_score_mode=score_mode,
                default_save_best=resample.valid_idx is not None and score_name is not None,
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

    def refit(
        self,
        *,
        dataset: Any,
        task: Any,
        method: MethodSpec,
        artifact_dir: Path,
        seed: int = 0,
        runtime: RuntimeSpec | None = None,
        logger: Logger | None = None,
    ) -> BackendRefitResult:
        """Fit and save a complete single-instance Torch method on all labeled rows."""
        if not isinstance(dataset, Dataset) or not isinstance(task, TaskSpec):
            raise TypeError("torch refit requires a single-instance Dataset and TaskSpec")
        model_builder = self._model_builder_for(method)
        task.validate_columns(dataset)
        if task.task_type == TaskType.UNSUPERVISED:
            raise ValueError("torch refit currently supports supervised tasks only")
        if task.sample_weight is not None:
            raise ValueError("torch refit does not support sample-weighted tasks")
        target = task.target_series(dataset)
        if target is None:
            raise ValueError("torch refit requires a supervised task target")
        if target.isna().any():
            raise ValueError("torch refit requires a target value for every training row")

        fit_config = TorchFitConfig.from_mapping(method.fit)
        self._validate_refit_config(fit_config)
        device = resolve_device(runtime)
        generator = seed_torch(seed)
        transform_config = dict(method.transform)
        policy = transform_config.pop("policy", "neural_default")

        fit_start = perf_counter()
        preprocessor = build_preprocessor(
            dataset=dataset,
            task=task,
            policy=policy,
            options=transform_config,
        )
        inputs = as_float32_array(preprocessor.fit_transform(task.source_frame(dataset), target))
        targets, _, classes = target_tensors(
            task_type=task.task_type,
            y_train=target,
            y_eval=target,
        )
        bundle = model_builder(
            task=task,
            input_dim=inputs.shape[1],
            n_classes=None if classes is None else len(classes),
            params=method.model.params,
            fit_config=fit_config,
            device=device,
        )
        train_loader = DataLoader(
            TensorDataset(torch.as_tensor(inputs), targets),
            batch_size=fit_config.batch_size,
            shuffle=True,
            generator=generator,
        )
        fit_model_bundle(
            bundle,
            train_loader,
            device=device,
            deterministic=None if runtime is None else runtime.deterministic,
            logger=logger,
        )
        fit_time = perf_counter() - fit_start

        preprocessor_path = artifact_dir / "preprocessor.joblib"
        model_path = artifact_dir / "model.pt"
        joblib.dump(preprocessor, preprocessor_path)
        torch.save(bundle.model.state_dict(), model_path)
        fitted_method = TorchFittedMethod(
            preprocessor=preprocessor,
            model_bundle=bundle,
            source_columns=task.source,
            classes=classes,
            device=device,
        )
        return BackendRefitResult(
            fitted_method=fitted_method,
            artifact_paths={
                "preprocessor": preprocessor_path,
                "model": model_path,
            },
            artifact_formats={
                "preprocessor": "joblib",
                "model": "torch_state_dict",
            },
            training_size=len(dataset),
            input_schema={name: dataset.schema.get(name) for name in task.source},
            fit_time=fit_time,
            metadata={
                "preprocessing_policy": policy,
                "input_dim": inputs.shape[1],
                "classes": None if classes is None else classes.tolist(),
                **dict(bundle.metadata),
            },
        )

    def load_refit(
        self,
        *,
        artifact_dir: Path,
        manifest: Mapping[str, Any],
        runtime: RuntimeSpec | None = None,
    ) -> TorchFittedMethod:
        """Reconstruct a trusted fitted Torch method from native artifacts."""
        method_data = manifest["method"]
        model_data = method_data["model"]
        method = MethodSpec(
            name=method_data["name"],
            transform=method_data["transform"],
            model=ModelSpec(
                kind=model_data["kind"],
                backend=model_data["backend"],
                params=model_data["params"],
            ),
            fit=method_data["fit"],
            metadata=method_data.get("metadata", {}),
        )
        task_data = manifest["task"]
        task = TaskSpec(
            name=task_data["name"],
            task_type=TaskType(task_data["task_type"]),
            source=task_data["source"],
            target=task_data["target"],
            sample_weight=task_data.get("sample_weight"),
            groups=task_data.get("groups", []),
            metrics=[MetricSpec(**metric) for metric in task_data.get("metrics", [])],
            primary_metric=task_data.get("primary_metric"),
            metadata=task_data.get("metadata", {}),
        )
        device = resolve_device(runtime)
        fit_config = TorchFitConfig.from_mapping(method.fit)
        classes_value = manifest["metadata"].get("classes")
        classes = None if classes_value is None else np.asarray(classes_value)
        bundle = self._model_builder_for(method)(
            task=task,
            input_dim=int(manifest["metadata"]["input_dim"]),
            n_classes=None if classes is None else len(classes),
            params=method.model.params,
            fit_config=fit_config,
            device=device,
        )
        model_artifact = manifest["artifacts"]["model"]
        if model_artifact["format"] != "torch_state_dict":
            raise ValueError(f"unsupported torch model format {model_artifact['format']!r}")
        state = torch.load(
            artifact_dir / model_artifact["path"],
            map_location=device,
            weights_only=True,
        )
        bundle.model.load_state_dict(state)
        preprocessor_artifact = manifest["artifacts"]["preprocessor"]
        if preprocessor_artifact["format"] != "joblib":
            raise ValueError(
                f"unsupported torch preprocessor format {preprocessor_artifact['format']!r}"
            )
        preprocessor = joblib.load(artifact_dir / preprocessor_artifact["path"])
        return TorchFittedMethod(
            preprocessor=preprocessor,
            model_bundle=bundle,
            source_columns=task.source,
            classes=classes,
            device=device,
        )

    def _model_builder_for(self, method: MethodSpec) -> TorchModelBuilder:
        self.validate_method(method)
        return self._model_builders[method.model.kind]

    @staticmethod
    def _validate_refit_config(fit_config: TorchFitConfig) -> None:
        if fit_config.early_stopping_patience is not None:
            raise ValueError("torch refit cannot use early stopping without validation data")
        checkpoint = fit_config.checkpoint
        checkpoint_enabled = bool(checkpoint.get("enabled", bool(checkpoint.get("resume_from"))))
        if checkpoint_enabled:
            raise ValueError("torch refit artifacts are separate from resumable checkpoints")

    def _prepare_data(
        self,
        *,
        case: BenchmarkCase,
        resample: Resample,
        policy: str,
        transform_config: dict[str, Any],
    ) -> TensorDatasetBundle:
        target_column = case.task.target
        if target_column is None:
            raise ValueError("torch methods require a supervised task target")
        train_data = case.dataset[resample.train_idx]
        test_data = case.dataset[resample.test_idx]
        validation_data = None if resample.valid_idx is None else case.dataset[resample.valid_idx]
        x_train = train_data.loc[:, case.task.source]
        y_train = train_data.loc[:, target_column]
        x_validation = None if validation_data is None else validation_data.loc[:, case.task.source]
        y_validation = None if validation_data is None else validation_data.loc[:, target_column]
        x_test = test_data.loc[:, case.task.source]
        y_test = test_data.loc[:, target_column]

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
        return build_supervised_prediction_set(
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
        if trainer.last_checkpoint_path is not None:
            metadata["last_checkpoint_path"] = trainer.last_checkpoint_path
        if trainer.best_checkpoint_path is not None:
            metadata["best_checkpoint_path"] = trainer.best_checkpoint_path
        if trainer.resume_checkpoint_path is not None:
            metadata["resume_checkpoint_path"] = trainer.resume_checkpoint_path
        if trainer.best_epoch is not None:
            metadata["selected_epoch"] = trainer.best_epoch
        return metadata
