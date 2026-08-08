from collections.abc import Mapping
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.methods import MethodSpec
from rtml.core.metrics import EvaluationMetrics
from rtml.core.resampling import Resample
from rtml.core.results import PredictionSet
from rtml.core.runtime import RuntimeSpec
from rtml.core.tasks import MetricSpec, TaskType
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
    resolve_score_metric,
    target_tensors,
)
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.methods._torch.attention_deep_mil.factory import (
    build_attention_deep_mil_bundle,
)
from rtml.multi_instance.methods._torch.data import (
    BagDatasetBundle,
    BagLoaderBundle,
    BagTensorDataset,
    as_float32_array,
    collate_bags,
)
from rtml.multi_instance.methods._torch.deep_sets.factory import build_deep_sets_bundle
from rtml.multi_instance.methods._torch.outputs import build_prediction_set
from rtml.multi_instance.preprocessing import build_preprocessor
from rtml.multi_instance.tasks import MultiInstanceTask


class MultiInstanceTorchModelBuilder(Protocol):
    """Build one multi-instance Torch model for prepared bag features."""

    def __call__(
        self,
        *,
        task: MultiInstanceTask,
        input_dim: int,
        n_classes: int | None,
        params: Mapping[str, Any],
        fit_config: TorchFitConfig,
        device: torch.device,
    ) -> TorchModelBundle: ...


class MultiInstanceTorchBackend(MethodBackend):
    """Adapt bag-structured RTML cases to the generic Torch/Ignite engine."""

    name = "torch"
    DEFAULT_MODEL_BUILDERS: Mapping[str, MultiInstanceTorchModelBuilder] = {
        "attention_deep_mil": build_attention_deep_mil_bundle,
        "deep_sets": build_deep_sets_bundle,
    }

    def __init__(
        self,
        model_builders: Mapping[str, MultiInstanceTorchModelBuilder] | None = None,
    ) -> None:
        builders = self.DEFAULT_MODEL_BUILDERS if model_builders is None else model_builders
        self._model_builders = dict(builders)
        if not self._model_builders:
            raise ValueError("multi-instance torch backend requires at least one model builder")

    def validate_method(self, method: MethodSpec) -> None:
        if method.model.kind not in self._model_builders:
            supported = ", ".join(sorted(self._model_builders))
            raise ValueError(
                f"multi-instance torch backend does not support model kind "
                f"{method.model.kind!r}; supported model kinds: {supported}"
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
        dataset, task = self._validate_case(case)
        model_builder = self._model_builder_for(method)
        resample = case.resampling.get_resample(resample_id)
        device = resolve_device(runtime)
        generator = seed_torch(
            seed,
            deterministic=None if runtime is None else runtime.deterministic,
        )
        fit_config = TorchFitConfig.from_mapping(method.fit)
        self._validate_fit_config(fit_config, resample)
        transform_options, policy = self._preprocessing_config(method)

        fit_start = perf_counter()
        data = self._prepare_data(
            dataset=dataset,
            task=task,
            resample=resample,
            policy=policy,
            transform_options=transform_options,
        )
        bundle = model_builder(
            task=task,
            input_dim=data.input_dim,
            n_classes=None if data.classes is None else len(data.classes),
            params=method.model.params,
            fit_config=fit_config,
            device=device,
        )
        loaders = self._build_loaders(data, bundle, generator=generator)
        score_metric: MetricSpec = resolve_score_metric(task.primary_metric, task.metrics)
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
            metrics=EvaluationMetrics(task.metrics).compute(predictions),
            fit_time=fit_time,
            predict_time=predict_time,
            metadata=self._metadata(
                bundle=bundle,
                policy=policy,
                device=device,
                trainer=trainer,
            ),
        )

    def _model_builder_for(self, method: MethodSpec) -> MultiInstanceTorchModelBuilder:
        self.validate_method(method)
        return self._model_builders[method.model.kind]

    @staticmethod
    def _validate_case(
        case: BenchmarkCase,
    ) -> tuple[MultiInstanceDataset, MultiInstanceTask]:
        if not isinstance(case.dataset, MultiInstanceDataset):
            raise TypeError("multi-instance torch backend requires MultiInstanceDataset")
        if not isinstance(case.task, MultiInstanceTask):
            raise TypeError("multi-instance torch backend requires MultiInstanceTask")
        case.task.validate_columns(case.dataset)
        if case.task.task_type == TaskType.UNSUPERVISED:
            raise ValueError("multi-instance torch backend supports supervised tasks only")
        return case.dataset, case.task

    @staticmethod
    def _validate_fit_config(fit_config: TorchFitConfig, resample: Resample) -> None:
        has_validation = resample.valid_idx is not None
        if fit_config.validation_fraction:
            raise ValueError("multi-instance validation must be materialized in resample.valid_idx")
        if fit_config.early_stopping_patience is not None and not has_validation:
            raise ValueError("early stopping requires saved resample.valid_idx")
        checkpoint = fit_config.checkpoint
        checkpoint_enabled = bool(checkpoint.get("enabled", bool(checkpoint.get("resume_from"))))
        if (
            checkpoint_enabled
            and checkpoint.get("save_best", has_validation)
            and not has_validation
        ):
            raise ValueError("best checkpoint selection requires saved resample.valid_idx")

    @staticmethod
    def _preprocessing_config(method: MethodSpec) -> tuple[dict[str, Any], str]:
        options = dict(method.transform)
        policy = str(options.pop("policy", "neural_default"))
        return options, policy

    @staticmethod
    def _prepare_data(
        *,
        dataset: MultiInstanceDataset,
        task: MultiInstanceTask,
        resample: Resample,
        policy: str,
        transform_options: Mapping[str, Any],
    ) -> BagDatasetBundle:
        train_frame, train_offsets = dataset[resample.train_idx]
        test_frame, test_offsets = dataset[resample.test_idx]
        preprocessor = build_preprocessor(
            dataset,
            task,
            policy=policy,
            options=transform_options,
        )
        columns = task.instance_source
        train_instances = as_float32_array(preprocessor.fit_transform(train_frame.loc[:, columns]))
        test_instances = as_float32_array(preprocessor.transform(test_frame.loc[:, columns]))

        target = task.target_series(dataset)
        if target is None:
            raise ValueError("multi-instance torch methods require a bag-level target")
        y_train = target.iloc[resample.train_idx]
        y_test = target.iloc[resample.test_idx]
        train_targets, test_targets, classes = target_tensors(
            task_type=task.task_type,
            y_train=y_train,
            y_eval=y_test,
        )
        validation_dataset = None
        if resample.valid_idx is not None:
            validation_frame, validation_offsets = dataset[resample.valid_idx]
            validation_instances = as_float32_array(
                preprocessor.transform(validation_frame.loc[:, columns])
            )
            _, validation_targets, _ = target_tensors(
                task_type=task.task_type,
                y_train=y_train,
                y_eval=target.iloc[resample.valid_idx],
            )
            validation_dataset = BagTensorDataset(
                validation_instances,
                validation_offsets,
                validation_targets,
            )

        return BagDatasetBundle(
            train=BagTensorDataset(train_instances, train_offsets, train_targets),
            validation=validation_dataset,
            test=BagTensorDataset(test_instances, test_offsets, test_targets),
            classes=classes,
            input_dim=train_instances.shape[1],
        )

    @staticmethod
    def _build_loaders(
        data: BagDatasetBundle,
        bundle: TorchModelBundle,
        *,
        generator: torch.Generator,
    ) -> BagLoaderBundle:
        batch_size = bundle.fit_config.batch_size
        return BagLoaderBundle(
            train=DataLoader(
                data.train,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                collate_fn=collate_bags,
            ),
            validation=None
            if data.validation is None
            else DataLoader(
                data.validation,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_bags,
            ),
            test=DataLoader(
                data.test,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_bags,
            ),
        )

    @staticmethod
    def _build_checkpoint_manager(
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

    @staticmethod
    def _evaluate(
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        resample: Resample,
        bundle: TorchModelBundle,
        test_loader: DataLoader,
        classes: np.ndarray | None,
        device: torch.device,
    ) -> PredictionSet:
        outputs, _ = Evaluator(bundle.evaluation_step, device=device).evaluate(test_loader)
        return build_prediction_set(
            case=case,
            method=method,
            resample_id=resample.id,
            test_indices=resample.test_idx,
            outputs=outputs,
            classes=classes,
        )

    @staticmethod
    def _metadata(
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
