from collections.abc import Mapping
from dataclasses import replace
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.methods import MethodSpec
from rtml.core.metrics import EvaluationMetrics
from rtml.core.results import PredictionSet
from rtml.core.runtime import RuntimeSpec
from rtml.core.tasks import TaskType
from rtml.loggers import Logger
from rtml.methods.backends.base import BackendResult, MethodBackend
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.methods.instance_pooling import InstancePoolingClassifier
from rtml.multi_instance.methods.mi_svm import MISVMClassifier
from rtml.multi_instance.preprocessing import build_bag_feature_dataset, build_preprocessor
from rtml.multi_instance.tasks import MultiInstanceTask
from rtml.single_instance.methods._sklearn import SklearnBackend


class MultiInstanceSklearnBackend(MethodBackend):
    """Run sklearn implementations whose training semantics operate on bags."""

    name = "sklearn"

    def validate_method(self, method: MethodSpec) -> None:
        aggregation = method.transform.get("aggregation")
        if method.model.kind in {"binary_mi_svm", "instance_pooling_classifier"}:
            if aggregation is not None:
                raise ValueError(f"{method.model.kind} does not use bag-feature aggregation")
            return
        if aggregation is None:
            raise ValueError(
                "multi-instance sklearn methods require transform.aggregation unless "
                "model.kind is 'instance_pooling_classifier' or 'binary_mi_svm'"
            )
        if not isinstance(aggregation, Mapping):
            raise TypeError("transform.aggregation must be a mapping")
        SklearnBackend().validate_method(method)

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
        self.validate_method(method)
        if method.model.kind == "instance_pooling_classifier":
            return self._run_instance_pooling(
                case=case,
                dataset=dataset,
                task=task,
                method=method,
                resample_id=resample_id,
                seed=seed,
            )
        if method.model.kind == "binary_mi_svm":
            return self._run_mi_svm(
                case=case,
                dataset=dataset,
                task=task,
                method=method,
                resample_id=resample_id,
                seed=seed,
            )
        return self._run_bag_feature_aggregation(
            case=case,
            dataset=dataset,
            task=task,
            method=method,
            resample_id=resample_id,
            seed=seed,
            runtime=runtime,
            logger=logger,
        )

    @staticmethod
    def _run_instance_pooling(
        *,
        case: BenchmarkCase,
        dataset: MultiInstanceDataset,
        task: MultiInstanceTask,
        method: MethodSpec,
        resample_id: str | None,
        seed: int,
    ) -> BackendResult:
        if task.task_type not in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }:
            raise ValueError("instance pooling supports classification tasks only")
        resample = case.resampling.get_resample(resample_id)
        training_indices = resample.train_idx
        if resample.valid_idx is not None:
            training_indices = np.concatenate((training_indices, resample.valid_idx))
        train_instances, train_offsets = dataset[training_indices]
        test_instances, test_offsets = dataset[resample.test_idx]

        transform_options = dict(method.transform)
        policy = str(transform_options.pop("policy", "linear_default"))
        preprocessor = build_preprocessor(
            dataset,
            task,
            policy=policy,
            options=transform_options,
        )
        model_params = dict(method.model.params)
        pooling = str(model_params.pop("pooling", "max"))
        model_params.setdefault("max_iter", 1000)
        model_params.setdefault("random_state", seed)
        estimator = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(**model_params)),
            ]
        )
        classifier = InstancePoolingClassifier(estimator, pooling=pooling)

        target = task.target_series(dataset)
        if target is None:
            raise ValueError("instance pooling requires a bag-level target")
        columns = task.instance_source

        fit_start = perf_counter()
        classifier.fit(
            train_instances.loc[:, columns],
            train_offsets,
            target.iloc[training_indices],
        )
        fit_time = perf_counter() - fit_start

        predict_start = perf_counter()
        probabilities = classifier.predict_proba(
            test_instances.loc[:, columns],
            test_offsets,
        )
        predictions = PredictionSet(
            dataset_name=dataset.name,
            task_name=task.name,
            method_name=method.name,
            resample_id=resample.id,
            sample_ids=dataset.sample_ids_for(resample.test_idx),
            y_true=target.iloc[resample.test_idx].to_numpy(),
            labels=classifier.classes_[probabilities.argmax(axis=1)],
            probabilities=probabilities,
            metadata={
                "case_name": case.name,
                "classes": classifier.classes_.tolist(),
            },
        )
        predict_time = perf_counter() - predict_start

        return BackendResult(
            predictions=predictions,
            metrics=EvaluationMetrics(task.metrics).compute(predictions),
            fit_time=fit_time,
            predict_time=predict_time,
            metadata={
                "preprocessing_policy": policy,
                "model_class": "InstancePoolingClassifier",
                "instance_estimator_class": "LogisticRegression",
                "pooling": pooling,
                "supervision": "bag_labels_as_instance_labels",
            },
        )

    @staticmethod
    def _run_mi_svm(
        *,
        case: BenchmarkCase,
        dataset: MultiInstanceDataset,
        task: MultiInstanceTask,
        method: MethodSpec,
        resample_id: str | None,
        seed: int,
    ) -> BackendResult:
        if task.task_type != TaskType.BINARY_CLASSIFICATION:
            raise ValueError("binary_mi_svm requires a binary classification task")
        resample = case.resampling.get_resample(resample_id)
        training_indices = resample.train_idx
        if resample.valid_idx is not None:
            training_indices = np.concatenate((training_indices, resample.valid_idx))
        train_frame, train_offsets = dataset[training_indices]
        test_frame, test_offsets = dataset[resample.test_idx]
        target = task.target_series(dataset)
        if target is None:
            raise ValueError("binary_mi_svm requires a bag-level target")

        transform_options = dict(method.transform)
        policy = str(transform_options.pop("policy", "linear_default"))
        preprocessor = build_preprocessor(
            dataset,
            task,
            policy=policy,
            options=transform_options,
        )
        columns = task.instance_source

        fit_start = perf_counter()
        train_instances = preprocessor.fit_transform(train_frame.loc[:, columns])
        classifier = MISVMClassifier(
            random_state=seed,
            **method.model.params,
        ).fit(
            train_instances,
            train_offsets,
            target.iloc[training_indices],
        )
        fit_time = perf_counter() - fit_start

        predict_start = perf_counter()
        test_instances = preprocessor.transform(test_frame.loc[:, columns])
        scores = classifier.decision_function(test_instances, test_offsets)
        predictions = PredictionSet(
            dataset_name=dataset.name,
            task_name=task.name,
            method_name=method.name,
            resample_id=resample.id,
            sample_ids=dataset.sample_ids_for(resample.test_idx),
            y_true=target.iloc[resample.test_idx].to_numpy(),
            labels=classifier.classes_[(scores >= 0).astype(int)],
            scores=scores,
            metadata={
                "case_name": case.name,
                "classes": classifier.classes_.tolist(),
            },
        )
        predict_time = perf_counter() - predict_start

        return BackendResult(
            predictions=predictions,
            metrics=EvaluationMetrics(task.metrics).compute(predictions),
            fit_time=fit_time,
            predict_time=predict_time,
            metadata={
                "preprocessing_policy": policy,
                "model_class": "MISVMClassifier",
                "n_iterations": classifier.n_iterations_,
                "converged": classifier.converged_,
                "termination": classifier.termination_,
                "n_witnesses": len(classifier.witness_indices_),
                "pooling": "max_instance_score",
            },
        )

    @staticmethod
    def _run_bag_feature_aggregation(
        *,
        case: BenchmarkCase,
        dataset: MultiInstanceDataset,
        task: MultiInstanceTask,
        method: MethodSpec,
        resample_id: str | None,
        seed: int,
        runtime: RuntimeSpec | None,
        logger: Logger | None,
    ) -> BackendResult:
        transform = dict(method.transform)
        aggregation = dict(transform.pop("aggregation"))
        aggregation_policy = str(aggregation.pop("policy", "summary_default"))
        aggregated_dataset, aggregated_task = build_bag_feature_dataset(
            dataset,
            task,
            policy=aggregation_policy,
            options=aggregation,
        )
        aggregated_case = BenchmarkCase(
            name=case.name,
            dataset=aggregated_dataset,
            task=aggregated_task,
            resampling=case.resampling,
            metadata=case.metadata,
        )
        result = SklearnBackend().run(
            case=aggregated_case,
            method=replace(method, transform=transform),
            resample_id=resample_id,
            seed=seed,
            runtime=runtime,
            logger=logger,
        )
        return replace(
            result,
            metadata={
                **result.metadata,
                "aggregation_policy": aggregation_policy,
                "aggregation_statistics": aggregation.get(
                    "statistics", ["mean", "std", "min", "max"]
                ),
                "include_bag_size": aggregation.get("include_size", True),
            },
        )

    @staticmethod
    def _validate_case(
        case: BenchmarkCase,
    ) -> tuple[MultiInstanceDataset, MultiInstanceTask]:
        if not isinstance(case.dataset, MultiInstanceDataset):
            raise TypeError("multi-instance sklearn backend requires MultiInstanceDataset")
        if not isinstance(case.task, MultiInstanceTask):
            raise TypeError("multi-instance sklearn backend requires MultiInstanceTask")
        case.task.validate_columns(case.dataset)
        if case.task.task_type == TaskType.UNSUPERVISED:
            raise ValueError("multi-instance sklearn backend supports supervised tasks only")
        return case.dataset, case.task
