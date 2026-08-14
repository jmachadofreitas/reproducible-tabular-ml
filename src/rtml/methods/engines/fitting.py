from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

from rtml.loggers import Logger
from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.checkpointing import CheckpointManager
from rtml.methods.engines.core import Evaluator, Trainer
from rtml.methods.engines.optim import create_lr_scheduler, create_optimizer


@contextmanager
def _deterministic_algorithms(enabled: bool | None) -> Iterator[None]:
    if enabled is None:
        yield
        return
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)


def fit_model_bundle(
    bundle: TorchModelBundle,
    train_dataloader: DataLoader,
    *,
    validation_dataloader: DataLoader | None = None,
    test_dataloader: DataLoader | None = None,
    score_name: str,
    score_mode: str,
    device: torch.device,
    deterministic: bool | None = None,
    logger: Logger | None = None,
    checkpoint_manager: CheckpointManager | None = None,
) -> Trainer:
    """Fit one model bundle with the common Torch/Ignite training machinery."""
    if bundle.fit_config.early_stopping_patience is not None and validation_dataloader is None:
        raise ValueError("early stopping requires validation data")
    if (
        checkpoint_manager is not None
        and checkpoint_manager.save_best
        and validation_dataloader is None
    ):
        raise ValueError("best checkpoint selection requires validation data")

    validation_evaluator = None
    if validation_dataloader is not None:
        validation_metrics = bundle.make_validation_metrics()
        if score_name not in validation_metrics.metrics:
            available = ", ".join(validation_metrics.metrics) or "none"
            raise ValueError(
                f"torch validation objective {score_name!r} is not available; "
                f"tracked validation metrics: {available}"
            )
        validation_evaluator = Evaluator(
            bundle.evaluation_step,
            metrics=validation_metrics,
            device=device,
        )

    test_evaluator = None
    if test_dataloader is not None and bundle.fit_config.tracking.get("log_test_metrics", False):
        test_evaluator = Evaluator(
            bundle.evaluation_step,
            metrics=bundle.make_test_metrics(),
            device=device,
        )

    optimizer = create_optimizer(bundle.model, **bundle.fit_config.optimizer)
    lr_scheduler = create_lr_scheduler(
        optimizer,
        config=None
        if bundle.fit_config.lr_scheduler is None
        else dict(bundle.fit_config.lr_scheduler),
        max_epochs=bundle.fit_config.max_epochs,
    )
    if bundle.fit_config.hp_scheduler is not None and bundle.hp_scheduler is None:
        raise ValueError("this torch model does not support hyperparameter scheduling")
    trainer = Trainer(
        bundle.create_training_step(optimizer),
        train_metrics=bundle.make_train_metrics(),
        lr_scheduler=lr_scheduler,
        hp_scheduler=bundle.hp_scheduler,
        model=bundle.model,
        val_evaluator=validation_evaluator,
        test_evaluator=test_evaluator,
        score_name=score_name,
        score_mode=score_mode,
        device=device,
        max_epochs=bundle.fit_config.max_epochs,
        val_every_n_epochs=bundle.fit_config.validation_every_n_epochs,
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
        if bundle.hp_scheduler is not None:
            objects["hp_scheduler"] = bundle.hp_scheduler
        checkpoint_manager.set_objects(objects)
        resume_path = checkpoint_manager.load_resume_checkpoint()
        trainer.resume_checkpoint_path = None if resume_path is None else str(resume_path)

    with _deterministic_algorithms(deterministic):
        trainer.train(
            train_dataloader,
            val_dataloader=validation_dataloader,
            test_dataloader=test_dataloader if test_evaluator is not None else None,
        )
    if checkpoint_manager is not None and checkpoint_manager.save_best:
        restored_epoch = checkpoint_manager.restore_best_model(bundle.model)
    else:
        restored_epoch = None
    if restored_epoch is None:
        trainer.restore_best_model()
    else:
        trainer.best_epoch = restored_epoch
    return trainer
