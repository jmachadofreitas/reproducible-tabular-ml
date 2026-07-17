from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import common
from rtml.runs.base import RunResult

build_executor = common.build_executor
build_methods = common.build_methods
build_runtime_specs = common.build_runtime_specs
build_scheduler_resources = common.build_scheduler_resources
build_study = common.build_study
build_suite = common.build_suite


def run_config(config: Mapping[str, Any]) -> list[RunResult]:
    return common.run_config(config, experiment_name="sklearn_classification_benchmark")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("Hydra config must resolve to a mapping")
    run_config(resolved)


if __name__ == "__main__":
    main()
