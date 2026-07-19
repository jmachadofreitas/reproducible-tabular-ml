from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.single_instance import common
from rtml.runs.base import RunResult


def run_config(config: Mapping[str, Any]) -> list[RunResult]:
    return common.run_config(config, experiment_name="openml_cc18")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("Hydra config must resolve to a mapping")
    run_config(resolved)


if __name__ == "__main__":
    main()
