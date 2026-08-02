from collections.abc import Mapping
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from experiments.multi_instance import common
from rtml.core.runs import RunResult


def run_config(config: Mapping[str, Any]) -> list[RunResult]:
    return common.run_config(config, experiment_name="classic_mil")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("Hydra config must resolve to a mapping")
    run_config(resolved)


if __name__ == "__main__":
    main()
