# Reproducible Tabular Machine Learning

RTML is a proof-of-concept for reproducible benchmarking of complete tabular ML methods: preprocessing, model, resampling, predictions, metrics, and local experiment tracking.

*Work in progress*

## Requirements

* Python 3.11+
* `uv`

## Setup

```shell
uv sync --dev
```

## Run

Run the default sklearn classification smoke study:

```bash
uv run python experiments/smoke-test/run.py
```

Use the sequential or the parallel executor (`ray`) to run the smoke study:

```bash
uv run python experiments/smoke-test/run.py execution.executor=sequential # or `ray`
```

Override the benchmark suite from the CLI:

```bash
uv run python experiments/smoke-test/run.py benchmark.suite=regression
```

Outputs are written under Hydra's `outputs/` directory. Each run stores prediction artifacts plus `summary.csv`, `summary.json`, `aggregate.csv`, and `aggregate.json`.

## Tracking

The smoke study logs to MLflow by default. Start the local UI with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///.runs/mlflow/mlflow.db
```

## Acknowledgements

This project is inspired by my work at the CD Laboratory for Dependable Intelligent Systems in Harsh Environments at TU Graz.
