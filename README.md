# Reproducible Tabular Machine Learning

RTML is a proof-of-concept for reproducible benchmarking of complete tabular ML methods: preprocessing, model, resampling, predictions, metrics, and local experiment tracking.

*Work in progress*

## Requirements

* Python 3.14+
* `uv`

## Setup

```shell
uv sync --dev
```

## Run

Run the complete local workflow from benchmark comparison through refit and inference:

```bash
uv run python examples/single_instance/benchmark_to_refit.py
```

The example writes per-run and aggregate reports, readable run evidence, and the selected fitted method under `outputs/`.

Run the default sklearn classification smoke study:

```bash
uv run python -m experiments.single_instance.smoke-test
```

Use the sequential or the parallel executor (`ray`) to run the smoke study:

```bash
uv run python -m experiments.single_instance.smoke-test execution.executor=sequential # or `ray`
```

Override the benchmark suite from the CLI:

```bash
uv run python -m experiments.single_instance.smoke-test benchmark.suite=regression
```

Outputs are written under Hydra's `outputs/` directory. Benchmark cases store their source, schema, task, and exact splits in `case.json`; each run stores `run.json` and predictions alongside the summary tables.

Run the simple Torch MLP example:

```bash
uv run python -m experiments.single_instance.simple-mlp
```

Run the classic multi-instance comparison:

```bash
uv run python -m experiments.multi_instance.classic_mil
```

## Tracking

The smoke study logs to MLflow by default. Start the local UI with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///.runs/mlflow/mlflow.db
```

## Docs

- [Design notes](docs/public/design-notes.md)
- [Future work](docs/public/future-work.md)

## Acknowledgements

This project is inspired by my work at the CD Laboratory for Dependable Intelligent Systems in Harsh Environments at TU Graz.
