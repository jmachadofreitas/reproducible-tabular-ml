setup:
    uv sync --dev

test path="tests" *args:
    uv run python -m pytest {{path}} {{args}}

lint:
    uv run ruff check .

format:
    uv run ruff format .

check: lint test

study executor="sequential" suite="classification":
    uv run python -m experiments.single_instance.smoke-test execution.executor={{executor}} benchmark.suite={{suite}}

mlp:
    uv run python -m experiments.single_instance.simple-mlp

mil-popstats:
    uv run python -m experiments.multi_instance.popstats

mil-classic:
    uv run python -m experiments.multi_instance.classic_mil

mlflow-ui db=".runs/mlflow/mlflow.db" host="127.0.0.1" port="5000":
    mkdir -p .runs/mlflow
    uv run mlflow ui --backend-store-uri sqlite:///{{db}} --host {{host}} --port {{port}}
