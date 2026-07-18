setup:
    uv sync --dev

test path="tests" *args:
    uv run pytest {{path}} {{args}}

lint:
    uv run ruff check .

format:
    uv run ruff format .

check: lint test

study executor="sequential" suite="classification":
    uv run python experiments/smoke-test/run.py execution.executor={{executor}} benchmark.suite={{suite}}

mlflow-ui db=".runs/mlflow/mlflow.db" host="127.0.0.1" port="5000":
    mkdir -p .runs/mlflow
    uv run mlflow ui --backend-store-uri sqlite:///{{db}} --host {{host}} --port {{port}}
