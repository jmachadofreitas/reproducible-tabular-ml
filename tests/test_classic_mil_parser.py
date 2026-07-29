from pathlib import Path

from rtml.core.datasets import FeatureKind
from rtml.core.tasks import TaskType
from rtml.multi_instance.datasets.classic import parse_classic_mil_arff
from rtml.multi_instance.datasets.classic.parser import parse_weka_relational_arff


def write_relational_arff(path: Path, *, escaped_newlines: bool = False) -> None:
    instances = "1.0,0.0\\n2.0,1.0" if escaped_newlines else "1.0,0.0\n2.0,1.0"
    path.write_text(
        "\n".join(
            [
                "@relation toy_mil",
                "@attribute bag_id {a,b}",
                "@attribute bag relational",
                "  @attribute x numeric",
                "  @attribute y real",
                "@end bag",
                "@attribute class {0,1}",
                "@data",
                f'a,"{instances}",1',
                'b,"3.0,0.0",0',
            ]
        )
    )


def test_parse_weka_relational_arff_returns_bag_and_instance_tables(tmp_path: Path) -> None:
    path = tmp_path / "toy_relational.arff"
    write_relational_arff(path)

    parsed = parse_weka_relational_arff(path)

    assert parsed.relation == "toy_mil"
    assert parsed.bag_id_column == "bag_id"
    assert parsed.target_column == "class"
    assert parsed.bag_table.to_dict("list") == {"bag_id": ["a", "b"], "class": ["1", "0"]}
    assert parsed.instance_table["x"].tolist() == [1.0, 2.0, 3.0]
    assert parsed.bag_offsets.tolist() == [0, 2, 3]


def test_parse_weka_relational_arff_handles_escaped_inner_newlines(tmp_path: Path) -> None:
    path = tmp_path / "toy_relational.arff"
    write_relational_arff(path, escaped_newlines=True)

    parsed = parse_weka_relational_arff(path)

    assert parsed.instance_table["x"].tolist() == [1.0, 2.0, 3.0]
    assert parsed.bag_offsets.tolist() == [0, 2, 3]


def test_parse_weka_relational_arff_accepts_uppercase_relational_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "toy_relational.arff"
    write_relational_arff(path)
    path.write_text(path.read_text().replace("bag relational", "bag RELATIONAL"))

    parsed = parse_weka_relational_arff(path)

    assert parsed.instance_table["x"].tolist() == [1.0, 2.0, 3.0]


def test_parse_classic_mil_arff_builds_rtml_dataset_and_task(tmp_path: Path) -> None:
    path = tmp_path / "toy_relational.arff"
    write_relational_arff(path)

    dataset, task = parse_classic_mil_arff(path)

    assert dataset.name == "toy_mil"
    assert dataset.bag_schema.get("bag_id").kind == FeatureKind.ID
    assert dataset.bag_schema.get("class").kind == FeatureKind.BINARY
    assert dataset.instance_schema.get("x").kind == FeatureKind.NUMERIC
    assert dataset.bag_instances(0)["x"].tolist() == [1.0, 2.0]
    assert task.task_type == TaskType.BINARY_CLASSIFICATION
    assert task.instance_source == ["x", "y"]
    assert task.target == "class"
    assert "path" not in dataset.metadata
