"""Parser for WEKA relational ARFF multiple-instance datasets."""

from io import StringIO
from pathlib import Path
from typing import NamedTuple
import csv
import sys

import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)


class ParsedRelationalArff(NamedTuple):
    relation: str
    bag_table: pd.DataFrame
    instance_table: pd.DataFrame
    bag_offsets: np.ndarray
    bag_id_column: str
    target_column: str
    outer_attributes: list[tuple[str, str]]
    instance_attributes: list[tuple[str, str]]


def parse_weka_relational_arff(arff_path: str | Path) -> ParsedRelationalArff:
    """Parse one WEKA relational ARFF file into bag and instance tables."""
    path = Path(arff_path).expanduser().resolve()
    relation, outer_attributes, instance_attributes, rows = _read_relational_arff(path)
    bag_id_column, target_column = _infer_bag_id_and_target(outer_attributes)

    bag_records: list[dict[str, object]] = []
    instance_frames: list[pd.DataFrame] = []
    bag_offsets = np.zeros(len(rows) + 1, dtype=int)

    instance_columns = [name for name, _ in instance_attributes]
    relation_index = _relational_attribute_index(outer_attributes)
    for bag_position, row in enumerate(rows):
        if len(row) != len(outer_attributes):
            raise ValueError(
                f"{path.name} row {bag_position} has {len(row)} values; "
                f"expected {len(outer_attributes)}"
            )

        outer_record = {
            name: _coerce_arff_value(value, type_spec)
            for (name, type_spec), value in zip(outer_attributes, row, strict=True)
            if type_spec.lower() != "relational"
        }
        bag_records.append(outer_record)

        instance_frame = _parse_instance_table(
            row[relation_index],
            instance_attributes=instance_attributes,
            path=path,
            bag_position=bag_position,
        )
        instance_frames.append(instance_frame)
        bag_offsets[bag_position + 1] = bag_offsets[bag_position] + len(instance_frame)

    bag_table = pd.DataFrame.from_records(bag_records)
    instance_table = (
        pd.concat(instance_frames, axis=0, ignore_index=True)
        if instance_frames
        else pd.DataFrame(columns=instance_columns)
    )

    return ParsedRelationalArff(
        relation=relation,
        bag_table=bag_table,
        instance_table=instance_table,
        bag_offsets=bag_offsets,
        bag_id_column=bag_id_column,
        target_column=target_column,
        outer_attributes=outer_attributes,
        instance_attributes=instance_attributes,
    )


def _read_relational_arff(
    path: Path,
) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]], list[list[str]]]:
    relation = path.stem.removesuffix("_relational")
    outer_attributes: list[tuple[str, str]] = []
    instance_attributes: list[tuple[str, str]] = []
    in_relation = False
    data_lines: list[str] = []
    in_data = False

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue

        lower = line.lower()
        if in_data:
            data_lines.append(f"{raw_line}\n")
        elif lower.startswith("@relation"):
            relation = _relation_name(line)
        elif lower.startswith("@attribute"):
            name, type_spec = _parse_attribute(line)
            if in_relation:
                instance_attributes.append((name, type_spec))
            else:
                outer_attributes.append((name, type_spec))
                if type_spec.lower() == "relational":
                    in_relation = True
        elif lower.startswith("@end"):
            in_relation = False
        elif lower.startswith("@data"):
            in_data = True

    if not outer_attributes:
        raise ValueError(f"{path} does not define outer ARFF attributes")
    if not instance_attributes:
        raise ValueError(f"{path} does not define relational instance attributes")
    if not data_lines:
        raise ValueError(f"{path} does not contain ARFF data rows")

    return relation, outer_attributes, instance_attributes, list(csv.reader(data_lines))


def _parse_attribute(line: str) -> tuple[str, str]:
    rest = line.strip()[len("@attribute") :].strip()
    if not rest:
        raise ValueError(f"invalid ARFF attribute line: {line!r}")

    if rest[0] in {"'", '"'}:
        quote = rest[0]
        end = rest.find(quote, 1)
        if end == -1:
            raise ValueError(f"unterminated quoted ARFF attribute name: {line!r}")
        return rest[1:end], rest[end + 1 :].strip()

    parts = rest.split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"invalid ARFF attribute line: {line!r}")
    return parts[0], parts[1].strip()


def _relation_name(line: str) -> str:
    rest = line.strip()[len("@relation") :].strip()
    return rest.strip("'\"") if rest else "unknown"


def _infer_bag_id_and_target(attributes: list[tuple[str, str]]) -> tuple[str, str]:
    non_relational = [
        (name, type_spec) for name, type_spec in attributes if type_spec.lower() != "relational"
    ]
    if len(non_relational) < 2:
        raise ValueError("classic MIL ARFF requires a bag id and a target column")
    bag_id_column = non_relational[0][0]
    target_column = (
        "class" if any(name == "class" for name, _ in non_relational) else non_relational[-1][0]
    )
    return bag_id_column, target_column


def _relational_attribute_index(attributes: list[tuple[str, str]]) -> int:
    for index, (_, type_spec) in enumerate(attributes):
        if type_spec.lower() == "relational":
            return index
    raise ValueError("ARFF file does not define a relational attribute")


def _parse_instance_table(
    value: str,
    *,
    instance_attributes: list[tuple[str, str]],
    path: Path,
    bag_position: int,
) -> pd.DataFrame:
    rows = [row for row in csv.reader(StringIO(value.replace("\\n", "\n"))) if row]
    columns = [name for name, _ in instance_attributes]
    if any(len(row) != len(columns) for row in rows):
        raise ValueError(
            f"{path.name} bag {bag_position} has instance rows that do not match "
            "the relational schema"
        )

    data = {
        name: [_coerce_arff_value(row[index], type_spec) for row in rows]
        for index, (name, type_spec) in enumerate(instance_attributes)
    }
    return pd.DataFrame(data, columns=columns)


def _coerce_arff_value(value: str, type_spec: str) -> object:
    if value == "?":
        return pd.NA

    normalized_type = type_spec.strip().lower()
    if normalized_type in {"numeric", "real"}:
        return float(value)
    if normalized_type == "integer":
        return int(value)
    return value
