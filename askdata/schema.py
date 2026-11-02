from functools import lru_cache
from pathlib import Path

import yaml

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "schema.yaml"


@lru_cache(maxsize=1)
def load_schema() -> dict:
    """Read data/schema.yaml once and keep it in memory."""
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def table_names() -> list[str]:
    """All table names, in the order schema.yaml lists them."""
    return list(load_schema()["tables"])


def join_pairs() -> list[tuple[str, str]]:
    """Each join as a pair of "table.column" strings."""
    return [(join["from"], join["to"]) for join in load_schema()["joins"]]


@lru_cache(maxsize=1)
def fk_graph() -> dict[str, set[str]]:
    """Map each table to the tables it joins with, in both directions."""
    graph = {name: set() for name in table_names()}
    for left, right in join_pairs():
        a, b = left.split(".")[0], right.split(".")[0]
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _describe_table(name: str, table: dict, with_notes: bool) -> str:
    if not with_notes:
        columns = ", ".join(f"{col} {info['type']}" for col, info in table["columns"].items())
        return f"{name}({columns})"

    lines = [f"Table {name}: {table['description']}"]
    lines += [f"  Note: {note}" for note in table.get("notes", [])]
    lines += [
        f"  - {col} ({info['type']}): {info['description']}"
        for col, info in table["columns"].items()
    ]
    return "\n".join(lines)


def schema_prompt(tables: list[str], with_notes: bool) -> str:
    """Prompt text for the given tables, with or without descriptions and notes."""
    schema = load_schema()
    parts = [_describe_table(name, schema["tables"][name], with_notes) for name in tables]

    chosen = set(tables)
    joins = [
        f"- {left} = {right}"
        for left, right in join_pairs()
        if left.split(".")[0] in chosen and right.split(".")[0] in chosen
    ]
    if joins:
        parts.append("Joins:\n" + "\n".join(joins))

    if with_notes:
        parts.append("Important rules for this data:\n" + "\n".join(
            f"- {note}" for note in schema["global_notes"]
        ))
    return "\n\n".join(parts)
