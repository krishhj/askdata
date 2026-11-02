from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from askdata import config
from askdata.schema import fk_graph, load_schema, table_names

EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "examples.yaml"


@lru_cache(maxsize=1)
def embedding_model():
    """Load the sentence embedding model once per process."""
    # Imported here because torch is slow to import, and runs with retrieval
    # and examples switched off should not pay for it.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts as unit-length rows, so a dot product is cosine similarity."""
    return embedding_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Indices of the k rows most similar to query, best first."""
    scores = matrix @ query
    return np.argsort(-scores)[:k].tolist()


@lru_cache(maxsize=1)
def load_examples() -> list[dict]:
    """Read data/examples.yaml once."""
    with EXAMPLES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _table_text(name: str, table: dict) -> str:
    # Column names help with questions that name a field, such as "zip code".
    columns = ", ".join(col.replace("_", " ") for col in table["columns"])
    return f"{name.replace('_', ' ')}: {table['description']} Columns: {columns}"


@lru_cache(maxsize=1)
def _table_index() -> tuple[list[str], np.ndarray]:
    tables = load_schema()["tables"]
    names = table_names()
    return names, embed([_table_text(name, tables[name]) for name in names])


@lru_cache(maxsize=1)
def _example_index() -> np.ndarray:
    return embed([ex["question"] for ex in load_examples()])


def retrieve_tables(question: str, k: int | None = None) -> list[str]:
    """The k most relevant tables plus the tables they join to, in schema order."""
    names, matrix = _table_index()
    query = embed([question])[0]
    chosen = {names[i] for i in top_k(query, matrix, k or config.TOP_K_TABLES)}
    graph = fk_graph()
    # Neighbours supply the join path, for example orders between customers and order_items.
    with_neighbours = chosen.union(*(graph[name] for name in chosen))
    return [name for name in names if name in with_neighbours]


def retrieve_examples(question: str, k: int | None = None) -> list[dict]:
    """The k solved examples whose questions are most similar, best first."""
    query = embed([question])[0]
    examples = load_examples()
    return [examples[i] for i in top_k(query, _example_index(), k or config.TOP_K_EXAMPLES)]
