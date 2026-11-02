import os

import numpy as np
import pytest

from askdata import config
from askdata.retrieval import load_examples, retrieve_examples, retrieve_tables, top_k
from askdata.schema import fk_graph
from askdata.validator import validate


def test_top_k_orders_by_similarity():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])
    assert top_k(np.array([1.0, 0.0]), matrix, 2) == [0, 2]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Which product categories bring in the most revenue?", {"order_items", "products", "category_translation"}),
        ("What is the average review score?", {"order_reviews"}),
        ("How many customers paid with boleto?", {"order_payments"}),
        ("Which sellers are based in Rio de Janeiro?", {"sellers"}),
    ],
)
def test_relevant_tables_are_retrieved(question, expected):
    assert expected <= set(retrieve_tables(question))


def test_join_neighbours_are_added():
    # With k=1 only order_reviews is chosen; orders comes in as its join neighbour.
    tables = retrieve_tables("What is the average review score?", k=1)
    assert set(tables) == {"order_reviews"} | fk_graph()["order_reviews"]


def test_examples_are_relevant_and_limited():
    examples = retrieve_examples("What is the average time it takes to deliver an order?")
    assert len(examples) == config.TOP_K_EXAMPLES
    assert "delivery" in examples[0]["question"].lower()


def test_examples_are_unique_and_complete():
    examples = load_examples()
    assert len(examples) >= 20
    questions = [ex["question"].strip().lower() for ex in examples]
    assert len(set(questions)) == len(questions)
    assert all(ex["sql"].strip() for ex in examples)


@pytest.mark.parametrize("example", load_examples(), ids=lambda ex: ex["question"][:40])
def test_example_sql_passes_validator(example):
    ok, _, _, reason = validate(example["sql"])
    assert ok, reason


@pytest.mark.db
@pytest.mark.skipif(not os.environ.get("READER_DB_URL"), reason="READER_DB_URL is not set")
@pytest.mark.parametrize("example", load_examples(), ids=lambda ex: ex["question"][:40])
def test_example_sql_runs_and_returns_rows(example):
    from askdata.executor import run_sql

    _, cleaned, _, _ = validate(example["sql"])
    assert not run_sql(cleaned).empty
