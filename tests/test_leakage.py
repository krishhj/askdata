import re
from pathlib import Path

import pytest
import yaml

from askdata.retrieval import embed, load_examples

GOLD_PATH = Path(__file__).resolve().parent.parent / "eval" / "gold.yaml"
MAX_SIMILARITY = 0.9

pytestmark = pytest.mark.skipif(not GOLD_PATH.exists(), reason="eval/gold.yaml does not exist yet")


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


@pytest.fixture(scope="module")
def questions():
    gold = yaml.safe_load(GOLD_PATH.read_text(encoding="utf-8"))
    return [ex["question"] for ex in load_examples()], [g["question"] for g in gold]


def test_no_example_repeats_a_gold_question(questions):
    examples, gold = questions
    overlap = {_normalise(q) for q in examples} & {_normalise(q) for q in gold}
    assert not overlap


def test_no_example_is_near_a_gold_question(questions):
    examples, gold = questions
    similarity = embed(examples) @ embed(gold).T
    leaks = [
        f"{similarity[i, j]:.2f}  {examples[i]!r} ~ {gold[j]!r}"
        for i, j in zip(*(similarity > MAX_SIMILARITY).nonzero())
    ]
    assert not leaks, "\n".join(leaks)
