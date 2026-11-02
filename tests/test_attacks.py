from pathlib import Path

import pytest
import yaml

from askdata.validator import UNSAFE, validate

ATTACKS_PATH = Path(__file__).resolve().parent.parent / "eval" / "attacks.yaml"
ATTACKS = yaml.safe_load(ATTACKS_PATH.read_text(encoding="utf-8"))
SQL_ATTACKS = [a for a in ATTACKS if a["kind"] == "sql"]


def test_attack_file_has_both_kinds():
    kinds = {a["kind"] for a in ATTACKS}
    assert kinds == {"sql", "nl"}
    assert len({a["id"] for a in ATTACKS}) == len(ATTACKS)


@pytest.mark.parametrize("attack", SQL_ATTACKS, ids=[a["id"] for a in SQL_ATTACKS])
def test_validator_blocks_raw_sql_attack(attack):
    ok, cleaned, category, reason = validate(attack["input"])
    assert not ok and cleaned is None
    # Unsafe, not invalid, so the pipeline never retries it.
    assert category == UNSAFE, reason
