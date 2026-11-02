import pandas as pd
import pytest

from eval.compare import results_match

GOLD = pd.DataFrame({"state": ["SP", "RJ", "MG"], "orders": [41746, 12852, 11635]})


def test_column_names_are_ignored():
    pred = GOLD.rename(columns={"state": "customer_state", "orders": "n"})
    assert results_match(GOLD, pred, ordered=True)


def test_unordered_ignores_row_order():
    assert results_match(GOLD, GOLD.iloc[::-1], ordered=False)


def test_ordered_needs_the_same_row_order():
    assert not results_match(GOLD, GOLD.iloc[::-1], ordered=True)


def test_column_order_is_ignored():
    assert results_match(GOLD, GOLD[["orders", "state"]], ordered=True)


def test_floats_are_rounded_to_two_decimals():
    gold = pd.DataFrame({"pct": [8.11]})
    assert results_match(gold, pd.DataFrame({"pct": [8.1123]}), ordered=False)
    assert not results_match(gold, pd.DataFrame({"pct": [8.2]}), ordered=False)


def test_int_and_float_with_same_value_match():
    assert results_match(pd.DataFrame({"n": [5]}), pd.DataFrame({"n": [5.0]}), ordered=False)


def test_duplicates_count_as_a_multiset():
    gold = pd.DataFrame({"x": [1, 1, 2]})
    assert results_match(gold, pd.DataFrame({"x": [2, 1, 1]}), ordered=False)
    assert not results_match(gold, pd.DataFrame({"x": [1, 2, 2]}), ordered=False)


def test_different_row_count_fails():
    assert not results_match(GOLD, GOLD.head(2), ordered=False)


def test_missing_column_fails():
    assert not results_match(GOLD, GOLD[["state"]], ordered=False)


def test_extra_columns_follow_the_setting():
    pred = GOLD.assign(share=[0.5, 0.25, 0.25])
    assert results_match(GOLD, pred, ordered=True, allow_extra_columns=True)
    assert not results_match(GOLD, pred, ordered=True, allow_extra_columns=False)


def test_rows_must_pair_up_not_just_columns():
    # Each column holds the same values, but the pairs are swapped.
    pred = pd.DataFrame({"state": ["SP", "RJ", "MG"], "orders": [12852, 41746, 11635]})
    assert not results_match(GOLD, pred, ordered=False)


def test_dates_and_timestamps_at_midnight_match():
    gold = pd.DataFrame({"month": pd.to_datetime(["2018-01-01"]), "n": [3]})
    pred = pd.DataFrame({"month": [pd.Timestamp("2018-01-01 00:00:00")], "n": [3]})
    assert results_match(gold, pred, ordered=True)


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NaT])
def test_missing_values_match_each_other(missing):
    gold = pd.DataFrame({"month": ["2018-01", "2018-02"], "change": [None, 4.5]})
    pred = pd.DataFrame({"month": ["2018-01", "2018-02"], "change": [missing, 4.5]})
    assert results_match(gold, pred, ordered=True)


def test_text_is_trimmed_but_case_sensitive():
    gold = pd.DataFrame({"category": ["health_beauty"]})
    assert results_match(gold, pd.DataFrame({"c": [" health_beauty "]}), ordered=False)
    assert not results_match(gold, pd.DataFrame({"c": ["Health Beauty"]}), ordered=False)


def test_empty_results_match():
    assert results_match(GOLD.iloc[0:0], GOLD.iloc[0:0], ordered=False)
