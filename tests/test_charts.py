import pandas as pd
import pytest

from askdata import config
from askdata.charts import chart_kind, pick_chart

MONTHS = pd.to_datetime(["2018-01-01", "2018-02-01", "2018-03-01"])


@pytest.mark.parametrize(
    "df, kind",
    [
        (pd.DataFrame({"revenue": [123.4]}), "metric"),
        (pd.DataFrame({"month": MONTHS, "orders": [5, 7, 6]}), "line"),
        (pd.DataFrame({"month": MONTHS, "orders": [5, 7, 6], "revenue": [1.0, 2.0, 3.0]}), "line"),
        (pd.DataFrame({"state": ["SP", "RJ"], "orders": [10, 4]}), "bar"),
        (pd.DataFrame({"price": [1.0, 2.0, 3.0], "freight": [0.5, 0.7, 0.9]}), "scatter"),
        (pd.DataFrame({"state": ["SP"], "city": ["x"], "orders": [1]}), None),
        (pd.DataFrame({"name": ["a"]}), None),
        (pd.DataFrame({"orders": []}), None),
        (None, None),
    ],
)
def test_chart_kind(df, kind):
    assert chart_kind(df) == kind


def test_metric_and_no_chart_return_none():
    assert pick_chart(pd.DataFrame({"revenue": [1.0]})) is None
    assert pick_chart(pd.DataFrame({"a": ["x"], "b": ["y"]})) is None


def test_too_many_rows_for_bar():
    df = pd.DataFrame({"city": [f"c{i}" for i in range(config.BAR_MAX_ROWS + 1)], "n": range(config.BAR_MAX_ROWS + 1)})
    assert chart_kind(df) is None


def test_single_row_with_date_is_not_a_line():
    df = pd.DataFrame({"month": MONTHS[:1], "orders": [5]})
    assert chart_kind(df) != "line"


def test_line_chart_is_sorted_by_date():
    df = pd.DataFrame({"month": MONTHS[::-1], "orders": [6, 7, 5]})
    fig = pick_chart(df)
    assert fig.data[0].type == "scatter"
    assert list(pd.to_datetime(fig.data[0].x)) == list(MONTHS)


def test_short_labels_give_vertical_bars():
    fig = pick_chart(pd.DataFrame({"state": ["SP", "RJ"], "orders": [10, 4]}))
    assert fig.data[0].type == "bar"
    assert fig.data[0].orientation in (None, "v")


def test_long_labels_give_horizontal_bars():
    df = pd.DataFrame({"category": ["health_beauty", "watches_gifts"], "revenue": [10.0, 4.0]})
    fig = pick_chart(df)
    assert fig.data[0].orientation == "h"
    assert fig.layout.yaxis.autorange == "reversed"


def test_scatter_uses_both_numeric_columns():
    fig = pick_chart(pd.DataFrame({"price": [1.0, 2.0], "freight": [0.5, 0.7]}))
    assert fig.data[0].mode == "markers"
    assert list(fig.data[0].x) == [1.0, 2.0]
