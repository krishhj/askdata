import os

import pandas as pd
import pytest

from askdata import config  # noqa: F401  loads .env before the skip check
from askdata.executor import QueryError, run_sql

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not os.environ.get("READER_DB_URL"), reason="READER_DB_URL is not set"),
]


def test_reader_can_select():
    df = run_sql("SELECT COUNT(*) AS n FROM orders")
    assert df["n"].iloc[0] > 0


def test_insert_fails():
    with pytest.raises(QueryError):
        run_sql("INSERT INTO category_translation VALUES ('test_x', 'test_y')")


def test_delete_fails():
    with pytest.raises(QueryError):
        run_sql("DELETE FROM category_translation WHERE product_category_name = 'none'")


def test_cannot_turn_off_read_only_and_write():
    # A session can switch read-only off, so the grants must still refuse the write.
    with pytest.raises(QueryError):
        run_sql(
            "SET default_transaction_read_only = off; "
            "INSERT INTO category_translation VALUES ('test_x', 'test_y')"
        )


def test_long_query_hits_statement_timeout():
    with pytest.raises(QueryError, match="timeout"):
        run_sql("SELECT pg_sleep(6)")


def test_error_contains_database_message():
    with pytest.raises(QueryError, match="no_such_column"):
        run_sql("SELECT no_such_column FROM orders")


def test_percent_and_colon_in_sql_are_passed_through():
    df = run_sql("SELECT COUNT(*) AS n FROM orders WHERE order_status LIKE 'deliv%' AND order_id <> ':x'")
    assert df["n"].iloc[0] > 0


def test_numeric_and_date_columns_are_converted():
    df = run_sql(
        "SELECT order_purchase_timestamp::date AS day, SUM(price) AS revenue "
        "FROM orders JOIN order_items USING (order_id) GROUP BY 1 ORDER BY 1 LIMIT 3"
    )
    assert pd.api.types.is_float_dtype(df["revenue"])
    assert pd.api.types.is_datetime64_any_dtype(df["day"])
