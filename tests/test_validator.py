import pytest

from askdata import config
from askdata.validator import INVALID, UNSAFE, validate

SAFE = [
    "SELECT COUNT(*) FROM orders",
    "SELECT * FROM public.orders",
    "SELECT order_status, COUNT(*) FROM orders GROUP BY 1 ORDER BY 2 DESC;",
    "WITH monthly AS (SELECT DATE_TRUNC('month', order_purchase_timestamp) AS m FROM orders) "
    "SELECT m, COUNT(*) FROM monthly GROUP BY m",
    "SELECT customer_state FROM customers UNION SELECT seller_state FROM sellers",
    "SELECT order_id FROM orders WHERE order_status LIKE 'deliv%'",
    "SELECT * FROM generate_series(1, 3)",
    # Keywords inside strings and column names are not statements.
    "SELECT 'delete' AS word, order_id FROM orders WHERE order_status = 'drop table'",
]

UNSAFE_SQL = [
    "SELECT 1; SELECT 2",
    "DELETE FROM orders",
    "INSERT INTO sellers VALUES ('a', 'b', 'c', 'd')",
    "UPDATE orders SET order_status = 'x'",
    "DROP TABLE orders",
    "CREATE TABLE t (x int)",
    "ALTER TABLE orders ADD COLUMN x int",
    "MERGE INTO orders USING customers ON true WHEN MATCHED THEN DELETE",
    "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",
    "SELECT * INTO t FROM orders",
    "SELECT * FROM orders FOR UPDATE",
    "VACUUM orders",
    "EXPLAIN ANALYZE SELECT 1",
    "SET ROLE postgres",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM pg_catalog.pg_tables",
    "SELECT * FROM pg_shadow",
    "SELECT pg_catalog.pg_sleep(1)",
    'SELECT "pg_sleep"(1)',
    "SELECT PG_SLEEP(1)",
    "SELECT * FROM pg_ls_dir('.')",
    "SELECT lo_export(1, '/tmp/x')",
    "SELECT dblink_exec('host=x', 'DROP TABLE orders')",
    "SELECT order_id FROM orders WHERE pg_sleep(1) IS NOT NULL",
]

INVALID_SQL = [
    "",
    "   ",
    "SELEC 1",
    "SELECT * FROM users",
    "SELECT * FROM postgres.public.orders",
    "SELECT * FROM other_schema.orders",
]


@pytest.mark.parametrize("sql", SAFE)
def test_safe_queries_pass(sql):
    ok, cleaned, category, reason = validate(sql)
    assert ok, reason
    assert cleaned and category is None


@pytest.mark.parametrize("sql", UNSAFE_SQL)
def test_unsafe_queries_are_blocked(sql):
    ok, cleaned, category, reason = validate(sql)
    assert not ok and cleaned is None
    assert category == UNSAFE
    assert reason


@pytest.mark.parametrize("sql", INVALID_SQL)
def test_invalid_queries_can_be_retried(sql):
    ok, cleaned, category, reason = validate(sql)
    assert not ok and cleaned is None
    assert category == INVALID
    assert reason


def test_limit_is_added_when_missing():
    _, cleaned, _, _ = validate("SELECT * FROM orders")
    assert cleaned.endswith(f"LIMIT {config.ROW_LIMIT}")


def test_limit_is_added_to_the_outer_query_only():
    _, cleaned, _, _ = validate("WITH x AS (SELECT * FROM orders LIMIT 5) SELECT * FROM x")
    assert "LIMIT 5" in cleaned
    assert cleaned.endswith(f"LIMIT {config.ROW_LIMIT}")


def test_existing_limit_is_kept():
    _, cleaned, _, _ = validate("SELECT * FROM orders LIMIT 10")
    assert cleaned.endswith("LIMIT 10")
    assert str(config.ROW_LIMIT) not in cleaned


def test_union_gets_one_limit():
    _, cleaned, _, _ = validate("SELECT 1 AS a UNION SELECT 2")
    assert cleaned.count("LIMIT") == 1


def test_cte_can_reuse_a_table_name():
    ok, _, _, reason = validate("WITH orders AS (SELECT 1 AS n) SELECT n FROM orders")
    assert ok, reason


def test_unknown_table_reason_lists_real_tables():
    _, _, _, reason = validate("SELECT * FROM users")
    assert "users" in reason
    assert "order_items" in reason


def test_cleaned_sql_drops_comments_and_semicolon():
    _, cleaned, _, _ = validate("SELECT 1 AS a -- note\n;")
    assert "--" not in cleaned
    assert ";" not in cleaned
