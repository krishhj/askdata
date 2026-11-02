import sys

from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

from askdata.config import require_env


def is_refused(conn, sql: str) -> bool:
    try:
        conn.execute(text(sql))
    except DBAPIError:
        conn.rollback()
        return True
    conn.rollback()
    return False


def main() -> int:
    results = {}

    with create_engine(require_env("READER_DB_URL")).connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar_one()
        results[f"reader can SELECT orders ({count:,} rows)"] = count > 0
        conn.rollback()
        results["reader cannot INSERT"] = is_refused(
            conn, "INSERT INTO category_translation VALUES ('x', 'y')"
        )
        results["reader cannot DELETE"] = is_refused(conn, "DELETE FROM category_translation")
        results["reader cannot read app.feedback"] = is_refused(conn, "SELECT * FROM app.feedback")
        results["reader statement timeout works"] = is_refused(conn, "SELECT pg_sleep(6)")

    with create_engine(require_env("FEEDBACK_DB_URL")).connect() as conn:
        # Rolled back, so the check leaves no row behind.
        conn.execute(text("INSERT INTO app.feedback (question, sql, rating) VALUES ('t', 't', 1)"))
        conn.rollback()
        results["feedback can INSERT and SELECT"] = (
            conn.execute(text("SELECT COUNT(*) FROM app.feedback")).scalar_one() >= 0
        )
        conn.rollback()
        results["feedback cannot DELETE"] = is_refused(conn, "DELETE FROM app.feedback")
        results["feedback cannot read orders"] = is_refused(conn, "SELECT * FROM orders")

    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
