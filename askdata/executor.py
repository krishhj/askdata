import datetime
import logging
from decimal import Decimal

import pandas as pd
import psycopg
from sqlalchemy import Engine

from askdata.db import reader_engine

logger = logging.getLogger(__name__)


class QueryError(Exception):
    """The database refused or failed to run a query."""


def _fix_types(df: pd.DataFrame) -> pd.DataFrame:
    # Postgres numeric arrives as Decimal and date as datetime.date. pandas stores
    # both as plain objects, which charts do not recognise as numbers or dates.
    for col in df.columns:
        values = df[col].dropna()
        if df[col].dtype != object or values.empty:
            continue
        if values.map(lambda v: isinstance(v, Decimal)).all():
            df[col] = df[col].astype(float)
        elif values.map(lambda v: isinstance(v, datetime.date)).all():
            df[col] = pd.to_datetime(df[col])
    return df


def run_sql(sql: str, engine: Engine | None = None) -> pd.DataFrame:
    """Run one validated query and return the rows as a DataFrame."""
    engine = engine or reader_engine()
    with engine.connect() as conn:
        # A raw cursor with no parameters sends the SQL untouched. SQLAlchemy's
        # text() would read ":name" as a parameter, and exec_driver_sql makes
        # psycopg read "%" in LIKE patterns as a placeholder.
        with conn.connection.cursor() as cursor:
            try:
                cursor.execute(sql)
            except psycopg.Error as exc:
                message = str(exc).strip()
                logger.warning("Query failed: %s", message)
                raise QueryError(message) from exc
            columns = [col.name for col in cursor.description]
            df = pd.DataFrame(cursor.fetchall(), columns=columns)
    return _fix_types(df)
