from functools import lru_cache

from sqlalchemy import Engine, create_engine

from askdata import config


def _make_engine(env_name: str) -> Engine:
    # The pooler closes idle connections, so check each one before use.
    return create_engine(
        config.require_env(env_name),
        pool_size=config.DB_POOL_SIZE,
        max_overflow=0,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def reader_engine() -> Engine:
    """Engine for the read-only role that runs generated queries."""
    return _make_engine("READER_DB_URL")


@lru_cache(maxsize=1)
def feedback_engine() -> Engine:
    """Engine for the role that can only write to app.feedback."""
    return _make_engine("FEEDBACK_DB_URL")
