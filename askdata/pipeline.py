import logging
import time
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from askdata import config
from askdata.charts import pick_chart
from askdata.db import feedback_engine
from askdata.executor import QueryError, run_sql
from askdata.generator import (
    LLM,
    GenerationError,
    RateLimitedError,
    generate_sql,
    groq_llm,
    summarise,
)
from askdata.retrieval import retrieve_examples, retrieve_tables
from askdata.schema import schema_prompt, table_names
from askdata.validator import UNSAFE, validate

logger = logging.getLogger(__name__)

RATE_LIMIT_MESSAGE = "Too many requests, please try again in a minute."


@dataclass
class Answer:
    question: str
    sql: str | None = None
    explanation: str = ""
    assumptions: list[str] = field(default_factory=list)
    df: pd.DataFrame | None = None
    chart: object | None = None
    summary: str = ""
    attempts: int = 0
    seconds: float = 0.0
    blocked_reason: str | None = None
    error: str | None = None
    tokens: int = 0


def _context(question: str) -> tuple[str, list[dict]]:
    tables = retrieve_tables(question) if config.USE_TABLE_RETRIEVAL else table_names()
    schema_text = schema_prompt(tables, with_notes=config.USE_SCHEMA_NOTES)
    examples = retrieve_examples(question) if config.USE_EXAMPLES else []
    return schema_text, examples


def _attempt(answer: Answer, schema_text: str, examples: list[dict], history: list[dict],
             llm: LLM, previous_sql: str | None, error: str | None) -> bool:
    # One generate, validate and execute pass. Returns True if a retry could help.
    answer.attempts += 1
    answer.error = None
    try:
        reply = generate_sql(answer.question, schema_text, examples, history, llm, previous_sql, error)
    except GenerationError as exc:
        answer.error = str(exc)
        return True

    answer.tokens += reply["tokens"]
    answer.sql, answer.explanation = reply["sql"], reply["explanation"]
    answer.assumptions = reply["assumptions"]
    if not answer.sql:
        # The model declined, for example because the question asks to change data.
        answer.error = answer.explanation or "This question cannot be answered from the data."
        return False
    return _validate_and_run(answer, answer.sql)


def _validate_and_run(answer: Answer, sql: str) -> bool:
    # Returns True if a retry could help.
    ok, cleaned_sql, category, reason = validate(sql)
    if not ok:
        if category == UNSAFE:
            answer.blocked_reason = reason
            return False
        answer.error = reason
        return True

    answer.sql = cleaned_sql
    try:
        answer.df = run_sql(cleaned_sql)
    except QueryError as exc:
        answer.error = str(exc)
        return True
    return False


def _generate_with_retries(answer: Answer, history: list[dict], llm: LLM) -> None:
    schema_text, examples = _context(answer.question)
    for _ in range(1 + config.MAX_RETRIES):
        # The failed SQL and its error are fed back so the next attempt can fix them.
        failed_sql, error = answer.sql, answer.error
        if not _attempt(answer, schema_text, examples, history, llm, failed_sql, error):
            return
        logger.info("Attempt %d failed: %s", answer.attempts, answer.error)


def answer_question(question: str, history: list[dict] | None = None, llm: LLM = groq_llm,
                    with_summary: bool = True) -> Answer:
    """Turn a question into SQL, check it, run it and return everything as an Answer."""
    start = time.perf_counter()
    answer = Answer(question=question)
    try:
        _generate_with_retries(answer, history or [], llm)
        if answer.df is not None:
            answer.chart = pick_chart(answer.df)
            if with_summary:
                answer.summary, tokens = summarise(question, answer.df, llm)
                answer.tokens += tokens
    except RateLimitedError:
        answer.error = RATE_LIMIT_MESSAGE
    answer.seconds = round(time.perf_counter() - start, 2)
    if answer.error or answer.blocked_reason:
        logger.warning("Question failed: %s | %s", question, answer.error or answer.blocked_reason)
    return answer


def run_user_sql(sql: str, question: str = "Edited SQL") -> Answer:
    """Run SQL the user edited, through the same validator as generated SQL."""
    start = time.perf_counter()
    answer = Answer(question=question, sql=sql, attempts=1)
    _validate_and_run(answer, sql)
    if answer.df is not None:
        answer.chart = pick_chart(answer.df)
    answer.seconds = round(time.perf_counter() - start, 2)
    return answer


def save_feedback(question: str, sql: str | None, rating: int) -> bool:
    """Store a rating (1 up, -1 down) for an answer. Returns False if it could not be saved."""
    # Bound parameters, never string formatting, because question is user text.
    statement = text("INSERT INTO app.feedback (question, sql, rating) VALUES (:question, :sql, :rating)")
    try:
        with feedback_engine().begin() as conn:
            conn.execute(statement, {"question": question, "sql": sql, "rating": rating})
    except SQLAlchemyError:
        logger.exception("Could not save feedback")
        return False
    return True
