import json
import logging
import re
from collections.abc import Callable
from functools import lru_cache

import groq
import pandas as pd

from askdata import config

logger = logging.getLogger(__name__)

# Takes chat messages and returns (reply text, tokens used). Tests pass a fake.
LLM = Callable[[list[dict]], tuple[str, int]]

SYSTEM_PROMPT = """You turn business questions into one PostgreSQL query over an e-commerce database.

Rules:
- Write exactly one read-only SELECT query (WITH and UNION are fine). Never modify data.
- Use only the tables and columns listed in the schema.
- Give result columns short, readable aliases.
- Round money and averages to 2 decimals.
- For "top" or "most" questions, ORDER BY the measure and LIMIT the rows.
- For trends over time, return one date column such as DATE_TRUNC('month', ...)::date, not separate year and month numbers.
- If the question cannot be answered from this data, or asks to change data, set "sql" to "" and say why in "explanation".

Reply with a JSON object only, with these keys:
- "sql": the query as a string
- "explanation": one or two plain sentences on how the query answers the question
- "assumptions": a list of short strings, each one choice you made where the question was ambiguous"""


class GenerationError(Exception):
    """The model's reply could not be used."""


class RateLimitedError(Exception):
    """The LLM provider is rate limiting us."""


@lru_cache(maxsize=1)
def _groq_client() -> groq.Groq:
    return groq.Groq(api_key=config.require_env("GROQ_API_KEY"))


def groq_llm(messages: list[dict]) -> tuple[str, int]:
    """Call Groq in JSON mode and return (reply text, total tokens)."""
    try:
        response = _groq_client().chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except groq.RateLimitError as exc:
        raise RateLimitedError(str(exc)) from exc
    tokens = response.usage.total_tokens if response.usage else 0
    return response.choices[0].message.content or "", tokens


def _format_examples(examples: list[dict]) -> str:
    return "\n\n".join(
        f"Example question: {ex['question']}\nExample SQL:\n{ex['sql'].strip()}" for ex in examples
    )


def _format_history(history: list[dict]) -> str:
    turns = history[-config.HISTORY_TURNS:]
    return "\n\n".join(f"Earlier question: {t['question']}\nIts SQL:\n{t['sql']}" for t in turns)


def build_messages(
    question: str,
    schema_text: str,
    examples: list[dict],
    history: list[dict],
    previous_sql: str | None = None,
    error: str | None = None,
) -> list[dict]:
    """Assemble the system and user messages for one generation attempt."""
    sections = [f"Schema:\n{schema_text}"]
    if examples:
        sections.append(f"Solved examples:\n{_format_examples(examples)}")
    if history:
        # Lets follow-ups like "now split that by state" refer back.
        sections.append(f"Conversation so far:\n{_format_history(history)}")
    # Labelled differently from the examples, so the model does not read it as another one.
    sections.append(f"Write the query for this question:\n{question}")
    if error is not None:
        failed = f"Query:\n{previous_sql}\n" if previous_sql else ""
        sections.append(
            f"Your previous attempt failed.\n{failed}Error: {error}\nFix the problem and try again."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def parse_reply(text: str) -> dict:
    """Turn the model's JSON reply into a dict with sql, explanation and assumptions."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"The model did not return valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("sql"), str):
        raise GenerationError("The model's reply has no sql string.")

    assumptions = data.get("assumptions") or []
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    return {
        "sql": data["sql"].strip(),
        "explanation": str(data.get("explanation") or "").strip(),
        "assumptions": [str(a) for a in assumptions],
    }


def generate_sql(question: str, schema_text: str, examples: list[dict], history: list[dict],
                 llm: LLM, previous_sql: str | None = None, error: str | None = None) -> dict:
    """Ask the LLM for a query. Returns sql, explanation, assumptions and tokens."""
    messages = build_messages(question, schema_text, examples, history, previous_sql, error)
    text, tokens = llm(messages)
    reply = parse_reply(text)
    reply["tokens"] = tokens
    logger.info("Generated SQL: %s", reply["sql"])
    return reply


SUMMARY_PROMPT = """You write a one-sentence answer to a business question from a query result.

Rules:
- One plain sentence that directly answers the question.
- Use only numbers that appear in the result rows or the question. Copy them as they appear; you may round to fewer decimals and add thousands separators, as in 924,645.
- Keep it short. For many rows, mention the highest and lowest rather than listing every value.
- Do not calculate new numbers, do not write "million" or similar, and do not add outside knowledge.

Reply with a JSON object only: {"summary": "..."}"""

# Prose may write 96,478, but in CSV text a comma separates values.
PROSE_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
CSV_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers_are_supported(summary: str, source_text: str) -> bool:
    """True if every number in summary appears in source_text, allowing rounding."""
    allowed = [float(n) for n in CSV_NUMBER.findall(source_text)]
    for number in (n.replace(",", "") for n in PROSE_NUMBER.findall(summary)):
        decimals = len(number.split(".")[1]) if "." in number else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(value - float(number)) <= tolerance for value in allowed):
            return False
    return True


def _fallback_summary(df: pd.DataFrame) -> str:
    if df.shape == (1, 1):
        return f"The answer is {df.iat[0, 0]}."
    return f"The query returned {len(df)} rows."


def summarise(question: str, df: pd.DataFrame, llm: LLM) -> tuple[str, int]:
    """One sentence answering question from df, and the tokens used."""
    if df.empty:
        return "No rows matched this question.", 0

    rows = df.head(config.SUMMARY_ROWS).to_csv(index=False)
    source = f"Question: {question}\nTotal rows: {len(df)}\nFirst rows:\n{rows}"
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": source},
    ]
    try:
        text, tokens = llm(messages)
        summary = str(json.loads(text).get("summary", "")).strip()
    except (RateLimitedError, json.JSONDecodeError, AttributeError) as exc:
        logger.warning("Summary failed: %s", exc)
        return _fallback_summary(df), 0

    # A sentence with an invented number is worse than a plain one.
    if not summary or not numbers_are_supported(summary, source):
        logger.warning("Summary rejected: %s", summary)
        return _fallback_summary(df), tokens
    return summary, tokens
