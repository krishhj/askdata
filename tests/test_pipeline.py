import json

import pandas as pd
import pytest

from askdata import config, pipeline
from askdata.executor import QueryError
from askdata.generator import RateLimitedError, numbers_are_supported, summarise

GOOD_SQL = "SELECT order_status, COUNT(*) AS orders FROM orders GROUP BY 1"
RESULT = pd.DataFrame({"order_status": ["delivered", "canceled"], "orders": [96478, 625]})


class FakeLLM:
    """Returns the given replies in order and records every prompt it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return (reply if isinstance(reply, str) else json.dumps(reply)), 10


def sql_reply(sql):
    return {"sql": sql, "explanation": "Counts orders.", "assumptions": ["All statuses"]}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Run without the embedding model or the database."""
    monkeypatch.setattr(config, "USE_TABLE_RETRIEVAL", False)
    monkeypatch.setattr(config, "USE_EXAMPLES", False)
    monkeypatch.setattr(config, "MAX_RETRIES", 2)
    executed = []

    def fake_run_sql(sql):
        executed.append(sql)
        if "no_such_column" in sql:
            raise QueryError('column "no_such_column" does not exist')
        return RESULT.copy()

    monkeypatch.setattr(pipeline, "run_sql", fake_run_sql)
    return executed


def ask(llm, **kwargs):
    return pipeline.answer_question("How many orders per status?", llm=llm, with_summary=False, **kwargs)


def test_bad_sql_then_good_sql_is_retried(offline):
    llm = FakeLLM(sql_reply("SELECT no_such_column FROM orders"), sql_reply(GOOD_SQL))
    answer = ask(llm)
    assert answer.attempts == 2
    assert answer.error is None
    assert answer.df.equals(RESULT)
    retry_prompt = llm.calls[1][1]["content"]
    assert "no_such_column" in retry_prompt and "does not exist" in retry_prompt


def test_unknown_table_is_retried_with_the_reason():
    llm = FakeLLM(sql_reply("SELECT * FROM users"), sql_reply(GOOD_SQL))
    answer = ask(llm)
    assert answer.attempts == 2 and answer.df is not None
    assert "Table users does not exist" in llm.calls[1][1]["content"]


def test_unsafe_sql_stops_without_retry(offline):
    llm = FakeLLM(sql_reply("DELETE FROM orders"), sql_reply(GOOD_SQL))
    answer = ask(llm)
    assert answer.attempts == 1
    assert answer.blocked_reason
    assert answer.df is None
    assert len(llm.calls) == 1
    assert offline == []


def test_declined_question_stops_without_retry():
    llm = FakeLLM({"sql": "", "explanation": "I can only read data.", "assumptions": []})
    answer = ask(llm)
    assert answer.attempts == 1
    assert answer.error == "I can only read data."


def test_gives_up_after_max_retries():
    llm = FakeLLM(*[sql_reply("SELECT no_such_column FROM orders")] * 3)
    answer = ask(llm)
    assert answer.attempts == 1 + config.MAX_RETRIES
    assert "no_such_column" in answer.error
    assert answer.df is None


def test_no_retries_when_max_retries_is_zero(monkeypatch):
    monkeypatch.setattr(config, "MAX_RETRIES", 0)
    llm = FakeLLM(sql_reply("SELECT no_such_column FROM orders"), sql_reply(GOOD_SQL))
    assert ask(llm).attempts == 1


def test_invalid_json_is_retried():
    llm = FakeLLM("not json at all", sql_reply(GOOD_SQL))
    answer = ask(llm)
    assert answer.attempts == 2 and answer.df is not None
    assert "valid JSON" in llm.calls[1][1]["content"]


def test_zero_rows_is_a_valid_answer(monkeypatch):
    monkeypatch.setattr(pipeline, "run_sql", lambda sql: RESULT.iloc[0:0])
    answer = ask(FakeLLM(sql_reply(GOOD_SQL)))
    assert answer.attempts == 1 and answer.error is None
    assert answer.df.empty


def test_rate_limit_gives_friendly_message():
    answer = ask(FakeLLM(RateLimitedError("429")))
    assert answer.error == pipeline.RATE_LIMIT_MESSAGE


def test_validated_sql_is_what_runs(offline):
    answer = ask(FakeLLM(sql_reply(GOOD_SQL)))
    assert offline == [answer.sql]
    assert f"LIMIT {config.ROW_LIMIT}" in answer.sql


def test_history_is_sent_to_the_model():
    llm = FakeLLM(sql_reply(GOOD_SQL))
    history = [{"question": f"q{i}", "sql": f"SELECT {i}"} for i in range(4)]
    ask(llm, history=history)
    prompt = llm.calls[0][1]["content"]
    assert "q3" in prompt and "q2" in prompt and "q1" not in prompt


def test_edited_sql_is_validated_and_run(offline):
    answer = pipeline.run_user_sql(GOOD_SQL)
    assert answer.df is not None and answer.chart is not None
    assert offline == [answer.sql]


def test_unsafe_edited_sql_is_blocked(offline):
    answer = pipeline.run_user_sql("DROP TABLE orders")
    assert answer.blocked_reason and answer.df is None
    assert offline == []


def test_feedback_failure_returns_false(monkeypatch):
    def broken_engine():
        raise pipeline.SQLAlchemyError("down")

    monkeypatch.setattr(pipeline, "feedback_engine", broken_engine)
    assert pipeline.save_feedback("q", "SELECT 1", 1) is False


def test_summary_and_chart_are_added():
    llm = FakeLLM(sql_reply(GOOD_SQL), {"summary": "Most orders, 96478, were delivered."})
    answer = pipeline.answer_question("How many orders per status?", llm=llm)
    assert answer.summary == "Most orders, 96478, were delivered."
    assert answer.chart is not None
    assert answer.tokens == 20


def test_summary_with_invented_number_falls_back():
    llm = FakeLLM({"summary": "Revenue grew 37% last year."})
    summary, _ = summarise("Orders per status?", RESULT, llm)
    assert summary == "The query returned 2 rows."


def test_summary_of_single_value_falls_back_to_the_value():
    llm = FakeLLM("broken")
    summary, _ = summarise("How many orders?", pd.DataFrame({"n": [99441]}), llm)
    assert summary == "The answer is 99441."


def test_summary_of_empty_result_skips_the_model():
    llm = FakeLLM()
    summary, tokens = summarise("Any orders in 2030?", RESULT.iloc[0:0], llm)
    assert "No rows" in summary and tokens == 0 and llm.calls == []


@pytest.mark.parametrize(
    "summary, ok",
    [
        ("Delivered orders total 96,478.", True),
        ("Revenue was 13591643.7.", True),
        ("Revenue was 13.6 thousand.", False),
        ("The late share was 8.1%.", True),
        ("The late share was 9%.", False),
        ("There were no numbers here.", True),
    ],
)
def test_numbers_are_supported(summary, ok):
    source = "order_status,orders,revenue,late_pct\ndelivered,96478,13591643.7,8.11"
    assert numbers_are_supported(summary, source) is ok
