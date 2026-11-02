import logging

import pandas as pd
import streamlit as st

from askdata.charts import chart_kind
from askdata.db import feedback_engine, reader_engine
from askdata.pipeline import (
    RATE_LIMIT_MESSAGE,
    Answer,
    answer_question,
    run_user_sql,
    save_feedback,
)
from askdata.retrieval import retrieve_examples, retrieve_tables
from askdata.schema import load_schema

st.set_page_config(page_title="AskData", page_icon=":material/query_stats:")
logging.basicConfig(level=logging.INFO)

SAMPLE_QUESTIONS = [
    "What were total sales by month in 2018?",
    "Which 10 product categories earned the most revenue?",
    "Which customer states have the highest share of late deliveries?",
    "How many orders did each payment method receive?",
    "Who are the top 5 sellers by number of items sold?",
    "How many orders were placed last month?",
]
AVATARS = {"user": ":material/person:", "assistant": ":material/query_stats:"}


@st.cache_resource(show_spinner="Loading the embedding model and database connections...")
def warm_up() -> bool:
    # The package already caches these per process. Calling them here loads
    # everything once at start-up, behind a spinner, not on the first question.
    retrieve_tables("warm up")
    retrieve_examples("warm up")
    reader_engine()
    feedback_engine()
    return True


def history() -> list[dict]:
    """Earlier successful questions and their SQL, for follow-up questions."""
    return [
        {"question": m["answer"].question, "sql": m["answer"].sql}
        for m in st.session_state.messages
        if m["role"] == "assistant" and m["answer"].df is not None
    ]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### :material/query_stats: AskData")
        st.write("Ask questions in plain English about 100,000 orders from Olist, a Brazilian online marketplace.")
        st.caption("Data covers September 2016 to October 2018.")

        st.markdown("**Try a question**")
        for question in SAMPLE_QUESTIONS:
            if st.button(question, width="stretch"):
                st.session_state.pending = question

        st.divider()
        with st.expander("Browse the schema", icon=":material/table_chart:"):
            tables = load_schema()["tables"]
            name = st.selectbox("Table", list(tables))
            st.caption(tables[name]["description"])
            st.dataframe(
                [{"column": col, "type": info["type"], "description": info["description"]}
                 for col, info in tables[name]["columns"].items()],
                hide_index=True,
            )

        if st.button("Clear chat", icon=":material/delete:", width="stretch"):
            # Widget keys are numbered by message, so old ones would attach to new answers.
            st.session_state.clear()
            st.rerun()


def on_starter() -> None:
    # A callback runs before the rerun, so the welcome card is already gone
    # when the answer is drawn.
    st.session_state.pending = st.session_state.starter
    st.session_state.starter = None


def render_welcome() -> None:
    with st.container(border=True):
        st.markdown("#### What would you like to know?")
        st.write(
            "Ask a business question and I will write the SQL, run it on a read-only copy "
            "of the data, and show you the result with a chart. You can always see and edit the query."
        )
        st.pills("Start with one of these", SAMPLE_QUESTIONS[:4], key="starter", on_change=on_starter)


def column_formats(df: pd.DataFrame) -> dict:
    """Readable dates and thousands separators for the result table."""
    formats = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            has_time = (df[col].dropna().dt.normalize() != df[col].dropna()).any()
            formats[col] = st.column_config.DatetimeColumn(
                format="YYYY-MM-DD HH:mm" if has_time else "YYYY-MM-DD"
            )
        elif pd.api.types.is_numeric_dtype(df[col]) and not col.lower().endswith("year"):
            # "localized" adds thousands separators; a year like 2018 should stay plain.
            formats[col] = st.column_config.NumberColumn(format="localized")
    return formats


def render_result(answer: Answer, key: str) -> None:
    """Chart or metric, table, CSV download for a successful answer."""
    df = answer.df
    if chart_kind(df) == "metric":
        st.metric(df.columns[0].replace("_", " ").capitalize(), f"{df.iat[0, 0]:,}", border=True)
    elif answer.chart is not None:
        st.plotly_chart(answer.chart, key=f"chart_{key}", config={"displayModeBar": False})
    st.dataframe(df, hide_index=True, column_config=column_formats(df))
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="askdata_result.csv",
        mime="text/csv",
        on_click="ignore",
        icon=":material/download:",
        type="tertiary",
        key=f"csv_{key}",
    )


def render_problem(answer: Answer) -> bool:
    """Show a block or failure message. Returns True if there was one."""
    if answer.blocked_reason:
        st.error(f"I can't run that query. {answer.blocked_reason}", icon=":material/block:")
    elif answer.error == RATE_LIMIT_MESSAGE:
        st.warning(RATE_LIMIT_MESSAGE, icon=":material/hourglass_top:")
    elif answer.error and answer.sql == "":
        # An empty string, not None, means the model declined and explained why.
        st.info(answer.error, icon=":material/info:")
    elif answer.error:
        st.info("Sorry, I couldn't answer that one. Could you try rephrasing the question?",
                icon=":material/help:")
    return bool(answer.blocked_reason or answer.error)


def render_details(answer: Answer, i: int) -> None:
    with st.expander("How I got this", icon=":material/code:"):
        if answer.sql:
            st.code(answer.sql, language="sql")
        if answer.explanation:
            st.write(answer.explanation)
        if answer.assumptions:
            st.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in answer.assumptions))
        if answer.error:
            st.caption(f"Last error: {answer.error}")
        st.caption(f"Attempts: {answer.attempts}  |  Time: {answer.seconds} s")

        edited = st.text_area("Edit the SQL", value=answer.sql or "", key=f"sql_{i}", height=200)
        if st.button("Run edited SQL", key=f"run_{i}", icon=":material/play_arrow:"):
            st.session_state[f"edited_{i}"] = run_user_sql(edited, question=answer.question)
    edited_answer = st.session_state.get(f"edited_{i}")
    if edited_answer is not None:
        st.caption("Result of your edited SQL")
        if not render_problem(edited_answer):
            render_result(edited_answer, f"edited_{i}")


def on_feedback(i: int) -> None:
    answer = st.session_state.messages[i]["answer"]
    value = st.session_state.get(f"feedback_{i}")
    if value is None:
        return
    # st.feedback gives 1 for thumbs up and 0 for thumbs down.
    if save_feedback(answer.question, answer.sql, 1 if value == 1 else -1):
        st.toast("Thanks for the feedback.", icon=":material/check:")
    else:
        st.toast("Sorry, your feedback could not be saved.", icon=":material/error:")


def render_answer(answer: Answer, i: int) -> None:
    if not render_problem(answer):
        st.write(answer.summary)
        render_result(answer, str(i))
    render_details(answer, i)
    if answer.df is not None:
        st.feedback("thumbs", key=f"feedback_{i}", on_change=on_feedback, args=(i,))


def ask(question: str) -> None:
    previous = history()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.write(question)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Writing and running the query..."):
            answer = answer_question(question, previous)
        st.session_state.messages.append({"role": "assistant", "answer": answer})
        render_answer(answer, len(st.session_state.messages) - 1)


def main() -> None:
    st.title("AskData")
    st.caption("Plain-English questions about Brazilian e-commerce data, answered with SQL you can check.")
    warm_up()
    st.session_state.setdefault("messages", [])
    render_sidebar()

    if not st.session_state.messages and "pending" not in st.session_state:
        render_welcome()
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_answer(message["answer"], i)

    typed = st.chat_input("Ask a question about the Olist orders")
    question = typed or st.session_state.pop("pending", None)
    if question:
        ask(question)


main()
