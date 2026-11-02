import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
DIFFICULTIES = ["easy", "medium", "hard"]

st.set_page_config(page_title="AskData evaluation", page_icon=":material/fact_check:")


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def ablation_table(setups: dict) -> pd.DataFrame:
    rows = []
    for name, s in setups.items():
        rows.append({
            "Setup": name,
            "What it adds": s["label"],
            "Accuracy": pct(s["overall"]),
            **{d.capitalize(): pct(s["by_difficulty"].get(d)) for d in DIFFICULTIES},
            "Valid SQL": pct(s["valid_sql_rate"]),
            "First try": pct(s["first_try_accuracy"]),
            "Retry rescue": pct(s["retry_rescue_rate"]),
            "Median s": round(s["median_seconds"], 2),
            "p95 s": round(s["p95_seconds"], 2),
            "Avg tokens": round(s["avg_tokens"]),
        })
    return pd.DataFrame(rows)


def render_headline(summary: dict) -> None:
    setups = summary["setups"]
    best = setups.get("E") or setups[sorted(setups)[-1]]
    st.subheader(f"Full pipeline: {best['label']}")
    cols = st.columns(4)
    cols[0].metric("Accuracy", pct(best["overall"]), border=True)
    cols[1].metric("First-try accuracy", pct(best["first_try_accuracy"]), border=True)
    cols[2].metric("Valid SQL", pct(best["valid_sql_rate"]), border=True)
    cols[3].metric("Median latency", f"{best['median_seconds']:.1f} s", border=True)

    attacks = summary.get("attacks")
    if attacks:
        stopped = attacks.get("blocked", 0) + attacks.get("declined", 0)
        st.metric(
            "Attacks that changed any data",
            f"{attacks['harmful']} of {attacks['total']}",
            help=f"{stopped} were blocked or declined; the rest ran as read-only queries.",
            border=True,
        )


def render_charts(setups: dict) -> None:
    overall = pd.DataFrame(
        [{"Setup": f"{n}: {s['label']}", "Accuracy": s["overall"] * 100} for n, s in setups.items()]
    )
    fig = px.bar(overall, x="Setup", y="Accuracy", text_auto=".1f", range_y=[0, 100])
    fig.update_layout(height=360, margin={"l": 0, "r": 0, "t": 10, "b": 0}, xaxis_title=None)
    st.plotly_chart(fig, config={"displayModeBar": False})

    st.subheader("Accuracy by difficulty")
    by_difficulty = pd.DataFrame(
        [{"Setup": n, "Difficulty": d, "Accuracy": s["by_difficulty"].get(d, 0) * 100}
         for n, s in setups.items() for d in DIFFICULTIES]
    )
    fig = px.bar(by_difficulty, x="Setup", y="Accuracy", color="Difficulty",
                 barmode="group", range_y=[0, 100])
    fig.update_layout(height=360, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, config={"displayModeBar": False})


def render_failures(setups: dict) -> None:
    st.subheader("Questions the model got wrong")
    name = st.selectbox("Setup", list(setups), index=len(setups) - 1)
    path = RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        st.caption(f"No results file for setup {name}.")
        return
    results = pd.read_csv(path)
    gold = pd.DataFrame(yaml.safe_load((ROOT / "eval" / "gold.yaml").read_text(encoding="utf-8")))
    failed = results[~results["correct"]].merge(gold[["id", "question", "gold_sql"]], on="id")
    if failed.empty:
        st.success("Every question was answered correctly in this setup.")
        return
    st.dataframe(
        failed[["id", "difficulty", "question", "gold_sql", "generated_sql", "error"]],
        hide_index=True,
    )


def main() -> None:
    st.title("Evaluation")
    if not SUMMARY_PATH.exists():
        st.info(
            "No evaluation results yet. Verify every entry in eval/gold.yaml, then run "
            "`python -m eval.run_eval` from the project folder."
        )
        return

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    run = summary.get("run", {})
    st.caption(
        f"{run.get('gold_questions', '?')} gold questions, model {run.get('model', '?')}, "
        f"run finished {run.get('finished_at', '?')}. A result counts as correct when it returns "
        "the same rows as the gold query."
    )
    setups = dict(sorted(summary["setups"].items()))
    render_headline(summary)

    st.subheader("Ablation: adding one feature at a time")
    st.dataframe(ablation_table(setups), hide_index=True)
    render_charts(setups)
    render_failures(setups)

    attacks_path = RESULTS_DIR / "attacks.csv"
    if attacks_path.exists():
        with st.expander("Attack results", icon=":material/shield:"):
            st.dataframe(pd.read_csv(attacks_path), hide_index=True)


main()
