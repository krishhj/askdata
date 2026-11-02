import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from askdata import config
from askdata.executor import run_sql
from askdata.pipeline import RATE_LIMIT_MESSAGE, Answer, answer_question
from askdata.validator import validate
from eval.compare import results_match

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# Each setup adds one feature to the one before, so differences can be attributed.
SETUPS = {
    "A": {"label": "All tables, names only", "USE_SCHEMA_NOTES": False, "USE_EXAMPLES": False,
          "MAX_RETRIES": 0, "USE_TABLE_RETRIEVAL": False},
    "B": {"label": "+ schema notes", "USE_SCHEMA_NOTES": True, "USE_EXAMPLES": False,
          "MAX_RETRIES": 0, "USE_TABLE_RETRIEVAL": False},
    "C": {"label": "+ examples", "USE_SCHEMA_NOTES": True, "USE_EXAMPLES": True,
          "MAX_RETRIES": 0, "USE_TABLE_RETRIEVAL": False},
    "D": {"label": "+ retries", "USE_SCHEMA_NOTES": True, "USE_EXAMPLES": True,
          "MAX_RETRIES": 2, "USE_TABLE_RETRIEVAL": False},
    "E": {"label": "+ table retrieval", "USE_SCHEMA_NOTES": True, "USE_EXAMPLES": True,
          "MAX_RETRIES": 2, "USE_TABLE_RETRIEVAL": True},
}
TABLES = sorted(config.ALLOWED_TABLES)


def load_gold() -> list[dict]:
    gold = yaml.safe_load((EVAL_DIR / "gold.yaml").read_text(encoding="utf-8"))
    unverified = [g["id"] for g in gold if g.get("verified") is not True]
    if unverified:
        sys.exit(f"Refusing to run: {len(unverified)} gold entries are not verified: {', '.join(unverified)}")
    return gold


def gold_results(gold: list[dict]) -> dict[str, pd.DataFrame]:
    """Run every gold query once, through the same validator and LIMIT as the app."""
    results = {}
    for g in gold:
        ok, sql, _, reason = validate(g["gold_sql"])
        if not ok:
            sys.exit(f"Gold query {g['id']} fails validation: {reason}")
        results[g["id"]] = run_sql(sql)
    return results


def apply_setup(name: str) -> None:
    for key, value in SETUPS[name].items():
        if key != "label":
            setattr(config, key, value)


def ask(question: str) -> Answer:
    """Answer one question, waiting and trying again if Groq rate limits us."""
    for _ in range(config.EVAL_RATE_LIMIT_TRIES):
        answer = answer_question(question, with_summary=False)
        time.sleep(config.EVAL_SLEEP_SECONDS)
        if answer.error != RATE_LIMIT_MESSAGE:
            return answer
        print(f"  rate limited, waiting {config.EVAL_RATE_LIMIT_WAIT_SECONDS} s")
        time.sleep(config.EVAL_RATE_LIMIT_WAIT_SECONDS)
    # Stop rather than score an unanswered question as wrong. On the free tier
    # this is usually the daily token limit, which clears over the next day.
    sys.exit(
        f"Still rate limited on {question!r}, probably Groq's daily token limit. "
        "Progress is saved; run the same command again later to continue."
    )


def done_ids(path: Path) -> set[str]:
    return set(pd.read_csv(path)["id"]) if path.exists() else set()


def append_row(path: Path, row: dict) -> None:
    # One row at a time, so a run stopped by the daily limit loses nothing.
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def evaluate_setup(name: str, gold: list[dict], expected: dict[str, pd.DataFrame], path: Path) -> pd.DataFrame:
    """Answer every gold question not yet in path, then return all rows for the setup."""
    apply_setup(name)
    finished = done_ids(path)
    for i, g in enumerate(gold, 1):
        if g["id"] in finished:
            continue
        answer = ask(g["question"])
        correct = answer.df is not None and results_match(expected[g["id"]], answer.df, g["ordered"])
        append_row(path, {
            "id": g["id"],
            "difficulty": g["difficulty"],
            "correct": correct,
            "attempts": answer.attempts,
            "seconds": answer.seconds,
            "tokens": answer.tokens,
            "generated_sql": answer.sql,
            "error": answer.blocked_reason or answer.error,
            "executed": answer.df is not None,
        })
        print(f"  {name} {i}/{len(gold)} {g['id']}: {'correct' if correct else 'wrong'}")
    return pd.read_csv(path)


def setup_metrics(df: pd.DataFrame) -> dict:
    retried = df[df["attempts"] > 1]
    return {
        "questions": len(df),
        "overall": df["correct"].mean(),
        "by_difficulty": df.groupby("difficulty")["correct"].mean().to_dict(),
        "valid_sql_rate": df["executed"].mean(),
        "first_try_accuracy": (df["correct"] & (df["attempts"] == 1)).mean(),
        # Of the questions whose first attempt failed, the share that retries got right.
        "retry_rescue_rate": retried["correct"].mean() if len(retried) else None,
        "median_seconds": df["seconds"].median(),
        "p95_seconds": df["seconds"].quantile(0.95),
        "avg_tokens": df["tokens"].mean(),
    }


def table_counts() -> dict[str, int]:
    return {t: int(run_sql(f"SELECT COUNT(*) AS n FROM {t}")["n"].iloc[0]) for t in TABLES}


def attack_outcome(answer: Answer) -> str:
    if answer.blocked_reason:
        return "blocked"
    if answer.sql == "":
        return "declined"
    if answer.df is not None:
        return "answered_read_only"
    return "failed"


def run_attacks(path: Path) -> dict:
    """Send every attack not yet in path through the full pipeline; check no table changed."""
    apply_setup("E")
    attacks = yaml.safe_load((EVAL_DIR / "attacks.yaml").read_text(encoding="utf-8"))
    finished = done_ids(path)
    for attack in attacks:
        if attack["id"] in finished:
            continue
        before = table_counts()
        answer = ask(attack["input"])
        outcome = attack_outcome(answer)
        append_row(path, {
            "id": attack["id"],
            "kind": attack["kind"],
            "outcome": outcome,
            "harmful": table_counts() != before,
            "generated_sql": answer.sql,
            "reason": answer.blocked_reason or answer.error,
        })
        print(f"  attack {attack['id']}: {outcome}")
    df = pd.read_csv(path)
    counts = df["outcome"].value_counts().to_dict()
    return {"total": len(df), "harmful": int(df["harmful"].sum()), **counts}


def load_summary(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"setups": {}}


def to_json(value):
    # numpy numbers are not JSON serialisable; .item() turns them into plain Python ones.
    return value.item() if hasattr(value, "item") else str(value)


def save_summary(summary: dict, path: Path, gold_count: int) -> None:
    summary["run"] = {
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": config.GROQ_MODEL,
        "gold_questions": gold_count,
        "allow_extra_columns": config.EVAL_ALLOW_EXTRA_COLUMNS,
    }
    path.write_text(json.dumps(summary, indent=2, default=to_json), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AskData evaluation.")
    parser.add_argument("--setups", default="ABCDE", help="Setups to run, for example ABCDE or E")
    parser.add_argument("--limit", type=int, help="Only the first N gold questions, for a cheap trial run")
    parser.add_argument("--skip-attacks", action="store_true")
    args = parser.parse_args()

    gold = load_gold()[: args.limit]
    expected = gold_results(gold)
    # Trial runs go to their own folder so they can never be mistaken for real results.
    out_dir = RESULTS_DIR / "trial" if args.limit else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    # Setups can be run separately (for rate limits), so earlier results are kept and merged.
    summary = load_summary(summary_path)

    # Questions already in a setup's CSV are skipped, so the same command resumes a stopped run.
    for name in args.setups.upper():
        print(f"Setup {name}: {SETUPS[name]['label']}")
        df = evaluate_setup(name, gold, expected, out_dir / f"{name}.csv")
        summary["setups"][name] = {"label": SETUPS[name]["label"], **setup_metrics(df)}
        save_summary(summary, summary_path, len(gold))

    if not args.skip_attacks:
        print("Attacks")
        summary["attacks"] = run_attacks(out_dir / "attacks.csv")
        save_summary(summary, summary_path, len(gold))

    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
