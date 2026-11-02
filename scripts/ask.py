import logging
import sys

from askdata.pipeline import answer_question


def main() -> int:
    if len(sys.argv) != 2:
        print('Usage: python -m scripts.ask "your question"')
        return 1
    logging.basicConfig(level=logging.WARNING)
    # The Windows console defaults to cp1252, which cannot print some characters models use.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    answer = answer_question(sys.argv[1])
    print(f"SQL:\n{answer.sql}\n")
    print(f"Explanation: {answer.explanation}")
    print(f"Assumptions: {answer.assumptions}")
    print(f"Attempts: {answer.attempts}, seconds: {answer.seconds}, tokens: {answer.tokens}\n")
    if answer.blocked_reason:
        print(f"Blocked: {answer.blocked_reason}")
    elif answer.error:
        print(f"Error: {answer.error}")
    else:
        print(answer.df.head(20).to_string())
        print(f"\n{len(answer.df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
